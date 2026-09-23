"""在评测集上跑完整流水线，把每题产出导出为 JSONL，供 Ragas 打分。

主环境只负责"跑"和"导出"，打分交给独立 venv 里的 Ragas（见 score_with_ragas.py），
这样 ragas 的 langchain-core 版本约束不会污染项目依赖。

用法（仓库根目录下执行）：
    python data/eval/run_eval.py --limit 10
    python data/eval/run_eval.py --ids q01,q15,q23
    python data/eval/run_eval.py --limit 50 --output data/eval/runs/runs.jsonl

脚本可重复执行：已跑过的 id 会跳过，方便中断后继续。
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "app"))

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from mult_agents.config import AppConfig  # noqa: E402
from mult_agents.graph import build_app as build_workflow_app  # noqa: E402
from mult_agents.main import build_agents  # noqa: E402
from mult_agents.state import create_initial_state  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs" / "runs.jsonl"
DEGRADE_MARK = "LLM 调用失败，已降级"
# 降级报告的固定开头，用来识别"这份答案是降级产物而非模型正常产出"
FALLBACK_REPORT_MARK = "写作模型本次调用失败"


# 模型的 token 单价（元 / 百万 token），只用于把用量折算成钱。这是促销参考价，
# 会随模型版本和地域变动，以阿里云百炼官网为准；用来比数量级够，别当账单。
PRICE_PER_MILLION = {
    "qwen-plus": (0.8, 2.0),
    "qwen-turbo": (0.3, 0.6),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICE_PER_MILLION.get(model)
    if not price:
        return None
    return input_tokens / 1_000_000 * price[0] + output_tokens / 1_000_000 * price[1]


class TokenCounter(BaseCallbackHandler):
    """累计本轮所有 LLM 调用的 token 用量。

    一轮评测动辄几十次调用，只看单次没有意义；而账单是延迟出账的，
    事后再想核算就对不上了。所以跑的时候就把它攒下来。
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        if not usage:
            # ChatTongyi 把 token_usage 挂在 message.response_metadata 上，
            # 不一定出现在 llm_output 里，两条路都试。
            for generations in getattr(response, "generations", None) or []:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    usage = (getattr(message, "response_metadata", None) or {}).get("token_usage") or {}
                    if usage:
                        break
                if usage:
                    break
        if not usage:
            return
        self.calls += 1
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)


class DegradeCounter(logging.Handler):
    """统计本轮有多少节点因为 LLM 失败走了降级路径，并区分失败原因。

    区分原因是为了决定"要不要重跑"：内容审核（DataInspectionFailed，HTTP 400）
    是 DashScope 对**模型生成内容**的拦截，输入不变时重跑大概率还是被拦，只会把
    同一题的费用乘以重试次数；账户欠费更是全量拒绝。只有网络抖动/超时这类失败
    才值得重跑。
    """

    MODERATION_MARKS = ("DataInspectionFailed", "inappropriate content")
    FATAL_MARKS = ("Arrearage", "欠费")

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0
        self.moderation_count = 0
        self.fatal_count = 0

    def reset(self) -> None:
        self.count = 0
        self.moderation_count = 0
        self.fatal_count = 0

    @property
    def retryable(self) -> bool:
        """本轮降级是否还有重跑价值。

        只有纯网络抖动/超时值得重跑。内容审核（DataInspectionFailed）与欠费是
        确定性的：审核拦的是**模型生成内容**，输入不变时重跑大概率仍被拦；而且
        只要有一处被拦，那个节点的证据就已换成未经 LLM 过滤的原始语料，记录已经
        污染，重跑换不回一条干净样本。实测 q36 单轮 8 次调用里 7 次被拦，若因为
        "还夹着 1 次网络失败" 就去重跑，只会把同一题白烧满重试次数。
        """
        return self.count > 0 and self.moderation_count == 0 and self.fatal_count == 0

    @property
    def block_reason(self) -> str:
        if self.moderation_count:
            return "内容审核拦截"
        if self.fatal_count:
            return "账户欠费/拒绝服务"
        return ""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if DEGRADE_MARK not in message:
            return
        self.count += 1
        if any(mark in message for mark in self.MODERATION_MARKS):
            self.moderation_count += 1
        elif any(mark in message for mark in self.FATAL_MARKS):
            self.fatal_count += 1


def load_eval_set() -> list[dict]:
    items = []
    for line in EVAL_SET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def load_done(output: Path) -> set[str]:
    if not output.exists():
        return set()
    done = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def collect_contexts(result: dict) -> list[str]:
    contexts: list[str] = []
    for key in ("local_evidence", "web_evidence"):
        for entry in result.get(key) or []:
            snippet = str(entry.get("snippet") or "").strip()
            if snippet:
                contexts.append(snippet)
    return contexts


def count_citations(text: str) -> int:
    import re

    return len(re.findall(r"\[(?:WEB|LOC)\d+_\d+-\d+\]", text or ""))


def extract_citation_ids(text: str) -> list[str]:
    import re

    return re.findall(r"\[((?:WEB|LOC)\d+_\d+-\d+)\]", text or "")


def count_search_queries(result: dict) -> tuple[int, int]:
    """统计本轮的检索计划条目数（网页, 本地）。

    注意：这是"计划发出的查询数"，不是"真实计费的 HTTP 次数"。web_search_node
    对每条查询都无条件追加一条 trace（nodes.py:1060），即便 Bocha 未配置、
    返回空结果也照记。所以关闭 Bocha 跑出来的 web_query_count 是计划数而非账单数；
    只有在 Bocha 开启时二者才相等（一条 trace = 一次计费调用）。
    """
    return len(result.get("web_search_trace") or []), len(result.get("local_rag_trace") or [])


def main() -> int:
    parser = argparse.ArgumentParser(description="deep_research eval runner")
    parser.add_argument("--limit", type=int, default=None, help="最多跑多少条")
    parser.add_argument("--ids", type=str, default=None, help="只跑指定 id，逗号分隔")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-degrade",
        type=int,
        default=0,
        help="允许的降级节点数上限，超过则重跑该题（默认 0，即任何降级都重跑）",
    )
    parser.add_argument("--retries", type=int, default=2, help="每题最多重跑次数")
    parser.add_argument("--retry-delay", type=float, default=20.0, help="首次重跑前等待秒数，之后指数退避")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    items = load_eval_set()
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [x for x in items if x["id"] in wanted]
    if args.limit is not None:
        items = items[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)
    pending = [x for x in items if x["id"] not in done]
    print(f"评测集 {len(items)} 条 | 已完成 {len(done)} 条 | 本次待跑 {len(pending)} 条")
    if not pending:
        print("没有待跑条目。")
        return 0

    config = AppConfig.from_file()
    if args.max_iterations is not None:
        config = config.with_overrides(max_iterations=args.max_iterations)
    agents = build_agents(config.model, config.api_key, config)
    app = build_workflow_app(agents, InMemorySaver())

    counter = DegradeCounter()
    logging.getLogger("mult_agents").addHandler(counter)
    tokens = TokenCounter()
    total_input_tokens = 0
    total_output_tokens = 0

    total_elapsed = 0.0
    bypassed_count = 0
    fallback_count = 0
    blocked_count = 0
    with args.output.open("a", encoding="utf-8") as fh:
        for index, item in enumerate(pending, 1):
            # 网络抖动会让某些节点的 LLM 调用失败并降级。降级过的样本答案不完整，
            # 混进评测会污染指标，所以自动重跑；重跑间隔指数退避。
            attempt = 0
            delay = args.retry_delay
            while True:
                counter.reset()
                tokens.reset()
                state = create_initial_state(
                    query=item["question"],
                    max_iterations=config.max_iterations,
                    user_id="eval_user",
                    tenant_id="eval_tenant",
                    memory_context="",
                )
                started = time.time()
                error = None
                try:
                    # callbacks 必须走 config：节点内部是通过 ensure_config() 把
                    # 外层 config 带进 agent.invoke() 的，否则这里挂的计数器收不到用量。
                    result = app.invoke(
                        state,
                        {"configurable": {"thread_id": f"eval_{item['id']}"}, "callbacks": [tokens]},
                    )
                except Exception as exc:
                    result = {}
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.time() - started
                answer = str(result.get("final") or "")
                is_fallback = FALLBACK_REPORT_MARK in answer
                if not is_fallback and counter.count <= args.max_degrade:
                    break
                # 内容审核/欠费这类确定性失败重跑结果不变，直接接受本轮结果并标记，
                # 别把同一题的钱白花在注定失败的重试上。
                if not counter.retryable:
                    print(
                        f"    [跳过重试] {item['id']} 本轮降级由{counter.block_reason}引起，"
                        f"重跑结果不变，接受本轮结果（已标记）"
                    )
                    break
                if attempt >= args.retries:
                    break
                attempt += 1
                reason = "答案是降级报告" if is_fallback else f"{counter.count} 个节点降级"
                print(f"    [重试 {attempt}/{args.retries}] {item['id']} {reason}，{delay:.0f}s 后重跑")
                time.sleep(delay)
                delay *= 2
            total_elapsed += elapsed
            total_input_tokens += tokens.input_tokens
            total_output_tokens += tokens.output_tokens

            cited = extract_citation_ids(answer)
            web_queries, local_queries = count_search_queries(result)
            valid_ids = {
                str(entry.get("source_id") or "").strip()
                for entry in (result.get("source_index") or [])
                if entry.get("source_id")
            }
            record = {
                "id": item["id"],
                "category": item.get("category", ""),
                "question": item["question"],
                "expected_sources": item.get("expected_sources", []),
                "answer": answer,
                "contexts": collect_contexts(result),
                "retrieved_titles": sorted(
                    {str(e.get("title") or "") for e in (result.get("local_evidence") or []) if e.get("title")}
                ),
                "iterations": result.get("iteration"),
                "intent": result.get("intent"),
                "citation_count": len(cited),
                "web_query_count": web_queries,
                "local_query_count": local_queries,
                "llm_calls": tokens.calls,
                "input_tokens": tokens.input_tokens,
                "output_tokens": tokens.output_tokens,
                "cited_ids": cited,
                "valid_source_ids": sorted(valid_ids),
                "invalid_citations": sorted({c for c in cited if c not in valid_ids}),
                "degraded_nodes": counter.count,
                "moderation_blocks": counter.moderation_count,
                "fatal_errors": counter.fatal_count,
                "answer_is_fallback": is_fallback,
                "retries_used": attempt,
                "elapsed_seconds": round(elapsed, 1),
                "error": error,
                "run_at": datetime.now().isoformat(timespec="seconds"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            # 题目被路由到 direct_answer 时不走检索，答案是模型的参数化知识而非语料证据。
            # 这类记录对评测无效，必须响亮标记，别让它伪装成一条正常结果。
            bypassed = record["intent"] == "direct"
            bypassed_count += int(bypassed)
            if bypassed:
                print(
                    f"    [警告] {item['id']} 被路由到 direct_answer，未经过检索流水线，"
                    f"该条对评测无效（题面需要包含调研/分析/对比等研究型措辞）"
                )
            if is_fallback:
                fallback_count += 1
                print(f"    [警告] {item['id']} 最终答案是降级报告，不能用于忠实度/相关性打分")
            if record["moderation_blocks"]:
                blocked_count += 1
                print(
                    f"    [警告] {item['id']} 有 {record['moderation_blocks']} 个节点被 DashScope "
                    f"内容审核拦截并降级，该条证据是未经 LLM 过滤的原始语料"
                )
            status = "ERROR" if error else ("BYPASSED" if bypassed else ("FALLBACK" if is_fallback else "ok"))
            print(
                f"[{index}/{len(pending)}] {item['id']} {status} | "
                f"{elapsed:.0f}s | contexts={len(record['contexts'])} | "
                f"引文={record['citation_count']} | 非法={len(record['invalid_citations'])} | "
                f"轮数={record['iterations']} | 降级={record['degraded_nodes']} | "
                f"查询={web_queries}网页/{local_queries}本地 | "
                f"token={tokens.input_tokens}进/{tokens.output_tokens}出（{tokens.calls}次调用）"
            )

    print(f"\n总耗时 {total_elapsed / 60:.1f} min | 平均 {total_elapsed / len(pending):.0f}s/题")
    total_tokens = total_input_tokens + total_output_tokens
    cost = estimate_cost(config.model, total_input_tokens, total_output_tokens)
    cost_text = f"≈ {cost:.3f} 元" if cost is not None else "（该模型不在单价表内，未折算）"
    print(
        f"token 消耗：输入 {total_input_tokens:,} | 输出 {total_output_tokens:,} | 合计 {total_tokens:,}"
    )
    print(f"折算成本 {cost_text}（模型 {config.model}，按促销参考价估算，不是账单数）")
    if bypassed_count:
        print(
            f"\n[!] {bypassed_count}/{len(pending)} 条被路由到 direct_answer，未经检索流水线。"
            f"\n    这些记录的答案是模型参数化知识，不能用于评测忠实度/检索指标。"
        )
    if fallback_count:
        print(
            f"\n[!] {fallback_count}/{len(pending)} 条最终产出的是降级报告（重试 {args.retries} 次仍失败）。"
            f"\n    多为网络/API 故障所致，建议网络恢复后重跑这些题。"
        )
    if blocked_count:
        print(
            f"\n[!] {blocked_count}/{len(pending)} 条被 DashScope 内容审核拦截过（moderation_blocks > 0）。"
            f"\n    被拦节点的证据是未经 LLM 过滤的原始语料，打分时应单独看待或排除。"
        )
    print(f"结果写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
