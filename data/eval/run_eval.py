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

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from mult_agents.config import AppConfig  # noqa: E402
from mult_agents.graph import build_app as build_workflow_app  # noqa: E402
from mult_agents.main import build_agents  # noqa: E402
from mult_agents.state import create_initial_state  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs" / "runs.jsonl"
DEGRADE_MARK = "LLM 调用失败，已降级"


class DegradeCounter(logging.Handler):
    """统计本轮有多少节点因为 LLM 失败走了降级路径。"""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if DEGRADE_MARK in record.getMessage():
            self.count += 1


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


def main() -> int:
    parser = argparse.ArgumentParser(description="deep_research eval runner")
    parser.add_argument("--limit", type=int, default=None, help="最多跑多少条")
    parser.add_argument("--ids", type=str, default=None, help="只跑指定 id，逗号分隔")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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

    total_elapsed = 0.0
    bypassed_count = 0
    with args.output.open("a", encoding="utf-8") as fh:
        for index, item in enumerate(pending, 1):
            counter.reset()
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
                result = app.invoke(state, {"configurable": {"thread_id": f"eval_{item['id']}"}})
            except Exception as exc:
                result = {}
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.time() - started
            total_elapsed += elapsed

            answer = str(result.get("final") or "")
            cited = extract_citation_ids(answer)
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
                "cited_ids": cited,
                "valid_source_ids": sorted(valid_ids),
                "invalid_citations": sorted({c for c in cited if c not in valid_ids}),
                "degraded_nodes": counter.count,
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
            status = "ERROR" if error else ("BYPASSED" if bypassed else "ok")
            print(
                f"[{index}/{len(pending)}] {item['id']} {status} | "
                f"{elapsed:.0f}s | contexts={len(record['contexts'])} | "
                f"引文={record['citation_count']} | 非法={len(record['invalid_citations'])} | "
                f"轮数={record['iterations']} | 降级={record['degraded_nodes']}"
            )

    print(f"\n总耗时 {total_elapsed / 60:.1f} min | 平均 {total_elapsed / len(pending):.0f}s/题")
    if bypassed_count:
        print(
            f"\n[!] {bypassed_count}/{len(pending)} 条被路由到 direct_answer，未经检索流水线。"
            f"\n    这些记录的答案是模型参数化知识，不能用于评测忠实度/检索指标。"
        )
    print(f"结果写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
