"""对 run_eval.py 导出的结果打分。

指标分两类：
  一、纯计算（不花 LLM）
      - 检索命中率：ground truth 文档是否出现在本轮检索结果里
      - 引用合规率：正文引用的 source_id 是否都在合法列表内
      - 降级率：本轮有多少节点因 LLM 失败走了降级
  二、LLM-as-judge
      - 忠实度：正文抽出的可验证断言中，有多少能被检索证据支撑
      - 相关性：报告是否真正回答了问题

自建 judge 而非引入 Ragas，是为了避开 langchain-core 1.x 的版本冲突，
也让每个指标的口径都可以逐条解释。

用法（仓库根目录下执行）：
    python data/eval/score_runs.py
    python data/eval/score_runs.py --input data/eval/runs/runs.jsonl --limit 5
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "app"))

from langchain_community.chat_models import ChatTongyi  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from mult_agents.config import AppConfig  # noqa: E402

DEFAULT_INPUT = Path(__file__).resolve().parent / "runs" / "runs.jsonl"

CLAIM_PROMPT = """从下面的研究报告正文中，抽取所有【可验证的事实性断言】。

规则：
- 只抽取含具体事实、数值、占比、时间、机制、因果的句子
- 跳过纯观点、过渡句、标题、目录
- 每条断言必须能脱离上下文独立理解
- 最多抽取 {max_claims} 条，优先抽最核心的

只输出 JSON：{{"claims": ["断言1", "断言2"]}}

正文：
{answer}"""

JUDGE_PROMPT = """你是严格的证据核查员。判断每条【待核查断言】能否被【证据片段】支撑。

判定标准：
- supported：证据中有明确依据直接支持该断言
- partial：证据只覆盖断言的一部分，或表述更弱
- unsupported：证据中找不到依据，或与证据矛盾

只输出 JSON：{{"results": [{{"claim": "断言原文", "verdict": "supported|partial|unsupported"}}]}}

证据片段：
{contexts}

待核查断言：
{claims}"""

RELEVANCE_PROMPT = """你在评估一份研究报告是否回答了用户的问题。

评分维度：
- 是否正面回答了用户问题（而不是泛泛而谈）
- 关键子问题是否都被覆盖
- 有无答非所问的偏题内容

只输出 JSON：{{"score": 0到100的整数, "reason": "一句话理由"}}

用户问题：{question}

报告：
{answer}"""


def extract_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text or "")
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except Exception:
            return {}
    return {}


def build_judge(config: AppConfig) -> ChatTongyi:
    return ChatTongyi(
        model=config.model,
        temperature=0.0,
        dashscope_api_key=config.api_key,
        max_retries=2,
        model_kwargs={"request_timeout": 120},
    )


def invoke(judge: ChatTongyi, prompt: str, node: str) -> dict:
    try:
        response = judge.invoke([HumanMessage(content=prompt)])
        return extract_json(str(response.content))
    except Exception as exc:
        print(f"    [judge:{node}] 调用失败: {type(exc).__name__}: {exc}")
        return {}


def score_faithfulness(judge: ChatTongyi, record: dict, max_claims: int) -> dict:
    contexts = record.get("contexts") or []
    if not contexts:
        return {"score": None, "reason": "无检索证据，无法评估"}
    parsed = invoke(judge, CLAIM_PROMPT.format(max_claims=max_claims, answer=record["answer"][:12000]), "claims")
    claims = [c for c in (parsed.get("claims") or []) if isinstance(c, str) and c.strip()]
    if not claims:
        return {"score": None, "reason": "未能抽出可核查断言"}
    context_block = "\n\n".join(f"[证据{i}] {c[:800]}" for i, c in enumerate(contexts, 1))
    claim_block = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    judged = invoke(judge, JUDGE_PROMPT.format(contexts=context_block, claims=claim_block), "judge")
    verdicts = judged.get("results") or []
    if not verdicts:
        return {"score": None, "reason": "判定失败", "claims": claims}
    weight = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    total = 0.0
    counts = {"supported": 0, "partial": 0, "unsupported": 0}
    for item in verdicts:
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict in weight:
            counts[verdict] += 1
            total += weight[verdict]
    return {
        "score": total / len(verdicts),
        "counts": counts,
        "claim_count": len(claims),
        "claims": claims,
        "unsupported_claims": [
            str(i.get("claim")) for i in verdicts if str(i.get("verdict")).lower() == "unsupported"
        ],
    }


def score_relevance(judge: ChatTongyi, record: dict) -> dict:
    parsed = invoke(
        judge,
        RELEVANCE_PROMPT.format(question=record["question"], answer=record["answer"][:8000]),
        "relevance",
    )
    try:
        score = float(parsed.get("score"))
    except (TypeError, ValueError):
        return {"score": None, "reason": "判定失败"}
    return {"score": max(0.0, min(1.0, score / 100.0)), "raw_score": score, "reason": parsed.get("reason", "")}


def main() -> int:
    parser = argparse.ArgumentParser(description="score eval runs")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-claims", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print(f"没有可打分的记录: {args.input}")
        return 1

    config = AppConfig.from_file()
    judge = build_judge(config)

    print(f"载入 {len(records)} 条 | judge={config.model}\n")
    scored = []
    skipped = []
    for index, record in enumerate(records, 1):
        # 直答走的是模型参数化知识，降级报告不是模型正常产出——两者都不能当评测样本
        if record.get("intent") not in (None, "multiagent"):
            skipped.append((record["id"], "路由到 direct_answer，未经检索"))
            continue
        if record.get("answer_is_fallback"):
            skipped.append((record["id"], "最终答案是降级报告"))
            continue

        if not record.get("answer", "").strip():
            skipped.append((record["id"], "答案为空"))
            continue

        print(f"[{len(scored) + 1}/{len(records)}] {record['id']} | {record['question'][:40]}")
        expected = set(record.get("expected_sources") or [])
        retrieved = set(record.get("retrieved_titles") or [])
        retrieval_hit = bool(expected & retrieved) if expected else None

        cited = record.get("cited_ids") or []
        valid = set(record.get("valid_source_ids") or [])
        citation_compliance = (
            len([c for c in cited if c in valid]) / len(cited) if cited and valid else None
        )

        faithfulness = score_faithfulness(judge, record, args.max_claims)
        relevance = score_relevance(judge, record)

        entry = {
            "id": record["id"],
            "category": record.get("category"),
            "retrieval_hit": retrieval_hit,
            "citation_count": record.get("citation_count", 0),
            "citation_compliance": citation_compliance,
            "degraded_nodes": record.get("degraded_nodes", 0),
            "elapsed_seconds": record.get("elapsed_seconds"),
            "iterations": record.get("iterations"),
            "faithfulness": faithfulness.get("score"),
            "faithfulness_counts": faithfulness.get("counts"),
            "unsupported_claims": faithfulness.get("unsupported_claims", []),
            "relevance": relevance.get("score"),
            "relevance_reason": relevance.get("reason", ""),
        }
        scored.append(entry)
        print(
            f"    忠实度={entry['faithfulness'] if entry['faithfulness'] is None else round(entry['faithfulness'], 3)}"
            f" | 相关性={entry['relevance'] if entry['relevance'] is None else round(entry['relevance'], 3)}"
            f" | 检索命中={entry['retrieval_hit']} | 引用合规={entry['citation_compliance']}"
        )

    # 汇总
    def avg(key: str):
        values = [x[key] for x in scored if isinstance(x.get(key), (int, float))]
        return sum(values) / len(values) if values else None

    print("\n" + "=" * 62)
    if skipped:
        print(f"已排除 {len(skipped)} 条无效样本:")
        for qid, why in skipped:
            print(f"  - {qid}: {why}")
        print()
    if not scored:
        print("没有可打分的有效样本。")
        return 1
    print(f"有效样本数: {len(scored)}")
    for label, key in [
        ("检索命中率", "retrieval_hit"),
        ("引用合规率", "citation_compliance"),
        ("忠实度 (LLM-judge)", "faithfulness"),
        ("相关性 (LLM-judge)", "relevance"),
    ]:
        if key in {"retrieval_hit"}:
            hits = [x for x in scored if x[key] is not None]
            rate = sum(1 for x in hits if x[key]) / len(hits) if hits else None
            print(f"{label}: {'n/a' if rate is None else f'{rate:.1%}'} ({len(hits)} 条可评)")
        else:
            value = avg(key)
            print(f"{label}: {'n/a' if value is None else f'{value:.3f}'}")

    degraded = sum(1 for x in scored if x["degraded_nodes"] > 0)
    print(f"含降级节点的样本: {degraded}/{len(scored)}")
    print("=" * 62)

    output = args.output or args.input.with_name(args.input.stem + "_scored.json")
    output.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
