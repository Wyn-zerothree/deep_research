"""校验评测集：题目绑定的 ground truth 文档是否存在，以及能否被检索召回。

出题时最容易犯的错是"问了一道语料根本答不了的题"——那样反思循环永远不收敛、
每次跑满 max_iterations。这个脚本在跑正式评测前先把这类题筛掉。

用法（仓库根目录下执行）：
    python data/eval/verify_eval_set.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "app"))

from mult_agents.config import AppConfig  # noqa: E402
from mult_agents.rag.core import RAGConfig  # noqa: E402
from mult_agents.tools import init_rag_system, search_knowledge_base_records  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.jsonl"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
TOP_K = 5


def load_eval_set() -> list[dict]:
    items = []
    for line in EVAL_SET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def main() -> int:
    items = load_eval_set()
    excluded = [x for x in items if x.get("excluded")]
    items = [x for x in items if not x.get("excluded")]
    # 标了 excluded 的题不跑批，也不该算进召回的分子分母（否则分母里混着一条
    # 永远不会被执行的题，Recall 会平白被拉低）。排除原因记在各条 excluded_reason。
    print(
        f"评测集条目: {len(items)}"
        + (f" | 已排除 {len(excluded)} 条: {', '.join(x['id'] for x in excluded)}" if excluded else "")
    )

    # 1) ground truth 文件存在性
    missing = []
    for item in items:
        for name in item["expected_sources"]:
            if not (CORPUS_DIR / name).exists():
                missing.append((item["id"], name))
    if missing:
        print("\n[致命] 以下 ground truth 文档不存在于语料目录，请修正题面：")
        for qid, name in missing:
            print(f"  {qid}: {name}")
        return 2
    print("ground truth 文档存在性: OK")

    # 2) 检索召回
    config = AppConfig.from_file()
    init_rag_system(
        api_key=config.api_key,
        config=RAGConfig(
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            collection_name=config.milvus_rag_collection,
        ),
    )

    hits = {1: 0, 3: 0, 5: 0}
    misses = []
    dup_queries = 0
    distinct_counts = []
    for item in items:
        expected = set(item["expected_sources"])
        records = search_knowledge_base_records(item["question"], limit=TOP_K)
        titles = [str(r.get("title", "")) for r in records]
        if not titles:
            misses.append((item["id"], item["question"], "检索返回空", []))
            continue
        for k in hits:
            if expected & set(titles[:k]):
                hits[k] += 1
        if not (expected & set(titles)):
            misses.append((item["id"], item["question"], "top5 未命中", titles))
        # 同文档重复占用 k 槽位，会压缩证据多样性
        distinct_counts.append(len(set(titles)))
        if len(titles) - len(set(titles)) >= 2:
            dup_queries += 1

    total = len(items)
    print("\n=== 检索召回 ===")
    for k in (1, 3, 5):
        print(f"Recall@{k}: {hits[k]}/{total} = {hits[k] / total:.1%}")
    print("\n=== 证据多样性 ===")
    print(f"top{TOP_K} 平均不同文档数: {sum(distinct_counts) / len(distinct_counts):.2f}")
    print(f"有 ≥3 个槽位被同一文档占用的 query: {dup_queries}/{total}")

    if misses:
        print(f"\n=== 未命中 {len(misses)} 条 ===")
        for qid, question, why, titles in misses:
            print(f"\n[{qid}] {question}  ({why})")
            for t in titles:
                print(f"    - {t}")
    else:
        print("\n全部命中。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
