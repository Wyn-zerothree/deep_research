"""知识库入库 CLI：把本地文档切分后写入 Milvus。

用法（从仓库根目录执行）：
    python app/mult_agents/rag/ingest.py <文件或目录>
    python app/mult_agents/rag/ingest.py data/corpus --chunk-size 300

目录会递归收集 *.txt / *.md / *.markdown。Milvus 连接信息与 collection
名称取自 config.json / 环境变量，优先级为 环境变量 > config.json > 默认值。

换分块粒度做对比实验时，配合环境变量换集合名，别覆盖生产集合：
    MILVUS_RAG_COLLECTION=mult_agent_knowledge_c300 python ... --chunk-size 300
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

# 将 app/ 加入 sys.path，使本脚本可从仓库任意目录以脚本方式执行
APP_DIR = Path(__file__).resolve().parents[2]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mult_agents.config import AppConfig  # noqa: E402
from mult_agents.rag.core import RAGConfig, RAGSystem  # noqa: E402

SUPPORTED_SUFFIXES = (".txt", ".md", ".markdown")


def _collect_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    paths: list[Path] = []
    for suffix in SUPPORTED_SUFFIXES:
        paths.extend(sorted(input_path.rglob(f"*{suffix}")))
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="把本地文档写入 Milvus 知识库")
    parser.add_argument("path", help="要入库的文件或目录")
    parser.add_argument("--config", default=None, help="config.json 路径，默认仓库根目录下的 config.json")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="分块字符数，默认沿用 RAGConfig 的 500；改粒度做对比时同时换集合名",
    )
    args = parser.parse_args()

    input_path = Path(args.path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"路径不存在: {input_path}")

    paths = _collect_paths(input_path)
    if not paths:
        raise ValueError(f"未找到可入库文件（{'/'.join(SUPPORTED_SUFFIXES)}）: {input_path}")

    config = AppConfig.from_file(args.config)
    rag_config = RAGConfig(
        milvus_host=config.milvus_host,
        milvus_port=config.milvus_port,
        collection_name=config.milvus_rag_collection,
    )
    if args.chunk_size is not None:
        rag_config = replace(rag_config, chunk_size=args.chunk_size)
    rag = RAGSystem(api_key=config.api_key, config=rag_config)

    total_chunks = rag.ingest_paths(paths)
    print(
        f"入库完成 | 文件数={len(paths)} | chunk数={total_chunks} | "
        f"chunk_size={rag_config.chunk_size} | collection={config.milvus_rag_collection}"
    )


if __name__ == "__main__":
    main()
