# 一是把 app 文件夹加入 Python 的搜索清单，确保能顺利导入深层模块；
# 二是把 .env 里的密钥加载到内存，确保后续调用 API 时有权限。
# 做完这两件事，它才转身去 app/mult_agents/main.py 喊真正的 CLI 主程序出来干活。”
from pathlib import Path
import sys


def _bootstrap() -> None:
    root = Path(__file__).resolve().parent
    src = root / "app"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    # 加载 .env 文件（在导入其他模块之前）
    from dotenv import load_dotenv

    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def main() -> None:
    _bootstrap()
    from mult_agents.main import main as run_main

    run_main()


if __name__ == "__main__":
    main()
