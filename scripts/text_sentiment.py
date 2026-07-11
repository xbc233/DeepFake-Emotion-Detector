"""分析转写文本的情感，并输出统一格式的情感分数。"""

from argparse import ArgumentParser
from pathlib import Path


def analyze_text_sentiment(text: str, model_path: Path | None = None) -> dict[str, float]:
    """分析文本情感；后续在此加载 CPU 友好的文本情感模型。"""
    raise NotImplementedError("文本情感分析模型将在后续阶段实现")


def build_parser() -> ArgumentParser:
    """创建命令行参数解析器。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("text", help="待分析文本")
    parser.add_argument("--model", type=Path, help="可选模型路径")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    analyze_text_sentiment(args.text, args.model)


if __name__ == "__main__":
    main()
