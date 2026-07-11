"""分析音频中的语音情感，并输出统一格式的情感分数。"""

from argparse import ArgumentParser
from pathlib import Path


def analyze_speech_emotion(audio_path: Path, model_path: Path | None = None) -> dict[str, float]:
    """分析语音情感；后续在此加载 CPU 友好的语音情感模型。"""
    raise NotImplementedError("语音情感分析模型将在后续阶段实现")


def build_parser() -> ArgumentParser:
    """创建命令行参数解析器。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="输入音频路径")
    parser.add_argument("--model", type=Path, help="可选模型路径")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    analyze_speech_emotion(args.audio, args.model)


if __name__ == "__main__":
    main()
