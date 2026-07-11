"""将视频音频转写为文本，为文本情感分析提供输入。"""

from argparse import ArgumentParser
from pathlib import Path


def transcribe_audio(audio_path: Path, language: str = "zh-CN", model_path: Path | None = None) -> str:
    """转写音频并返回文本；后续在此接入离线语音识别模型。"""
    raise NotImplementedError("语音转写模型将在后续阶段实现")


def build_parser() -> ArgumentParser:
    """创建命令行参数解析器。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="输入音频路径")
    parser.add_argument("--language", default="zh-CN", help="音频语言")
    parser.add_argument("--model", type=Path, help="可选模型路径")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    transcribe_audio(args.audio, args.language, args.model)


if __name__ == "__main__":
    main()
