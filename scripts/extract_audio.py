"""从视频中提取音频，为语音情感分析和语音转写准备输入。"""

from argparse import ArgumentParser
from pathlib import Path


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """提取视频音轨并返回输出路径；后续在此接入 FFmpeg。"""
    raise NotImplementedError("音频提取功能将在后续阶段实现")


def build_parser() -> ArgumentParser:
    """创建命令行参数解析器。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="输入视频路径")
    parser.add_argument("output", type=Path, help="输出音频路径")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    extract_audio(args.video, args.output)


if __name__ == "__main__":
    main()
