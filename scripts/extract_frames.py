"""按时间间隔从视频中提取帧，为人脸表情分析准备输入。"""

from argparse import ArgumentParser
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, interval_seconds: float = 1.0) -> list[Path]:
    """提取视频帧并返回文件列表；后续在此接入 OpenCV。"""
    raise NotImplementedError("视频帧提取功能将在后续阶段实现")


def build_parser() -> ArgumentParser:
    """创建命令行参数解析器。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="输入视频路径")
    parser.add_argument("output_dir", type=Path, help="帧输出目录")
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔（秒）")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    extract_frames(args.video, args.output_dir, args.interval)


if __name__ == "__main__":
    main()
