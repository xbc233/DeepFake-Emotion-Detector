"""分析人脸图像中的表情情感，并输出统一格式的情感分数。"""

from argparse import ArgumentParser
from pathlib import Path


def analyze_face_emotion(frame_paths: list[Path], model_path: Path | None = None) -> list[dict[str, float]]:
    """分析一组视频帧；后续在此加载 CPU 友好的人脸情感模型。"""
    raise NotImplementedError("人脸情感分析模型将在后续阶段实现")


def build_parser() -> ArgumentParser:
    """创建命令行参数解析器。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, nargs="+", help="一个或多个视频帧路径")
    parser.add_argument("--model", type=Path, help="可选模型路径")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    analyze_face_emotion(args.frames, args.model)


if __name__ == "__main__":
    main()
