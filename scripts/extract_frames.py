"""Recursively extract JPEG frames from an MP4 video dataset.

The default input is ``data/raw`` and the default output is
``data/processed/frames``. Source subdirectories are preserved and every
video receives its own output directory.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import cv2


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "frames"
DEFAULT_EXTRACTION_FPS = 2.0


class FrameExtractionError(RuntimeError):
    """Raised when frames cannot be read from or written for a video."""


def find_mp4_files(input_dir: Path) -> list[Path]:
    """Return all MP4 files below *input_dir* in deterministic order."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )


def get_video_output_dir(
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Return a video's frame directory while preserving source folders."""
    relative_path = video_path.relative_to(input_dir)
    return output_dir / relative_path.parent / relative_path.stem


def extract_frames(
    video_path: Path,
    output_dir: Path,
    extraction_fps: float = DEFAULT_EXTRACTION_FPS,
) -> tuple[int, int]:
    """Extract JPEG frames from one video at the requested sampling rate.

    Args:
        video_path: MP4 file to decode.
        output_dir: Directory dedicated to this video's frames.
        extraction_fps: Number of frames to sample per second.

    Returns:
        A tuple containing the number of created and skipped images.

    Raises:
        ValueError: If *extraction_fps* is not a finite positive number.
        FrameExtractionError: If the video cannot be opened or an image
            cannot be written.
    """
    if not math.isfinite(extraction_fps) or extraction_fps <= 0:
        raise ValueError("Extraction FPS must be a finite number greater than 0.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise FrameExtractionError(f"Unable to open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise FrameExtractionError(
            f"Video reports an invalid source FPS ({source_fps}): {video_path}"
        )

    effective_fps = min(extraction_fps, source_fps)
    if extraction_fps > source_fps:
        LOGGER.warning(
            "Requested %.3f FPS but source is %.3f FPS; extracting every frame: %s",
            extraction_fps,
            source_fps,
            video_path,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_interval = 1.0 / effective_fps
    next_sample_time = 0.0
    source_frame_index = 0
    output_index = 1
    created = 0
    skipped = 0

    try:
        while True:
            read_ok, frame = capture.read()
            if not read_ok:
                break

            frame_time = source_frame_index / source_fps
            if frame_time + 1e-9 >= next_sample_time:
                image_path = output_dir / f"{output_index:06d}.jpg"
                if image_path.exists():
                    LOGGER.debug("Skipping existing frame: %s", image_path)
                    skipped += 1
                else:
                    write_ok = cv2.imwrite(str(image_path), frame)
                    if not write_ok:
                        raise FrameExtractionError(
                            f"OpenCV could not write frame: {image_path}"
                        )
                    created += 1

                output_index += 1
                next_sample_time += sample_interval

            source_frame_index += 1
    finally:
        capture.release()

    if source_frame_index == 0:
        raise FrameExtractionError(f"No decodable frames found: {video_path}")

    return created, skipped


def process_videos(
    input_dir: Path,
    output_dir: Path,
    extraction_fps: float,
) -> tuple[int, int, int]:
    """Extract frames from every MP4 below *input_dir*.

    Processing continues after an individual video fails.

    Returns:
        A tuple containing processed, skipped-image, and failed-video counts.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        NotADirectoryError: If an input or output path is not a directory.
        ValueError: If the requested extraction FPS is invalid.
    """
    if not math.isfinite(extraction_fps) or extraction_fps <= 0:
        raise ValueError("Extraction FPS must be a finite number greater than 0.")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = find_mp4_files(input_dir)
    LOGGER.info("Found %d MP4 video(s) in %s", len(videos), input_dir)

    processed = 0
    skipped_images = 0
    failed = 0
    for video_path in videos:
        video_output_dir = get_video_output_dir(
            video_path,
            input_dir,
            output_dir,
        )
        LOGGER.info("Extracting frames: %s -> %s", video_path, video_output_dir)
        try:
            created, skipped = extract_frames(
                video_path,
                video_output_dir,
                extraction_fps,
            )
        except FrameExtractionError as exc:
            LOGGER.error("%s", exc)
            failed += 1
            continue

        LOGGER.info(
            "Finished video: %d created, %d existing images skipped",
            created,
            skipped,
        )
        processed += 1
        skipped_images += skipped

    return processed, skipped_images, failed


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Source dataset directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination frame directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_EXTRACTION_FPS,
        help=f"Frames to extract per second (default: {DEFAULT_EXTRACTION_FPS:g})",
    )
    return parser


def configure_logging() -> None:
    """Configure concise console logging for command-line execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Run batch frame extraction and return a process exit code."""
    configure_logging()
    args = build_parser().parse_args()

    try:
        processed, skipped_images, failed = process_videos(
            args.input,
            args.output,
            args.fps,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        ValueError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "Finished: %d video(s) processed, %d existing image(s) skipped, "
        "%d video(s) failed",
        processed,
        skipped_images,
        failed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
