"""Recursively extract model-ready WAV audio from MP4 video datasets.

By default, videos are read from ``data/raw`` and audio is written to
``data/processed/audio`` while preserving each video's relative directory.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "audio"


class AudioExtractionError(RuntimeError):
    """Raised when FFmpeg cannot extract audio from a video."""


def find_ffmpeg() -> str:
    """Return the system FFmpeg executable path.

    Raises:
        FileNotFoundError: If FFmpeg cannot be found on the system PATH.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise FileNotFoundError(
            "FFmpeg was not found. Install it and add ffmpeg to the system PATH."
        )
    return ffmpeg_path


def find_mp4_files(input_dir: Path) -> list[Path]:
    """Return all MP4 files below *input_dir* in deterministic order."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )


def get_output_path(
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Map a source video to its WAV path while preserving subdirectories."""
    relative_path = video_path.relative_to(input_dir)
    return (output_dir / relative_path).with_suffix(".wav")


def extract_audio(
    video_path: Path,
    output_path: Path,
    ffmpeg_path: str,
) -> Path:
    """Extract mono, 16 kHz PCM WAV audio from one MP4 video.

    Args:
        video_path: Source MP4 video.
        output_path: Destination WAV file.
        ffmpeg_path: Path to the FFmpeg executable.

    Returns:
        The generated WAV path.

    Raises:
        AudioExtractionError: If FFmpeg exits unsuccessfully or creates no file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise AudioExtractionError(
            f"Unable to start FFmpeg for {video_path}: {exc}"
        ) from exc

    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        error_message = result.stderr.strip() or "No FFmpeg error details available."
        raise AudioExtractionError(
            f"FFmpeg failed for {video_path} (exit code {result.returncode}): "
            f"{error_message}"
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise AudioExtractionError(
            f"FFmpeg reported success but produced no valid audio: {output_path}"
        )

    return output_path


def process_videos(input_dir: Path, output_dir: Path) -> tuple[int, int, int]:
    """Extract audio from every MP4 below the input directory.

    Existing WAV files are skipped. A failed video is logged without stopping
    the remaining batch.

    Returns:
        A tuple containing successful, skipped, and failed file counts.

    Raises:
        FileNotFoundError: If the input directory or FFmpeg does not exist.
        NotADirectoryError: If an input or output path is not a directory.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    ffmpeg_path = find_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = find_mp4_files(input_dir)
    LOGGER.info("Found %d MP4 video(s) in %s", len(videos), input_dir)

    succeeded = skipped = failed = 0
    for video_path in videos:
        output_path = get_output_path(video_path, input_dir, output_dir)
        if output_path.exists():
            LOGGER.info("Skipping existing audio: %s", output_path)
            skipped += 1
            continue

        LOGGER.info("Extracting: %s -> %s", video_path, output_path)
        try:
            extract_audio(video_path, output_path, ffmpeg_path)
        except AudioExtractionError as exc:
            LOGGER.error("%s", exc)
            failed += 1
        else:
            LOGGER.info("Created audio: %s", output_path)
            succeeded += 1

    return succeeded, skipped, failed


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
        help=f"Destination audio directory (default: {DEFAULT_OUTPUT_DIR})",
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
    """Run the batch audio extraction command and return its exit code."""
    configure_logging()
    args = build_parser().parse_args()

    try:
        succeeded, skipped, failed = process_videos(args.input, args.output)
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "Finished: %d succeeded, %d skipped, %d failed",
        succeeded,
        skipped,
        failed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
