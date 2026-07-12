"""Transcribe WAV datasets with the OpenAI Whisper tiny model on CPU.

Audio is read recursively from ``data/processed/audio`` and JSON transcripts
are written to ``data/processed/transcript`` with the same directory layout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import whisper


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "audio"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "transcript"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "whisper"
MODEL_NAME = "tiny"


class TranscriptionError(RuntimeError):
    """Raised when Whisper cannot transcribe or save an audio transcript."""


def find_wav_files(input_dir: Path) -> list[Path]:
    """Return all WAV files below *input_dir* in deterministic order."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def get_output_path(
    audio_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Map an audio path to a JSON path while preserving subdirectories."""
    relative_path = audio_path.relative_to(input_dir)
    return (output_dir / relative_path).with_suffix(".json")


def load_whisper_model(model_dir: Path | None = None) -> Any:
    """Load Whisper tiny on CPU, downloading its weights when necessary.

    Args:
        model_dir: Optional model cache directory. Whisper's default cache is
            used when this is ``None``.

    Returns:
        A loaded Whisper model instance.
    """
    download_root = str(model_dir.resolve()) if model_dir is not None else None
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading Whisper model '%s' on CPU", MODEL_NAME)
    try:
        return whisper.load_model(
            MODEL_NAME,
            device="cpu",
            download_root=download_root,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Unable to load or download Whisper model '{MODEL_NAME}': {exc}"
        ) from exc


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a Whisper result into the project's stable JSON structure."""
    segments: list[dict[str, Any]] = []
    for raw_segment in result.get("segments", []):
        start = float(raw_segment.get("start", 0.0))
        end = float(raw_segment.get("end", start))
        text = str(raw_segment.get("text", "")).strip()

        if not math.isfinite(start) or not math.isfinite(end):
            raise TranscriptionError("Whisper returned a non-finite timestamp.")

        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )

    return {
        "language": str(result.get("language", "unknown")),
        "text": str(result.get("text", "")).strip(),
        "segments": segments,
    }


def transcribe_audio(
    model: Any,
    audio_path: Path,
    language: str | None = None,
) -> dict[str, Any]:
    """Transcribe one WAV file with Whisper using CPU-safe settings.

    Args:
        model: A loaded Whisper model.
        audio_path: WAV file to transcribe.
        language: Optional ISO language code. ``None`` enables automatic
            language detection.

    Returns:
        A normalized transcript containing language, text, and segments.

    Raises:
        TranscriptionError: If Whisper fails or returns an invalid result.
    """
    try:
        result = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            fp16=False,
            verbose=False,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Whisper failed to transcribe {audio_path}: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise TranscriptionError(
            f"Whisper returned an unexpected result for {audio_path}."
        )
    return normalize_result(result)


def save_transcript(transcript: dict[str, Any], output_path: Path) -> None:
    """Write a transcript as UTF-8 JSON using an atomic file replacement."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(transcript, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(output_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise TranscriptionError(
            f"Unable to save transcript {output_path}: {exc}"
        ) from exc


def process_audio_files(
    input_dir: Path,
    output_dir: Path,
    language: str | None = None,
    model_dir: Path | None = None,
) -> tuple[int, int, int]:
    """Transcribe every WAV file below *input_dir* with one model instance.

    Existing JSON files are skipped, and processing continues if an individual
    audio file fails.

    Returns:
        A tuple containing successful, skipped, and failed file counts.

    Raises:
        FileNotFoundError: If the input directory does not exist.
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

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = find_wav_files(input_dir)
    LOGGER.info("Found %d WAV file(s) in %s", len(audio_files), input_dir)
    if not audio_files:
        return 0, 0, 0

    pending_files = [
        audio_path
        for audio_path in audio_files
        if not get_output_path(audio_path, input_dir, output_dir).exists()
    ]
    skipped = len(audio_files) - len(pending_files)
    if skipped:
        LOGGER.info("Skipping %d audio file(s) with existing JSON", skipped)
    if not pending_files:
        return 0, skipped, 0

    model = load_whisper_model(model_dir)
    succeeded = 0
    failed = 0
    for audio_path in pending_files:
        output_path = get_output_path(audio_path, input_dir, output_dir)
        LOGGER.info("Transcribing: %s -> %s", audio_path, output_path)
        try:
            transcript = transcribe_audio(model, audio_path, language)
            save_transcript(transcript, output_path)
        except TranscriptionError as exc:
            LOGGER.error("%s", exc)
            failed += 1
        else:
            LOGGER.info("Created transcript: %s", output_path)
            succeeded += 1

    return succeeded, skipped, failed


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Source audio directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Transcript directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code, such as zh or en (default: auto-detect)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory used to cache the Whisper tiny model "
        f"(default: {DEFAULT_MODEL_DIR})",
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
    """Run batch speech recognition and return a process exit code."""
    configure_logging()
    args = build_parser().parse_args()

    try:
        succeeded, skipped, failed = process_audio_files(
            args.input,
            args.output,
            args.language,
            args.model_dir,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        RuntimeError,
    ) as exc:
        LOGGER.error("Unable to initialize transcription: %s", exc)
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
