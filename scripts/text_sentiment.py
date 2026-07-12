"""Classify Whisper transcript segments with zero-shot text classification.

Whisper JSON files are read recursively from ``data/processed/transcript``.
One time-aligned CSV is written per transcript below
``data/processed/emotions/text`` while preserving the directory structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

from transformers import pipeline


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "transcript"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "emotions" / "text"
)
DEFAULT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
CANDIDATE_LABELS = ("happy", "neutral", "sad", "fear", "anger", "surprise")
HYPOTHESIS_TEMPLATE = "The emotion expressed in this text is {}."


class TextSentimentError(RuntimeError):
    """Raised when transcript sentiment processing cannot be completed."""


def load_classifier(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load a Hugging Face zero-shot classifier for CPU inference.

    The model is downloaded automatically on first use.

    Args:
        model_name: Hugging Face model repository name or local model path.

    Returns:
        A zero-shot classification pipeline running on CPU.

    Raises:
        TextSentimentError: If the model cannot be loaded or downloaded.
    """
    LOGGER.info("Loading zero-shot model '%s' on CPU", model_name)
    try:
        return pipeline(
            task="zero-shot-classification",
            model=model_name,
            device=-1,
        )
    except Exception as exc:
        raise TextSentimentError(
            f"Unable to load zero-shot model '{model_name}': {exc}"
        ) from exc


def find_json_files(input_dir: Path) -> list[Path]:
    """Return all JSON transcript files recursively in deterministic order."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".json"
    )


def get_output_path(
    transcript_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Map a transcript JSON to a CSV while preserving subdirectories."""
    relative_path = transcript_path.relative_to(input_dir)
    return (output_dir / relative_path).with_suffix(".csv")


def load_segments(transcript_path: Path) -> list[dict[str, Any]]:
    """Load and validate time-aligned segments from one Whisper JSON file.

    Args:
        transcript_path: Whisper JSON file to parse.

    Returns:
        Validated dictionaries containing ``start``, ``end``, and ``text``.

    Raises:
        TextSentimentError: If the JSON or segment structure is invalid.
    """
    try:
        with transcript_path.open("r", encoding="utf-8-sig") as json_file:
            document = json.load(json_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TextSentimentError(
            f"Unable to read transcript {transcript_path}: {exc}"
        ) from exc

    if not isinstance(document, dict):
        raise TextSentimentError(
            f"Transcript root must be a JSON object: {transcript_path}"
        )
    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list):
        raise TextSentimentError(
            f"Transcript has no valid segments list: {transcript_path}"
        )

    segments: list[dict[str, Any]] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise TextSentimentError(
                f"Segment {index} is not an object: {transcript_path}"
            )
        try:
            start = float(raw_segment["start"])
            end = float(raw_segment["end"])
            text = str(raw_segment.get("text", "")).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise TextSentimentError(
                f"Segment {index} is invalid in {transcript_path}: {exc}"
            ) from exc

        if not math.isfinite(start) or not math.isfinite(end):
            raise TextSentimentError(
                f"Segment {index} has a non-finite timestamp: {transcript_path}"
            )
        if start < 0 or end < start:
            raise TextSentimentError(
                f"Segment {index} has an invalid time range: {transcript_path}"
            )
        segments.append({"start": start, "end": end, "text": text})
    return segments


def classify_texts(
    texts: list[str],
    classifier: Any,
    batch_size: int,
) -> list[tuple[str, float]]:
    """Classify a batch of non-empty texts into the candidate emotions.

    Args:
        texts: Non-empty transcript texts.
        classifier: Loaded zero-shot classification pipeline.
        batch_size: Maximum number of texts processed together.

    Returns:
        Emotion and score pairs in input order.

    Raises:
        ValueError: If *batch_size* is not positive.
        TextSentimentError: If inference returns malformed data or fails.
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if not texts:
        return []

    try:
        raw_results = classifier(
            texts,
            candidate_labels=list(CANDIDATE_LABELS),
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
            batch_size=batch_size,
            truncation=True,
        )
    except Exception as exc:
        raise TextSentimentError(
            f"Zero-shot text classification failed: {exc}"
        ) from exc

    if isinstance(raw_results, dict):
        raw_results = [raw_results]
    if not isinstance(raw_results, list) or len(raw_results) != len(texts):
        raise TextSentimentError(
            "Zero-shot classifier returned an unexpected number of results."
        )

    predictions: list[tuple[str, float]] = []
    for result in raw_results:
        if not isinstance(result, dict):
            raise TextSentimentError("Classifier returned a malformed result.")
        labels = result.get("labels")
        scores = result.get("scores")
        if not isinstance(labels, list) or not isinstance(scores, list):
            raise TextSentimentError("Classifier result has no labels or scores.")
        if not labels or not scores:
            raise TextSentimentError("Classifier returned an empty prediction.")

        emotion = str(labels[0]).lower()
        score = float(scores[0])
        if emotion not in CANDIDATE_LABELS or not math.isfinite(score):
            raise TextSentimentError("Classifier returned an invalid prediction.")
        predictions.append((emotion, min(max(score, 0.0), 1.0)))
    return predictions


def analyze_segments(
    segments: list[dict[str, Any]],
    classifier: Any,
    batch_size: int,
) -> list[dict[str, str]]:
    """Classify transcript segments and build time-aligned CSV rows.

    Empty segment text is assigned ``neutral`` with score ``0`` without model
    inference.
    """
    non_empty_texts = [segment["text"] for segment in segments if segment["text"]]
    predictions = iter(classify_texts(non_empty_texts, classifier, batch_size))
    rows: list[dict[str, str]] = []

    for segment in segments:
        if segment["text"]:
            emotion, score = next(predictions)
        else:
            emotion, score = "neutral", 0.0
        rows.append(
            {
                "start": f"{segment['start']:.3f}",
                "end": f"{segment['end']:.3f}",
                "emotion": emotion,
                "score": f"{score:.6f}",
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Atomically write text sentiment rows to a UTF-8 CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=("start", "end", "emotion", "score"),
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise TextSentimentError(
            f"Unable to write text sentiment CSV {output_path}: {exc}"
        ) from exc


def process_transcript(
    transcript_path: Path,
    output_path: Path,
    classifier: Any,
    batch_size: int,
) -> bool:
    """Process one transcript, returning false if its CSV already exists."""
    if output_path.exists():
        LOGGER.info("Skipping existing CSV: %s", output_path)
        return False
    segments = load_segments(transcript_path)
    rows = analyze_segments(segments, classifier, batch_size)
    write_csv(rows, output_path)
    return True


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 8,
) -> tuple[int, int, int]:
    """Process every Whisper JSON with one shared classifier instance.

    Returns:
        A tuple containing successful, skipped, and failed file counts.
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_files = find_json_files(input_dir)
    LOGGER.info("Found %d transcript JSON file(s)", len(transcript_files))
    if not transcript_files:
        return 0, 0, 0

    pending_files: list[tuple[Path, Path]] = []
    skipped = 0
    for transcript_path in transcript_files:
        output_path = get_output_path(transcript_path, input_dir, output_dir)
        if output_path.exists():
            LOGGER.info("Skipping existing CSV: %s", output_path)
            skipped += 1
        else:
            pending_files.append((transcript_path, output_path))

    if not pending_files:
        return 0, skipped, 0

    classifier = load_classifier(model_name)
    succeeded = 0
    failed = 0
    for transcript_path, output_path in pending_files:
        LOGGER.info("Processing: %s -> %s", transcript_path, output_path)
        try:
            process_transcript(
                transcript_path,
                output_path,
                classifier,
                batch_size,
            )
        except (TextSentimentError, ValueError) as exc:
            LOGGER.error("%s", exc)
            failed += 1
        else:
            LOGGER.info("Created text sentiment CSV: %s", output_path)
            succeeded += 1
    return succeeded, skipped, failed


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Whisper JSON directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination CSV directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of transcript segments per inference batch (default: 8)",
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
    """Run batch text sentiment classification and return an exit code."""
    configure_logging()
    args = build_parser().parse_args()
    try:
        succeeded, skipped, failed = process_dataset(
            args.input,
            args.output,
            args.model,
            args.batch_size,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        TextSentimentError,
        ValueError,
    ) as exc:
        LOGGER.error("Unable to initialize text sentiment processing: %s", exc)
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
