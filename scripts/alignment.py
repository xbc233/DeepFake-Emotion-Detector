"""Align face, speech, and text emotions on a shared time axis.

The three modality CSV trees are matched by relative path. Emotion samples
are aligned to a 0.5-second timeline with nearest-neighbor matching, then
pairwise and final consistency scores are calculated for every time point.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMOTION_ROOT = PROJECT_ROOT / "data" / "processed" / "emotions"
DEFAULT_FACE_DIR = EMOTION_ROOT / "face"
DEFAULT_SPEECH_DIR = EMOTION_ROOT / "speech"
DEFAULT_TEXT_DIR = EMOTION_ROOT / "text"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "alignment"
DEFAULT_INTERVAL = 0.5
DEFAULT_FACE_FPS = 2.0
UNKNOWN_EMOTIONS = {"", "unknown", "none", "nan"}
EMOTION_ALIASES = {
    "ang": "anger",
    "angry": "anger",
    "anger": "anger",
    "fearful": "fear",
    "fear": "fear",
    "hap": "happy",
    "happiness": "happy",
    "happy": "happy",
    "neu": "neutral",
    "neutral": "neutral",
    "sadness": "sad",
    "sad": "sad",
    "surprised": "surprise",
    "surprise": "surprise",
}


class AlignmentError(RuntimeError):
    """Raised when modality results cannot be loaded or aligned."""


@dataclass(frozen=True, slots=True)
class EmotionPoint:
    """Represent one emotion observation at a time in seconds."""

    time: float
    emotion: str


def normalize_emotion(emotion: str) -> str:
    """Map modality-specific emotion names to a shared label vocabulary."""
    normalized = emotion.strip().lower()
    if normalized in UNKNOWN_EMOTIONS:
        return "unknown"
    return EMOTION_ALIASES.get(normalized, normalized)


def read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], set[str]]:
    """Read a CSV and return its rows and normalized field names.

    Raises:
        AlignmentError: If the file cannot be read or has no header.
    """
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise AlignmentError(f"CSV has no header: {csv_path}")
            rows = [dict(row) for row in reader]
            fields = {field.strip().lower() for field in reader.fieldnames}
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AlignmentError(f"Unable to read CSV {csv_path}: {exc}") from exc
    return rows, fields


def parse_finite_float(value: str, field: str, csv_path: Path) -> float:
    """Parse a finite floating-point CSV value with a useful error message."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AlignmentError(
            f"Invalid {field!r} value in {csv_path}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise AlignmentError(
            f"Non-finite {field!r} value in {csv_path}: {value!r}"
        )
    return parsed


def load_face_points(
    csv_path: Path,
    face_fps: float,
) -> tuple[list[EmotionPoint], float]:
    """Load face emotions, deriving timestamps from frame numbers.

    A frame named ``000001.jpg`` is assigned time 0.0. If a ``time`` column
    exists, it takes precedence over frame-number inference.
    """
    if not math.isfinite(face_fps) or face_fps <= 0:
        raise ValueError("Face FPS must be a finite positive number.")
    rows, fields = read_csv_rows(csv_path)
    required = {"frame", "emotion"}
    if not required.issubset(fields):
        raise AlignmentError(
            f"Face CSV must contain frame and emotion columns: {csv_path}"
        )

    points: list[EmotionPoint] = []
    for row_index, row in enumerate(rows):
        if "time" in fields and row.get("time", "").strip():
            timestamp = parse_finite_float(row["time"], "time", csv_path)
        else:
            frame_name = row.get("frame", "")
            try:
                frame_number = int(Path(frame_name).stem)
                timestamp = max(frame_number - 1, 0) / face_fps
            except ValueError:
                timestamp = row_index / face_fps
        if timestamp < 0:
            raise AlignmentError(f"Negative face timestamp in {csv_path}")
        points.append(
            EmotionPoint(timestamp, normalize_emotion(row.get("emotion", "")))
        )

    points.sort(key=lambda point: point.time)
    extent = points[-1].time if points else 0.0
    return points, extent


def load_speech_points(csv_path: Path) -> tuple[list[EmotionPoint], float]:
    """Load time-stamped speech emotion observations from a CSV file."""
    rows, fields = read_csv_rows(csv_path)
    required = {"time", "emotion"}
    if not required.issubset(fields):
        raise AlignmentError(
            f"Speech CSV must contain time and emotion columns: {csv_path}"
        )

    points: list[EmotionPoint] = []
    for row in rows:
        timestamp = parse_finite_float(row.get("time", ""), "time", csv_path)
        if timestamp < 0:
            raise AlignmentError(f"Negative speech timestamp in {csv_path}")
        points.append(
            EmotionPoint(timestamp, normalize_emotion(row.get("emotion", "")))
        )
    points.sort(key=lambda point: point.time)
    extent = points[-1].time if points else 0.0
    return points, extent


def load_text_points(csv_path: Path) -> tuple[list[EmotionPoint], float]:
    """Load text emotions using each transcript segment midpoint as its time."""
    rows, fields = read_csv_rows(csv_path)
    required = {"start", "end", "emotion"}
    if not required.issubset(fields):
        raise AlignmentError(
            f"Text CSV must contain start, end, and emotion columns: {csv_path}"
        )

    points: list[EmotionPoint] = []
    extent = 0.0
    for row in rows:
        start = parse_finite_float(row.get("start", ""), "start", csv_path)
        end = parse_finite_float(row.get("end", ""), "end", csv_path)
        if start < 0 or end < start:
            raise AlignmentError(f"Invalid text time range in {csv_path}")
        points.append(
            EmotionPoint(
                (start + end) / 2.0,
                normalize_emotion(row.get("emotion", "")),
            )
        )
        extent = max(extent, end)
    points.sort(key=lambda point: point.time)
    return points, extent


def nearest_emotion(points: list[EmotionPoint], target_time: float) -> str:
    """Return the emotion nearest to *target_time*, preferring earlier ties."""
    if not points:
        return "unknown"
    times = [point.time for point in points]
    position = bisect.bisect_left(times, target_time)
    if position == 0:
        return points[0].emotion
    if position == len(points):
        return points[-1].emotion

    before = points[position - 1]
    after = points[position]
    if target_time - before.time <= after.time - target_time:
        return before.emotion
    return after.emotion


def pair_consistency(first: str, second: str) -> float | None:
    """Return binary pairwise consistency, or none for an unknown modality."""
    if first == "unknown" or second == "unknown":
        return None
    return 1.0 if first == second else 0.0


def calculate_consistency(
    face: str,
    speech: str,
    text: str,
) -> tuple[float | None, float | None, float | None, float]:
    """Calculate all modality-pair scores and their final mean.

    Unknown modalities are excluded. The final score is zero when fewer than
    two modalities are available.
    """
    face_speech = pair_consistency(face, speech)
    speech_text = pair_consistency(speech, text)
    face_text = pair_consistency(face, text)
    available = [
        score
        for score in (face_speech, speech_text, face_text)
        if score is not None
    ]
    final_score = sum(available) / len(available) if available else 0.0
    return face_speech, speech_text, face_text, final_score


def align_modalities(
    face_points: list[EmotionPoint],
    speech_points: list[EmotionPoint],
    text_points: list[EmotionPoint],
    duration: float,
    interval: float = DEFAULT_INTERVAL,
) -> tuple[list[dict[str, str]], float]:
    """Align three modality streams and return rows plus average consistency."""
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Timeline interval must be a finite positive number.")
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Alignment duration must be finite and non-negative.")

    step_count = math.floor((duration + 1e-9) / interval)
    rows: list[dict[str, str]] = []
    scores: list[float] = []
    for step in range(step_count + 1):
        timestamp = step * interval
        face = nearest_emotion(face_points, timestamp)
        speech = nearest_emotion(speech_points, timestamp)
        text = nearest_emotion(text_points, timestamp)
        _, _, _, consistency = calculate_consistency(face, speech, text)
        scores.append(consistency)
        rows.append(
            {
                "time": f"{timestamp:.3f}",
                "face": face,
                "speech": speech,
                "text": text,
                "consistency": f"{consistency:.6f}",
            }
        )

    average = sum(scores) / len(scores) if scores else 0.0
    return rows, average


def write_alignment_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Atomically write aligned modality rows to a UTF-8 CSV file."""
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
                fieldnames=("time", "face", "speech", "text", "consistency"),
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise AlignmentError(
            f"Unable to write alignment CSV {output_path}: {exc}"
        ) from exc


def process_video(
    face_path: Path,
    speech_path: Path,
    text_path: Path,
    output_path: Path,
    interval: float,
    face_fps: float,
) -> float:
    """Align one video's three CSV files and return average consistency."""
    face_points, face_extent = load_face_points(face_path, face_fps)
    speech_points, speech_extent = load_speech_points(speech_path)
    text_points, text_extent = load_text_points(text_path)
    if not face_points and not speech_points and not text_points:
        raise AlignmentError(f"All modality CSV files are empty for {face_path}")

    duration = max(face_extent, speech_extent, text_extent)
    rows, average = align_modalities(
        face_points,
        speech_points,
        text_points,
        duration,
        interval,
    )
    write_alignment_csv(rows, output_path)
    return average


def read_average_consistency(csv_path: Path) -> float:
    """Read the mean consistency from an existing alignment CSV."""
    rows, fields = read_csv_rows(csv_path)
    if "consistency" not in fields:
        raise AlignmentError(f"Alignment CSV lacks consistency: {csv_path}")
    scores = [
        parse_finite_float(row.get("consistency", ""), "consistency", csv_path)
        for row in rows
    ]
    return sum(scores) / len(scores) if scores else 0.0


def write_summary(summary: list[dict[str, str]], output_dir: Path) -> None:
    """Atomically write per-video average consistency rates."""
    output_path = output_dir / "summary.csv"
    temporary_path = output_path.with_suffix(".csv.tmp")
    try:
        with temporary_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=("video", "average_consistency"),
            )
            writer.writeheader()
            writer.writerows(summary)
        temporary_path.replace(output_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise AlignmentError(f"Unable to write summary CSV: {exc}") from exc


def validate_directory(path: Path, name: str) -> Path:
    """Resolve and validate one required input directory."""
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{name} directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{name} path is not a directory: {resolved}")
    return resolved


def process_dataset(
    face_dir: Path,
    speech_dir: Path,
    text_dir: Path,
    output_dir: Path,
    interval: float = DEFAULT_INTERVAL,
    face_fps: float = DEFAULT_FACE_FPS,
) -> tuple[int, int, int]:
    """Batch-align matching modality CSV files and generate a summary CSV.

    Returns:
        A tuple containing successful, skipped, and failed video counts.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Timeline interval must be a finite positive number.")
    if not math.isfinite(face_fps) or face_fps <= 0:
        raise ValueError("Face FPS must be a finite positive number.")

    face_dir = validate_directory(face_dir, "Face")
    speech_dir = validate_directory(speech_dir, "Speech")
    text_dir = validate_directory(text_dir, "Text")
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    face_files = sorted(face_dir.rglob("*.csv"))
    LOGGER.info("Found %d face emotion CSV file(s)", len(face_files))
    succeeded = 0
    skipped = 0
    failed = 0
    summary: list[dict[str, str]] = []

    for face_path in face_files:
        relative_path = face_path.relative_to(face_dir)
        speech_path = speech_dir / relative_path
        text_path = text_dir / relative_path
        output_path = output_dir / relative_path
        video_name = relative_path.with_suffix("").as_posix()

        if not speech_path.is_file() or not text_path.is_file():
            LOGGER.error(
                "Missing matching modality CSV for %s (speech=%s, text=%s)",
                relative_path,
                speech_path.is_file(),
                text_path.is_file(),
            )
            failed += 1
            continue

        try:
            if output_path.exists():
                LOGGER.info("Skipping existing alignment: %s", output_path)
                average = read_average_consistency(output_path)
                skipped += 1
            else:
                LOGGER.info("Aligning video: %s", video_name)
                average = process_video(
                    face_path,
                    speech_path,
                    text_path,
                    output_path,
                    interval,
                    face_fps,
                )
                succeeded += 1
        except (AlignmentError, ValueError) as exc:
            LOGGER.error("%s", exc)
            failed += 1
            continue

        LOGGER.info("Average consistency for %s: %.2f%%", video_name, average * 100)
        summary.append(
            {
                "video": video_name,
                "average_consistency": f"{average:.6f}",
            }
        )

    write_summary(summary, output_dir)
    return succeeded, skipped, failed


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", type=Path, default=DEFAULT_FACE_DIR)
    parser.add_argument("--speech", type=Path, default=DEFAULT_SPEECH_DIR)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"Timeline interval in seconds (default: {DEFAULT_INTERVAL:g})",
    )
    parser.add_argument(
        "--face-fps",
        type=float,
        default=DEFAULT_FACE_FPS,
        help=f"FPS used to extract face frames (default: {DEFAULT_FACE_FPS:g})",
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
    """Run batch cross-modal alignment and return a process exit code."""
    configure_logging()
    args = build_parser().parse_args()
    try:
        succeeded, skipped, failed = process_dataset(
            args.face,
            args.speech,
            args.text,
            args.output,
            args.interval,
            args.face_fps,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        AlignmentError,
        ValueError,
    ) as exc:
        LOGGER.error("Unable to initialize alignment: %s", exc)
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
