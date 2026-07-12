"""Recognize facial emotions in extracted video frames and write CSV files.

Frames are read recursively from ``data/processed/frames``. Each directory
containing images is treated as one video and produces one CSV below
``data/processed/emotions/face`` with the same relative directory structure.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "frames"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "emotions" / "face"
)
DEFAULT_MODEL_NAME = "enet_b0_8_best_afew"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class FaceEmotionError(RuntimeError):
    """Raised when face emotion processing cannot be completed."""


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> HSEmotionRecognizer:
    """Load an HSEmotion recognizer configured for CPU inference.

    Args:
        model_name: Name of an HSEmotion-compatible emotion model.

    Returns:
        An initialized CPU HSEmotion recognizer.

    Raises:
        FaceEmotionError: If model initialization or weight download fails.
    """
    LOGGER.info("Loading HSEmotion model '%s' on CPU", model_name)

    # hsemotion 0.3.0 stores a complete timm model object and calls
    # torch.load(path) without specifying weights_only. PyTorch 2.6+ changed
    # that default to True, which cannot deserialize this legacy checkpoint.
    # Limit the compatibility override to HSEmotion initialization only.
    original_torch_load = torch.load

    def load_legacy_hsemotion_checkpoint(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        with patch("torch.load", side_effect=load_legacy_hsemotion_checkpoint):
            return HSEmotionRecognizer(model_name=model_name, device="cpu")
    except Exception as exc:
        raise FaceEmotionError(
            f"Unable to load HSEmotion model '{model_name}': {exc}"
        ) from exc


def load_face_detector() -> cv2.CascadeClassifier:
    """Load OpenCV's built-in frontal-face Haar cascade detector.

    Returns:
        A ready-to-use OpenCV cascade classifier.

    Raises:
        FaceEmotionError: If the bundled cascade file cannot be loaded.
    """
    cascade_path = Path(cv2.data.haarcascades) / (
        "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise FaceEmotionError(
            f"Unable to load OpenCV face detector: {cascade_path}"
        )
    return detector


def find_video_directories(input_dir: Path) -> list[Path]:
    """Return directories that directly contain supported frame images."""
    video_directories = {
        path.parent
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    return sorted(video_directories)


def find_frame_files(video_dir: Path) -> list[Path]:
    """Return supported frame images directly inside one video directory."""
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def get_output_path(
    video_dir: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Map a video frame directory to its corresponding CSV path."""
    relative_dir = video_dir.relative_to(input_dir)
    return (output_dir / relative_dir).with_suffix(".csv")


def detect_primary_face(
    image: np.ndarray[Any, Any],
    detector: cv2.CascadeClassifier,
) -> np.ndarray[Any, Any] | None:
    """Detect and return the largest face crop in a BGR image.

    The largest detected face is used as the primary subject when a frame
    contains multiple people.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    face_bgr = image[y : y + height, x : x + width]
    return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)


def recognize_frame(
    frame_path: Path,
    detector: cv2.CascadeClassifier,
    model: HSEmotionRecognizer,
) -> tuple[str, float]:
    """Recognize the primary face emotion in one frame.

    Args:
        frame_path: Image file to analyze.
        detector: OpenCV face detector.
        model: Loaded HSEmotion recognizer.

    Returns:
        An emotion label and posterior confidence. If no face is found, the
        result is ``("unknown", 0.0)``.

    Raises:
        FaceEmotionError: If the image cannot be read or inference fails.
    """
    image = cv2.imread(str(frame_path))
    if image is None:
        raise FaceEmotionError(f"Unable to read frame image: {frame_path}")

    face = detect_primary_face(image, detector)
    if face is None:
        return "unknown", 0.0

    try:
        emotion, scores = model.predict_emotions(face, logits=False)
        probabilities = np.asarray(scores, dtype=float).reshape(-1)
    except Exception as exc:
        raise FaceEmotionError(
            f"Emotion inference failed for {frame_path}: {exc}"
        ) from exc

    if probabilities.size == 0:
        raise FaceEmotionError(
            f"HSEmotion returned no confidence scores for {frame_path}"
        )
    confidence = float(np.max(probabilities))
    if not math.isfinite(confidence):
        raise FaceEmotionError(
            f"HSEmotion returned an invalid confidence for {frame_path}"
        )

    confidence = min(max(confidence, 0.0), 1.0)
    return str(emotion).lower(), confidence


def process_video(
    video_dir: Path,
    output_path: Path,
    detector: cv2.CascadeClassifier,
    model: HSEmotionRecognizer,
) -> tuple[int, int]:
    """Analyze all frames for one video and atomically write its CSV.

    Frames that cannot be decoded or inferred are logged and represented as
    ``unknown`` so the CSV retains one row per input frame.

    Returns:
        A tuple containing processed and unknown frame counts.

    Raises:
        FaceEmotionError: If the CSV cannot be written.
    """
    frame_files = find_frame_files(video_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    unknown_count = 0

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=("frame", "emotion", "confidence"),
            )
            writer.writeheader()

            for frame_path in frame_files:
                try:
                    emotion, confidence = recognize_frame(
                        frame_path,
                        detector,
                        model,
                    )
                except FaceEmotionError as exc:
                    LOGGER.error("%s", exc)
                    emotion, confidence = "unknown", 0.0

                if emotion == "unknown":
                    unknown_count += 1
                writer.writerow(
                    {
                        "frame": frame_path.name,
                        "emotion": emotion,
                        "confidence": f"{confidence:.6f}",
                    }
                )

        temporary_path.replace(output_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise FaceEmotionError(
            f"Unable to write face emotion CSV {output_path}: {exc}"
        ) from exc

    return len(frame_files), unknown_count


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[int, int, int]:
    """Analyze every video frame directory in the dataset.

    Existing CSV outputs are skipped. The HSEmotion model and face detector
    are each loaded only once.

    Returns:
        A tuple containing completed, skipped, and failed video counts.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        NotADirectoryError: If an input or output path is not a directory.
        FaceEmotionError: If model or detector initialization fails.
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
    video_dirs = find_video_directories(input_dir)
    LOGGER.info("Found %d video frame directorie(s)", len(video_dirs))
    if not video_dirs:
        return 0, 0, 0

    pending_dirs: list[tuple[Path, Path]] = []
    skipped = 0
    for video_dir in video_dirs:
        output_path = get_output_path(video_dir, input_dir, output_dir)
        if output_path.exists():
            LOGGER.info("Skipping existing CSV: %s", output_path)
            skipped += 1
        else:
            pending_dirs.append((video_dir, output_path))

    if not pending_dirs:
        return 0, skipped, 0

    detector = load_face_detector()
    model = load_model(model_name)
    completed = 0
    failed = 0
    for video_dir, output_path in pending_dirs:
        LOGGER.info("Processing: %s -> %s", video_dir, output_path)
        try:
            frame_count, unknown_count = process_video(
                video_dir,
                output_path,
                detector,
                model,
            )
        except FaceEmotionError as exc:
            LOGGER.error("%s", exc)
            failed += 1
        else:
            LOGGER.info(
                "Created CSV with %d frame(s), %d unknown: %s",
                frame_count,
                unknown_count,
                output_path,
            )
            completed += 1

    return completed, skipped, failed


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Source frame directory (default: {DEFAULT_INPUT_DIR})",
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
        help=f"HSEmotion model name (default: {DEFAULT_MODEL_NAME})",
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
    """Run face emotion recognition and return a process exit code."""
    configure_logging()
    args = build_parser().parse_args()

    try:
        completed, skipped, failed = process_dataset(
            args.input,
            args.output,
            args.model,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        FaceEmotionError,
    ) as exc:
        LOGGER.error("Unable to initialize face emotion processing: %s", exc)
        return 1

    LOGGER.info(
        "Finished: %d completed, %d skipped, %d failed",
        completed,
        skipped,
        failed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
