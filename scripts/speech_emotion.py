"""Recognize speech emotions in WAV datasets with Hugging Face Wav2Vec2.

Audio is read recursively from ``data/processed/audio``. One time-aligned CSV
is written per audio file below ``data/processed/emotions/speech`` while
preserving the source directory structure.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "audio"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "emotions" / "speech"
)
DEFAULT_MODEL_NAME = "superb/wav2vec2-base-superb-er"
DEFAULT_WINDOW_SECONDS = 3.0
TARGET_SAMPLE_RATE = 16_000
LABEL_ALIASES = {
    "ang": "anger",
    "anger": "anger",
    "hap": "happiness",
    "happy": "happiness",
    "happiness": "happiness",
    "neu": "neutral",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
}


class SpeechEmotionError(RuntimeError):
    """Raised when speech emotion processing cannot be completed."""


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> tuple[Any, Any]:
    """Load a Hugging Face audio classifier and feature extractor on CPU.

    Model files are downloaded automatically by Transformers on first use.

    Args:
        model_name: Hugging Face model repository name or local model path.

    Returns:
        A tuple containing the feature extractor and evaluation-mode model.

    Raises:
        SpeechEmotionError: If model initialization or download fails.
    """
    LOGGER.info("Loading speech emotion model '%s' on CPU", model_name)
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        model.to(torch.device("cpu"))
        model.eval()
    except Exception as exc:
        raise SpeechEmotionError(
            f"Unable to load Hugging Face model '{model_name}': {exc}"
        ) from exc
    return feature_extractor, model


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
    """Map a source WAV to a CSV path while preserving subdirectories."""
    relative_path = audio_path.relative_to(input_dir)
    return (output_dir / relative_path).with_suffix(".csv")


def read_wav(audio_path: Path) -> tuple[np.ndarray[Any, Any], int]:
    """Read an uncompressed PCM WAV file as a mono float32 waveform.

    Args:
        audio_path: WAV file to read.

    Returns:
        A mono waveform in ``[-1, 1]`` and its original sample rate.

    Raises:
        SpeechEmotionError: If the WAV format is unsupported or unreadable.
    """
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()
            raw_audio = wav_file.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise SpeechEmotionError(
            f"Unable to read WAV file {audio_path}: {exc}"
        ) from exc

    if compression != "NONE":
        raise SpeechEmotionError(f"Compressed WAV is unsupported: {audio_path}")
    if channels < 1 or sample_rate <= 0 or frame_count == 0:
        raise SpeechEmotionError(f"WAV contains no valid audio: {audio_path}")

    dtype_map = {
        1: np.dtype("uint8"),
        2: np.dtype("<i2"),
        4: np.dtype("<i4"),
    }
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise SpeechEmotionError(
            f"Unsupported {sample_width * 8}-bit PCM WAV: {audio_path}"
        )

    samples = np.frombuffer(raw_audio, dtype=dtype).astype(np.float32)
    if sample_width == 1:
        samples = (samples - 128.0) / 128.0
    else:
        samples /= float(2 ** (sample_width * 8 - 1))

    try:
        samples = samples.reshape(-1, channels).mean(axis=1)
    except ValueError as exc:
        raise SpeechEmotionError(
            f"WAV sample data is malformed: {audio_path}"
        ) from exc
    return samples, sample_rate


def resample_audio(
    waveform: np.ndarray[Any, Any],
    source_rate: int,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray[Any, Any]:
    """Resample a mono waveform using linear interpolation when necessary."""
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Audio sample rates must be positive integers.")

    output_length = max(1, round(len(waveform) * target_rate / source_rate))
    source_positions = np.arange(len(waveform), dtype=np.float64)
    target_positions = np.linspace(
        0,
        max(len(waveform) - 1, 0),
        output_length,
        dtype=np.float64,
    )
    resampled = np.interp(target_positions, source_positions, waveform)
    return resampled.astype(np.float32)


def normalize_label(label: str) -> str:
    """Convert abbreviated model labels to stable English emotion names."""
    normalized = label.strip().lower()
    return LABEL_ALIASES.get(normalized, normalized)


def predict_emotion(
    waveform: np.ndarray[Any, Any],
    feature_extractor: Any,
    model: Any,
) -> tuple[str, float]:
    """Predict an emotion and confidence for one 16 kHz audio window."""
    minimum_length = TARGET_SAMPLE_RATE // 10
    if waveform.size < minimum_length:
        waveform = np.pad(waveform, (0, minimum_length - waveform.size))

    try:
        inputs = feature_extractor(
            waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        cpu_inputs = {name: tensor.to("cpu") for name, tensor in inputs.items()}
        with torch.inference_mode():
            logits = model(**cpu_inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_id].item())
        label = model.config.id2label[predicted_id]
    except Exception as exc:
        raise SpeechEmotionError(f"Speech emotion inference failed: {exc}") from exc

    if not math.isfinite(confidence):
        raise SpeechEmotionError("Model returned a non-finite confidence score.")
    return normalize_label(str(label)), min(max(confidence, 0.0), 1.0)


def analyze_audio(
    audio_path: Path,
    feature_extractor: Any,
    model: Any,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> list[dict[str, str]]:
    """Analyze one WAV file in consecutive time windows.

    Returns:
        Rows containing window start time, emotion, and confidence.

    Raises:
        ValueError: If *window_seconds* is invalid.
        SpeechEmotionError: If audio loading or inference fails.
    """
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Window duration must be a finite positive number.")

    waveform, source_rate = read_wav(audio_path)
    waveform = resample_audio(waveform, source_rate)
    window_size = max(1, round(window_seconds * TARGET_SAMPLE_RATE))
    rows: list[dict[str, str]] = []

    for start_sample in range(0, len(waveform), window_size):
        window = waveform[start_sample : start_sample + window_size]
        emotion, confidence = predict_emotion(
            window,
            feature_extractor,
            model,
        )
        start_time = start_sample / TARGET_SAMPLE_RATE
        rows.append(
            {
                "time": f"{start_time:.3f}",
                "emotion": emotion,
                "confidence": f"{confidence:.6f}",
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Atomically write speech emotion rows to a UTF-8 CSV file."""
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
                fieldnames=("time", "emotion", "confidence"),
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise SpeechEmotionError(
            f"Unable to write speech emotion CSV {output_path}: {exc}"
        ) from exc


def process_audio(
    audio_path: Path,
    output_path: Path,
    feature_extractor: Any,
    model: Any,
    window_seconds: float,
) -> bool:
    """Analyze one WAV and write its CSV, or skip an existing output.

    Returns:
        ``True`` if a CSV is created and ``False`` if it already exists.
    """
    if output_path.exists():
        LOGGER.info("Skipping existing CSV: %s", output_path)
        return False
    rows = analyze_audio(
        audio_path,
        feature_extractor,
        model,
        window_seconds,
    )
    write_csv(rows, output_path)
    return True


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> tuple[int, int, int]:
    """Analyze every WAV below the input directory with one model instance.

    Returns:
        A tuple containing successful, skipped, and failed file counts.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        NotADirectoryError: If an input or output path is not a directory.
        ValueError: If *window_seconds* is invalid.
        SpeechEmotionError: If model initialization fails.
    """
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Window duration must be a finite positive number.")

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

    pending_files: list[tuple[Path, Path]] = []
    skipped = 0
    for audio_path in audio_files:
        output_path = get_output_path(audio_path, input_dir, output_dir)
        if output_path.exists():
            LOGGER.info("Skipping existing CSV: %s", output_path)
            skipped += 1
        else:
            pending_files.append((audio_path, output_path))

    if not pending_files:
        return 0, skipped, 0

    feature_extractor, model = load_model(model_name)
    succeeded = 0
    failed = 0
    for audio_path, output_path in pending_files:
        LOGGER.info("Processing: %s -> %s", audio_path, output_path)
        try:
            process_audio(
                audio_path,
                output_path,
                feature_extractor,
                model,
                window_seconds,
            )
        except (SpeechEmotionError, ValueError) as exc:
            LOGGER.error("%s", exc)
            failed += 1
        else:
            LOGGER.info("Created speech emotion CSV: %s", output_path)
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
        help=f"Destination CSV directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"Seconds per prediction window (default: {DEFAULT_WINDOW_SECONDS:g})",
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
    """Run batch speech emotion recognition and return a process exit code."""
    configure_logging()
    args = build_parser().parse_args()
    try:
        succeeded, skipped, failed = process_dataset(
            args.input,
            args.output,
            args.model,
            args.window_seconds,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        SpeechEmotionError,
        ValueError,
    ) as exc:
        LOGGER.error("Unable to initialize speech emotion processing: %s", exc)
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
