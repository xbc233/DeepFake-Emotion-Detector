"""Run the complete three-modal deepfake emotion analysis for one video."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "processed" / "pipeline"
LOGGER = logging.getLogger("pipeline")


class PipelineError(RuntimeError):
    """Raised when a pipeline stage cannot produce its required artifact."""


def timed(timings: dict[str, float], name: str, action: Callable[[], Any]) -> Any:
    """Run an action and record its wall-clock duration."""
    started = time.perf_counter()
    try:
        return action()
    finally:
        timings[name] = round(time.perf_counter() - started, 3)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dominant_emotion(path: Path) -> dict[str, Any]:
    """Return the most frequent known emotion and its share."""
    emotions = [
        row.get("emotion", "").strip().lower()
        for row in read_csv(path)
        if row.get("emotion", "").strip().lower() not in {"", "unknown", "none"}
    ]
    if not emotions:
        return {"emotion": "unknown", "proportion": 0.0, "samples": 0}
    emotion, count = Counter(emotions).most_common(1)[0]
    return {
        "emotion": emotion,
        "proportion": round(count / len(emotions), 6),
        "samples": len(emotions),
    }


def inconsistent_segments(
    rows: list[dict[str, str]], interval: float, threshold: float
) -> list[dict[str, Any]]:
    """Merge consecutive alignment points below the consistency threshold."""
    segments: list[dict[str, Any]] = []
    current: list[dict[str, str]] = []
    for row in rows + [{}]:
        score = float(row.get("consistency", "inf"))
        if score < threshold:
            current.append(row)
            continue
        if current:
            scores = [float(item["consistency"]) for item in current]
            segments.append(
                {
                    "start": round(float(current[0]["time"]), 3),
                    "end": round(float(current[-1]["time"]) + interval, 3),
                    "average_consistency": round(sum(scores) / len(scores), 6),
                    "samples": len(current),
                }
            )
            current = []
    return segments


def video_info(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineError(f"无法读取视频: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    duration = frame_count / fps if math.isfinite(fps) and fps > 0 else 0.0
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "duration_seconds": round(duration, 3),
        "fps": round(fps, 3) if math.isfinite(fps) else 0.0,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def run_pipeline(args: argparse.Namespace) -> Path:
    video = args.input.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"输入视频不存在: {video}")
    if video.suffix.lower() != ".mp4":
        raise PipelineError("当前流水线仅支持 MP4 视频。")

    work = args.work_dir.resolve() / video.stem
    output = (args.output or (DEFAULT_RESULTS_DIR / f"{video.stem}_report.json")).resolve()
    audio_path = work / "audio" / f"{video.stem}.wav"
    frames_dir = work / "frames"
    transcript_path = work / "transcript" / f"{video.stem}.json"
    face_csv = work / "emotions" / "face.csv"
    speech_csv = work / "emotions" / "speech.csv"
    text_csv = work / "emotions" / "text.csv"
    alignment_csv = work / "alignment.csv"
    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    LOGGER.info("[1/7] 提取音频")
    extract_audio = importlib.import_module("extract_audio")
    timed(timings, "audio_extraction", lambda: extract_audio.extract_audio(
        video, audio_path, extract_audio.find_ffmpeg()))

    LOGGER.info("[2/7] 提取视频帧")
    extract_frames = importlib.import_module("extract_frames")
    timed(timings, "frame_extraction", lambda: extract_frames.extract_frames(
        video, frames_dir, args.frame_fps))

    LOGGER.info("[3/7] 语音转写")
    speech_to_text = importlib.import_module("speech_to_text")
    def transcribe() -> None:
        model = speech_to_text.load_whisper_model(args.whisper_model_dir)
        result = speech_to_text.transcribe_audio(model, audio_path, args.language)
        speech_to_text.save_transcript(result, transcript_path)
    timed(timings, "speech_transcription", transcribe)

    LOGGER.info("[4/7] 三模态情感识别")
    face_emotion = importlib.import_module("face_emotion")
    def analyze_face() -> None:
        face_emotion.process_video(frames_dir, face_csv,
                                   face_emotion.load_face_detector(),
                                   face_emotion.load_model(args.face_model))
    timed(timings, "face_emotion", analyze_face)

    speech_emotion = importlib.import_module("speech_emotion")
    def analyze_speech() -> None:
        extractor, model = speech_emotion.load_model(args.speech_model)
        speech_emotion.process_audio(audio_path, speech_csv, extractor, model,
                                     args.speech_window)
    timed(timings, "speech_emotion", analyze_speech)

    text_sentiment = importlib.import_module("text_sentiment")
    def analyze_text() -> None:
        classifier = text_sentiment.load_classifier(args.text_model)
        text_sentiment.process_transcript(transcript_path, text_csv, classifier,
                                          args.text_batch_size)
    timed(timings, "text_emotion", analyze_text)

    LOGGER.info("[5/7] 时间对齐")
    alignment = importlib.import_module("alignment")
    average = timed(timings, "alignment", lambda: alignment.process_video(
        face_csv, speech_csv, text_csv, alignment_csv, args.interval,
        args.frame_fps))

    LOGGER.info("[6/7] 风险评分与结论")
    rows = read_csv(alignment_csv)
    risk = round(1.0 - min(max(float(average), 0.0), 1.0), 6)
    report = {
        "schema_version": "1.0",
        "video": video_info(video),
        "main_emotions": {
            "face": dominant_emotion(face_csv),
            "speech": dominant_emotion(speech_csv),
            "text": dominant_emotion(text_csv),
        },
        "average_consistency": round(float(average), 6),
        "inconsistent_segments": inconsistent_segments(
            rows, args.interval, args.inconsistency_threshold),
        "deepfake_risk_score": risk,
        "prediction": "fake" if risk >= args.fake_threshold else "real",
        "thresholds": {
            "fake": args.fake_threshold,
            "inconsistency": args.inconsistency_threshold,
        },
        "timings_seconds": timings,
        "artifacts": {
            "audio": str(audio_path), "frames": str(frames_dir),
            "transcript": str(transcript_path), "face_emotion": str(face_csv),
            "speech_emotion": str(speech_csv), "text_emotion": str(text_csv),
            "alignment": str(alignment_csv),
        },
    }
    report["timings_seconds"]["total"] = round(time.perf_counter() - total_started, 3)

    LOGGER.info("[7/7] 写入 JSON 报告")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="待检测 MP4 视频")
    parser.add_argument("--output", type=Path, help="JSON 报告路径")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--frame-fps", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--speech-window", type=float, default=3.0)
    parser.add_argument("--language", help="Whisper 语言代码，例如 zh；默认自动检测")
    parser.add_argument("--whisper-model-dir", type=Path,
                        default=PROJECT_ROOT / "models" / "whisper")
    parser.add_argument("--face-model", default="enet_b0_8_best_afew")
    parser.add_argument("--speech-model", default="superb/wav2vec2-base-superb-er")
    parser.add_argument("--text-model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--fake-threshold", type=float, default=0.5)
    parser.add_argument("--inconsistency-threshold", type=float, default=1.0)
    return parser


def main() -> int:
    # Keep Chinese CLI help/logs readable when output is redirected on Windows.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()
    if not (0 <= args.fake_threshold <= 1 and 0 <= args.inconsistency_threshold <= 1):
        LOGGER.error("阈值必须位于 0 到 1 之间。")
        return 2
    if args.frame_fps <= 0 or args.interval <= 0 or args.speech_window <= 0:
        LOGGER.error("帧率、对齐间隔和语音窗口必须大于 0。")
        return 2
    if args.text_batch_size <= 0:
        LOGGER.error("文本批大小必须大于 0。")
        return 2
    try:
        output = run_pipeline(args)
    except Exception as exc:
        LOGGER.error("流水线失败: %s", exc)
        return 1
    LOGGER.info("检测完成，报告已写入: %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
