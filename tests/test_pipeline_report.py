import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_pipeline import (PipelineError, build_parser, dominant_emotion,
                          inconsistent_segments, run_pipeline)


def test_segment_end_stays_within_video():
    rows = [{"time": "15.0", "consistency": "0"},
            {"time": "15.5", "consistency": "0"}]
    result = inconsistent_segments(rows, 0.5, 1.0, duration=15.067)
    assert result == [{"start": 15.0, "end": 15.067,
                       "average_consistency": 0.0, "samples": 1}]


def test_summary_combines_emotion_aliases(tmp_path):
    path = tmp_path / "face.csv"
    path.write_text("emotion\nhappiness\nhappy\nneutral\nnan\n", encoding="utf-8")
    assert dominant_emotion(path) == {
        "emotion": "happy", "proportion": 0.666667, "samples": 3}


def test_offline_missing_checkpoint_fails_before_processing(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.touch()
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    args = build_parser().parse_args([
        "--input", str(video), "--offline",
        "--whisper-model-dir", str(tmp_path / "missing")])
    with pytest.raises(PipelineError, match="离线模式缺少模型"):
        run_pipeline(args)
