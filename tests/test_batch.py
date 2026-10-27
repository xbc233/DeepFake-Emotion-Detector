import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_batch
from prepare_ravdess import digest, selected_names


def row(emotion="happy", actor="01", **changes):
    value = {
        "status": "success", "emotion": emotion, "actor": actor,
        "predictions": {"face": "happy", "speech": "happy", "text": "neutral"},
        "face_correct": True, "speech_correct": True,
        "speech_target_supported": True, "face_known_frames": 3, "face_frames": 4,
        "fully_observed_points": 3, "alignment_points": 4,
        "prediction": "fake", "risk_score": 0.7,
        "word_errors": 1, "reference_words": 6,
    }
    value.update(changes)
    return value


def test_pilot_selection_balanced_and_excludes_silent_video():
    from collections import Counter
    names = selected_names("01") + selected_names("02")
    assert len(names) == len(set(names)) == 24
    assert all(name.startswith("01-01-") for name in names)
    assert set(Counter(name.split("-")[2] for name in names).values()) == {4}
    assert all(name.split("-")[3] == "01" and name.split("-")[5] == "01" for name in names)


def test_remote_zip_reader_combines_local_prefix_ranges_and_tail(tmp_path, monkeypatch):
    import io
    import zipfile
    import ravdess_download
    payload = bytes(range(256)) * 1000
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("sample.mp4", payload)
    data = buffer.getvalue()
    monkeypatch.setitem(ravdess_download.SIZES, "01", len(data))
    monkeypatch.setattr(ravdess_download.RemoteArchive, "fetch",
                        lambda self, start, end: data[start:end + 1])
    (tmp_path / "Video_Speech_Actor_01.zip.part").write_bytes(data[:1024])
    reader = ravdess_download.RemoteArchive("01", tmp_path)
    with zipfile.ZipFile(reader) as archive:
        assert archive.read("sample.mp4") == payload


def test_failed_samples_and_unsupported_emotions_have_explicit_denominators():
    results = [row(), row("fear", speech_target_supported=False, speech_correct=False,
                          face_correct=False, face_known_frames=0,
                          predictions={"face": "unknown", "speech": "sad", "text": "neutral"}),
               {"status": "failed"}]
    summary = run_batch.aggregate(results, 4)
    assert (summary["succeeded"], summary["failed"], summary["pending"]) == (2, 1, 1)
    assert summary["overall"]["face_accuracy"] == 0.5
    assert summary["overall"]["speech_accuracy_all_six_labels"] == 0.5
    assert summary["overall"]["speech_accuracy_supported_four_labels"] == 1.0
    assert summary["overall"]["speech_supported_n"] == 1
    assert summary["deepfake_metrics"] is None
    assert summary["confusion"]["face"]["fear"] == {"unknown": 1}


def test_empty_run_has_no_fabricated_scores():
    metrics = run_batch.aggregate([], 24)
    assert metrics["overall"]["face_accuracy"] is None
    assert metrics["overall"]["high_risk_fraction"] is None
    assert metrics["pending"] == 24


def test_asr_word_errors_include_insertions_deletions_and_substitutions():
    assert run_batch.word_errors("Kids are talking by the door.", "KIDS are talking by the door!") == (0, 6)
    assert run_batch.word_errors("dogs are sitting", "kids sitting here") == (3, 3)
    assert run_batch.word_errors("dogs", "dogs and cats") == (2, 1)
    assert run_batch.word_errors("dogs are sitting", "") == (3, 3)


def test_manifest_detects_changed_input(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"original")
    manifest = {"samples": [{"id": "sample", "path": str(video),
                            "sha256": digest(video), "dataset": "RAVDESS", "split": "pilot"}]}
    run_batch.validate_manifest(manifest)
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        run_batch.validate_manifest(manifest)


def test_batch_records_failures_continues_and_rejects_reusing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run_batch, "ROOT", tmp_path)
    samples = []
    for name in ("one", "two"):
        video = tmp_path / f"{name}.mp4"
        video.write_bytes(b"sample")
        samples.append({"id": name, "path": video.name, "sha256": digest(video),
                        "dataset": "RAVDESS", "split": "pilot", "actor": "01",
                        "emotion": "happy", "language": "en"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(run_batch.subprocess, "run", fail)
    output = tmp_path / "experiment"
    assert run_batch.run_batch(manifest, output, 10) == 1
    assert len(calls) == 2
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["failed"] == 2 and metrics["pending"] == 0
    assert metrics["overall"]["face_accuracy"] is None
    with pytest.raises(FileExistsError):
        run_batch.run_batch(manifest, output, 10)
