"""Run a labeled emotion pilot with per-video logs and auditable summaries."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from alignment import normalize_emotion
from prepare_ravdess import digest
from run_pipeline import read_csv

ROOT = Path(__file__).resolve().parents[1]
SPEECH_LABELS = {"neutral", "happy", "sad", "anger"}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def word_errors(reference, hypothesis):
    """Levenshtein word errors; punctuation/case are ignored, no ASR correction."""
    ref = re.findall(r"[a-z]+(?:'[a-z]+)?", reference.lower())
    hyp = re.findall(r"[a-z]+(?:'[a-z]+)?", hypothesis.lower())
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1], len(ref)


def validate_manifest(manifest):
    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError("Manifest contains no samples")
    ids = set()
    for sample in samples:
        key = sample["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or key in ids:
            raise ValueError(f"Unsafe or duplicate sample ID: {key}")
        ids.add(key)
        if sample["dataset"] != "RAVDESS" or sample["split"] != "pilot":
            raise ValueError("This runner evaluates the RAVDESS pilot only")
        path = (ROOT / sample["path"]).resolve()
        if not path.is_file() or digest(path) != sample["sha256"]:
            raise ValueError(f"Missing or changed sample: {path}")


def summarize_sample(sample, report):
    artifacts = report["artifacts"]
    if not all(Path(path).exists() for path in artifacts.values()):
        raise ValueError("Report references missing artifacts")
    face_rows = read_csv(Path(artifacts["face_emotion"]))
    speech_rows = read_csv(Path(artifacts["speech_emotion"]))
    text_rows = read_csv(Path(artifacts["text_emotion"]))
    aligned = read_csv(Path(artifacts["alignment"]))
    if not face_rows or not speech_rows or not text_rows or not aligned:
        raise ValueError("A modality produced empty results")
    transcript = json.loads(Path(artifacts["transcript"]).read_text(encoding="utf-8"))
    risk = report["deepfake_risk_score"]
    if not math.isfinite(risk) or not 0 <= risk <= 1:
        raise ValueError("Invalid risk score")
    predictions = {key: normalize_emotion(value["emotion"])
                   for key, value in report["main_emotions"].items()}
    errors, words = word_errors(sample["reference_text"], transcript["text"])
    return {
        "status": "success", "emotion": sample["emotion"],
        "predictions": predictions,
        "face_correct": predictions["face"] == sample["emotion"],
        "speech_correct": predictions["speech"] == sample["emotion"],
        "speech_target_supported": sample["emotion"] in SPEECH_LABELS,
        "face_frames": len(face_rows),
        "face_known_frames": sum(normalize_emotion(row["emotion"]) != "unknown" for row in face_rows),
        "alignment_points": len(aligned),
        "fully_observed_points": sum(all(normalize_emotion(row[k]) != "unknown"
                                         for k in ("face", "speech", "text")) for row in aligned),
        "risk_score": risk, "prediction": report["prediction"],
        "threshold": report["thresholds"]["fake"],
        "transcript": transcript["text"], "word_errors": errors, "reference_words": words,
        "pipeline_seconds": report["timings_seconds"]["total"],
    }


def ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def group_metrics(rows):
    supported = [r for r in rows if r["speech_target_supported"]]
    return {
        "n": len(rows),
        "face_accuracy": ratio(sum(r["face_correct"] for r in rows), len(rows)),
        "speech_accuracy_all_six_labels": ratio(sum(r["speech_correct"] for r in rows), len(rows)),
        "speech_supported_n": len(supported),
        "speech_accuracy_supported_four_labels": ratio(sum(r["speech_correct"] for r in supported), len(supported)),
        "face_frame_coverage": ratio(sum(r["face_known_frames"] for r in rows), sum(r["face_frames"] for r in rows)),
        "three_modality_coverage": ratio(sum(r["fully_observed_points"] for r in rows), sum(r["alignment_points"] for r in rows)),
        "high_risk_count": sum(r["prediction"] == "fake" for r in rows),
        "high_risk_fraction": ratio(sum(r["prediction"] == "fake" for r in rows), len(rows)),
        "asr_word_error_rate": ratio(sum(r["word_errors"] for r in rows), sum(r["reference_words"] for r in rows)),
        "mean_risk_score": ratio(sum(r["risk_score"] for r in rows), len(rows)),
    }


def aggregate(results, total):
    good = [r for r in results if r["status"] == "success"]
    return {
        "planned": total, "succeeded": len(good),
        "failed": sum(r["status"] == "failed" for r in results),
        "pending": total - len(results),
        "overall": group_metrics(good),
        "by_emotion": {key: group_metrics([r for r in good if r["emotion"] == key])
                       for key in sorted({r["emotion"] for r in good})},
        "by_actor": {key: group_metrics([r for r in good if r["actor"] == key])
                     for key in sorted({r["actor"] for r in good})},
        "confusion": {modality: {key: dict(Counter(r["predictions"][modality] for r in good if r["emotion"] == key))
                                  for key in sorted({r["emotion"] for r in good})}
                      for modality in ("face", "speech")},
        "deepfake_metrics": None,
        "limitations": [
            "24 selected clips from two actors; pilot only, no held-out generalization claim.",
            "Actor emotion is an audiovisual target, not a text-emotion target; no text-emotion accuracy.",
            "Speech model cannot predict fear/surprise; supported-label accuracy is reported separately.",
            "No fake examples or deepfake benchmark labels; no accuracy, recall or ROC-AUC for authenticity.",
            "High-risk fraction describes original actor recordings, not general deepfake performance.",
            "Failed clips are excluded from metric denominators and reported separately; missing faces remain incorrect.",
            "Nearest-neighbor alignment, silent windows, and missing-modality scoring remain baseline limitations.",
        ],
    }


def percent(value):
    return "N/A" if value is None else f"{value:.1%}"


def save_summary(output, results, total):
    metrics = aggregate(results, total)
    write_json(output / "results.json", results)
    write_json(output / "metrics.json", metrics)
    m = metrics["overall"]
    lines = ["# RAVDESS 小批量实验", "",
             f"计划 {total} 段；成功 {metrics['succeeded']}；失败 {metrics['failed']}；待运行 {metrics['pending']}。", "",
             "固定两位演员、六类情感、每类四段。仅用于模块检查，未调阈值（0.5）。", "",
             "| 指标 | 结果 |", "|---|---:|",
             f"| 人脸主情感准确率（片段级） | {percent(m['face_accuracy'])} |",
             f"| 语音主情感准确率（全部六类） | {percent(m['speech_accuracy_all_six_labels'])} |",
             f"| 语音主情感准确率（支持的四类，n={m['speech_supported_n']}） | {percent(m['speech_accuracy_supported_four_labels'])} |",
             f"| 检出人脸并获得情感的帧比例 | {percent(m['face_frame_coverage'])} |",
             f"| 三模态均有标签的对齐点比例 | {percent(m['three_modality_coverage'])} |",
             f"| 原始录制样本高风险判定比例 | {m['high_risk_count']}/{m['n']}（{percent(m['high_risk_fraction'])}） |",
             f"| 转写词错误率 WER | {percent(m['asr_word_error_rate'])} |", "",
             "语音模型仅支持 neutral/happy/sad/anger。fear/surprise 的六类计分保留为错误，另报四类指标。", "",
             "文本内容是固定句子，不能用演员表演情感当作文字情感真值；本轮只核对转写。", "",
             "没有伪造样本，不报告真假分类准确率、召回率或 AUC；高风险比例也不是通用误报率。", "",
             "准确率按每段视频的主情感计算；未检出人脸的整段样本仍计为错误。失败任务单列，其他指标仅覆盖成功样本。", "",
             "| 样本 | 目标情感 | 人脸 | 语音 | 文本 | 风险分数 | 状态 |",
             "|---|---|---|---|---|---:|---|"]
    for row in results:
        if row["status"] == "success":
            p = row["predictions"]
            lines.append(f"| {row['id']} | {row['emotion']} | {p['face']} | {p['speech']} | {p['text']} | {row['risk_score']:.3f} | {row['prediction']} |")
        else:
            lines.append(f"| {row['id']} | {row['emotion']} | — | — | — | — | failed（见日志） |")
    lines += ["", "完整结果与混淆矩阵见 results.json / metrics.json；参数、版本和源代码校验值见 run.json。",
              "", "来源：[RAVDESS 官方数据](https://zenodo.org/records/1188976)，Livingstone & Russo (2018)，CC BY-NC-SA 4.0。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_batch(manifest_path, output, timeout):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    # Each experiment owns its output directory; never mix stale successes.
    output.mkdir(parents=True, exist_ok=False)
    (output / "reports").mkdir()
    (output / "logs").mkdir()
    packages = {}
    for name in ("torch", "transformers", "openai-whisper", "hsemotion", "timm", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json(output / "manifest.json", manifest)
    write_json(output / "run.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": digest(manifest_path), "python": sys.version,
        "packages": packages, "offline": True, "fake_threshold": 0.5,
        "timeout_seconds": timeout,
        "scripts_sha256": {p.name: digest(p) for p in sorted((ROOT / "scripts").glob("*.py"))},
    })
    results = []
    samples = manifest["samples"]
    save_summary(output, results, len(samples))
    for index, sample in enumerate(samples, 1):
        report_path = output / "reports" / f"{sample['id']}.json"
        log_path = output / "logs" / f"{sample['id']}.log"
        command = [sys.executable, str(ROOT / "scripts/run_pipeline.py"),
                   "--input", str(ROOT / sample["path"]), "--offline",
                   "--language", sample["language"], "--fake-threshold", "0.5",
                   "--output", str(report_path), "--work-dir", str(output / "artifacts")]
        result = {"id": sample["id"], "actor": sample["actor"],
                  "emotion": sample["emotion"], "report": str(report_path), "log": str(log_path)}
        started = time.perf_counter()
        print(f"[{index}/{len(samples)}] {sample['id']}", flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=timeout)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result.update(summarize_sample(sample, report))
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        result["wall_seconds"] = round(time.perf_counter() - started, 3)
        results.append(result)
        save_summary(output, results, len(samples))
        print(f"  {result['status']} ({result['wall_seconds']:.1f}s)", flush=True)
    return 1 if any(r["status"] == "failed" for r in results) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/metadata/ravdess_pilot24.json")
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    return run_batch(args.manifest.resolve(), args.output.resolve(), args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
