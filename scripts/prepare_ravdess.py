"""Verify official RAVDESS archives and select a reproducible 24-clip pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://zenodo.org/records/1188976"
ARCHIVES = {
    "01": "3c8ececaf392b4a9b11b32271f4f6d01",
    "02": "a6f40d413b2e6ef25b3a595099e59abb",
}
EMOTIONS = {"01": "neutral", "03": "happy", "04": "sad",
            "05": "anger", "06": "fear", "08": "surprise"}
STATEMENTS = {"01": "Kids are talking by the door.",
              "02": "Dogs are sitting by the door."}


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def selected_names(actor):
    # Full AV, speech, normal intensity, first repetition, both statements.
    return [f"01-01-{emotion}-01-{statement}-01-{actor}.mp4"
            for emotion in EMOTIONS for statement in STATEMENTS]


def prepare(archive_dir, output_dir, manifest_path, download=False):
    # Validate both archives before extracting any sample.
    verified = []
    for actor, expected in ARCHIVES.items():
        path = archive_dir / f"Video_Speech_Actor_{actor}.zip"
        if not path.is_file():
            path = path.with_suffix(".zip.part")
        if download:
            from ravdess_download import RemoteArchive
            path = RemoteArchive(actor, archive_dir)
        elif not path.is_file() or digest(path, "md5") != expected:
            raise ValueError(f"Missing or incorrect official archive: {path}")
        verified.append((actor, path, expected))
    samples = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for actor, path, expected in verified:
        with zipfile.ZipFile(path) as archive:
            members = {}
            for item in archive.infolist():
                if item.is_dir() or "__MACOSX" in item.filename:
                    continue
                name = Path(item.filename).name
                if name in members:
                    raise ValueError(f"Duplicate archive basename: {name}")
                members[name] = item
            for name in selected_names(actor):
                print(f"Extracting and CRC-checking {name}", flush=True)
                # Write a known basename, never extract arbitrary ZIP paths.
                target = output_dir / name
                content = archive.read(members[name])
                target.write_bytes(content)
                parts = target.stem.split("-")
                samples.append({
                    "id": target.stem, "path": target.relative_to(ROOT).as_posix(),
                    "sha256": digest(target), "dataset": "RAVDESS",
                    "source_url": SOURCE, "source_archive": f"Video_Speech_Actor_{actor}.zip",
                    "official_archive_md5": expected, "archive_md5_verified": not download,
                    "member_crc32": f"{members[name].CRC:08x}",
                    "verification": "ZIP member CRC32 + extracted SHA256" if download else "archive MD5 + member CRC32 + extracted SHA256",
                    "actor": actor,
                    "split": "pilot", "language": "en",
                    "emotion": EMOTIONS[parts[2]],
                    "reference_text": STATEMENTS[parts[4]],
                    "authenticity_label": None,
                    "recording_type": "original_actor_recording",
                })
    manifest = {
        "schema_version": "1.0", "dataset": "RAVDESS",
        "source_url": SOURCE, "doi": "10.5281/zenodo.1188976",
        "citation": "Livingstone SR, Russo FA (2018). PLoS ONE 13(5): e0196391.",
        "license": "CC BY-NC-SA 4.0",
        "selection": "Actors 01/02; full AV speech; emotions 01/03/04/05/06/08; "
                     "normal intensity; statements 01/02; repetition 01. No selection by model result.",
        "purpose": "Emotion-module pilot, not a held-out deepfake benchmark. No threshold tuning.",
        "samples": samples,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=ROOT / "data/raw/ravdess/archives")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/ravdess/pilot24")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/metadata/ravdess_pilot24.json")
    parser.add_argument("--download", action="store_true", help="Download only selected ZIP members using HTTP ranges")
    args = parser.parse_args()
    result = prepare(args.archives.resolve(), args.output.resolve(), args.manifest.resolve(), args.download)
    print(f"Verified selected {len(result['samples'])} clips. Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
