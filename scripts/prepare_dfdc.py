"""Audit an extracted official DFDC training sample; never infer fraud labels."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

FIELDS = ['video_id', 'video_path', 'source_id', 'person_id', 'emotion_label',
          'is_fake', 'is_fraud', 'label_source']
SOURCE = 'https://www.kaggle.com/competitions/deepfake-detection-challenge/data'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate metadata key: {key}')
        result[key] = value
    return result


def video_name(value):
    if (not isinstance(value, str) or not value or '/' in value or '\\' in value
            or ':' in value or Path(value).suffix.lower() != '.mp4'):
        raise ValueError(f'Invalid video filename: {value!r}')
    return value


def audit(root):
    root = root.resolve()
    metadata = root / 'metadata.json'
    records = json.loads(metadata.read_text(encoding='utf-8-sig'),
                         object_pairs_hook=unique_object)
    if not isinstance(records, dict) or not records:
        raise ValueError('metadata.json must be a nonempty filename-to-label mapping')
    labels, samples, missing, external = [], [], [], []
    seen = set()
    for name, record in sorted(records.items()):
        video_name(name)
        if name.casefold() in seen:
            raise ValueError(f'Case-insensitive filename collision: {name}')
        seen.add(name.casefold())
        if not isinstance(record, dict) or record.get('label') not in ('REAL', 'FAKE'):
            raise ValueError(f'Missing or invalid truth label: {name}')
        if record.get('split') != 'train':
            raise ValueError(f'Expected labeled training sample: {name}')
        fake = record['label'] == 'FAKE'
        original = video_name(record.get('original')) if fake else name
        if fake and original == name:
            raise ValueError(f'Fake cannot reference itself: {name}')
        if fake and original in records and records[original].get('label') != 'REAL':
            raise ValueError(f'Original must be REAL: {name}')
        if original not in records:
            external.append(original)
        path = root / name
        if path.resolve().parent != root:
            raise ValueError(f'Video resolves outside dataset: {name}')
        present = path.is_file() and path.stat().st_size > 0
        if not present:
            missing.append(name)
        key = Path(name).stem
        labels.append(dict(zip(FIELDS, [key, str(path), Path(original).stem,
                                       '', '', int(fake), '', SOURCE + '#metadata.json'])))
        samples.append({'id': key, 'path': str(path), 'dataset': 'DFDC',
                        'source_id': Path(original).stem, 'is_fake': int(fake),
                        'is_fraud': None, 'present': present,
                        'bytes': path.stat().st_size if present else 0,
                        'sha256': digest(path) if present else None})
    counts = Counter(r['label'] for r in records.values())
    extras = sorted(p.name for p in root.glob('*.mp4') if p.name not in records)
    report = {'status': 'ready_for_media_validation' if not missing and len(counts) == 2 else 'incomplete',
              'source': SOURCE, 'metadata_sha256': digest(metadata),
              'counts': dict(counts), 'total': len(records), 'missing_or_empty': missing,
              'unlabeled_videos': extras, 'originals_outside_sample': sorted(set(external)),
              'source_groups': len({r['source_id'] for r in samples}),
              'limitations': ['Fraud, emotion and person labels are unknown.',
                             'Source groups do not establish person-disjoint splits.',
                             'File presence and hashes do not verify decoding or audio availability.',
                             'Local metadata provenance must be checked against the official download.']}
    return labels, samples, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Directory containing metadata.json and MP4s')
    parser.add_argument('--output', type=Path, required=True, help='New audit output directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists; choose a new directory')
    labels, samples, report = audit(args.input)
    args.output.mkdir(parents=True)
    with (args.output / 'evaluation_labels.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(labels)
    for name, value in [('manifest.json', {'samples': samples}), ('audit.json', report)]:
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'ready_for_media_validation' else 2


if __name__ == '__main__':
    raise SystemExit(main())
