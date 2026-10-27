"""Freeze a source-group-disjoint LAV-DF subset without opening media or fitting."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 'lavdf-expansion-v1-20261025'
QUOTAS = {'train': 80, 'dev': 20, 'test': 20}
COMBINATIONS = [(False, False), (False, True), (True, False), (True, True)]


def select(records, excluded):
    groups = defaultdict(list)
    filenames = set()
    for record in records:
        name = record['file']
        if name in filenames:
            raise ValueError(f'Duplicate file: {name}')
        filenames.add(name)
        groups[record['original'] or name].append(record)
    eligible = defaultdict(list)
    for key, members in groups.items():
        splits = {r['split'] for r in members}
        if len(splits) != 1:
            raise ValueError(f'Source leakage: {key}')
        if key in excluded:
            continue
        combos = {(r['modify_audio'], r['modify_video']) for r in members}
        if not set(COMBINATIONS) <= combos:
            continue
        if not any(r['file'] == key and not r['modify_audio'] and not r['modify_video'] for r in members):
            raise ValueError(f'Missing real source: {key}')
        eligible[next(iter(splits))].append(key)
    selected = []
    for split, quota in QUOTAS.items():
        ordered = sorted(eligible[split], key=lambda k: hashlib.sha256((SEED+k).encode()).hexdigest())
        if len(ordered) < quota:
            raise ValueError(f'Insufficient groups in {split}')
        for group in ordered[:quota]:
            for audio, video in COMBINATIONS:
                candidates = sorted((r for r in groups[group] if r['modify_audio'] == audio and r['modify_video'] == video), key=lambda r: r['file'])
                record = candidates[0]
                if not audio and not video:
                    record = next(r for r in candidates if r['file'] == group)
                selected.append({'id': record['file'].replace('/', '__').removesuffix('.mp4'),
                                 'source_id': group, 'split': split, 'metadata': record,
                                 'path': str(ROOT / 'data/raw/lavdf' / record['file']),
                                 'is_fake': int(audio or video), 'is_fraud': None})
    return selected


def main():
    source = ROOT / 'data/processed/lavdf/metadata_validated_intervals.json'
    pilot = ROOT / 'experiments/lavdf_pilot16_20261024/manifest.json'
    excluded = {s['metadata']['original'] or s['metadata']['file'] for s in json.loads(pilot.read_text())['samples']}
    selected = select(json.loads(source.read_text()), excluded)
    counts = Counter(s['split'] for s in selected)
    out = ROOT / 'experiments/lavdf_expansion480_20261025'
    out.mkdir(exist_ok=True)
    target = out / 'selection.json'
    result = {'seed': SEED, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'groups_per_split': QUOTAS, 'videos_per_split': dict(counts),
              'excluded_pilot_sources': sorted(excluded), 'samples': selected,
              'limitations': ['source group separation is not verified identity separation',
                             'balanced modification categories imply 75% fake; not deployment prevalence',
                             'test reserved; no threshold selection or feature selection on test'],
              'status': 'selection_frozen_media_not_verified'}
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if target.exists() and target.read_text(encoding='utf-8') != serialized:
        raise FileExistsError('Refusing to replace a different frozen selection')
    target.write_text(serialized, encoding='utf-8')
    print(json.dumps({'videos': len(selected), 'groups': len({s['source_id'] for s in selected}), 'splits': counts}))


if __name__ == '__main__':
    main()
