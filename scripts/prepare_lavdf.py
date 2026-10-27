"""Validate official LAV-DF metadata and export separate fake/fraud labels."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath


def prepare(records, root):
    if not isinstance(records, list) or not records:
        raise ValueError('Expected nonempty official metadata list')
    root = root.resolve()
    present = {p.relative_to(root).as_posix() for p in root.rglob('*.mp4') if p.is_file()}
    rows = []; seen = set()
    for r in records:
        name = r['file']
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or name in seen:
            raise ValueError('Unsafe or duplicate path: ' + name)
        seen.add(name)
        for key in ('modify_audio', 'modify_video'):
            if type(r[key]) is not bool:
                raise ValueError('Expected boolean ' + key)
        if r['split'] not in ('train', 'dev', 'test'):
            raise ValueError('Unknown official split: ' + str(r['split']))
        duration = r['duration']
        if isinstance(duration, bool) or not isinstance(duration, (int,float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('Invalid duration')
        periods = r['fake_periods']
        if not isinstance(periods, list) or type(r['n_fakes']) is not int or r['n_fakes'] != len(periods):
            raise ValueError('Invalid fake period count')
        for period in periods:
            if len(period) != 2 or not all(type(t) in (int,float) and math.isfinite(t) for t in period) or not 0 <= period[0] < period[1] <= duration + .05:
                raise ValueError('Invalid fake interval')
        fake = r['modify_audio'] or r['modify_video']
        if bool(periods) != fake:
            raise ValueError('Modification flags conflict with fake intervals')
        original = r.get('original')
        if fake and not isinstance(original, str):
            raise ValueError('Fake video missing original relation')
        local = root / name
        if name in present and not local.resolve().is_relative_to(root):
            raise ValueError('Resolved path outside input root')
        rows.append({'video_id':path.with_suffix('').as_posix().replace('/','__'),
                     'video_path':str(local), 'source_id':original if fake else name,
                     'person_id':'', 'emotion_label':'', 'is_fake':int(fake), 'is_fraud':'',
                     'label_source':'LAV-DF official metadata', 'split':r['split'],
                     'modify_audio':r['modify_audio'], 'modify_video':r['modify_video'],
                     'fake_periods':json.dumps(periods), 'duration':duration,
                     'media_present':name in present, 'official_file':name})
    if len({r['video_id'] for r in rows}) != len(rows):
        raise ValueError('Derived video IDs collide')
    groups = {}
    for r in rows:groups.setdefault(r['source_id'],set()).add(r['split'])
    report = {'total':len(rows), 'splits':dict(Counter(r['split'] for r in rows)),
              'fake_counts':dict(Counter(r['is_fake'] for r in rows)),
              'media_present':sum(r['media_present'] for r in rows),
              'cross_split_source_groups':{k:sorted(v) for k,v in groups.items() if len(v)>1},
              'missing_originals':sorted({r['source_id'] for r in rows if r['is_fake'] and r['source_id'] not in seen}),
              'status':'metadata_checked_media_decoding_pending'}
    return rows, report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--media-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output must be a new directory')
    rows, report=prepare(json.loads(a.metadata.read_text(encoding='utf-8')),a.media_root)
    report['metadata_sha256']=hashlib.sha256(a.metadata.read_bytes()).hexdigest()
    a.output.mkdir(parents=True)
    with (a.output/'evaluation_labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':main()
