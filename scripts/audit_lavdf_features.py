"""Audit a completed batch, or a labelled-as-partial snapshot without ranking."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from evaluate_risk import binary_metrics

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'data/results/lavdf_features400_20261025'
SPLITS=ROOT/'experiments/lavdf_expansion480_20261025/splits'
OUT=ROOT/'experiments/lavdf_features400_audit_20261025'


def ranking(rows):
    pairs=[(r['is_fake'],r['js']) for r in rows if r['js'] is not None]
    if not pairs:return {'n':0,'roc_auc':None,'pr_auc_average_precision':None}
    metrics=binary_metrics(pairs,.5)
    # Deliberately discard arbitrary threshold metrics; JS is not a probability.
    return {k:metrics[k] for k in ('n','positive','negative','roc_auc','pr_auc_average_precision')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true',help='Quality-only snapshot; never rank incomplete samples')
    args=parser.parse_args()
    expected={}
    for split in ('train','dev'):
        for s in json.loads((SPLITS/f'{split}_inputs.json').read_text()):expected[s['id']]=s
    results=json.loads((BATCH/'results.json').read_text(encoding='utf-8'))
    ids=[r['id'] for r in results]
    if len(set(ids))!=len(ids) or set(ids)-set(expected):raise ValueError('Duplicate or unexpected batch IDs')
    status=BATCH/'status.json'
    complete=(len(results)==len(expected) and all(r['status']=='success' for r in results)
              and status.exists() and json.loads(status.read_text())['status']=='complete')
    if not complete and not args.partial:raise ValueError('Batch incomplete; use --partial for quality only')
    labels={}
    # Partial audits deliberately do not open labels or report classification metrics.
    if not args.partial:
        for split in ('train','dev'):
            with (SPLITS/f'{split}_labels.csv').open(encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):labels[row['video_id']]=row
        if set(labels)!=set(expected):raise ValueError('Label/input ID mismatch')
    config=json.loads((BATCH/'config.json').read_text())
    rows=[]
    for result in results:
        if result['status']!='success':continue
        s=expected[result['id']]
        report=json.loads(Path(result['report']).read_text(encoding='utf-8'))
        if report['fingerprint']!=config['fingerprint'] or report['source_sha256']!=s['sha256']:
            raise ValueError('Report provenance mismatch')
        for path,digest in report['artifact_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Artifact changed: '+path)
        js=report['interval_js_comparison']
        with Path(report['artifacts']['face_emotion']).open(encoding='utf-8-sig') as f:faces=list(csv.DictReader(f))
        quality=json.loads((Path(result['report']).parent/'quality.json').read_text())
        row={'id':s['id'],'split':s['split'],'source_id':s['source_id'],
             'duration':js['duration_seconds'],'valid_seconds':js['comparable_duration_seconds'],
             'coverage':js['comparable_duration_fraction'],'js':js['mean_js_distance'],
             'face_frames':len(faces),'observed_face_frames':sum(bool(r['probabilities']) for r in faces),
             'speech_windows':len(quality['windows']),'eligible_speech_windows':sum(q['eligible'] for q in quality['windows'])}
        if not args.partial:
            truth=labels[s['id']]
            if truth['source_id']!=s['source_id'] or truth['split']!=s['split']:raise ValueError('Label grouping mismatch')
            row.update(is_fake=int(truth['is_fake']),modify_audio=truth['modify_audio']=='True',modify_video=truth['modify_video']=='True')
        rows.append(row)
    summary={'status':'partial_quality_only' if args.partial else 'complete',
             'expected':len(expected),'audited_success':len(rows),'missing':sorted(set(expected)-set(ids)),
             'failures':[r for r in results if r['status']!='success'],'test_evaluated':False,'splits':{}}
    for split in ('train','dev'):
        selected=[r for r in rows if r['split']==split]
        duration=sum(r['duration'] for r in selected)
        group={'videos':len(selected),'source_groups':len({r['source_id'] for r in selected}),
               'videos_with_js':sum(r['js'] is not None for r in selected),
               'duration_coverage':sum(r['valid_seconds'] for r in selected)/duration if duration else None,
               'face_frames':sum(r['face_frames'] for r in selected),
               'observed_face_frames':sum(r['observed_face_frames'] for r in selected)}
        if not args.partial:
            group['ranking']=ranking(selected)
            group['by_manipulation']={}
            for audio,video in ((False,True),(True,False),(True,True)):
                paired=[r for r in selected if not r['is_fake'] or (r['modify_audio']==audio and r['modify_video']==video)]
                group['by_manipulation'][f'audio_{audio}_video_{video}']=ranking(paired)
        summary['splits'][split]=group
    summary['limitations']=['Missing JS is excluded, never filled with zero. Report coverage alongside ranking.',
                           'Correlated source groups; ranking point estimates are exploratory, no independent-video significance claims.',
                           'No threshold tuning, no fraud labels, text not fused into JS; test remains reserved.']
    OUT.mkdir(exist_ok=True)
    name='partial' if args.partial else 'complete'
    (OUT/f'{name}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/f'{name}_rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('missing','limitations')},ensure_ascii=True))


if __name__=='__main__':main()
