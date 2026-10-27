"""Post-test descriptive error audit; no refitting, thresholds, or inference."""
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from analyze_lavdf_local import overlap

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/syncnet_test80_20261025'
BATCH=ROOT/'data/results/syncnet_test80_20261025'
OUT=ROOT/'experiments/syncnet_test_errors_20261025'


def main():
    predictions=json.loads((SOURCE/'predictions.json').read_text())
    results={r['id']:r for r in json.loads((BATCH/'results.json').read_text())}
    with (ROOT/'experiments/lavdf_expansion480_20261025/splits/test_labels.csv').open(encoding='utf-8-sig') as f:labels={r['video_id']:r for r in csv.DictReader(f)}
    threshold=json.loads((SOURCE/'frozen_protocol.json').read_text())['threshold']
    rows=[]
    for p in predictions:
        r=results[p['id']];label=labels[p['id']];periods=json.loads(label['fake_periods'])
        kind=('audio' if label['modify_audio']=='True' else '')+('video' if label['modify_video']=='True' else '') or 'real'
        outcome='abstain' if p['prediction'] is None else ('TP' if p['y'] else 'FP') if p['prediction'] else ('FN' if p['y'] else 'TN')
        duration=r['total_frames']/25;fake_seconds=overlap(0,duration,periods)
        retained=sum(overlap(s['start_frame']/25,s['end_frame_exclusive']/25,periods) for s in r['segments'])
        visual_support=sum(overlap((s['start_frame']+15)/25,(s['start_frame']+15+s['sync']['support_windows']+4)/25,periods) for s in r['segments'])
        row={**p,'type':kind,'outcome':outcome,'duration':duration,'fake_seconds':fake_seconds,
             'fake_fraction':fake_seconds/duration,'fake_retained_fraction':retained/fake_seconds if fake_seconds else None,
             'fake_visual_support_fraction':visual_support/fake_seconds if fake_seconds else None,
             'detection_fraction':r['detection_quality']['fraction'],'max_missing_gap':r['detection_quality']['max_missing_gap'],
             'multiface_frames':r['detection_quality']['multiface_frames'],'retained_fraction':r.get('retained_track_seconds',0)/duration,
             'segments':len(r['segments']),'reject_reason':r.get('reason'),
             'score_margin':p['score']-threshold if p['score'] is not None else None,
             'min_distance':r.get('sync',{}).get('min_distance'),'confidence':r.get('sync',{}).get('confidence'),
             'abs_offset':abs(r['sync']['offset_frames']) if 'sync' in r else None}
        rows.append(row)
    by_type={k:dict(Counter(r['outcome'] for r in rows if r['type']==k)) for k in ('real','audio','video','audiovideo')}
    groups=defaultdict(list)
    for r in rows:groups[r['source_id']].append(r)
    group_summary=[{'source_id':g,'outcomes':dict(Counter(r['outcome'] for r in rs)),
                    'real_score':next(r['score'] for r in rs if r['type']=='real'),
                    'fake_scores':[r['score'] for r in rs if r['type']!='real']} for g,rs in groups.items()]
    fields=['fake_fraction','fake_retained_fraction','fake_visual_support_fraction','detection_fraction','retained_fraction','min_distance','confidence','abs_offset','score_margin']
    distributions={}
    for outcome in ('TP','FN','FP','TN','abstain'):
        distributions[outcome]={}
        for field in fields:
            values=[r[field] for r in rows if r['outcome']==outcome and r[field] is not None]
            distributions[outcome][field]={'n':len(values),'median':float(np.median(values)) if values else None,'min':min(values) if values else None,'max':max(values) if values else None}
    summary={'by_type':by_type,'groups':group_summary,'distributions':distributions,
             'fn_full_fake_retention':sum(r['outcome']=='FN' and r['fake_retained_fraction']>=.999 for r in rows),
             'fn_no_fake_visual_support':sum(r['outcome']=='FN' and r['fake_visual_support_fraction']==0 for r in rows),
             'fn_within_005_below_threshold':sum(r['outcome']=='FN' and r['score_margin']>=-.05 for r in rows),
             'fn_groups':sum(any(r['outcome']=='FN' for r in rs) for rs in groups.values()),
             'all_three_fake_missed_groups':sum(sum(r['outcome']=='FN' for r in rs)==3 for rs in groups.values()),
             'source_sha256':hashlib.sha256((SOURCE/'predictions.json').read_bytes()).hexdigest(),
             'note':'Descriptive post-test audit, not causal attribution. Visual support refers only to the union of 5-frame visual windows; shifted audio support differs. No model changes.'}
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'rows.json').write_text(json.dumps(rows,indent=2))
    # Fixed case selection: all false positives, one representative per rejection
    # source, worst-score FN of each manipulation type, plus nearest-threshold FN.
    selected=[r for r in rows if r['outcome']=='FP'];seen=set()
    for r in rows:
        if r['outcome']=='abstain' and r['source_id'] not in seen:selected.append(r);seen.add(r['source_id'])
    for kind in ('audio','video','audiovideo'):
        candidates=[r for r in rows if r['outcome']=='FN' and r['type']==kind]
        if candidates:selected.append(min(candidates,key=lambda r:r['score']))
    selected.append(max((r for r in rows if r['outcome']=='FN'),key=lambda r:r['score']))
    selected=list({r['id']:r for r in selected}.values());tiles=[];cases=[]
    for r in selected:
        periods=json.loads(labels[r['id']]['fake_periods'])
        moment=sum(periods[0])/2 if periods else r['duration']/2
        frame=max(1,min(round(moment*25)+1,results[r['id']]['total_frames']))
        path=BATCH/'samples'/r['id']/'frames'/f'{frame:06d}.jpg'
        im=cv2.imread(str(path))
        if im is None:raise ValueError('Missing diagnostic frame')
        im=cv2.resize(im,(224,224));im=cv2.copyMakeBorder(im,44,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(im,r['id']+' '+r['outcome'],(3,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        cv2.putText(im,f"{moment:.2f}s score={r['score'] if r['score'] is not None else 'NA'}"[:30],(3,33),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
        tiles.append(im);cases.append({**r,'review_time':moment,'frame_path':str(path)})
    while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(OUT/'cases.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
    (OUT/'cases.json').write_text(json.dumps(cases,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('groups','distributions')},ensure_ascii=True))
    print(json.dumps({o:{k:v['median'] for k,v in distributions[o].items()} for o in ('TP','FN','FP')},ensure_ascii=True))


if __name__=='__main__':main()
