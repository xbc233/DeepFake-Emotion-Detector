"""Validate and merge one human initial review without fabricating consensus."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare_simulated40 import OUT

EMOTIONS={'neutral','happy','sad','angry','fear','surprise','disgust','mixed','uncertain'}
ALLOWED={'semantics':{'normal','simulated_scam','uncertain'},'visual_emotion':EMOTIONS,'audio_emotion':EMOTIONS,'text_emotion':EMOTIONS,
         'emotion_relation':{'consistent','inconsistent','uncertain'},'lip_sync':{'aligned','misaligned','uncertain'},'quality':{'usable','unusable','uncertain'}}


def validate(payload,expected):
    if payload.get('dataset')!='pilot40_20261026' or payload.get('stage')!='single_reviewer_initial':raise ValueError('Wrong dataset or review stage')
    rows=payload.get('records',[])
    if not rows:raise ValueError('No human records')
    seen=set();reviewers=set()
    for r in rows:
        if r.get('id') not in expected or r['id'] in seen:raise ValueError('Unknown or duplicate sample')
        seen.add(r['id'])
        if not isinstance(r.get('reviewer'),str) or not r['reviewer'].strip() or not r.get('reviewed_at'):raise ValueError('Missing reviewer or time')
        reviewers.add(r['reviewer'])
        for k,allowed in ALLOWED.items():
            if r.get(k) not in allowed:raise ValueError('Invalid or missing review field: '+k)
    if len(reviewers)!=1:raise ValueError('Expected one initial reviewer; keep reviewers separate')
    return {r['id']:r for r in rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    m=json.loads((OUT/'private/generated_manifest.json').read_text(encoding='utf-8'))
    raw=a.input.read_bytes();payload=json.loads(raw.decode('utf-8-sig'));reviews=validate(payload,{s['id'] for s in m['samples']})
    accepted=[]
    for s in m['samples']:
        if s['id'] not in reviews:continue
        r=reviews[s['id']];s['human_review']=r;s['human_semantic_label']=r['semantics'];s['human_emotion_relation']=r['emotion_relation'];s['annotation_status']='single_reviewer_initial_only'
        if r['quality']=='usable' and r['semantics']!='uncertain' and r['emotion_relation']!='uncertain':accepted.append(s['id'])
    token=hashlib.sha256(raw).hexdigest()[:12];folder=OUT/'reviews'/token;folder.mkdir(exist_ok=False)
    (folder/'raw_review.json').write_bytes(raw)
    (folder/'merged_manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')
    summary={'received':len(reviews),'planned':40,'missing':sorted({s['id'] for s in m['samples']}-set(reviews)),
             'usable_with_single_review_labels':len(accepted),'status':'single_review_complete' if len(reviews)==40 else 'partial_single_review',
             'independent_second_review_complete':False,'consensus_ground_truth':False,'review_sha256':hashlib.sha256(raw).hexdigest()}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))


if __name__=='__main__':main()
