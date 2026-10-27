"""Coverage and controlled-delay acceptance, not classification-driven tuning."""
import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from evaluate_risk import binary_metrics
from refine_syncnet_tracks import BASE,OUT


def main():
    assert json.loads((OUT/'status.json').read_text())['processed']==32
    old=json.loads((BASE/'results.json').read_text());new=json.loads((OUT/'results.json').read_text())
    oi={r['id']:r for r in old};ni={r['id']:r for r in new};assert set(oi)==set(ni)
    oldok={r['id'] for r in old if r['status']=='success'};newok={r['id'] for r in new if r['status']=='success'}
    total=sum(r['total_frames']/25 for r in new)
    controls=[{'id':r['id'],'start':s['start_frame']/25,**s['control']} for r in new for s in r['segments'] if 'control' in s]
    def rank(ids,index):
        m=binary_metrics([(index[i]['is_fake'],index[i]['sync']['min_distance']) for i in sorted(ids)],.5)
        return {k:m[k] for k in ('n','positive','negative','roc_auc','pr_auc_average_precision')}
    summary={'planned':32,'baseline_valid':len(oldok),'candidate_valid':len(newok),'recovered':sorted(newok-oldok),'lost':sorted(oldok-newok),
             'remaining_rejected':[{'id':r['id'],'tracks':r['tracks'],'reason':r.get('reason')} for r in new if r['status']!='success'],
             'baseline_retained_track_seconds':sum(ni[i]['total_frames']/25 for i in oldok),
             'candidate_retained_track_seconds':sum(r.get('retained_track_seconds',0) for r in new),
             'total_video_frame_seconds':total,
             'baseline_support_anchor_seconds':sum(oi[i]['sync']['support_windows']/25 for i in oldok),
             'candidate_support_anchor_seconds':sum(r['sync']['support_anchor_seconds'] for r in new if r['status']=='success'),
             'segments':sum(len(r['segments']) for r in new),'reused_embedding_segments':sum(s['reused_embeddings'] for r in new for s in r['segments']),
             'controls':controls,'controls_passed':sum(c['passed'] and c['within_search'] for c in controls),
             'controls_in_search':sum(c['within_search'] for c in controls),
             'same_original_subset':{'old':rank(oldok&newok,oi),'new':rank(oldok&newok,ni)},
             'candidate_all_available':rank(newok,ni),
             'coverage_by_class':{str(y):{'planned':sum(r['is_fake']==y for r in new),'valid':sum(r['is_fake']==y and r['status']=='success' for r in new)} for y in (0,1)}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    tiles=[]
    for sid in sorted(newok-oldok):
        for p in sorted((OUT/'samples'/sid).glob('crop_*.jpg')):
            im=cv2.imread(str(p));im=cv2.copyMakeBorder(im,25,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(im,sid+' '+p.stem,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(im)
    for start in range(0,len(tiles),6):cv2.imwrite(str(OUT/f'recovered_crops_{start//6+1}.jpg'),np.vstack(tiles[start:start+6]))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('remaining_rejected','controls')},ensure_ascii=True))


if __name__=='__main__':main()
