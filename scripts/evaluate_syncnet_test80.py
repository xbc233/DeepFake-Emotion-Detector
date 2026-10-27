"""Read test truth only after extraction; frozen inference and group uncertainty."""
import csv
import json
import hashlib
from collections import Counter
import numpy as np
from train_lavdf_lightweight import ROOT,predict,auc
from evaluate_risk import binary_metrics

OUT=ROOT/'experiments/syncnet_test80_20261025'


def main():
    source=ROOT/'data/results/syncnet_test80_20261025'
    assert json.loads((source/'status.json').read_text())['status']=='complete'
    protocol=json.loads((OUT/'frozen_protocol.json').read_text());modelpath=OUT/'frozen_model.json'
    assert hashlib.sha256(modelpath.read_bytes()).hexdigest()==protocol['model_sha256']
    model=json.loads(modelpath.read_text());threshold=protocol['threshold']
    results=json.loads((source/'results.json').read_text());assert len(results)==80
    with (ROOT/'experiments/lavdf_expansion480_20261025/splits/test_labels.csv').open(encoding='utf-8-sig') as f:labels={r['video_id']:r for r in csv.DictReader(f)}
    assert set(labels)=={r['id'] for r in results}
    rows=[]
    for r in results:
        truth=labels[r['id']];assert truth['source_id']==r['source_id'] and truth['split']=='test'
        row={'id':r['id'],'source_id':r['source_id'],'y':int(truth['is_fake']),'status':r['status'],'score':None,'prediction':None}
        if r['status']=='success':
            s=r['sync'];f={'sync_min_distance':s['min_distance'],'sync_zero_distance':s['zero_distance'],'sync_confidence':s['confidence'],'sync_abs_offset':abs(s['offset_frames'])}
            row['score']=float(predict(model,np.array([[f[k] for k in model['features']]]))[0]);row['prediction']=int(row['score']>=threshold)
        rows.append(row)
    valid=[r for r in rows if r['score'] is not None]
    y=np.array([r['y'] for r in valid]);p=np.array([r['score'] for r in valid])
    m=binary_metrics(list(zip(map(int,y),map(float,p))),threshold)
    for k in ('recall','precision','false_positive_rate'):m[k].pop('ci95_wilson',None)
    m.pop('conclusion',None)
    ids=sorted({r['source_id'] for r in rows});groups=[np.array([i for i,r in enumerate(valid) if r['source_id']==g],dtype=int) for g in ids]
    rng=np.random.default_rng(20261025);values=[]
    for _ in range(2000):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if len(set(y[ix]))<2:continue
        values.append([auc(y[ix],p[ix]),float(np.mean(p[ix][y[ix]==1]>=threshold)),float(np.mean(p[ix][y[ix]==0]>=threshold))])
    summary={'threshold':threshold,'protocol':protocol,'coverage':{'planned':80,'valid':len(valid),'status':dict(Counter(r['status'] for r in rows)),
                'by_label':{str(k):dict(Counter(r['status'] for r in rows if r['y']==k)) for k in (0,1)}},
             'metrics_on_valid':m,'all_fake_alert_fraction_including_abstentions':sum(r['y']==1 and r['prediction']==1 for r in rows)/sum(r['y']==1 for r in rows),
             'source_group_bootstrap':{'replicates':len(values),'groups':len(ids),'ci95':{k:list(map(float,np.quantile(np.array(values)[:,i],[.025,.975]))) for i,k in enumerate(['auc','recall','fpr'])}},
             'results_sha256':hashlib.sha256((source/'results.json').read_bytes()).hexdigest(),
             'limitations':['This is the first frozen-protocol test evaluation; future method changes require new external evaluation.',
                            'Observed validation FPR constraint does not guarantee test FPR. Only 20 real test videos before quality gating.',
                            'Source groups not verified independent identities; no fraud labels; scores are uncalibrated.',
                            'Quality rejections are abstentions, never classified as real or correct.']}
    (OUT/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False));(OUT/'predictions.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({k:summary[k] for k in ('threshold','coverage','metrics_on_valid','source_group_bootstrap')}))


if __name__=='__main__':main()
