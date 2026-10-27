"""Fixed L2 logistic baselines; train-only preprocessing, reserved test untouched."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from evaluate_risk import binary_metrics

ROOT=Path(__file__).resolve().parents[1]
QUALITY=['duration','face_observed_fraction','valid_time_fraction','speech_eligible_fraction','face_shared_fraction','js_missing']
EMOTION=['js_mean','js_max','js_p90','js_std','js_range','js_adjacent_change']


def features(report):
    js=report['interval_js_comparison'];windows=js['windows']
    good=[w for w in windows if w['js_distance'] is not None]
    x=np.array([w['js_distance'] for w in good]);weights=np.array([w['end']-w['start'] for w in good])
    with Path(report['artifacts']['face_emotion']).open(encoding='utf-8-sig') as f:faces=list(csv.DictReader(f))
    quality=json.loads((Path(report['artifacts']['speech_emotion']).parent/'quality.json').read_text())['windows']
    frame_count=sum(w['face_frames'] for w in windows)
    result={'duration':js['duration_seconds'],'face_observed_fraction':sum(bool(r['probabilities']) for r in faces)/len(faces),
            'valid_time_fraction':js['comparable_duration_fraction'],
            'speech_eligible_fraction':sum(q['eligible'] for q in quality)/len(quality),
            'face_shared_fraction':sum(w['face_comparable_frames'] for w in windows)/frame_count if frame_count else 0.,
            'js_missing':float(not len(x))}
    mean=float(np.average(x,weights=weights)) if len(x) else float('nan')
    changes=[abs(b['js_distance']-a['js_distance']) for a,b in zip(windows,windows[1:])
             if a['js_distance'] is not None and b['js_distance'] is not None and abs(a['end']-b['start'])<1e-6]
    result.update(js_mean=mean,js_max=float(x.max()) if len(x) else float('nan'),
                  js_p90=float(np.quantile(x,.9)) if len(x) else float('nan'),
                  js_std=float(np.sqrt(np.average((x-mean)**2,weights=weights))) if len(x) else float('nan'),
                  js_range=float(np.ptp(x)) if len(x) else float('nan'),
                  js_adjacent_change=float(np.mean(changes)) if changes else float('nan'))
    return result


def fit(x,y):
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T])
    filled=np.where(np.isfinite(x),x,med);mean=filled.mean(axis=0);scale=filled.std(axis=0);scale[scale<1e-12]=1
    z=np.column_stack([np.ones(len(x)),(filled-mean)/scale])
    def objective(w):
        logits=z@w
        return float(np.sum(np.logaddexp(0,logits)-y*logits)+.5*np.dot(w[1:],w[1:])), z.T@(expit(logits)-y)+np.r_[0,w[1:]]
    result=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':2000,'gtol':1e-8,'ftol':1e-12})
    if not result.success:raise ValueError('Optimizer failed: '+result.message)
    return {'median':med.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':result.x.tolist(),
            'objective':'sum logistic loss + 0.5 * squared L2 coefficients; intercept unpenalized','iterations':int(result.nit)}


def predict(model,x):
    a=np.where(np.isfinite(x),x,np.array(model['median']))
    return expit(np.column_stack([np.ones(len(a)),(a-np.array(model['mean']))/np.array(model['scale'])])@np.array(model['coefficients']))


def metrics(y,p):
    m=binary_metrics(list(zip(map(int,y),map(float,p))),.5)
    m['accuracy']=(m['confusion']['TP']+m['confusion']['TN'])/len(y)
    c=m['confusion']
    m['balanced_accuracy']=.5*(c['TP']/(c['TP']+c['FN'])+c['TN']/(c['TN']+c['FP']))
    # Videos within a source group are correlated: discard per-video Wilson CIs.
    for field in ('recall','precision','false_positive_rate'):
        m[field].pop('ci95_wilson',None)
    m.pop('conclusion',None)
    return m


def auc(y,p):
    pos=p[y==1];neg=p[y==0]
    return float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))


def main():
    audit=ROOT/'experiments/lavdf_features400_audit_20261025/complete.json'
    assert json.loads(audit.read_text())['status']=='complete'
    batch=ROOT/'data/results/lavdf_features400_20261025'
    results=json.loads((batch/'results.json').read_text());index={r['id']:r for r in results}
    rows=[]
    for split in ('train','dev'):
        label_file=ROOT/f'experiments/lavdf_expansion480_20261025/splits/{split}_labels.csv'
        with label_file.open(encoding='utf-8-sig') as f:labels=list(csv.DictReader(f))
        for label in labels:
            r=json.loads(Path(index[label['video_id']]['report']).read_text(encoding='utf-8'))
            assert r['split']==split and r['source_id']==label['source_id']
            rows.append({'id':r['id'],'split':split,'source_id':r['source_id'],'y':int(label['is_fake']),'features':features(r)})
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    assert len(train)==320 and len(dev)==80
    assert not {r['source_id'] for r in train}&{r['source_id'] for r in dev}
    yt=np.array([r['y'] for r in train]);yv=np.array([r['y'] for r in dev])
    out=ROOT/'experiments/lavdf_lightweight_20261025';out.mkdir(exist_ok=True)
    scores={};summary={'threshold':.5,'threshold_selected_on_dev':False,'test_used':False,'models':{}}
    for name,fields in [('quality',QUALITY),('emotion',EMOTION),('fusion',QUALITY+EMOTION)]:
        xt=np.array([[r['features'][k] for k in fields] for r in train]);xv=np.array([[r['features'][k] for k in fields] for r in dev])
        model=fit(xt,yt);model['features']=fields
        pt=predict(model,xt);pv=predict(model,xv);scores[name]=pv
        serialized=json.dumps(model,indent=2,allow_nan=False)
        (out/f'{name}_model.json').write_text(serialized)
        assert np.allclose(predict(json.loads(serialized),xv),pv,rtol=0,atol=1e-12)
        summary['models'][name]={'train':metrics(yt,pt),'dev':metrics(yv,pv)}
    summary['models']['majority_reference']={'dev':metrics(yv,np.full(len(dev),yt.mean()))}
    group_ids=sorted({r['source_id'] for r in dev});indices=[np.array([i for i,r in enumerate(dev) if r['source_id']==g]) for g in group_ids]
    rng=np.random.default_rng(20261025);deltas=[]
    for _ in range(2000):
        ix=np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
        deltas.append(auc(yv[ix],scores['fusion'][ix])-auc(yv[ix],scores['quality'][ix]))
    summary['fusion_minus_quality_dev_auc']={'difference':auc(yv,scores['fusion'])-auc(yv,scores['quality']),
        'source_group_bootstrap95':list(map(float,np.quantile(deltas,[.025,.975]))),'resamples':2000,'seed':20261025,'groups':len(group_ids)}
    summary['provenance']={'batch_config_sha256':hashlib.sha256((batch/'config.json').read_bytes()).hexdigest(),
                            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    predictions=[{'id':r['id'],'source_id':r['source_id'],'y':r['y'],**{k:float(v[i]) for k,v in scores.items()}} for i,r in enumerate(dev)]
    (out/'dev_predictions.json').write_text(json.dumps(predictions,indent=2))
    # Explicit nulls document missing features; fitting imputation is train-only.
    for r in rows:r['features']={k:v if np.isfinite(v) else None for k,v in r['features'].items()}
    (out/'features.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    print(json.dumps(summary))


if __name__=='__main__':main()
