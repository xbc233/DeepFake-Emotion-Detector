"""Freeze validation-selected sync model/threshold before test features or labels."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from train_lavdf_lightweight import ROOT

OUT=ROOT/'experiments/syncnet_test80_20261025'


def threshold_choice(y,p,max_fpr=.1):
    y=np.asarray(y);p=np.asarray(p)
    if not np.isfinite(p).all() or set(y)!={0,1}:raise ValueError('Need finite scores and both classes')
    candidates=[]
    for t in np.r_[np.unique(p),np.nextafter(p.max(),np.inf)]:
        recall=float(np.mean(p[y==1]>=t));fpr=float(np.mean(p[y==0]>=t))
        if fpr<=max_fpr+1e-12:candidates.append((recall,-fpr,float(t)))
    recall,neg_fpr,t=max(candidates)
    return {'threshold':t,'validation_recall':recall,'validation_fpr':-neg_fpr}


def main():
    OUT.mkdir(exist_ok=True)
    target=OUT/'frozen_protocol.json'
    if target.exists():raise FileExistsError('Protocol already frozen; never overwrite after test access')
    source=ROOT/'experiments/syncnet400_comparison_20261025'
    records=json.loads((source/'dev_predictions.json').read_text())
    choice=threshold_choice([r['y'] for r in records],[r['sync'] for r in records])
    model=(source/'sync_model.json').read_bytes();(OUT/'frozen_model.json').write_bytes(model)
    files=[ROOT/'scripts/run_syncnet_test80.py',ROOT/'scripts/evaluate_syncnet_test80.py',ROOT/'scripts/run_syncnet_pilot.py',ROOT/'scripts/refine_syncnet_tracks.py']
    protocol={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'model':'sync','model_selection':'Highest dev AUC among fixed candidates from previous experiment',
              'threshold_policy':'Maximize validation recall with observed FPR <= 0.10; ties prefer lower FPR then higher threshold',
              'threshold_is_exploratory_not_deployment_requirement':True,'max_validation_fpr':.1,**choice,
              'model_sha256':hashlib.sha256(model).hexdigest(),
              'dev_predictions_sha256':hashlib.sha256((source/'dev_predictions.json').read_bytes()).hexdigest(),
              'extraction_config_sha256':hashlib.sha256((ROOT/'data/results/syncnet400_20261025/config.json').read_bytes()).hexdigest(),
              'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'test_policy':'Once-only test evaluation at frozen threshold; rejection is abstention. No fitting or threshold reselection on test.',
              'uncertainty':'2000 source-group bootstrap replicates, seed 20261025; exclude single-class replicates; no claim of person independence',
              'caveat':'Only 20 real validation videos; observed 10% FPR does not guarantee population/test FPR.'}
    target.write_text(json.dumps(protocol,indent=2),encoding='utf-8');print(json.dumps(choice))


if __name__=='__main__':main()
