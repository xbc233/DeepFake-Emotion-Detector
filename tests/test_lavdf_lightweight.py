import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_lavdf_lightweight import fit,predict,auc,metrics


def test_learns_direction_and_serializes():
    import json
    x=np.array([[-2.,1],[-1,1],[1,1],[2,1]])
    m=fit(x,np.array([0,0,1,1]));p=predict(m,x)
    assert np.all(np.diff(p)>0)
    assert np.allclose(p,predict(json.loads(json.dumps(m)),x))


def test_prediction_does_not_refit_preprocessing():
    import copy
    x=np.array([[0.,np.nan],[2.,np.nan],[4.,np.nan],[6.,np.nan]])
    m=fit(x,np.array([0,0,1,1]));saved=copy.deepcopy(m)
    p=predict(m,np.array([[1000.,np.nan]]))
    assert np.isfinite(p).all() and m==saved and m['median']==[3.,0.]


def test_auc_ties():
    assert auc(np.array([0,1,0,1]),np.ones(4))==.5


def test_majority_prediction_is_not_useful_detection():
    m=metrics(np.array([0,1,1,1]),np.full(4,.75))
    assert m['accuracy']==.75 and m['balanced_accuracy']==.5
    assert m['false_positive_rate']['value']==1
    assert 'ci95_wilson' not in m['recall']
