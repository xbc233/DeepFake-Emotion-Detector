import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_emotion2vec_pilot import classification


def test_no_emotion_truth_has_no_accuracy():
    m=classification([{'target':None,'prediction':'happy'}],'prediction')
    assert m['n']==0 and m['accuracy'] is None and m['macro_f1_six_targets'] is None


def test_unknown_predictions_are_errors_not_dropped():
    rows=[{'target':k,'prediction':k} for k in ('anger','fear','happy','neutral','sad','surprise')]
    rows[0]['prediction']='unknown'
    m=classification(rows,'prediction')
    assert m['n']==6 and m['accuracy']==pytest.approx(5/6)
    assert m['macro_f1_six_targets']==pytest.approx(5/6)
    assert m['confusion']['anger']=={'unknown':1}
