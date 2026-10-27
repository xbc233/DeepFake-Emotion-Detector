import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from freeze_syncnet_protocol import threshold_choice


def test_threshold_respects_ties_and_fpr():
    y=[0,0,0,0,1,1];p=[.1,.2,.8,.8,.8,.9]
    r=threshold_choice(y,p,.25)
    assert r['validation_fpr']==0 and r['validation_recall']==.5


def test_no_feasible_positive_alert_still_allows_abstain_from_alert():
    r=threshold_choice([0,1],[.9,.1],0)
    assert r['threshold']>.9 and r['validation_recall']==0


def test_optimal_recall_under_constraint():
    r=threshold_choice([0,0,1,1],[.2,.7,.6,.8],.5)
    assert r['validation_recall']==1 and r['validation_fpr']==.5
