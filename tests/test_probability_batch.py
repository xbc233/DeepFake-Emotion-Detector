import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_probability_batch import summarize


def test_failed_samples_remain_explicit():
    rows,windows,m=summarize([{'id':'a','status':'failed','error':'timeout','emotion':'sad'}],[{'id':'a','risk_score':.9}])
    assert m['failed']==1 and m['succeeded']==0
    assert m['duration_coverage'] is None
    assert rows[0]['baseline_risk']==.9
    assert rows[0]['js_distance'] is None
    assert windows==[]
