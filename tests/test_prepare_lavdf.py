import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_lavdf import prepare


def records():
    base=dict(file='real/a.mp4',n_fakes=0,fake_periods=[],duration=5.0,original=None,modify_audio=False,modify_video=False,split='train')
    fake=dict(base,file='fake/b.mp4',n_fakes=1,fake_periods=[[1.,2.]],original='real/a.mp4',modify_audio=True)
    return [base,fake]


def test_truth_and_grouping(tmp_path):
    rows,report=prepare(records(),tmp_path)
    assert [r['is_fake'] for r in rows]==[0,1]
    assert all(r['is_fraud']=='' for r in rows)
    assert len({r['source_id'] for r in rows})==1
    assert report['media_present']==0


def test_source_leakage(tmp_path):
    r=records();r[1]['split']='test'
    assert prepare(r,tmp_path)[1]['cross_split_source_groups']=={'real/a.mp4':['test','train']}


@pytest.mark.parametrize('field,value',[('file','../escape.mp4'),('modify_audio','false'),('fake_periods',[[1,9]]),('n_fakes',2)])
def test_bad_metadata(tmp_path,field,value):
    r=records();r[1][field]=value
    with pytest.raises(ValueError):prepare(r,tmp_path)
