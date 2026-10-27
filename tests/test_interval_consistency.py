import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from interval_consistency import compare, project, js_distance

P={'anger':0,'happy':0,'neutral':1,'sad':0}
Q={'anger':1,'happy':0,'neutral':0,'sad':0}
def row(p,**kw): return dict(probabilities=json.dumps(p),emotion=max(p,key=p.get),**kw)


def test_js_endpoints_and_symmetry():
    assert js_distance([1,0],[1,0])==0
    assert js_distance([1,0],[0,1])==1
    assert js_distance([.2,.8],[.7,.3])==pytest.approx(js_distance([.7,.3],[.2,.8]))


def test_actual_interval_boundaries_and_no_text_extrapolation():
    face=[row(P,frame='000001.jpg'),row(Q,frame='000003.jpg')]
    speech=[row(P,time='0',end='1'),row(Q,time='1',end='2')]
    text=[row(P,start='.2',end='.6')]
    result=compare(face,speech,text,2)
    assert result['mean_js_distance']==0
    assert result['comparable_duration_fraction']==1
    assert result['windows'][0]['text_overlaps'][0]['start']==.2
    assert result['windows'][1]['text_overlaps']==[]
    assert result['prediction'] is None


def test_reject_unsupported_mass_and_missing_probabilities():
    p={k:v*.2 for k,v in P.items()}; p['fear']=.8
    assert project(p,.8)[0] is None
    assert project({'neutral':1},.8)[2]=='unsupported_label_space'
    result=compare([{'frame':'000001.jpg'}],[row(P,time='0',end='1')],[],1)
    assert result['mean_js_distance'] is None
    assert result['comparable_duration_fraction']==0


def test_duration_weighting_and_clipping():
    result=compare([row(P,frame='000001.jpg'),row(P,frame='000003.jpg')],
                   [row(P,time='0',end='1'),row(Q,time='1',end='3')],[],1.5)
    assert result['mean_js_distance']==pytest.approx(1/3)
    assert result['windows'][-1]['end']==1.5


def test_overlapping_windows_rejected():
    with pytest.raises(ValueError,match='overlapping'):
        compare([],[row(P,time='0',end='2'),row(P,time='1',end='3')],[],3)

def test_sparse_face_evidence_abstains():
    result=compare([row(P,frame='000001.jpg')],[row(P,time='0',end='3')],[],3)
    assert result['mean_js_distance'] is None
    assert result['windows'][0]['face_time_coverage']==pytest.approx(1/6)
    assert result['comparable_duration_fraction']==0

def test_extended_space_accepts_fear_without_changing_default():
    from interval_consistency import project, SHARED
    p={'anger':0,'happy':0,'neutral':0,'sad':0,'fear':1,'surprise':0,'other':0,'unknown':0}
    assert project(p,.8)[2]=='dominant_label_unsupported'
    extended=SHARED+('fear','surprise')
    q,mass,status=project(p,.8,extended)
    assert status=='comparable' and q[4]==1 and mass==1
    assert SHARED==('anger','happy','neutral','sad')
