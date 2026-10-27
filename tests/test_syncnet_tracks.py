import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refine_syncnet_tracks import segments,combine


def test_long_gap_splits_without_fabrication():
    b=[[50,50,30]]*70+[None]*6+[[50,50,30]]*70
    t=segments(b)
    assert [(s['start_frame'],s['end_frame_exclusive']) for s in t]==[(0,70),(76,146)]
    assert all(s['eligible'] for s in t)


def test_bounded_gap_and_edge_trim():
    b=[None]*8+[[50,50,30]]*35+[None]*5+[[50,50,30]]*35+[None]*8
    t=segments(b)
    assert len(t)==1 and t[0]['start_frame']==8 and t[0]['frames']==75 and t[0]['eligible']


def test_jump_and_short_track_rejected():
    t=segments([[10,10,20]]*45+[[100,100,20]]*70)
    assert len(t)==2 and not t[0]['eligible'] and t[1]['eligible']
    assert segments([None]*100)==[]


def test_curve_support_weighting():
    a=[5.]*31;b=[5.]*31;a[15]=1.;b[15]=3.
    r=combine([{'sync':{'curve':a,'support_windows':30}},{'sync':{'curve':b,'support_windows':90}}])
    assert r['min_distance']==2.5 and r['offset_frames']==0 and r['support_windows']==120
