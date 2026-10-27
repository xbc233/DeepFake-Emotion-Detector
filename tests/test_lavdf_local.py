import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_lavdf_local import overlap,local,grouped


def test_overlapping_annotations_not_double_counted():
    assert overlap(0,3,[(1,2),(1.5,4)])==2


def test_mixed_window_is_not_clean_and_missing_is_not_zero():
    windows=[dict(start=0,end=3,js_distance=.8),dict(start=3,end=6,js_distance=.2),dict(start=6,end=9,js_distance=None)]
    result=local(windows,[(1,2),(7,8)])
    assert abs(result['difference']-.6)<1e-9
    assert result['fake_overlap_seconds']==2
    assert result['valid_fake_overlap_seconds']==1
    assert local(windows,[(0,9)])['difference'] is None


def test_groups_have_equal_weight():
    rows=[dict(source_id='a',x=1.)]*3+[dict(source_id='b',x=-1.)]
    assert grouped(rows,'x')['mean']==0
