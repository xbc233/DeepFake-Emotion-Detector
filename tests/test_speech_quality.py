import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from speech_quality import window_quality


def test_overlap_and_unknown_tail_not_silence():
    frames=[{'start':0,'end':.3,'speech':True},{'start':.3,'end':.6,'speech':False},{'start':.6,'end':.62,'speech':None}]
    q=window_quality(.1,.62,frames,min_voiced=.1)
    assert q['voiced_seconds']==pytest.approx(.2)
    assert q['vad_unobserved_seconds']==pytest.approx(.02)
    assert q['eligible']


def test_silence_and_short_windows_rejected():
    q=window_quality(0,.3,[{'start':0,'end':.3,'speech':False}])
    assert set(q['reasons'])=={'short_window','insufficient_voiced_seconds','low_voiced_fraction'}
    assert not q['eligible']


def test_exact_threshold_is_accepted():
    assert window_quality(0,.5,[{'start':0,'end':.3,'speech':True},{'start':.3,'end':.5,'speech':False}])['eligible']


def test_invalid_range_rejected():
    with pytest.raises(ValueError):window_quality(1,0,[])
