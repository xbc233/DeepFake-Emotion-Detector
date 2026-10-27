import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_syncnet_pilot import distance_curve


def test_known_audio_delay_has_negative_offset():
    rng=np.random.default_rng(9)
    video=torch.tensor(rng.normal(size=(80,16)),dtype=torch.float32)
    audio=torch.cat([torch.zeros((10,16)),video[:-10]])
    r=distance_curve(video,audio)
    assert r['offset_frames']==-10
    assert r['min_distance']<1e-4
    assert r['support_windows']==50


def test_aligned_and_short_input():
    import pytest
    x=torch.randn(70,8)
    assert distance_curve(x,x)['offset_frames']==0
    with pytest.raises(ValueError,match='Insufficient'):
        distance_curve(x[:20],x[:20])
