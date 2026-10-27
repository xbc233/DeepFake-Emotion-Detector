"""Source isolation is critical to meaningful downstream evaluation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_lavdf_expansion as expansion


def group(split, name):
    original = f'{split}/{name}.mp4'
    return [dict(file=original if not a and not v else f'{split}/{name}_{int(a)}{int(v)}.mp4',
                 original=None if not a and not v else original,
                 modify_audio=a, modify_video=v, split=split)
            for a,v in expansion.COMBINATIONS]


def test_selection_is_deterministic_grouped_and_excludes_pilot(monkeypatch):
    monkeypatch.setattr(expansion, 'QUOTAS', {'train':2, 'dev':1, 'test':1})
    records = [r for split in expansion.QUOTAS for name in ('a','b','c') for r in group(split,name)]
    selected = expansion.select(records, {'train/a.mp4'})
    assert selected == expansion.select(list(reversed(records)), {'train/a.mp4'})
    assert len(selected) == 16
    assert 'train/a.mp4' not in {s['source_id'] for s in selected}
    for source in {s['source_id'] for s in selected}:
        members = [s for s in selected if s['source_id']==source]
        assert len(members)==4
        assert len({s['split'] for s in members})==1
        assert sum(s['is_fake'] for s in members)==3


def test_cross_split_source_is_rejected():
    records = group('train','a')
    records[-1]['split']='test'
    with pytest.raises(ValueError, match='leakage'):
        expansion.select(records, set())


def test_duplicate_filename_is_rejected():
    records = group('train','a')
    with pytest.raises(ValueError, match='Duplicate'):
        expansion.select(records+records[:1], set())
