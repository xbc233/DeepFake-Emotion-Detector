import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from emotion2vec_candidate import normalize_result


def test_native_unknown_mass_is_preserved():
    labels=['angry','disgusted','fearful','happy','neutral','other','sad','surprised','<unk>']
    p=normalize_result({'labels':labels,'scores':[.1]*8+[.2]})
    assert p['unknown']==.2 and p['other']==.1 and p['fear']==.1


def test_incomplete_or_invalid_scores_rejected():
    with pytest.raises(ValueError):
        normalize_result({'labels':['happy'],'scores':[1]})
    with pytest.raises(ValueError):
        normalize_result({'labels':['happy']*9,'scores':[1/9]*9})
