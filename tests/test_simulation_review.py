import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from import_simulation_review import validate,ALLOWED


def payload():
    r={'id':'S001','reviewer':'r1','reviewed_at':'2026-10-26T00:00:00Z',**{k:'uncertain' for k in ALLOWED}}
    return {'dataset':'pilot40_20261026','stage':'single_reviewer_initial','records':[r]}


def test_uncertain_is_valid_not_forced_into_truth():
    assert validate(payload(),{'S001'})['S001']['emotion_relation']=='uncertain'


def test_duplicate_and_unknown_rejected():
    p=payload();p['records']*=2
    with pytest.raises(ValueError,match='duplicate'):validate(p,{'S001'})
    with pytest.raises(ValueError,match='Unknown'):validate(payload(),{'S002'})


def test_missing_judgement_rejected():
    p=payload();del p['records'][0]['audio_emotion']
    with pytest.raises(ValueError,match='field'):validate(p,{'S001'})
