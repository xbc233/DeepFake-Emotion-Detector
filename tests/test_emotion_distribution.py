import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from emotion_distribution import encode_distribution


def test_distribution_normalizes_aliases_without_fabricating_classes():
    result=json.loads(encode_distribution(['hap','sadness','neutral','ang'],[.4,.2,.3,.1]))
    assert set(result)=={'happy','sad','neutral','anger'}
    assert sum(result.values())==pytest.approx(1)
    assert 'fear' not in result


@pytest.mark.parametrize('labels,scores', [(['happy'],[float('nan')]), (['happy'],[.2]), (['happy'],[-1]), (['happy','sad'],[1]), (['unknown'],[1])])
def test_invalid_distributions_rejected(labels,scores):
    with pytest.raises(ValueError):
        encode_distribution(labels,scores)
