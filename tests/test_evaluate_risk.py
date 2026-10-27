import math
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_risk import evaluate, binary_metrics


def test_known_confusion_and_auc():
    m = binary_metrics([(1, .9), (0, .8), (1, .4), (0, .1)], .5)
    assert m['confusion'] == dict(TP=1, FP=1, FN=1, TN=1)
    assert m['roc_auc'] == .75
    assert m['pr_auc_average_precision'] == pytest.approx(5/6)
    assert m['recall']['value'] == .5
    assert m['recall']['ci95_wilson'][0] < .5 < m['recall']['ci95_wilson'][1]


def test_ties_single_class_and_empty():
    assert binary_metrics([(1, .5), (0, .5)], .5)['roc_auc'] == .5
    assert binary_metrics([(1, .5), (0, .5)], .5)['pr_auc_average_precision'] == .5
    assert binary_metrics([(0, .8)], .5)['recall']['value'] is None
    assert binary_metrics([(0, .8)], .5)['roc_auc'] is None
    assert binary_metrics([], .5)['precision']['value'] is None


def test_unknown_labels_failed_missing_invalid_and_extra():
    labels = [{'video_id': k, 'is_fake': '', 'is_fraud': ''} for k in 'abcd']
    results = [{'id': 'a', 'status': 'success', 'risk_score': .8},
               {'id': 'b', 'status': 'failed'},
               {'id': 'c', 'status': 'success', 'risk_score': math.nan},
               {'id': 'extra', 'status': 'success', 'risk_score': .1}]
    rows, m = evaluate(labels, results, .5)
    assert m['valid_scores'] == 1 and m['high_risk_count'] == 1
    assert m['is_fraud']['n'] == 0
    assert m['unmatched_result_ids'] == ['extra']
    assert rows[0]['is_fraud_outcome'] == ''
    assert set(m['status_counts']) == {'success', 'failed', 'invalid_score', 'missing_result'}


def test_validation_and_threshold_boundary():
    label = {'video_id': 'a', 'is_fraud': '1'}
    result = {'id': 'a', 'status': 'success', 'risk_score': .5}
    rows, m = evaluate([label], [result], .5)
    assert rows[0]['is_fraud_outcome'] == 'TP'
    with pytest.raises(ValueError, match='duplicate'):
        evaluate([label, label], [result], .5)
    with pytest.raises(ValueError, match='duplicate'):
        evaluate([label], [result, result], .5)
    with pytest.raises(ValueError, match='blank'):
        evaluate([dict(label, is_fraud='unknown')], [result], .5)
    with pytest.raises(ValueError, match='threshold'):
        evaluate([label], [result], math.nan)
