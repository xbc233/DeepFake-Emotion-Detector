"""Validated native emotion distributions; no fabricated unsupported classes."""
import json
import math
from alignment import normalize_emotion


def encode_distribution(labels, scores):
    labels, scores = list(labels), list(scores)
    if not labels or len(labels) != len(scores):
        raise ValueError('Distribution labels/scores length mismatch')
    result = {}
    for label, score in zip(labels, scores):
        key, value = normalize_emotion(str(label)), float(score)
        if key == 'unknown' or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid probability distribution')
        result[key] = result.get(key, 0.0) + value
    total = sum(result.values())
    if not math.isclose(total, 1.0, abs_tol=1e-4):
        raise ValueError('Probabilities must sum to one')
    return json.dumps({k: v/total for k, v in result.items()}, ensure_ascii=False, allow_nan=False)
