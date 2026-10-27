import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('prepare_dfdc', Path(__file__).parents[1] / 'scripts/prepare_dfdc.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_data(root):
    data = {'real.mp4': {'label': 'REAL', 'split': 'train'},
            'fake.mp4': {'label': 'FAKE', 'split': 'train', 'original': 'real.mp4'}}
    (root / 'metadata.json').write_text(json.dumps(data), encoding='utf-8')
    for name in data:
        (root / name).write_bytes(b'test fixture, not a playable video')
    return data


def test_truth_and_groups(tmp_path):
    fixture_data(tmp_path)
    labels, samples, report = module.audit(tmp_path)
    assert report['status'] == 'ready_for_media_validation'
    assert report['counts'] == {'FAKE': 1, 'REAL': 1}
    assert {r['source_id'] for r in labels} == {'real'}
    assert all(r['is_fraud'] == '' for r in labels)
    assert {r['is_fake'] for r in labels} == {0, 1}
    assert all(len(s['sha256']) == 64 for s in samples)


def test_missing_not_silently_removed(tmp_path):
    fixture_data(tmp_path)
    (tmp_path / 'fake.mp4').unlink()
    labels, _, report = module.audit(tmp_path)
    assert len(labels) == 2
    assert report['status'] == 'incomplete'
    assert report['missing_or_empty'] == ['fake.mp4']


def test_invalid_truth_rejected(tmp_path):
    data = fixture_data(tmp_path)
    data['fake.mp4']['label'] = 'SCAM'
    (tmp_path / 'metadata.json').write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='truth label'):
        module.audit(tmp_path)


def test_external_source_preserved(tmp_path):
    data = fixture_data(tmp_path)
    data['fake.mp4']['original'] = 'external.mp4'
    (tmp_path / 'metadata.json').write_text(json.dumps(data), encoding='utf-8')
    labels, _, report = module.audit(tmp_path)
    assert report['originals_outside_sample'] == ['external.mp4']
    assert labels[0]['source_id'] == 'external'


def test_duplicate_metadata_rejected(tmp_path):
    (tmp_path / 'metadata.json').write_text('{"a.mp4":{},"a.mp4":{}}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        module.audit(tmp_path)


@pytest.mark.parametrize('name', ['../bad.mp4', 'C:bad.mp4', 'dir\\bad.mp4'])
def test_unsafe_path_rejected(name):
    with pytest.raises(ValueError):
        module.video_name(name)
