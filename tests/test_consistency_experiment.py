import hashlib,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_consistency_experiment import validate_cached_pilot


def test_reuse_rejects_different_source_and_incomplete_runs(tmp_path):
    source=tmp_path/'results.json';source.write_text('[]')
    pilot=tmp_path/'pilot';pilot.mkdir();(pilot/'predictions.csv').write_text('video_id')
    m={'provenance':{'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},'planned':24,'success':24,'failures':[]}
    (pilot/'metrics.json').write_text(json.dumps(m));validate_cached_pilot(pilot,source)
    source.write_text('[1]')
    with pytest.raises(ValueError,match='mismatch'):validate_cached_pilot(pilot,source)
    source.write_text('[]');m['success']=23;(pilot/'metrics.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='incomplete'):validate_cached_pilot(pilot,source)
