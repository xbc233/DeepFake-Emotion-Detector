"""Export checked split inputs and separate ground truth for the next stage."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/lavdf_expansion480_20261025'


def main():
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'verified'
    assert manifest['selection_sha256'] == hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    samples = manifest['samples']
    for s in samples:
        assert s['status'] == 'verified'
        assert hashlib.sha256(Path(s['path']).read_bytes()).hexdigest() == s['sha256']
        assert all(end <= s['actual_duration_seconds'] + .05 for start,end in s['metadata']['fake_periods'])
    destination = OUT/'splits'
    destination.mkdir(exist_ok=True)
    for split in ('train','dev','test'):
        selected = [s for s in samples if s['split']==split]
        # Feature extractors need only media identity and integrity, not labels.
        inputs = [{k:s[k] for k in ('id','path','sha256','source_id','split','actual_duration_seconds')} for s in selected]
        (destination/f'{split}_inputs.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
        labels = [{'video_id':s['id'],'source_id':s['source_id'],'split':split,
                   'is_fake':s['is_fake'],'is_fraud':'','emotion_label':'','person_id':'',
                   'modify_audio':s['metadata']['modify_audio'],'modify_video':s['metadata']['modify_video'],
                   'fake_periods':json.dumps(s['metadata']['fake_periods']),
                   'label_source':'LAV-DF official metadata'} for s in selected]
        with (destination/f'{split}_labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(labels[0]));writer.writeheader();writer.writerows(labels)
    audit = {'status':'ready_for_feature_extraction','videos':len(samples),'source_groups':len({s['source_id'] for s in samples}),
             'split_counts':manifest['split_counts'],'group_cross_split_leaks':manifest['group_leaks'],
             'exact_content_cross_split_leaks':manifest['exact_content_cross_split_leaks'],
             'media_sha256_rechecked':len(samples),'decode_success':len(samples),
             'fake_interval_exceeds_media_tolerance':0,'interval_tolerance_seconds':.05,
             'total_duration_seconds':sum(s['actual_duration_seconds'] for s in samples),
             'total_bytes':manifest['bytes'],'fake_counts':dict(Counter(s['is_fake'] for s in samples)),
             'model_inference_started':False,'training_started':False,
             'identity_separation_verified':False,'near_duplicate_audit_completed':False,
             'full_archive_sha256_verified':False}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
