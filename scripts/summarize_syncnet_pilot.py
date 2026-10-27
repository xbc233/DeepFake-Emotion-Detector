"""Same-subset descriptive rankings; no trained fusion or test evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_risk import binary_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/syncnet_pilot32_20261025'


def ranking(rows,field):
    m=binary_metrics([(r['is_fake'],r[field]) for r in rows],.5)
    return {k:m[k] for k in ('n','positive','negative','roc_auc','pr_auc_average_precision')}


def main():
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='complete' and status['samples']==32
    results=json.loads((OUT/'results.json').read_text());rows=[];controls=[]
    for r in results:
        if r['status']!='success':continue
        report=ROOT/'data/results/lavdf_features400_20261025/samples'/r['id']/'report.json'
        old=json.loads(report.read_text(encoding='utf-8'))
        assert r['source_sha256']==old['source_sha256']
        js=old['interval_js_comparison']['mean_js_distance']
        rows.append({'id':r['id'],'source_id':r['source_id'],'is_fake':r['is_fake'],
                     'modify_audio':r['modify_audio'],'modify_video':r['modify_video'],
                     'zero_distance':r['sync']['zero_distance'],'min_distance':r['sync']['min_distance'],
                     'negative_confidence':-r['sync']['confidence'],'absolute_offset':abs(r['sync']['offset_frames']),
                     'emotion_js':js})
        if 'control_offset_change' in r:
            baseline=r['sync']['offset_frames'];expected=baseline-10
            controls.append({'id':r['id'],'baseline_offset':baseline,'delayed_offset':r['audio_delayed_400ms']['offset_frames'],
                             'expected_delayed_offset':expected,'within_search':-15<=expected<=15,
                             'passed_one_frame_tolerance':abs(r['control_offset_change']+10)<=1,
                             'zero_distance_change':r['audio_delayed_400ms']['zero_distance']-r['sync']['zero_distance']})
    fields=['zero_distance','min_distance','negative_confidence','absolute_offset','emotion_js']
    common=[r for r in rows if r['emotion_js'] is not None]
    summary={'planned':32,'success':len(rows),'failures_or_quality_rejections':[r for r in results if r['status']!='success'],
             'source_groups':len({r['source_id'] for r in rows}),'paired_emotion_subset':len(common),
             'same_subset_rankings':{k:ranking(common,k) for k in fields},'controls':controls,
             'control_success_in_search':sum(c['passed_one_frame_tolerance'] and c['within_search'] for c in controls),
             'control_count_in_search':sum(c['within_search'] for c in controls),'by_type':{},
             'limitations':['Training pilot only; 8 correlated source groups, not independent generalization.',
                            'YuNet crop adapter differs from upstream S3FD tracker. No speech/VAD gating in this pilot.',
                            'SyncNet synchronization is not fake/fraud truth. Synthetic delays only test offset sensitivity.',
                            'Fixed score directions, no threshold selection or score inversion; no classifier fitted.',
                            'Reported rankings are descriptive; multiple feature comparisons not corrected. Test/dev not used.']}
    for audio,video in ((False,True),(True,False),(True,True)):
        subset=[r for r in common if not r['is_fake'] or (r['modify_audio']==audio and r['modify_video']==video)]
        summary['by_type'][f'audio_{audio}_video_{video}']={k:ranking(subset,k) for k in fields}
    group_counts={g:sum(r['source_id']==g for r in common) for g in {r['source_id'] for r in common}}
    complete=[r for r in common if group_counts[r['source_id']]==4]
    summary['complete_four_variant_groups']={'groups':len({r['source_id'] for r in complete}),
        'rankings':{k:ranking(complete,k) for k in fields}}
    source_files=[ROOT/'scripts/run_syncnet_pilot.py',ROOT/'scripts/summarize_syncnet_pilot.py',*list((ROOT/'models/syncnet').glob('*.py'))]
    summary['code_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'comparison_rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('success','source_groups','same_subset_rankings','control_success_in_search','control_count_in_search')}))


if __name__=='__main__':main()
