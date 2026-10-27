"""Training-only local and source-paired diagnostics from frozen JS windows."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def overlap(start,end,periods):
    clipped=sorted((max(start,a),min(end,b)) for a,b in periods if max(start,a)<min(end,b))
    total=0.;last=start
    for a,b in clipped:
        total+=max(0,b-max(a,last));last=max(last,b)
    return total


def local(windows,periods):
    buckets={'touching':[], 'clean':[]};total_overlap=valid_overlap=0.;fractions=[]
    for w in windows:
        duration=w['end']-w['start'];o=overlap(w['start'],w['end'],periods)
        total_overlap+=o
        if w['js_distance'] is None:continue
        valid_overlap+=o
        buckets['touching' if o>0 else 'clean'].append((w['js_distance'],duration))
        if o>0:fractions.append(o/duration)
    means={k:sum(v*d for v,d in rows)/sum(d for v,d in rows) if rows else None for k,rows in buckets.items()}
    return {'touching_mean':means['touching'],'clean_mean':means['clean'],
            'difference':means['touching']-means['clean'] if all(v is not None for v in means.values()) else None,
            'touching_windows':len(buckets['touching']),'clean_windows':len(buckets['clean']),
            'fake_overlap_seconds':total_overlap,'valid_fake_overlap_seconds':valid_overlap,
            'mean_touching_overlap_fraction':sum(fractions)/len(fractions) if fractions else None}


def grouped(rows,field):
    groups=defaultdict(list)
    for r in rows:
        if r[field] is not None:groups[r['source_id']].append(r[field])
    values=np.array([sum(v)/len(v) for v in groups.values()])
    if not len(values):return {'groups':0,'mean':None,'bootstrap95':None}
    rng=np.random.default_rng(20261025)
    means=values[rng.integers(0,len(values),size=(5000,len(values)))].mean(axis=1)
    return {'groups':len(values),'videos':sum(len(v) for v in groups.values()),'mean':float(values.mean()),
            'positive_group_fraction':float((values>0).mean()),
            'bootstrap95':[float(x) for x in np.quantile(means,[.025,.975])]}


def main():
    selection=ROOT/'experiments/lavdf_expansion480_20261025/selection.json'
    samples=[s for s in json.loads(selection.read_text(encoding='utf-8'))['samples'] if s['split']=='train']
    reports={s['metadata']['file']:json.loads((ROOT/'data/results/lavdf_features400_20261025/samples'/s['id']/'report.json').read_text(encoding='utf-8')) for s in samples}
    rows=[]
    for s in samples:
        if not s['is_fake']:continue
        meta=s['metadata'];r=reports[meta['file']];real=reports[s['source_id']]
        js=r['interval_js_comparison'];base=real['interval_js_comparison']
        stat=local(js['windows'],meta['fake_periods'])
        rows.append({'id':s['id'],'source_id':s['source_id'],
                     'type':('audio' if meta['modify_audio'] else '')+('video' if meta['modify_video'] else ''),**stat,
                     'whole_video_paired_difference':js['mean_js_distance']-base['mean_js_distance'] if js['mean_js_distance'] is not None and base['mean_js_distance'] is not None else None})
    summary={'split':'train','fake_videos':len(rows),'source_groups':len({r['source_id'] for r in rows}),
             'definition':'Touching = any annotated fake overlap; clean = zero overlap. Entire mixed windows stay in touching, never fractional negative assignment.',
             'local_difference':grouped(rows,'difference'),'whole_video_paired_difference':grouped(rows,'whole_video_paired_difference'),
             'missing_local_comparisons':sum(r['difference'] is None for r in rows),
             'fake_time_coverage':sum(r['valid_fake_overlap_seconds'] for r in rows)/sum(r['fake_overlap_seconds'] for r in rows),
             'by_type':{},'bootstrap':'5000 resamples of source-group means, seed 20261025; descriptive training-set interval, not independent-video inference'}
    for kind in ('audio','video','audiovideo'):
        subset=[r for r in rows if r['type']==kind]
        summary['by_type'][kind]={'local':grouped(subset,'difference'),'paired':grouped(subset,'whole_video_paired_difference')}
    out=ROOT/'experiments/lavdf_local_train_20261025';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
