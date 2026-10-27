"""Paired quality-gating experiment on stored emotion probabilities."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
from evaluate_risk import write_csv
from interval_consistency import compare,read
from run_emotion2vec_pilot import SPACES
from speech_quality import detect,window_quality


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'samples').mkdir()
    predictions=read(args.predictions);samples=json.loads(args.results.read_text(encoding='utf-8'))
    rows=[];windows=[];counts=Counter();total_voice=0;total_duration=0
    for sample in samples:
        if sample['status']!='success':raise ValueError('Resolve baseline failure before paired audit')
        report=json.loads(Path(sample['report']).read_text(encoding='utf-8'));art=report['artifacts']
        frames=detect(art['audio']);duration=report['video']['duration_seconds']
        speech=[r for r in predictions if r['video_id']==sample['id'] and r['kind']=='window']
        if not speech:raise ValueError('Missing candidate windows')
        gated=[]
        for r in speech:
            start,end=float(r['time']),min(float(r['end']),duration)
            if start>=duration:continue
            q=window_quality(start,end,frames);counts.update(q['reasons'])
            windows.append({'video_id':sample['id'],'start':start,'end':end,'emotion':r['emotion'],
                            **{k:v for k,v in q.items() if k!='reasons'},'reasons':';'.join(q['reasons'])})
            item=dict(r)
            if not q['eligible']:item['probabilities']='';item['quality_status']=';'.join(q['reasons'])
            gated.append(item)
        row={'video_id':sample['id'],'target':sample.get('emotion'),'duration':duration,
             'vad_voiced_seconds':sum(max(0,min(f['end'],duration)-f['start']) for f in frames if f['speech'] is True),
             'eligible_windows':sum(w['eligible'] for w in windows if w['video_id']==sample['id'])}
        total_voice+=row['vad_voiced_seconds'];total_duration+=duration
        detail={'frames':frames,'comparisons':{}}
        for space in ('six','seven'):
            for kind,data in [('before',speech),('after',gated)]:
                js=compare(read(art['face_emotion']),data,read(art['text_emotion']),duration,report['run_settings']['frame_fps'],shared=SPACES[space])
                detail['comparisons'][space+'_'+kind]=js
                row[space+'_'+kind+'_valid_seconds']=js['comparable_duration_seconds']
                row[space+'_'+kind+'_js']=js['mean_js_distance']
                # Speech duration that lies within windows eligible for this JS analysis.
                row[space+'_'+kind+'_voiced_seconds']=sum(window_quality(w['start'],w['end'],frames)['voiced_seconds'] for w in js['windows'] if w['js_distance'] is not None)
        rows.append(row)
        (args.output/'samples'/(sample['id']+'.json')).write_text(json.dumps(detail,indent=2,allow_nan=False),encoding='utf-8')
    metrics={'videos':len(rows),'windows':len(windows),'eligible_windows':sum(w['eligible'] for w in windows),
             'rejection_reasons_nonexclusive':dict(counts),'total_seconds':total_duration,'vad_voiced_seconds':total_voice,
             'settings':{'vad':'webrtcvad','version':importlib.metadata.version('webrtcvad-wheels'),'mode':2,'frame_ms':30,
                         'min_window_seconds':.5,'min_voiced_seconds':.3,'min_voiced_fraction':.2},'comparisons':{},
             'inputs':{k:{'path':str(v.resolve()),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in [('results',args.results),('predictions',args.predictions)]}}
    for space in ('six','seven'):
        for kind in ('before','after'):
            key=space+'_'+kind;valid=sum(r[key+'_valid_seconds'] for r in rows);voice=sum(r[key+'_voiced_seconds'] for r in rows)
            metrics['comparisons'][key]={'valid_seconds':valid,'whole_video_coverage':valid/total_duration,
                                        'vad_speech_coverage':voice/total_voice if total_voice else None,
                                        'videos_with_js':sum(r[key+'_js'] is not None for r in rows)}
    write_csv(args.output/'comparison.csv',rows,list(rows[0]));write_csv(args.output/'windows.csv',windows,list(windows[0]))
    (args.output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
