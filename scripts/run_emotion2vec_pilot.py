"""Paired speech-model pilot with fixed four/six/seven-class JS ablations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from alignment import normalize_emotion
from emotion2vec_candidate import load_verified_model, normalize_result
from evaluate_risk import write_csv
from interval_consistency import compare,read,SHARED

SPACES={'four':SHARED,'six':('anger','fear','happy','neutral','sad','surprise'),
        'seven':('anger','disgust','fear','happy','neutral','sad','surprise')}
TARGETS=SPACES['six']


def classification(rows, field):
    rows=[r for r in rows if r.get('target') in TARGETS]
    n=len(rows);conf=Counter((r['target'],r[field]) for r in rows)
    f1=[]
    for label in TARGETS:
        tp=conf[label,label];fp=sum(v for (a,b),v in conf.items() if b==label and a!=label)
        fn=sum(v for (a,b),v in conf.items() if a==label and b!=label)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return {'n':n,'accuracy':sum(r['target']==r[field] for r in rows)/n if n else None,
            'macro_f1_six_targets':sum(f1)/len(f1) if n else None,
            'confusion':{k:dict(Counter(r[field] for r in rows if r['target']==k)) for k in TARGETS}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,default=Path('models/emotion2vec_plus_base'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);(args.output/'samples').mkdir()
    import soundfile as sf
    import importlib.metadata
    model,verification=load_verified_model(args.model)
    results=json.loads(args.results.read_text(encoding='utf-8'));rows=[];failures=[];all_predictions=[]
    for i,sample in enumerate(results,1):
        print(f'[{i}/{len(results)}] {sample["id"]}',flush=True)
        try:
            if sample['status']!='success':raise ValueError('Baseline failed')
            report=json.loads(Path(sample['report']).read_text(encoding='utf-8'));a=report['artifacts']
            audio_path=Path(a['audio']);audio,sr=sf.read(audio_path,dtype='float32')
            if sr!=16000 or audio.ndim!=1:raise ValueError('Expected mono 16k audio')
            old=read(a['speech_emotion']);new=[];utterance=None;inference_seconds=0
            spans=[('utterance',0,len(audio))]+[('window',round(float(w['time'])*sr),min(len(audio),round(float(w['end'])*sr))) for w in old]
            for kind,start,end in spans:
                if end-start < round(.5*sr):
                    item={'time':start/sr,'end':end/sr,'emotion':'unknown','confidence':0.0,
                          'probabilities':'','quality_status':'short_window_not_inferred'}
                    all_predictions.append({'video_id':sample['id'],'kind':kind,**item})
                    if kind=='window':new.append(item)
                    else:utterance='unknown'
                    continue
                tick=time.perf_counter();raw=model.generate(input=audio[start:end],granularity='utterance',extract_embedding=False)
                inference_seconds+=time.perf_counter()-tick
                probs=normalize_result(raw[0]);label=max(probs,key=probs.get)
                item={'time':start/sr,'end':end/sr,'emotion':label,'confidence':probs[label],
                      'probabilities':json.dumps(probs),'quality_status':'observed_vad_not_checked'}
                all_predictions.append({'video_id':sample['id'],'kind':kind,**item})
                if kind=='utterance':utterance=label
                else:new.append(item)
            def dominant(records):
                counts=Counter(normalize_emotion(r['emotion']) for r in records)
                return counts.most_common(1)[0][0]
            row={'video_id':sample['id'],'target':sample.get('emotion'),'baseline_window_vote':dominant(old),
                 'candidate_window_vote':dominant(new),'candidate_utterance':utterance,'seconds':inference_seconds}
            detail={'source_report':sample['report'],'audio_sha256':hashlib.sha256(audio_path.read_bytes()).hexdigest(),'comparisons':{}}
            face,text=read(a['face_emotion']),read(a['text_emotion'])
            variants=[('baseline_four',old,SHARED)]+[('candidate_'+k,new,v) for k,v in SPACES.items()]
            for name,speech,space in variants:
                value=compare(face,speech,text,report['video']['duration_seconds'],report['run_settings']['frame_fps'],shared=space)
                detail['comparisons'][name]=value
                row[name+'_coverage']=value['comparable_duration_fraction'];row[name+'_js']=value['mean_js_distance']
                row[name+'_valid_seconds']=value['comparable_duration_seconds']
            row['duration']=report['video']['duration_seconds'];rows.append(row)
            (args.output/'samples'/(sample['id']+'.json')).write_text(json.dumps(detail,indent=2,allow_nan=False),encoding='utf-8')
        except Exception as exc:
            failures.append({'video_id':sample['id'],'error':str(exc)})
    metrics={'provenance':{'source_results':str(args.results.resolve()),'source_sha256':hashlib.sha256(args.results.read_bytes()).hexdigest()},'planned':len(results),'success':len(rows),'failures':failures,'verification':verification,
             'model':json.loads((args.model/'download.json').read_text(encoding='utf-8')),
             'versions':{k:importlib.metadata.version(k) for k in ['funasr','modelscope','torch']},
             'classification':{k:classification(rows,k) for k in ['baseline_window_vote','candidate_window_vote','candidate_utterance']},'coverage':{}}
    for name in ['baseline_four','candidate_four','candidate_six','candidate_seven']:
        duration=sum(r['duration'] for r in rows);valid=sum(r[name+'_valid_seconds'] for r in rows)
        metrics['coverage'][name]={'duration_coverage':valid/duration if duration else None,'valid_seconds':valid,
                                 'videos_with_js':sum(r[name+'_js'] is not None for r in rows),
                                 'videos_at_least_50pct':sum(r[name+'_coverage']>=.5 for r in rows)}
    if rows:write_csv(args.output/'comparison.csv',rows,list(rows[0]))
    if all_predictions:write_csv(args.output/'predictions.csv',all_predictions,list(all_predictions[0]))
    (args.output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'success':len(rows),'failures':failures,'classification':metrics['classification'],'coverage':metrics['coverage']},ensure_ascii=True),flush=True)
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
