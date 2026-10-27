"""Offline, resumable train/dev features; one model load per process, no fitting."""
import os
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):
    os.environ[key]='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
import csv
import hashlib
import json
import math
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/results/lavdf_features400_20261025'
INPUT=ROOT/'experiments/lavdf_expansion480_20261025/splits'
SHARED=('anger','disgust','fear','happy','neutral','sad','surprise')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(p)


def write_rows(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    import cv2
    import numpy as np
    import soundfile as sf
    import torch
    from extract_audio import extract_audio,find_ffmpeg
    from extract_frames import extract_frames
    from face_emotion import load_model
    from speech_to_text import load_whisper_model,transcribe_audio,save_transcript
    from text_sentiment import load_classifier,process_transcript
    from emotion2vec_candidate import load_verified_model,normalize_result
    from emotion_distribution import encode_distribution
    from speech_quality import detect,window_quality
    from interval_consistency import compare,read
    torch.set_num_threads(4)
    samples=[]
    for split in ('train','dev'):
        samples.extend(json.loads((INPUT/f'{split}_inputs.json').read_text()))
    assert len(samples)==400 and all(s['split'] in ('train','dev') for s in samples)
    OUT.mkdir(parents=True,exist_ok=True)
    settings={'inputs':{s:digest(INPUT/f'{s}_inputs.json') for s in ('train','dev')},
              'scripts':{n:digest(ROOT/'scripts'/n) for n in ('extract_lavdf_features.py','face_emotion.py','emotion2vec_candidate.py','speech_quality.py','interval_consistency.py','text_sentiment.py','speech_to_text.py','extract_audio.py','extract_frames.py','emotion_distribution.py')},
              'fps':2,'speech_window':3,'yunet_score':.9,'yunet_nms':.3,'min_box_side':30,
              'face_model':'enet_b0_8_best_afew','speech_model':'emotion2vec_plus_base','text_model':'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',
              'shared':SHARED,'min_mass':.8,'min_face_coverage':.5,'whisper':'tiny','language':'auto',
              'yunet_sha256':digest(ROOT/'models/yunet/face_detection_yunet_2023mar.onnx'),
              'face_sha256':digest(Path.home()/'.hsemotion/enet_b0_8_best_afew.pt'),
              'speech_sha256':digest(ROOT/'models/emotion2vec_plus_base/model.pt'),
              'whisper_sha256':digest(ROOT/'models/whisper/tiny.pt')}
    fingerprint=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    config=OUT/'config.json'
    if config.exists():
        assert json.loads(config.read_text())['fingerprint']==fingerprint,'Changed inputs/config/code; use a new output directory'
    else:save(config,{'fingerprint':fingerprint,'settings':settings})
    pending=[];results=[]
    for s in samples:
        assert digest(s['path'])==s['sha256'],'Media changed'
        report=OUT/'samples'/s['id']/'report.json'
        if report.exists():
            previous=json.loads(report.read_text(encoding='utf-8'))
            assert previous['fingerprint']==fingerprint and previous['source_sha256']==s['sha256']
            assert all(Path(p).exists() and digest(p)==h for p,h in previous['artifact_hashes'].items()),'Cached artifact changed'
            results.append({'id':s['id'],'split':s['split'],'status':'success','report':str(report)})
        else:pending.append(s)
    if pending:
        print('Loading models once; pending',len(pending),flush=True)
        face_model=load_model();whisper=load_whisper_model(ROOT/'models/whisper')
        text_model=load_classifier(settings['text_model'])
        speech_model,verification=load_verified_model(ROOT/'models/emotion2vec_plus_base')
        save(OUT/'model_verification.json',verification)
        detector=cv2.FaceDetectorYN.create(str(ROOT/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(224,224),.9,.3,5000)
    for s in pending:
        started=time.monotonic();work=OUT/'samples'/s['id'];work.mkdir(parents=True,exist_ok=True)
        try:
            audio=work/'audio.wav';frames=work/'frames';transcript=work/'transcript.json'
            face_csv=work/'face.csv';speech_csv=work/'speech.csv';text_csv=work/'text.csv'
            extract_audio(Path(s['path']),audio,find_ffmpeg())
            extract_frames(Path(s['path']),frames,2)
            if not transcript.exists():save_transcript(transcribe_audio(whisper,audio,None),transcript)
            boxes=[];faces=[]
            for path in sorted(frames.glob('*.jpg')):
                image=cv2.imread(str(path))
                if image is None:raise ValueError('Unreadable frame')
                h,w=image.shape[:2];detector.setInputSize((w,h));_,detections=detector.detect(image)
                candidates=[]
                for d in ([] if detections is None else detections):
                    if not np.isfinite(d).all():continue
                    x,y,bw,bh=map(float,d[:4]);x0,y0=max(0,math.floor(x)),max(0,math.floor(y))
                    x1,y1=min(w,math.ceil(x+bw)),min(h,math.ceil(y+bh))
                    if x1-x0>=30 and y1-y0>=30:candidates.append((x0,y0,x1,y1,float(d[-1])))
                box=max(candidates,key=lambda b:(b[2]-b[0])*(b[3]-b[1])) if candidates else None
                row={'frame':path.name,'emotion':'unknown','confidence':0.,'probabilities':'','quality_status':'no_face'}
                if box:
                    x0,y0,x1,y1,_=box
                    label,p=face_model.predict_emotions(cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),logits=False)
                    p=np.asarray(p,dtype=float).reshape(-1)
                    row.update(emotion=str(label).lower(),confidence=float(p.max()),probabilities=encode_distribution([face_model.idx_to_class[i] for i in range(len(p))],p),quality_status='observed')
                faces.append(row);boxes.append({'frame':path.name,'box':box,'eligible_faces':len(candidates)})
            if not faces:raise ValueError('No extracted frames')
            write_rows(face_csv,faces);save(work/'boxes.json',boxes)
            waveform,sr=sf.read(audio,dtype='float32')
            assert sr==16000 and waveform.ndim==1
            vad=detect(audio);speech=[];quality=[];duration=s['actual_duration_seconds']
            for start in range(0,len(waveform),3*sr):
                end=min(start+3*sr,len(waveform));lo=start/sr;hi=min(end/sr,duration)
                if lo>=hi:continue
                q=window_quality(lo,hi,vad);quality.append({'start':lo,'end':hi,**q})
                row={'time':lo,'end':hi,'emotion':'unknown','confidence':0.,'probabilities':'','quality_status':';'.join(q['reasons']),'raw_probabilities':''}
                if end-start>=sr*.5:
                    raw=speech_model.generate(input=waveform[start:end],granularity='utterance',extract_embedding=False)
                    p=normalize_result(raw[0]);label=max(p,key=p.get)
                    row.update(emotion=label,confidence=p[label],raw_probabilities=json.dumps(p))
                    if q['eligible']:row.update(probabilities=json.dumps(p),quality_status='observed')
                speech.append(row)
            write_rows(speech_csv,speech);save(work/'quality.json',{'vad':vad,'windows':quality})
            process_transcript(transcript,text_csv,text_model,8)
            js=compare(faces,speech,read(text_csv),duration,2,shared=SHARED)
            artifacts={'audio':str(audio),'frames':str(frames),'transcript':str(transcript),'face_emotion':str(face_csv),'speech_emotion':str(speech_csv),'text_emotion':str(text_csv)}
            hashes={str(p):digest(p) for p in work.rglob('*') if p.is_file() and p.name not in ('report.json','error.json')}
            report={'id':s['id'],'split':s['split'],'source_id':s['source_id'],'source_sha256':s['sha256'],'fingerprint':fingerprint,
                    'video':{'path':s['path'],'duration_seconds':duration},'run_settings':settings,'artifacts':artifacts,'artifact_hashes':hashes,
                    'interval_js_comparison':js,'seconds':time.monotonic()-started,'note':'Frozen features only; JS is not a fake probability; text not in JS; no labels read.'}
            save(work/'report.json',report)
            results.append({'id':s['id'],'split':s['split'],'status':'success','report':str(work/'report.json')})
        except Exception as exc:
            save(work/'error.json',{'error':str(exc)})
            results.append({'id':s['id'],'split':s['split'],'status':'failed','error':str(exc)})
        save(OUT/'results.json',results)
        save(OUT/'progress.json',{'planned':len(samples),'completed':len(results),'success':sum(r['status']=='success' for r in results),'failed':sum(r['status']=='failed' for r in results),'latest':s['id']})
        print(len(results),s['id'],results[-1]['status'],round(time.monotonic()-started,1),flush=True)
    save(OUT/'results.json',results)
    save(OUT/'status.json',{'status':'complete' if all(r['status']=='success' for r in results) else 'failed','videos':len(results),'test_processed':False})


if __name__=='__main__':main()
