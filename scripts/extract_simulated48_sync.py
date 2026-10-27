"""Frozen train/dev SyncNet extraction followed by automatic baseline evaluation."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import cv2
import torch
from run_syncnet_pilot import ROOT,S,save,media,crop,embeddings,distance_curve
from refine_syncnet_tracks import RULES,segments,crops_for,combine

OUT=ROOT/'data/results/simulated48_sync_20261026'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'experiments/simulated48_features_20261026'
    samples=json.loads((source/'inputs.json').read_text())
    assert len(samples)==48
    config={'rules':RULES,'inputs':{'pilot':sha(source/'inputs.json')},
            'scripts':{s:sha(ROOT/'scripts'/s) for s in ('extract_simulated48_sync.py','run_syncnet_pilot.py','refine_syncnet_tracks.py')},
            'weights':{n:sha(ROOT/'models'/p) for n,p in [('syncnet','syncnet/syncnet_v2.model'),('yunet','yunet/face_detection_yunet_2023mar.onnx')]}}
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (OUT/'config.json').exists():assert json.loads((OUT/'config.json').read_text())['fingerprint']==fingerprint,'Changed batch configuration'
    else:save(OUT/'config.json',{'fingerprint':fingerprint,**config})
    previous={}
    model=S();model.load_state_dict(torch.load(ROOT/'models/syncnet/syncnet_v2.model',map_location='cpu',weights_only=True),strict=True);model.eval()
    detector=cv2.FaceDetectorYN.create(str(ROOT/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(224,224),.9,.3,5000)
    results=[]
    for sample in samples:
        tick=time.monotonic();sid=sample['id'];work=OUT/'samples'/sid;work.mkdir(parents=True,exist_ok=True)
        target=work/'result.json';assert sha(sample['path'])==sample['sha256']
        if target.exists():
            r=json.loads(target.read_text());assert r['fingerprint']==fingerprint and r['source_sha256']==sample['sha256']
        else:
            r={'id':sid,'split':sample['split'],'source_id':sample['source_id'],'source_sha256':sample['sha256'],'fingerprint':fingerprint}
            try:
                if sid in previous:
                    old=previous[sid];assert old['source_sha256']==sample['sha256']
                    r.update({k:v for k,v in old.items() if k not in ('is_fake','modify_audio','modify_video')})
                    r['reused_pilot']=True
                else:
                    images,audio=media(Path(sample['path']),work)
                    _,quality,boxes=crop(images,detector);save(work/'boxes.json',boxes)
                    tracks=segments(boxes);parts=[]
                    r.update(total_frames=len(images),detection_quality=quality,tracks=tracks,segments=parts,reused_pilot=False)
                    if quality['multiface_frames']:r.update(status='quality_rejected',reason='multiface_requires_redetection')
                    else:
                        for i,t in enumerate(tracks):
                            if not t['eligible']:continue
                            start,end=t['start_frame'],t['end_frame_exclusive']
                            wave=audio[start*640:min(end*640,len(audio))]
                            if len(wave)//640-5-30<25:
                                t['eligible']=False;t['reasons'].append('short_audio_support');continue
                            faces=crops_for(images,boxes,start,end);v,a=embeddings(model,faces,wave)
                            sync=distance_curve(v,a);parts.append({**t,'sync':sync})
                            cv2.imwrite(str(work/f'crop_{i}.jpg'),faces[len(faces)//2])
                        if parts:r.update(status='success',sync=combine(parts),retained_track_seconds=sum(p['frames'] for p in parts)/25)
                        else:r.update(status='quality_rejected',reason='no_track_with_sufficient_quality_and_duration')
            except Exception as exc:r.update(status='failed',error=str(exc))
            r['seconds']=time.monotonic()-tick;save(target,r)
        results.append(r);save(OUT/'results.json',results)
        save(OUT/'progress.json',{'planned':48,'completed':len(results),'success':sum(x['status']=='success' for x in results),
             'quality_rejected':sum(x['status']=='quality_rejected' for x in results),'failed':sum(x['status']=='failed' for x in results),'latest':sid})
        print(len(results),sid,r['status'],flush=True)
    save(OUT/'status.json',{'status':'complete' if not any(r['status']=='failed' for r in results) else 'failed','processed':len(results),'test_processed':False})
    if any(r['status']=='failed' for r in results):raise RuntimeError('Resolve extraction failures before evaluation')



if __name__=='__main__':main()
