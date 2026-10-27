"""Frozen SyncNet pilot: official example, YuNet crops, and delayed-audio controls."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.io import wavfile
from scipy.ndimage import median_filter
import python_speech_features

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models/syncnet'))
from SyncNetModel import S
OUT=ROOT/'experiments/syncnet_pilot32_20261025'


def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)


def distance_curve(v,a,shift=15):
    n=min(len(v),len(a))
    if n<=2*shift:raise ValueError('Insufficient common interior')
    # Identical visual support for every tested lag; no zero-padded embeddings.
    curve=[float(torch.nn.functional.pairwise_distance(v[shift:n-shift],a[shift+k:n-shift+k]).mean()) for k in range(-shift,shift+1)]
    best=int(np.argmin(curve))
    return {'offset_frames':shift-best,'offset_seconds':(shift-best)/25,
            'confidence':float(np.median(curve)-min(curve)),'min_distance':min(curve),'zero_distance':curve[shift],
            'curve':curve,'support_windows':n-2*shift}


def audio_features(model,audio):
    mfcc=np.asarray(python_speech_features.mfcc(audio,16000)).T
    count=min(math.floor(len(audio)/640)-5,(mfcc.shape[1]-20)//4+1)
    outputs=[]
    with torch.inference_mode():
        for start in range(0,count,16):
            batch=np.stack([mfcc[:,i*4:i*4+20] for i in range(start,min(count,start+16))])[:,None]
            outputs.append(model.forward_aud(torch.from_numpy(batch).float()).cpu())
    return torch.cat(outputs)


def embeddings(model,images,audio):
    count=min(len(images),math.floor(len(audio)/640))-5
    outputs=[]
    with torch.inference_mode():
        for start in range(0,count,8):
            batch=np.stack([np.stack(images[i:i+5],axis=1).transpose(3,1,0,2) for i in range(start,min(count,start+8))])
            outputs.append(model.forward_lip(torch.from_numpy(batch).float()).cpu())
    a=audio_features(model,audio)
    return torch.cat(outputs),a[:count]


def media(path,work):
    frames=work/'frames';frames.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(path),'-vf','fps=25','-threads','1',str(frames/'%06d.jpg')],check=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(path),'-ac','1','-ar','16000','-acodec','pcm_s16le',str(work/'audio.wav')],check=True)
    images=[cv2.imread(str(p)) for p in sorted(frames.glob('*.jpg'))]
    sr,audio=wavfile.read(work/'audio.wav');assert sr==16000 and audio.ndim==1
    return images,audio


def crop(images,detector):
    boxes=[];multiface=0
    for im in images:
        h,w=im.shape[:2];detector.setInputSize((w,h));_,d=detector.detect(im)
        candidates=[] if d is None else [b for b in d if np.isfinite(b).all() and min(b[2:4])>=30]
        multiface+=int(len(candidates)>1)
        b=max(candidates,key=lambda b:b[2]*b[3]) if candidates else None
        boxes.append(None if b is None else [float(b[0]+b[2]/2),float(b[1]+b[3]/2),float(max(b[2],b[3])/2)])
    good=[i for i,b in enumerate(boxes) if b is not None]
    longest=run=0
    for b in boxes:
        run=run+1 if b is None else 0;longest=max(longest,run)
    quality={'frames':len(images),'detected':len(good),'fraction':len(good)/len(images),'max_missing_gap':longest,'multiface_frames':multiface}
    if len(good)/len(images)<.8 or longest>5 or multiface:
        return None,quality,boxes
    values=np.array([boxes[i] for i in good]);smooth=np.stack([median_filter(np.interp(np.arange(len(images)),good,values[:,j]),size=13,mode='nearest') for j in range(3)],axis=1)
    crops=[]
    for im,(x,y,s) in zip(images,smooth):
        pad=int(s*1.8);padded=np.pad(im,((pad,pad),(pad,pad),(0,0)),constant_values=110)
        x+=pad;y+=pad
        face=padded[int(y-s):int(y+s*1.8),int(x-s*1.4):int(x+s*1.4)]
        if face.size==0:raise ValueError('Invalid crop')
        crops.append(cv2.resize(face,(224,224)))
    return crops,quality,boxes


def main():
    p=argparse.ArgumentParser();p.add_argument('--example-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    model=S();state=torch.load(ROOT/'models/syncnet/syncnet_v2.model',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);model.eval()
    example=OUT/'official_example.json'
    if not example.exists():
        work=OUT/'example';work.mkdir(exist_ok=True)
        images,audio=media(ROOT/'models/syncnet/example.avi',work)
        v,a=embeddings(model,images,audio);n=min(len(v),len(a));v=v[:n];a=a[:n]
        # Reproduce upstream zero-padding metric for example only.
        padded=torch.nn.functional.pad(a,(0,0,15,15))
        curve=torch.stack([torch.nn.functional.pairwise_distance(v[i:i+1].repeat(31,1),padded[i:i+31]) for i in range(n)]).mean(0)
        official={'offset':15-int(curve.argmin()),'min_distance':float(curve.min()),'confidence':float(curve.median()-curve.min())}
        passed=official['offset']==3 and abs(official['min_distance']-5.353)<.15 and abs(official['confidence']-10.021)<.3
        save(example,{'upstream_metric':official,'expected':{'offset':3,'min_distance':5.353,'confidence':10.021},'passed':passed,'interior_metric':distance_curve(v,a)})
        print('example',official,'passed',passed,flush=True)
    if not json.loads(example.read_text())['passed']:raise ValueError('Official example did not reproduce')
    if args.example_only:return
    all_samples=json.loads((ROOT/'experiments/lavdf_expansion480_20261025/selection.json').read_text())['samples']
    train=[s for s in all_samples if s['split']=='train'];groups=list(dict.fromkeys(s['source_id'] for s in train))[:8]
    samples=[s for s in train if s['source_id'] in groups]
    save(OUT/'selection.json',{'rule':'First 8 groups in prior frozen hash-ordered training selection; no score selection','samples':samples})
    detector=cv2.FaceDetectorYN.create(str(ROOT/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(224,224),.9,.3,5000)
    results=[];tiles=[]
    for s in samples:
        work=OUT/'samples'/s['id'];work.mkdir(parents=True,exist_ok=True);target=work/'result.json'
        if target.exists():results.append(json.loads(target.read_text()));continue
        tick=time.monotonic();result={'id':s['id'],'source_id':s['source_id'],'is_fake':s['is_fake'],'modify_audio':s['metadata']['modify_audio'],'modify_video':s['metadata']['modify_video']}
        try:
            path=Path(s['path']);result['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            images,audio=media(path,work);crops,q,boxes=crop(images,detector);result['crop_quality']=q;save(work/'boxes.json',boxes)
            if crops is None:result['status']='quality_rejected'
            else:
                cv2.imwrite(str(work/'crop_preview.jpg'),np.hstack([crops[0],crops[len(crops)//2],crops[-1]]))
                v,a=embeddings(model,crops,audio);np.savez_compressed(work/'embeddings.npz',video=v.numpy(),audio=a.numpy())
                result.update(status='success',sync=distance_curve(v,a))
                if not s['is_fake']:
                    delayed=np.concatenate([np.zeros(6400,dtype=audio.dtype),audio[:-6400]])
                    shifted=audio_features(model,delayed);result['audio_delayed_400ms']=distance_curve(v,shifted)
                    result['control_offset_change']=result['audio_delayed_400ms']['offset_frames']-result['sync']['offset_frames']
        except Exception as exc:result.update(status='failed',error=str(exc))
        result['seconds']=time.monotonic()-tick;save(target,result);results.append(result)
        save(OUT/'results.json',results);print(len(results),s['id'],result['status'],round(result['seconds'],1),flush=True)
    save(OUT/'results.json',results)
    save(OUT/'status.json',{'status':'complete','samples':len(results),'success':sum(r['status']=='success' for r in results),'note':'Quality rejections/failures remain explicit; not equivalent to scientific effectiveness'})


if __name__=='__main__':main()
