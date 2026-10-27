"""Segment cached detections at gaps/jumps; never interpolate across long gaps."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.io import wavfile
from scipy.ndimage import median_filter
from run_syncnet_pilot import ROOT,S,save,embeddings,audio_features,distance_curve

BASE=ROOT/'experiments/syncnet_pilot32_20261025'
OUT=ROOT/'experiments/syncnet_tracks32_20261025'
RULES={'max_missing_gap':5,'min_frames':60,'min_detection_fraction':.8,
       'max_center_shift_halfsize':.75,'max_size_ratio':1.8,'median_filter_frames':13,
       'fps':25,'search_frames':15,'min_support_windows':25}


def segments(boxes):
    good=[i for i,b in enumerate(boxes) if b is not None]
    if not good:return []
    spans=[];start=previous=good[0]
    for i in good[1:]:
        a=np.array(boxes[previous]);b=np.array(boxes[i])
        jump=np.linalg.norm(a[:2]-b[:2])/max(min(a[2],b[2]),1e-9)>.75
        ratio=max(a[2],b[2])/max(min(a[2],b[2]),1e-9)>1.8
        if i-previous>6 or jump or ratio:
            spans.append((start,previous+1));start=i
        previous=i
    spans.append((start,previous+1))
    result=[]
    for start,end in spans:
        fraction=sum(b is not None for b in boxes[start:end])/(end-start)
        reasons=[]
        if end-start<60:reasons.append('short_track')
        if fraction<.8:reasons.append('low_detection_fraction')
        result.append({'start_frame':start,'end_frame_exclusive':end,'frames':end-start,
                       'detection_fraction':fraction,'eligible':not reasons,'reasons':reasons})
    return result


def crops_for(images,boxes,start,end):
    local=boxes[start:end];good=[i for i,b in enumerate(local) if b is not None]
    values=np.array([local[i] for i in good])
    smooth=np.stack([median_filter(np.interp(np.arange(len(local)),good,values[:,j]),size=13,mode='nearest') for j in range(3)],axis=1)
    crops=[]
    for image,(x,y,s) in zip(images[start:end],smooth):
        pad=int(s*1.8);image=np.pad(image,((pad,pad),(pad,pad),(0,0)),constant_values=110);x+=pad;y+=pad
        region=image[int(y-s):int(y+s*1.8),int(x-s*1.4):int(x+s*1.4)]
        crops.append(cv2.resize(region,(224,224)))
    return crops


def combine(parts,field='sync'):
    weights=np.array([p[field]['support_windows'] for p in parts])
    curve=np.average(np.array([p[field]['curve'] for p in parts]),axis=0,weights=weights)
    best=int(np.argmin(curve))
    return {'curve':curve.tolist(),'offset_frames':15-best,'min_distance':float(curve.min()),
            'zero_distance':float(curve[15]),'confidence':float(np.median(curve)-curve.min()),
            'support_windows':int(weights.sum()),'support_anchor_seconds':float(weights.sum()/25)}


def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    old=json.loads((BASE/'results.json').read_text());selection=json.loads((BASE/'selection.json').read_text())
    inputs={s['id']:s for s in selection['samples']}
    config={'rules':RULES,'baseline_results_sha256':hashlib.sha256((BASE/'results.json').read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256((ROOT/'models/syncnet/syncnet_v2.model').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(OUT/'config.json',config)
    model=S();model.load_state_dict(torch.load(ROOT/'models/syncnet/syncnet_v2.model',map_location='cpu',weights_only=True),strict=True);model.eval()
    results=[]
    for r in old:
        sid=r['id'];src=BASE/'samples'/sid;out=OUT/'samples'/sid;out.mkdir(parents=True,exist_ok=True)
        assert hashlib.sha256(Path(inputs[sid]['path']).read_bytes()).hexdigest()==r['source_sha256']
        boxes=json.loads((src/'boxes.json').read_text());tracks=segments(boxes)
        result={k:r[k] for k in ('id','source_id','is_fake','modify_audio','modify_video','source_sha256')}
        result.update(baseline_status=r['status'],tracks=tracks,segments=[],total_frames=len(boxes))
        # Earlier cache does not retain all multi-face boxes; do not guess a track.
        if r['crop_quality']['multiface_frames']:
            result.update(status='quality_rejected',reason='multiface_requires_redetection')
        else:
            images=None;sr,audio=wavfile.read(src/'audio.wav');assert sr==16000
            for j,t in enumerate(tracks):
                if not t['eligible']:continue
                start,end=t['start_frame'],t['end_frame_exclusive']
                if images is None:images=[cv2.imread(str(p)) for p in sorted((src/'frames').glob('*.jpg'))]
                crops=crops_for(images,boxes,start,end);wave=audio[start*640:min(end*640,len(audio))]
                cached=start==0 and end==len(boxes) and r['status']=='success'
                if cached:
                    data=np.load(src/'embeddings.npz');v=torch.from_numpy(data['video']);a=torch.from_numpy(data['audio'])
                else:v,a=embeddings(model,crops,wave)
                sync=distance_curve(v,a)
                if sync['support_windows']<25:
                    t['eligible']=False;t['reasons'].append('short_audio_support');continue
                part={**t,'sync':sync,'reused_embeddings':cached}
                if not r['is_fake']:
                    if cached:part['delayed']=r['audio_delayed_400ms']
                    else:
                        delayed=np.concatenate([np.zeros(6400,dtype=wave.dtype),wave[:-6400]])
                        part['delayed']=distance_curve(v,audio_features(model,delayed))
                    expected=sync['offset_frames']-10
                    part['control']={'expected_offset':expected,'within_search':-15<=expected<=15,
                                     'passed':abs(part['delayed']['offset_frames']-expected)<=1}
                cv2.imwrite(str(out/f'crop_{j}.jpg'),np.hstack([crops[0],crops[len(crops)//2],crops[-1]]))
                if not cached:np.savez_compressed(out/f'embeddings_{j}.npz',video=v.numpy(),audio=a.numpy())
                result['segments'].append(part)
            if result['segments']:
                result.update(status='success',sync=combine(result['segments']),
                              retained_track_seconds=sum(p['frames'] for p in result['segments'])/25)
            else:result.update(status='quality_rejected',reason='no_track_with_sufficient_quality_and_duration')
        save(out/'result.json',result);results.append(result);save(OUT/'results.json',results)
        print(sid,result['status'],len(result['segments']),flush=True)
    save(OUT/'status.json',{'status':'complete','processed':len(results),'success':sum(r['status']=='success' for r in results)})


if __name__=='__main__':main()
