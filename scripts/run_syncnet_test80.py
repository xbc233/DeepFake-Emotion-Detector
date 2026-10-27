"""Apply immutable preprocessing/model protocol to held-out media; no labels read."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import cv2
import torch
from run_syncnet_pilot import ROOT,S,save,media,crop,embeddings,distance_curve
from refine_syncnet_tracks import RULES,segments,crops_for,combine

OUT=ROOT/'data/results/syncnet_test80_20261025'
PROTOCOL=ROOT/'experiments/syncnet_test80_20261025'


def main():
    torch.set_num_threads(4)
    frozen=json.loads((PROTOCOL/'frozen_protocol.json').read_text())
    for p,h in frozen['code_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'Frozen code changed'
    configpath=ROOT/'data/results/syncnet400_20261025/config.json'
    assert hashlib.sha256(configpath.read_bytes()).hexdigest()==frozen['extraction_config_sha256']
    config=json.loads(configpath.read_text());assert config['rules']==RULES
    for name,path in [('syncnet','syncnet/syncnet_v2.model'),('yunet','yunet/face_detection_yunet_2023mar.onnx')]:
        assert hashlib.sha256((ROOT/'models'/path).read_bytes()).hexdigest()==config['weights'][name]
    inputs=ROOT/'experiments/lavdf_expansion480_20261025/splits/test_inputs.json'
    samples=json.loads(inputs.read_text());assert len(samples)==80 and all(s['split']=='test' for s in samples)
    old=json.loads((ROOT/'data/results/syncnet400_20261025/results.json').read_text())
    assert not {s['source_id'] for s in samples}&{r['source_id'] for r in old}
    OUT.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256((PROTOCOL/'frozen_protocol.json').read_bytes()+inputs.read_bytes()).hexdigest()
    save(OUT/'config.json',{'fingerprint':fingerprint,'protocol':frozen,'inputs_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest()})
    model=S();model.load_state_dict(torch.load(ROOT/'models/syncnet/syncnet_v2.model',map_location='cpu',weights_only=True),strict=True);model.eval()
    detector=cv2.FaceDetectorYN.create(str(ROOT/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(224,224),.9,.3,5000)
    results=[]
    for sample in samples:
        sid=sample['id'];work=OUT/'samples'/sid;work.mkdir(parents=True,exist_ok=True)
        assert hashlib.sha256(Path(sample['path']).read_bytes()).hexdigest()==sample['sha256']
        target=work/'result.json'
        if target.exists():
            r=json.loads(target.read_text());assert r['fingerprint']==fingerprint and r['source_sha256']==sample['sha256']
        else:
            r={'id':sid,'split':'test','source_id':sample['source_id'],'source_sha256':sample['sha256'],'fingerprint':fingerprint}
            try:
                images,audio=media(Path(sample['path']),work)
                _,quality,boxes=crop(images,detector);save(work/'boxes.json',boxes)
                tracks=segments(boxes);parts=[];r.update(total_frames=len(images),detection_quality=quality,tracks=tracks,segments=parts)
                if quality['multiface_frames']:r.update(status='quality_rejected',reason='multiface_requires_redetection')
                else:
                    for i,t in enumerate(tracks):
                        if not t['eligible']:continue
                        start,end=t['start_frame'],t['end_frame_exclusive'];wave=audio[start*640:min(end*640,len(audio))]
                        if len(wave)//640-35<25:
                            t['eligible']=False;t['reasons'].append('short_audio_support');continue
                        faces=crops_for(images,boxes,start,end);v,a=embeddings(model,faces,wave)
                        parts.append({**t,'sync':distance_curve(v,a)});cv2.imwrite(str(work/f'crop_{i}.jpg'),faces[len(faces)//2])
                    if parts:r.update(status='success',sync=combine(parts),retained_track_seconds=sum(p['frames'] for p in parts)/25)
                    else:r.update(status='quality_rejected',reason='no_track_with_sufficient_quality_and_duration')
            except Exception as exc:r.update(status='failed',error=str(exc))
            save(target,r)
        results.append(r);save(OUT/'results.json',results)
        save(OUT/'progress.json',{'planned':80,'completed':len(results),'success':sum(r['status']=='success' for r in results),'quality_rejected':sum(r['status']=='quality_rejected' for r in results),'failed':sum(r['status']=='failed' for r in results)})
        print(len(results),sid,r['status'],flush=True)
    save(OUT/'status.json',{'status':'complete' if not any(r['status']=='failed' for r in results) else 'failed','processed':len(results)})
    if any(r['status']=='failed' for r in results):raise RuntimeError('Test extraction failures; no evaluation')
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_syncnet_test80.py')],check=True,cwd=ROOT)


if __name__=='__main__':main()
