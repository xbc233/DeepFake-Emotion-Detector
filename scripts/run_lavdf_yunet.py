"""Same-frame detector ablation; frozen HSEmotion, speech and JS thresholds."""
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from face_emotion import load_model
from emotion_distribution import encode_distribution
from interval_consistency import compare, read

ROOT = Path(__file__).resolve().parents[1]
SHARED = ('anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    out = ROOT / 'experiments/lavdf_yunet_20261025'
    out.mkdir(exist_ok=True)
    if any(out.rglob('*.json')):
        raise FileExistsError('Completed or partial output exists; preserve it before rerunning')
    (out / 'samples').mkdir(exist_ok=True)
    source = ROOT / 'data/results/lavdf_pilot16_20261024/results.json'
    prior = ROOT / 'experiments/lavdf_emotion2vec_v2_20261024'
    weights = ROOT / 'models/yunet/face_detection_yunet_2023mar.onnx'
    model = load_model()
    detector = cv2.FaceDetectorYN.create(str(weights), '', (224,224), .9, .3, 5000)
    predictions = read(prior / 'candidate/predictions.csv')
    gates = {(r['video_id'], float(r['start'])): r['eligible'] == 'True'
             for r in read(prior / 'quality/windows.csv')}
    rows, tiles, hashes = [], [], {}
    for sample in json.loads(source.read_text(encoding='utf-8')):
        sid = sample['id']
        report = json.loads(Path(sample['report']).read_text(encoding='utf-8'))
        art = report['artifacts']
        old = read(art['face_emotion'])
        frames_dir = Path(art['frames'])
        faces, boxes = [], []
        for row in old:
            path = frames_dir / row['frame']
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Unreadable frame: {path}')
            h, w = image.shape[:2]
            detector.setInputSize((w, h))
            _, detections = detector.detect(image)
            candidates = []
            for d in ([] if detections is None else detections):
                if not np.isfinite(d).all():
                    continue
                x,y,bw,bh = map(float,d[:4])
                x0,y0 = max(0,math.floor(x)),max(0,math.floor(y))
                x1,y1 = min(w,math.ceil(x+bw)),min(h,math.ceil(y+bh))
                if x1-x0 >= 30 and y1-y0 >= 30:
                    candidates.append((x0,y0,x1,y1,float(d[-1])))
            box = max(candidates,key=lambda b:(b[2]-b[0])*(b[3]-b[1])) if candidates else None
            item = dict(frame=row['frame'],emotion='unknown',confidence=0.,probabilities='',quality_status='no_face')
            if box:
                x0,y0,x1,y1,score = box
                emotion, scores = model.predict_emotions(cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),logits=False)
                probs = np.asarray(scores,dtype=float).reshape(-1)
                item.update(emotion=str(emotion).lower(),confidence=float(probs.max()),
                            probabilities=encode_distribution([model.idx_to_class[i] for i in range(len(probs))],probs),quality_status='observed')
                cv2.rectangle(image,(x0,y0),(x1,y1),(0,255,0),2)
            boxes.append({'frame':row['frame'],'box_xyxy_score':box,'eligible_faces':len(candidates),'haar_observed':bool(row['probabilities'])})
            faces.append(item)
            # Preserve every annotated frame, not just favourable examples.
            tile = cv2.resize(image,(224,224))
            tile = cv2.copyMakeBorder(tile,24,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(tile,f"{sid[-6:]} {path.stem} H:{int(bool(row['probabilities']))} Y:{int(box is not None)}",(3,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
            tiles.append(tile)
        speech=[]
        for p in predictions:
            if p['video_id'] != sid or p['kind'] != 'window' or float(p['time']) >= report['video']['duration_seconds']:
                continue
            p = dict(p)
            if not gates[(sid,float(p['time']))]:
                p['probabilities']=''
            speech.append(p)
        args=(speech,read(art['text_emotion']),report['video']['duration_seconds'],report['run_settings']['frame_fps'])
        baseline=compare(old,*args,shared=SHARED)
        saved=json.loads((prior/'quality/samples'/f'{sid}.json').read_text())['comparisons']['seven_after']
        assert json.loads(json.dumps(baseline)) == saved, 'Baseline reproduction mismatch'
        candidate=compare(faces,*args,shared=SHARED)
        dump(out/'samples'/f'{sid}.json',{'boxes':boxes,'baseline':baseline,'candidate':candidate})
        with (out/'samples'/f'{sid}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(faces[0]));writer.writeheader();writer.writerows(faces)
        rows.append({'video_id':sid,'frames':len(old),'haar_observed':sum(bool(r['probabilities']) for r in old),
                     'yunet_observed':sum(bool(r['probabilities']) for r in faces),
                     'new_detections':sum(not b['haar_observed'] and b['box_xyxy_score'] is not None for b in boxes),
                     'lost_detections':sum(b['haar_observed'] and b['box_xyxy_score'] is None for b in boxes),
                     'duration':candidate['duration_seconds'],
                     'haar_js_seconds':baseline['comparable_duration_seconds'],'yunet_js_seconds':candidate['comparable_duration_seconds'],
                     'haar_js':baseline['mean_js_distance'],'yunet_js':candidate['mean_js_distance']})
        print(sid,rows[-1]['yunet_observed'],flush=True)
    for page,start in enumerate(range(0,len(tiles),48),1):
        batch=tiles[start:start+48]
        batch += [np.zeros_like(tiles[0])]*(48-len(batch))
        cv2.imwrite(str(out/f'boxes_{page}.jpg'),np.vstack([np.hstack(batch[i:i+8]) for i in range(0,48,8)]))
    total=sum(r['duration'] for r in rows)
    metrics={'videos':len(rows),'frames':sum(r['frames'] for r in rows),'rows':rows,
             'settings':{'detector_score':.9,'nms':.3,'min_box_side':30,'selection':'largest clipped area','shared':SHARED,'min_mass':.8,'min_face_coverage':.5},
             'model_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'opencv':cv2.__version__,
             'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frame_sha256':hashes}
    for prefix in ('haar','yunet'):
        metrics[prefix]={'observed_frames':sum(r[prefix+'_observed'] for r in rows),
                         'js_seconds':sum(r[prefix+'_js_seconds'] for r in rows),
                         'duration_coverage':sum(r[prefix+'_js_seconds'] for r in rows)/total,
                         'videos_with_js':sum(r[prefix+'_js'] is not None for r in rows)}
    dump(out/'metrics.json',metrics)
    print(json.dumps({p:metrics[p] for p in ('haar','yunet')}))


if __name__ == '__main__':
    main()
