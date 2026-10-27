"""Interval-aligned, conditional shared-label JS comparison; not a fake probability."""
import argparse
import csv
import json
import math
from pathlib import Path
from alignment import normalize_emotion

SHARED = ('anger', 'happy', 'neutral', 'sad')


def distribution(row):
    raw = row.get('probabilities', '')
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict) or not value:
        raise ValueError('Malformed probability dictionary')
    p = {}
    for key, number in value.items():
        x = float(number)
        if not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError('Invalid probability')
        label = normalize_emotion(key)
        p[label] = p.get(label, 0) + x
    if abs(sum(p.values()) - 1) > 1e-4:
        raise ValueError('Distribution must sum to one')
    return p


def project(p, min_mass, shared=SHARED):
    if p is None:
        return None, None, 'missing_probabilities'
    if not set(shared) <= set(p):
        return None, None, 'unsupported_label_space'
    mass = sum(p[k] for k in shared)
    if max(p, key=p.get) not in shared:
        return None, mass, 'dominant_label_unsupported'
    if mass < min_mass:
        return None, mass, 'insufficient_shared_mass'
    return [p[k]/mass for k in shared], mass, 'comparable'


def js_distance(p, q):
    divergence = 0.0
    for a,b in zip(p,q):
        m=(a+b)/2
        if a: divergence += .5*a*math.log2(a/m)
        if b: divergence += .5*b*math.log2(b/m)
    return math.sqrt(max(0, divergence))


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def compare(face, speech, text, duration, fps=2, min_mass=.8, min_face_coverage=.5, shared=SHARED):
    if not all(math.isfinite(x) for x in (duration,fps,min_mass,min_face_coverage)) or duration <= 0 or fps <= 0 or not 0 < min_mass <= 1 or not 0 < min_face_coverage <= 1:
        raise ValueError('Invalid duration, fps or shared mass threshold')
    if not shared or len(set(shared)) != len(shared) or any(k in ("unknown", "other") for k in shared):
        raise ValueError("Invalid shared emotion space")
    frames=[]
    for i,row in enumerate(face):
        t=float(row['time']) if row.get('time') else (int(Path(row['frame']).stem)-1)/fps
        frames.append((t,row))
    windows=[]; previous_end=0
    for row in sorted(speech,key=lambda r:float(r['time'])):
        start,end=float(row['time']),float(row['end'])
        if not math.isfinite(start) or not math.isfinite(end) or start < previous_end or end <= start or start < 0:
            raise ValueError('Invalid or overlapping speech windows')
        previous_end=end
        end=min(end,duration)
        if start>=duration: continue
        selected=[r for t,r in frames if start<=t<end]
        projected=[]; masses=[]; reasons={}
        for frame in selected:
            p,mass,reason=project(distribution(frame),min_mass,shared)
            reasons[reason]=reasons.get(reason,0)+1
            if p is not None: projected.append(p); masses.append(mass)
        audio,audio_mass,audio_status=project(distribution(row),min_mass,shared)
        face_mean=[sum(p[i] for p in projected)/len(projected) for i in range(len(shared))] if projected else None
        text_matches=[]
        for segment in text:
            lo,hi=float(segment['start']),float(segment['end'])
            overlap=max(0,min(end,hi)-max(start,lo))
            if overlap:
                text_matches.append({'start':max(start,lo),'end':min(end,hi),'emotion':segment['emotion'],
                                     'probabilities':distribution(segment)})
        expected_frames=max(1, math.ceil((end-start)*fps-1e-9))
        coverage=min(1.0,len(projected)/expected_frames)
        distance=js_distance(face_mean,audio) if face_mean is not None and audio is not None and coverage >= min_face_coverage else None
        windows.append({'start':start,'end':end,'js_distance':distance,
                        'status':'comparable' if distance is not None else 'insufficient_evidence',
                        'face_time_coverage':coverage,'expected_face_frames':expected_frames,
                        'face_frames':len(selected),'face_comparable_frames':len(projected),
                        'face_comparable_fraction':len(projected)/len(selected) if selected else 0,
                        'face_rejection_counts':reasons,'speech_status':audio_status,
                        'face_shared_mass_mean':sum(masses)/len(masses) if masses else None,
                        'speech_shared_mass':audio_mass,'face_conditional_distribution':face_mean,
                        'speech_conditional_distribution':audio,'text_overlaps':text_matches})
    valid=[w for w in windows if w['js_distance'] is not None]
    covered=sum(w['end']-w['start'] for w in valid)
    return {'method':'interval_conditional_shared_js_v1','shared_labels':shared,
            'min_shared_mass':min_mass,'min_face_coverage':min_face_coverage,'duration_seconds':duration,
            'comparable_duration_seconds':covered,'comparable_duration_fraction':covered/duration,
            'mean_js_distance':sum(w['js_distance']*(w['end']-w['start']) for w in valid)/covered if covered else None,
            'prediction':None,'windows':windows,
            'limitations':['Conditional on the declared shared labels; not full-emotion JS or calibrated fake risk.',
                           'Shared-mass cutoff is an exploratory setting, not a validated detection threshold.',
                           'Text is aligned by actual overlap and excluded from JS score.',
                           'No VAD/confidence calibration; face coverage uses expected frames from configured FPS.',
                           'No fake/real threshold or sustained-anomaly decision has been calibrated.']}


def from_report(report):
    a=report['artifacts']
    return compare(read(a['face_emotion']),read(a['speech_emotion']),read(a['text_emotion']),
                   report['video']['duration_seconds'],report['run_settings']['frame_fps'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=from_report(json.loads(args.report.read_text(encoding='utf-8')))
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)


if __name__=='__main__': main()
