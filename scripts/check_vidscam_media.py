"""Decode acquired pilot videos and screen face/speech presence, not authenticity."""
import hashlib
import argparse
import json
import subprocess
from pathlib import Path
import cv2
from speech_quality import detect, window_quality

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/vidscam_media_pilot_20261024')
    out = parser.parse_args().input
    acquisition = json.loads((out / 'acquisition.json').read_text(encoding='utf-8'))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    results = []
    for row in acquisition['samples']:
        if row['status'] != 'media_acquired':
            continue
        path = Path(row['path']); key = row['video_id']
        duration = float(row['probe']['format']['duration'])
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'], capture_output=True, text=True)
        wav = out / (key + '.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(path), '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(wav)], check=True)
        quality = window_quality(0, duration, detect(wav))
        cap = cv2.VideoCapture(str(path)); frames = []
        for n in range(12):
            t = duration * (n + .5) / 12
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                frames.append({'time': t, 'decoded': False}); continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            frames.append({'time': t, 'decoded': True, 'face_count': len(faces)})
        cap.release()
        results.append({'video_id': key, 'is_fraud': row['is_fraud'], 'is_fake': None,
                        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'duration': duration, 'decode_exit_code': decoded.returncode,
                        'decode_errors': decoded.stderr, 'vad': quality, 'sampled_frames': frames,
                        'frames_with_face': sum(f.get('face_count',0)>0 for f in frames),
                        'speaker_face_correspondence': 'not_verified',
                        'note': 'Haar face and WebRTC VAD are screening signals; neither proves talking-face alignment or intelligible speech.'})
    (out / 'media_quality.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    for r in results:
        print(r['video_id'], 'duration', round(r['duration'],1), 'faces', r['frames_with_face'], '/12', 'vad_fraction', round(r['vad']['voiced_fraction'],3), 'decode', r['decode_exit_code'])


if __name__ == '__main__':
    main()
