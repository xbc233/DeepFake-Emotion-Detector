"""Produce fixed-window ASR and contact sheets; never certify speaker alignment."""
import json
import argparse
import subprocess
from pathlib import Path
import cv2
import numpy as np
import torch
from speech_to_text import load_whisper_model, transcribe_audio

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/vidscam_media_expand_20261024')
    parser.add_argument('--frames-only', action='store_true')
    args = parser.parse_args()
    out = args.input
    samples = json.loads((out / 'media_quality.json').read_text())
    samples += json.loads((ROOT / 'experiments/vidscam_media_pilot_20261024/media_quality.json').read_text())[:1]
    torch.set_num_threads(4)
    model = None if args.frames_only else load_whisper_model(ROOT / 'models/whisper')
    results = []
    for r in samples:
        key = r['video_id']; cap = cv2.VideoCapture(r['path']); tiles = []
        for n in range(12):
            timestamp = r['duration'] * (n + .5) / 12
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()
            tile = cv2.resize(frame, (320,180)) if ok else np.zeros((180,320,3),np.uint8)
            cv2.putText(tile,f'{timestamp:.1f}s',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
            tiles.append(tile)
        cap.release()
        sheet = np.vstack([np.hstack(tiles[n:n+4]) for n in range(0,12,4)])
        cv2.imwrite(str(out / (key + '_frames.jpg')),sheet)
        if args.frames_only:
            continue
        wav = out / (key + '_first30.wav')
        subprocess.run(['ffmpeg','-v','error','-y','-i',r['path'],'-t','30','-vn','-ar','16000','-ac','1','-c:a','pcm_s16le',str(wav)],check=True)
        transcript = transcribe_audio(model,wav)
        results.append({'video_id':key,'window':[0,min(30,r['duration'])],
                        'transcript':transcript,'asr_status':'automatic_unverified',
                        'speaker_face_correspondence':'not_verified'})
        (out/'transcript_preview.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(key,transcript['language'],len(transcript['text']),flush=True)


if __name__ == '__main__':
    main()
