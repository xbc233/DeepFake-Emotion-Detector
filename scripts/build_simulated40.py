"""Mux watermarked paired research videos and verify media integrity."""
import hashlib
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path
import numpy as np
import cv2
from prepare_simulated40 import ROOT,OUT


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest=OUT/'private/manifest.json';m=json.loads(manifest.read_text(encoding='utf-8'))
    durations={}
    for s in m['samples']:
        with wave.open(s['raw_audio']) as f:s['raw_audio_duration']=f.getnframes()/f.getframerate()
        durations[s['group_id']]=max(durations.get(s['group_id'],0),s['raw_audio_duration']+.4)
    sources={s['group_id']:s for s in m['sources']};rows=[];tiles=[]
    for g in sources.values():
        assert durations[g['group_id']]<=g['duration'],'No source stretching or speech truncation permitted'
        copy=Path(g['original_copy'])
        if not copy.exists():shutil.copyfile(g['source_path'],copy)
        assert digest(copy)==g['sha256']
        cap=cv2.VideoCapture(str(copy));cap.set(cv2.CAP_PROP_POS_MSEC,durations[g['group_id']]*500);ok,im=cap.read();cap.release()
        assert ok
        im=cv2.resize(im,(224,224));im=cv2.copyMakeBorder(im,24,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(im,g['group_id']+' '+g['screening_emotion'],(3,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);tiles.append(im)
    while len(tiles)%5:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(OUT/'private/source_contact_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,len(tiles),5)]))
    for s in m['samples']:
        g=sources[s['group_id']];duration=math.ceil(durations[s['group_id']]*25)/25
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',s['raw_audio'],'-af','loudnorm=I=-20:TP=-2:LRA=7,aresample=16000,apad',
                        '-t',str(duration),'-ac','1','-c:a','pcm_s16le',s['audio']],check=True)
        filt="fps=25,pad=iw:ih+32:0:32:color=black,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='SIMULATED RESEARCH':fontcolor=white:fontsize=12:x=(w-tw)/2:y=8"
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',g['source_path'],'-i',s['audio'],'-map','0:v:0','-map','1:a:0','-vf',filt,
                        '-t',str(duration),'-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','96k','-movflags','+faststart',s['video']],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',s['video']]))
        decode=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',s['video'],'-f','null','-'],capture_output=True,text=True,check=True)
        assert not decode.stderr.strip()
        assert {t['codec_type'] for t in probe['streams']}=={'audio','video'}
        assert abs(float(probe['format']['duration'])-duration)<.1
        with wave.open(s['audio']) as f:
            assert f.getframerate()==16000 and f.getnchannels()==1 and f.getsampwidth()==2
            samples=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(float)/32768
        peak=float(np.max(np.abs(samples)));assert peak<1 and peak>.001
        rms=float(np.sqrt(np.mean(samples**2)));assert rms>.001
        s.update(final_duration=duration,video_sha256=digest(s['video']),audio_sha256=digest(s['audio']),raw_audio_sha256=digest(s['raw_audio']),
                 source_sha256=g['sha256'],technically_verified=True,peak=peak,rms=rms)
        rows.append({'id':s['id'],'duration':duration,'peak':peak,'rms':rms,'decode_errors':0,'video_sha256':s['video_sha256']})
        print(s['id'],'verified',flush=True)
    m['status']='technical_complete_human_review_pending'
    (OUT/'private/generated_manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'private/technical_audit.json').write_text(json.dumps({'planned':len(m['samples']),'generated':len(rows),'originals':len(sources),'rows':rows,
           'human_labels_complete':False,'sample_rate':16000,'voice':'Microsoft Zira Desktop','text_language':'English',
           'license_clearance':'Noncommercial research terms recorded; no independent likeness or redistribution clearance',
           'emotion_acceptance':'Pending manual review; prosody presets do not establish emotional ground truth'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
