"""WebRTC speech activity and explicit window eligibility; no waveform trimming."""
import math
import wave


def detect(path, mode=2, frame_ms=30):
    import webrtcvad
    if frame_ms not in (10,20,30) or mode not in (0,1,2,3):raise ValueError('Invalid VAD configuration')
    with wave.open(str(path),'rb') as f:
        rate=f.getframerate()
        if f.getnchannels()!=1 or f.getsampwidth()!=2 or rate!=16000 or f.getcomptype()!='NONE':
            raise ValueError('Expected mono 16-bit PCM at 16 kHz')
        pcm=f.readframes(f.getnframes())
    size=rate*frame_ms//1000*2;vad=webrtcvad.Vad(mode);frames=[]
    for offset in range(0,len(pcm),size):
        chunk=pcm[offset:offset+size];complete=len(chunk)==size
        frames.append({'start':offset/2/rate,'end':(offset+len(chunk))/2/rate,
                       'speech':bool(vad.is_speech(chunk,rate)) if complete else None})
    return frames


def window_quality(start,end,frames,min_window=.5,min_voiced=.3,min_fraction=.2):
    if not all(math.isfinite(x) for x in (start,end,min_window,min_voiced,min_fraction)) or start<0 or end<=start or min_window<0 or min_voiced<0 or not 0<=min_fraction<=1:
        raise ValueError('Invalid window or thresholds')
    voiced=observed=0.0
    for frame in frames:
        overlap=max(0,min(end,frame['end'])-max(start,frame['start']))
        if frame['speech'] is not None:observed+=overlap
        if frame['speech'] is True:voiced+=overlap
    duration=end-start;fraction=voiced/duration
    reasons=[]
    if duration+1e-9<min_window:reasons.append('short_window')
    if voiced+1e-9<min_voiced:reasons.append('insufficient_voiced_seconds')
    if fraction+1e-9<min_fraction:reasons.append('low_voiced_fraction')
    if observed<=0:reasons.append('no_vad_observation')
    return {'duration':duration,'voiced_seconds':voiced,'voiced_fraction':fraction,
            'vad_observed_seconds':observed,'vad_unobserved_seconds':max(0,duration-observed),
            'eligible':not reasons,'reasons':reasons}
