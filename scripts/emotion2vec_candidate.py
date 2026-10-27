"""Isolated emotion2vec+ CPU adapter; retains other/unknown probability mass."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

ALIASES={'angry':'anger','disgusted':'disgust','fearful':'fear','happy':'happy',
         'neutral':'neutral','other':'other','sad':'sad','surprised':'surprise','<unk>':'unknown','unknown':'unknown'}


def normalize_result(result):
    labels,scores=result['labels'],result['scores']
    if len(labels)!=len(scores) or len(labels)!=9:
        raise ValueError('Expected all nine native classes')
    p={}
    for label,score in zip(labels,scores):
        key=ALIASES.get(label.split('/')[-1].strip())
        value=float(score)
        if key is None or key in p or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid or duplicate native probability')
        p[key]=value
    if abs(sum(p.values())-1)>1e-4:
        raise ValueError('Native probabilities do not sum to one')
    return p


def load_verified_model(model_path):
    from funasr import AutoModel
    started=time.perf_counter()
    model=AutoModel(model=str(model_path.resolve()),device='cpu',disable_update=True)
    import torch
    checkpoint=torch.load(model_path/'model.pt',map_location='cpu',weights_only=True)
    for container in ('state_dict','model_state_dict','model'):
        if container in checkpoint: checkpoint=checkpoint[container]
    checked=0
    unused_decoder=[]
    for name,tensor in model.model.state_dict().items():
        if name.startswith('modality_encoders.AUDIO.decoder.'):
            unused_decoder.append(name)
            continue
        source=checkpoint.get(name,checkpoint.get('d2v_model.'+name,checkpoint.get('module.'+name)))
        if source is None or tensor.shape!=source.shape or not torch.equal(tensor.cpu(),source):
            raise ValueError(f'Checkpoint did not fully initialize model parameter: {name}')
        checked+=1
    def reject_decoder_use(module, inputs):
        raise RuntimeError('Training-only decoder unexpectedly used during inference')
    model.model.modality_encoders['AUDIO'].decoder.register_forward_pre_hook(reject_decoder_use)
    del checkpoint
    load_seconds=time.perf_counter()-started
    return model, {"load_seconds":load_seconds,"verified_state_tensors":checked,"unused_training_decoder_tensors":unused_decoder}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True,help='Existing pipeline report with extracted audio')
    parser.add_argument('--model',type=Path,default=Path('models/emotion2vec_plus_base'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=json.loads(args.report.read_text(encoding='utf-8'))
    import soundfile as sf
    from funasr import AutoModel
    audio_path=Path(report['artifacts']['audio'])
    audio,sr=sf.read(audio_path,dtype='float32')
    if sr!=16000 or audio.ndim!=1: raise ValueError('Expected mono 16 kHz audio')
    model, verification=load_verified_model(args.model)
    rows=[]; raw=[]
    # Include whole-utterance diagnostic plus the same 3-second windows as baseline.
    step=round(report['run_settings']['speech_window']*sr)
    spans=[('utterance',0,len(audio))]+[('window',i,min(i+step,len(audio))) for i in range(0,len(audio),step)]
    for kind,start,end in spans:
        tick=time.perf_counter()
        result=model.generate(input=audio[start:end],granularity='utterance',extract_embedding=False)
        p=normalize_result(result[0]);label=max(p,key=p.get)
        rows.append({'kind':kind,'time':start/sr,'end':end/sr,'emotion':label,'confidence':p[label],
                     'probabilities':json.dumps(p,ensure_ascii=False),'quality_status':'observed_vad_not_checked',
                     'seconds':time.perf_counter()-tick})
        raw.append({'kind':kind,'start':start/sr,'end':end/sr,'result':result})
    with (args.output/'speech.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata={'source_report':str(args.report.resolve()),'audio_sha256':hashlib.sha256(audio_path.read_bytes()).hexdigest(),
              'model':json.loads((args.model/'download.json').read_text(encoding='utf-8')),
              **verification,'versions':{k:importlib.metadata.version(k) for k in ['funasr','modelscope','torch']},
              'results':raw,'note':'Single-sample functional validation; no accuracy or generalization claim.'}
    (args.output/'result.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print([(r['kind'],r['time'],r['emotion'],r['confidence']) for r in rows])


if __name__=='__main__':main()
