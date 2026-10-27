"""Run frozen existing pipeline on audited LAV-DF training pilot; no fitting."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'experiments/lavdf_pilot16_20261024/manifest.json'
    out=ROOT/'data/results/lavdf_pilot16_20261024'
    out.mkdir(parents=True,exist_ok=False)
    (out/'reports').mkdir();(out/'logs').mkdir()
    samples=json.loads(source.read_text())['samples'];results=[];labels=[]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    provenance={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'purpose':'training-split workflow pilot only; no model fitting or threshold selection',
                'legacy_threshold':.5,'scripts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')}}
    (out/'run.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    for i,s in enumerate(samples,1):
        key=s['id'];meta=s['metadata'];path=Path(s['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=s['sha256']:raise ValueError('Input changed')
        labels.append({'video_id':key,'video_path':str(path),'source_id':meta['original'] or meta['file'],
                       'person_id':'','emotion_label':'','is_fake':s['is_fake'],'is_fraud':'',
                       'label_source':'LAV-DF metadata.min.json'})
        report=out/'reports'/f'{key}.json';log=out/'logs'/f'{key}.log'
        result={'id':key,'report':str(report),'status':'pending','is_fake':s['is_fake']}
        start=time.monotonic()
        print(f'[{i}/{len(samples)}] {key}',flush=True)
        cmd=[sys.executable,str(ROOT/'scripts/run_pipeline.py'),'--input',str(path),'--offline',
             '--output',str(report),'--work-dir',str(out/'artifacts'/key)]
        try:
            with log.open('w',encoding='utf-8') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
            data=json.loads(report.read_text(encoding='utf-8'))
            result.update(status='success',risk_score=data['deepfake_risk_score'],prediction=data['prediction'],
                          interval_js_comparison=data['interval_js_comparison'])
        except Exception as exc:result.update(status='failed',error=str(exc))
        result['seconds']=round(time.monotonic()-start,2);results.append(result)
        (out/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
        print(result['status'],result['seconds'],flush=True)
    with (out/'labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(labels[0]));w.writeheader();w.writerows(labels)
    return int(any(r['status']!='success' for r in results))


if __name__=='__main__':raise SystemExit(main())
