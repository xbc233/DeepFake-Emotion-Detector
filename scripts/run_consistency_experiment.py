"""One entry point: candidate speech -> VAD quality -> offline timeline report."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def validate_cached_pilot(pilot,results):
    m=json.loads((pilot/'metrics.json').read_text(encoding='utf-8'))
    expected=hashlib.sha256(results.read_bytes()).hexdigest()
    if m.get('provenance',{}).get('source_sha256')!=expected:
        raise ValueError('Cached pilot source mismatch or missing provenance')
    if m.get('failures') or m.get('success')!=m.get('planned'):
        raise ValueError('Cached pilot contains failures or incomplete results')
    if not (pilot/'predictions.csv').is_file():raise ValueError('Missing cached predictions')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True,help='Existing successful pipeline batch results.json')
    p.add_argument('--output',type=Path,required=True,help='New experiment directory')
    p.add_argument('--reuse-pilot',type=Path,help='Reuse provenance-verified candidate predictions')
    a=p.parse_args();results=a.results.resolve();output=a.output.resolve()
    if a.reuse_pilot:validate_cached_pilot(a.reuse_pilot.resolve(),results)
    output.mkdir(parents=True,exist_ok=False)
    run={'status':'running','source_results':str(results),'source_sha256':hashlib.sha256(results.read_bytes()).hexdigest(),'stages':[]}
    def save(): (output/'run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    def stage(name,script,*args):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)]
        print('Running:',name,flush=True)
        with (output/(name+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        run['stages'].append(name);save()
    save()
    try:
        pilot=a.reuse_pilot.resolve() if a.reuse_pilot else output/'candidate'
        if not a.reuse_pilot:stage('candidate','run_emotion2vec_pilot.py','--results',results,'--output',pilot)
        run['candidate_path']=str(pilot)
        stage('quality','run_speech_quality_audit.py','--results',results,'--predictions',pilot/'predictions.csv','--output',output/'quality')
        stage('timeline','render_timeline_report.py','--results',results,'--predictions',pilot/'predictions.csv','--quality',output/'quality','--output',output/'timeline.html')
        run['status']='complete';save();print(output/'timeline.html')
    except Exception as exc:
        run['status']='failed';run['error']=str(exc);save();raise


if __name__=='__main__':main()
