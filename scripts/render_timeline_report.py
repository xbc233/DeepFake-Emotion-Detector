"""Build a self-contained local interactive experiment report (no network)."""
import argparse
import json
from pathlib import Path
from interval_consistency import read


def build(results_path,predictions_path,quality_dir,output):
    samples=json.loads(results_path.read_text(encoding='utf-8'))
    predictions=read(predictions_path);quality=read(quality_dir/'windows.csv');data=[]
    for s in samples:
        if s['status']!='success':continue
        report=json.loads(Path(s['report']).read_text(encoding='utf-8'))
        detail=json.loads((quality_dir/'samples'/(s['id']+'.json')).read_text(encoding='utf-8'))
        data.append({'id':s['id'],'target':s.get('emotion'),'duration':report['video']['duration_seconds'],
                     'legacyRisk':report['deepfake_risk_score'],
                     'face':read(report['artifacts']['face_emotion']),
                     'fps':report['run_settings']['frame_fps'],
                     'speech':[r for r in predictions if r['video_id']==s['id'] and r['kind']=='window'],
                     'text':read(report['artifacts']['text_emotion']),
                     'quality':[r for r in quality if r['video_id']==s['id']],**detail})
    template=(Path(__file__).resolve().parents[1]/'web/timeline_template.html').read_text(encoding='utf-8')
    # JSON is embedded as data, never as executable user-controlled JavaScript.
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    output.write_text(template.replace('__REPORT_DATA__',payload),encoding='utf-8')
    return len(data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--quality',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    print('Rendered samples:',build(a.results,a.predictions,a.quality,a.output))


if __name__=='__main__':main()
