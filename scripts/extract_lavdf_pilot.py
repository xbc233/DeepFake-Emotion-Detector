"""Extract 16 deterministic training samples without unpacking the full archive."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / 'data/raw/lavdf'
    out = ROOT / 'experiments/lavdf_pilot16_20261024'
    out.mkdir(parents=True,exist_ok=True)
    records = json.loads((root/'metadata.min.json').read_text())
    selected=[]
    for audio,video in [(False,False),(False,True),(True,False),(True,True)]:
        group=sorted([r for r in records if r['split']=='train' and r['modify_audio']==audio and r['modify_video']==video],key=lambda r:r['file'])
        selected.extend(group[:4])
    wanted={'LAV-DF/'+r['file']:r for r in selected}
    samples=[]; encountered=set(); member_count=0
    with tarfile.open(ROOT/'data/raw/lavdf.tar') as archive:
        for m in archive:
            member_count+=1
            if m.name not in wanted:continue
            if m.name in encountered or not m.isfile():raise ValueError('Duplicate/nonregular requested member')
            encountered.add(m.name)
            r=wanted[m.name];path=(root/r['file']).resolve()
            if not path.is_relative_to(root.resolve()):raise ValueError('Unsafe target')
            b=archive.extractfile(m).read()
            if len(b)!=m.size:raise ValueError('Truncated member')
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and path.read_bytes()!=b:raise ValueError('Existing media differs')
            path.write_bytes(b)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
            decode=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],capture_output=True,text=True)
            samples.append({'id':r['file'].replace('/','__').removesuffix('.mp4'),'path':str(path),
                            'metadata':r,'sha256':hashlib.sha256(b).hexdigest(),
                            'probe':probe,'decode_exit_code':decode.returncode,'decode_errors':decode.stderr,
                            'is_fake':int(r['modify_audio'] or r['modify_video']),'is_fraud':None})
    report={'selection':'First 4 sorted training files per audio/video modification combination; pipeline pilot only',
            'archive_bytes':(ROOT/'data/raw/lavdf.tar').stat().st_size,
            'archive_members_scanned':member_count,'missing':sorted(set(wanted)-encountered),'samples':samples}
    (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'metadata.min.json').write_text(json.dumps(selected),encoding='utf-8')
    print('extracted',len(samples),'missing',len(report['missing']),'decode_failures',sum(s['decode_exit_code']!=0 for s in samples),flush=True)


if __name__=='__main__':main()
