from pathlib import Path
import requests,json,hashlib
out=Path('models/emotion2vec_plus_base');out.mkdir(parents=True,exist_ok=True)
repo='emotion2vec/emotion2vec_plus_base'
s=requests.Session()
r=s.get('https://huggingface.co/api/models/'+repo,timeout=45);r.raise_for_status();revision=r.json()['sha']
metadata={'repo':repo,'revision':revision,'files':{}}
for name in ['README.md','config.yaml','configuration.json','tokens.txt','model.pt']:
 target=out/name
 if target.exists():
  print('exists',name,flush=True)
 else:
  print('Downloading',name,flush=True)
  r=s.get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',stream=True,timeout=(30,90));r.raise_for_status()
  with target.with_suffix(target.suffix+'.part').open('wb') as f:
   for chunk in r.iter_content(1024*1024):f.write(chunk)
  target.with_suffix(target.suffix+'.part').replace(target)
 h=hashlib.sha256()
 with target.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 metadata['files'][name]={'bytes':target.stat().st_size,'sha256':h.hexdigest()}
(out/'download.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
r=s.get('https://raw.githubusercontent.com/modelscope/FunASR/main/MODEL_LICENSE',timeout=45);r.raise_for_status();(out/'MODEL_LICENSE').write_bytes(r.content)
print('Complete',revision,flush=True)
