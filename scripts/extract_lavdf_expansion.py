"""Extract only frozen selected members and verify streams and full decoding."""
import hashlib
import json
import subprocess
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/lavdf_expansion480_20261025'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify(sample):
    result = dict(sample)
    try:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', sample['path']], capture_output=True, text=True, timeout=60, check=True)
        data = json.loads(probe.stdout)
        types = {s['codec_type'] for s in data['streams']}
        if not {'audio', 'video'} <= types:
            raise ValueError('Missing required audio/video stream')
        duration = float(data['format']['duration'])
        if duration <= 0:
            raise ValueError('Invalid media duration')
        decode = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', sample['path'], '-threads', '1', '-f', 'null', '-'], capture_output=True, text=True, timeout=120)
        result.update(probe=data, decode_exit_code=decode.returncode, decode_errors=decode.stderr,
                      actual_duration_seconds=duration,
                      metadata_duration_delta=duration-float(sample['metadata']['duration']))
        if decode.returncode or decode.stderr.strip():
            raise ValueError('Decode failed or emitted errors')
        # Media duration is used downstream; preserve metadata without silent repair.
        result['status'] = 'verified'
    except Exception as exc:
        result.update(status='failed', error=str(exc))
    return result


def main():
    selection_path = OUT / 'selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    archive_path = ROOT / 'data/raw/lavdf.tar'
    destination = (ROOT / 'data/raw/lavdf').resolve()
    wanted = {'LAV-DF/' + s['metadata']['file']: s for s in selection['samples']}
    encountered = set()
    extracted = []
    with tarfile.open(archive_path, 'r:') as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            if member.name in encountered or not member.isfile():
                raise ValueError('Duplicate or nonregular selected archive member')
            encountered.add(member.name)
            sample = dict(wanted[member.name])
            path = Path(sample['path']).resolve()
            if not path.is_relative_to(destination):
                raise ValueError('Unsafe destination')
            payload = archive.extractfile(member).read()
            if len(payload) != member.size:
                raise ValueError('Truncated archive member')
            digest = hashlib.sha256(payload).hexdigest()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError('Existing media differs from archive')
            else:
                temporary = path.with_suffix('.mp4.tmp')
                temporary.write_bytes(payload)
                temporary.replace(path)
            sample.update(sha256=digest, bytes=len(payload))
            extracted.append(sample)
            if len(extracted) % 80 == 0:
                print(f'Extracted {len(extracted)}/{len(wanted)}', flush=True)
    missing = sorted(set(wanted) - encountered)
    if missing:
        save(OUT / 'missing_members.json', missing)
        raise ValueError('Selected media missing from archive')
    verified = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(verify, s) for s in extracted]
        for task in as_completed(tasks):
            verified.append(task.result())
            if len(verified) % 40 == 0:
                save(OUT / 'verification_progress.json', {'completed':len(verified), 'planned':len(extracted), 'failures':[s['id'] for s in verified if s['status'] != 'verified']})
                print(f'Verified {len(verified)}/{len(extracted)}', flush=True)
    verified.sort(key=lambda s: s['id'])
    groups, hashes = defaultdict(set), defaultdict(set)
    for sample in verified:
        groups[sample['source_id']].add(sample['split'])
        hashes[sample['sha256']].add(sample['split'])
    group_leaks = [k for k,v in groups.items() if len(v)>1]
    content_leaks = [k for k,v in hashes.items() if len(v)>1]
    failures = [s['id'] for s in verified if s['status'] != 'verified']
    manifest = {'status':'verified' if not (failures or group_leaks or content_leaks) else 'failed',
                'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                'archive_bytes':archive_path.stat().st_size, 'archive_hash_verified':False,
                'files':len(verified), 'groups':len(groups), 'bytes':sum(s['bytes'] for s in verified),
                'split_counts':dict(Counter(s['split'] for s in verified)),
                'group_leaks':group_leaks, 'exact_content_cross_split_leaks':content_leaks,
                'failures':failures, 'samples':verified,
                'note':'Source grouping and exact hashes do not establish person identity separation or rule out near duplicates. Test media decoding only; no model inference.'}
    save(OUT / 'manifest.json', manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='samples'}, ensure_ascii=True), flush=True)
    return int(manifest['status'] != 'verified')


if __name__ == '__main__':
    raise SystemExit(main())
