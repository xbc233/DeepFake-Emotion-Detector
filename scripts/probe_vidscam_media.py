"""Bounded media pilot: two reviewed samples per class, outside source fine-tuning."""
import csv
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data/tools/yt_dlp'))
import yt_dlp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/vidscam_media_pilot_20261024')
    parser.add_argument('--label', choices=['0', '1', 'both'], default='both')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--count', type=int, default=2)
    args = parser.parse_args()
    if args.offset < 0 or not 1 <= args.count <= 10:
        parser.error('offset must be nonnegative and count must be 1..10')
    out = args.output
    if (out / 'acquisition.json').exists():
        parser.error('Choose a new output directory to preserve acquisition history')
    out.mkdir(parents=True, exist_ok=True)
    media = ROOT / 'data/raw/vidscam_media'
    media.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT / 'experiments/vidscam_intake_20261024/reviewed_labels.csv').open(encoding='utf-8-sig')))
    selected = []
    for label in (('0', '1') if args.label == 'both' else (args.label,)):
        selected.extend(sorted([r for r in rows if r['is_fraud'] == label and r['in_train'] == 'False'], key=lambda r: r['video_id'])[args.offset:args.offset+args.count])
    report = {'selection': {'order':'sorted video_id', 'label':args.label, 'offset':args.offset, 'count_per_class':args.count, 'exclude_source_finetuning':True, 'purpose':'workflow pilot only'},
              'tool_version': yt_dlp.version.__version__, 'samples': []}
    (out / 'candidates.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    for r in selected:
        record = {'video_id': r['video_id'], 'is_fraud': int(r['is_fraud']), 'is_fake': None,
                  'face_verified': None, 'speech_verified': None, 'status': 'pending'}
        options = {'quiet': True, 'noprogress': True, 'noplaylist': True, 'socket_timeout': 15, 'retries': 1,
                   'extractor_retries': 0, 'fragment_retries': 0,
                   'outtmpl': str(media / '%(id)s.%(ext)s'),
                   'js_runtimes': {'node': {}},
                   'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
                   'merge_output_format': 'mkv',
                   'max_filesize': 100 * 1024 * 1024}
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(r['youtube_url'], download=False)
                record.update(duration=info.get('duration'), language=info.get('language'),
                              channel_id=info.get('channel_id'), availability=info.get('availability'))
                if not info.get('duration') or info['duration'] > 900:
                    record['status'] = 'skipped_duration_limit'
                else:
                    ydl.download([r['youtube_url']])
                    path = media / (r['video_id'] + '.mkv')
                    if not path.is_file():
                        path = Path(ydl.prepare_filename(info))
                    if not path.is_file():
                        raise ValueError('No completed media file (possibly size limit)')
                    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], capture_output=True, text=True, check=True)
                    probe = json.loads(result.stdout)
                    record.update(status='media_acquired', path=str(path), probe=probe)
                    record['has_audio_stream'] = any(s['codec_type'] == 'audio' for s in probe['streams'])
                    record['has_video_stream'] = any(s['codec_type'] == 'video' for s in probe['streams'])
        except Exception as exc:
            record.update(status='acquisition_failed', error=str(exc))
        report['samples'].append(record)
        (out / 'acquisition.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(record['video_id'], record['status'], flush=True)


if __name__ == '__main__':
    main()
