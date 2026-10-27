"""Audit downloaded VidScam metadata without treating model responses as truth."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://github.com/ummay-kulsum18/VidScamNet/tree/main/DATA'


def main():
    raw = ROOT / 'data/raw/vidscam_metadata'
    out = ROOT / 'experiments/vidscam_intake_20261024'
    out.mkdir(parents=True, exist_ok=True)
    annotated = list(csv.DictReader((raw / 'Annotated_Data.csv').open(encoding='utf-8-sig')))
    train = json.loads((raw / 'finetune_data_VidScam.json').read_text(encoding='utf-8'))
    evaluation = json.loads((raw / 'eval_data_VidScam.json').read_text(encoding='utf-8'))
    metadata = json.loads((raw / 'VidScam_video_metadata.json').read_text(encoding='utf-8'))
    train_ids = [str(x['video']) for x in train]
    eval_ids = [str(x['video_id']) for x in evaluation]
    overlap = sorted(set(train_ids) & set(eval_ids))
    mapping = {'scam': '1', 'non scam': '0', 'potential scam': ''}
    rows = []
    conflicts = []
    for r in annotated:
        label = r['Label'].strip().lower()
        if label not in mapping:
            raise ValueError(f'Unknown label: {label}')
        key = r['video_id'].strip()
        corrected = mapping[label]
        if corrected and corrected != {'TRUE': '1', 'FALSE': '0'}.get(r['scam']):
            conflicts.append(key)
        rows.append({'video_id': key, 'video_path': '', 'source_id': key,
                     'person_id': '', 'emotion_label': '', 'is_fake': '',
                     'is_fraud': corrected, 'label_source': SOURCE + '/Annotated_Data.csv:Label',
                     'review_label': label, 'original_scam': r['scam'],
                     'youtube_url': f'https://www.youtube.com/watch?v={key}',
                     'in_train': key in set(train_ids), 'in_eval': key in set(eval_ids),
                     'media_status': 'not_acquired', 'face_verified': '', 'audio_verified': ''})
    if len({r['video_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate reviewed video IDs')
    with (out / 'reviewed_labels.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        'source': SOURCE, 'download_date': '2026-10-24',
        'dataset_shortlink_destination': SOURCE,
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(raw.iterdir()) if p.is_file()},
        'reviewed_rows': len(rows), 'reviewed_labels': dict(Counter(r['review_label'] for r in rows)),
        'original_vs_reviewed_conflicts': conflicts,
        'train_rows': len(train), 'train_unique': len(set(train_ids)),
        'eval_rows': len(evaluation), 'eval_unique': len(set(eval_ids)),
        'eval_label_counts': dict(Counter(str(r['label']) for r in evaluation)),
        'train_eval_overlap': overlap, 'metadata_entries': len(metadata),
        'reviewed_in_train': sum(r['in_train'] for r in rows),
        'reviewed_in_eval': sum(r['in_eval'] for r in rows),
        'downloaded_video_count': 0, 'independent_deepfake_labels': 0,
        'status': 'metadata_audited_media_pending',
        'limitations': ['Potential scam stays unknown in binary evaluation.',
                       'Reviewed Label takes precedence over original scam column.',
                       'Fraud labels describe dataset annotations, not adjudicated crimes.',
                       'Source metadata contains sampling times, not raw audio or video.',
                       'No observed media means face/audio availability is unverified.',
                       'Do not include annotation reasoning or answer text in classifier inputs.']}
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# VidScam 接入核验', '',
             f'- 复核标注：{len(rows)} 条；分布：{report["reviewed_labels"]}。',
             f'- 原 scam 列与确定复核标签冲突：{len(conflicts)} 条，采用复核 Label 列。',
             f'- 训练：{len(train)} 条/{len(set(train_ids))} 个唯一视频；评估：{len(evaluation)} 条/{len(set(eval_ids))} 个唯一视频。',
             f'- 训练与评估 ID 交集：{len(overlap)} 个；完整 ID 见 audit.json。',
             f'- 视频采样元数据：{len(metadata)} 条。',
             '- 已下载视频：0。官方数据短链接跳转至 GitHub DATA 标注目录，不是 MP4 下载包。',
             '- reviewed_labels.csv 可接现有评估器，但尚无实际视频路径和预测结果。',
             '- potential scam 不强制转成诈骗；is_fake 全部未知。',
             '- 不使用微调回答、诈骗理由、标签字段作为输入特征，避免答案泄漏。',
             '- 下一步需获取作者视频包或逐条验证原视频来源，检查可用人脸、音轨和语言。',
             '- 在取得独立深伪真值之前，不进行双标签融合准确率评估。']
    (out / 'audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'limitations', 'train_eval_overlap', 'original_vs_reviewed_conflicts')}, ensure_ascii=True))
    print('overlap_count', len(overlap), 'label_conflicts', len(conflicts))


if __name__ == '__main__':
    main()
