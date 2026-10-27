"""Audit coverage losses from saved artifacts; do not relax scoring thresholds."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'experiments/lavdf_coverage_diagnosis_20261025';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/results/lavdf_pilot16_20261024/results.json'
    samples=json.loads(source.read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'experiments/lavdf_pilot16_20261024/manifest.json').read_text())['samples']
    groups=Counter(s['metadata']['original'] or s['metadata']['file'] for s in manifest)
    reasons=Counter();speech=Counter();losses=Counter();all_face=Counter();rows=[]
    for s in samples:
        report=json.loads(Path(s['report']).read_text(encoding='utf-8'))
        faces=list(csv.DictReader(Path(report['artifacts']['face_emotion']).open(encoding='utf-8-sig')))
        unknown=sum(not f.get('probabilities') for f in faces)
        all_face.update({'total':len(faces),'missing_probabilities':unknown})
        detail=json.loads((ROOT/'experiments/lavdf_emotion2vec_v2_20261024/quality/samples'/f'{s["id"]}.json').read_text())
        js=detail['comparisons']['seven_after']
        for w in js['windows']:
            reasons.update(w['face_rejection_counts']);speech[w['speech_status']]+=1
            face_ok=w['face_time_coverage']>=.5
            speech_ok=w['speech_status']=='comparable'
            reason='valid' if face_ok and speech_ok else 'both' if not face_ok and not speech_ok else 'face_only' if not face_ok else 'speech_only'
            losses[reason]+=1
        rows.append({'video_id':s['id'],'frames':len(faces),'missing_face_probability':unknown,
                     'js_seconds':js['comparable_duration_seconds'],'duration':js['duration_seconds'],
                     'js_coverage':js['comparable_duration_fraction']})
    result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'all_face_rows':dict(all_face),
            'face_reasons_inside_windows':dict(reasons),'speech_statuses':dict(speech),
            'window_losses_mutually_exclusive':dict(losses),'source_groups':dict(groups),
            'visual_audit':'One unavailable frame per video inspected; visible human faces in all 16 examples. Not an exhaustive frame annotation.',
            'note':'Missing probabilities can include detector or inference failures; review logs before equating all missing values to detector misses.'}
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'coverage.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# LAV-DF 低覆盖率诊断','','本次只读取已有输出，没有重新推理或修改阈值。',
           f'- 全部人脸输出：{dict(all_face)}。',f'- 比较窗口内人脸状态：{dict(reasons)}。',
           f'- 语音窗口状态：{dict(speech)}。',f'- 窗口失败互斥分类：{dict(losses)}。',
           f'- 16段仅来自 {len(groups)} 个原视频来源组，不能当作16个独立样本。',
           '- 每段抽查一张缺失人脸输出的画面，16张均可见人物脸部。主要瓶颈指向Haar漏检，不能只靠放宽共同类别或有效覆盖阈值解决。',
           '- 视觉抽查未覆盖所有帧；未标注完整人脸框真值，不能报告检测器召回率。',
           '- 画面为224×224近脸裁剪，包含侧脸、俯视、遮挡及低清晰度。不能默认整帧都适合作为表情模型输入。',
           '', '## 下一步',
           '1. 为当前16段增加更适合非正面脸的检测器候选；保存框、置信度和失败原因，保持原输出不变。',
           '2. 使用相同HSEmotion权重和评分阈值，只替换人脸检测环节，做同帧对照并抽查新增框，防止误检换来虚假覆盖。',
           '3. 人脸质量改善后，再扩大原视频来源组；训练、验证与测试按原始关联隔离，不在当前16段上调阈值。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
