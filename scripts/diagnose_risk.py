"""Audit baseline label disagreements without changing historical artifacts."""
import argparse
import csv
import json
from pathlib import Path
from alignment import normalize_emotion
from evaluate_risk import write_csv


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def diagnose(report):
    artifacts = report['artifacts']
    aligned = read(artifacts['alignment'])
    speech = read(artifacts['speech_emotion'])
    text = read(artifacts['text_emotion'])
    face = read(artifacts['face_emotion'])
    pairs = [('face', 'speech'), ('face', 'text'), ('speech', 'text')]
    points = []
    for row in aligned:
        t = float(row['time'])
        r = {'time': t, **{k: normalize_emotion(row[k]) for k in ('face', 'speech', 'text')}}
        available = [(a,b) for a,b in pairs if 'unknown' not in (r[a],r[b])]
        for a,b in pairs:
            r[a+'_'+b+'_conflict'] = '' if (a,b) not in available else int(r[a] != r[b])
            r[a+'_'+b+'_risk_contribution'] = (int(r[a] != r[b])/len(available)) if (a,b) in available else 0
        r['no_pair_risk_contribution'] = int(not available)
        r['text_outside_segment'] = int(not any(float(x['start']) <= t < float(x['end']) for x in text))
        window = report['run_settings']['speech_window']
        active = [x for x in speech if float(x['time']) <= t < min(float(x['time'])+window,report['video']['duration_seconds'])]
        r['speech_nearest_differs_from_containing_window'] = int(bool(active) and normalize_emotion(active[0]['emotion']) != r['speech'])
        points.append(r)
    if not points:
        raise ValueError('Empty alignment')
    out = {'risk_score':report['deepfake_risk_score'], 'alignment_points':len(points),
           'face_unknown_fraction':sum(normalize_emotion(x['emotion'])=='unknown' for x in face)/len(face) if face else None,
           'text_outside_segment_points':sum(x['text_outside_segment'] for x in points),
           'speech_window_mismatch_points':sum(x['speech_nearest_differs_from_containing_window'] for x in points)}
    for a,b in pairs:
        key=a+'_'+b
        observed=[r[key+'_conflict'] for r in points if r[key+'_conflict'] != '']
        out[key+'_conflict_fraction']=sum(observed)/len(observed) if observed else None
        out[key+'_risk_contribution']=sum(r[key+'_risk_contribution'] for r in points)/len(points)
    out['no_pair_risk_contribution']=sum(r['no_pair_risk_contribution'] for r in points)/len(points)
    out['reconstructed_risk']=sum(out[k] for k in out if k.endswith('_risk_contribution'))
    if abs(out['reconstructed_risk']-out['risk_score']) > 2e-6:
        raise ValueError('Risk reconstruction mismatch')
    return out,points


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    results=json.loads(args.results.read_text(encoding='utf-8'))
    rows=[]; timeline=[]
    for sample in results:
        if sample['status']!='success':
            raise ValueError('This diagnostic requires successful baseline reports')
        report=json.loads(Path(sample['report']).read_text(encoding='utf-8'))
        summary,points=diagnose(report)
        rows.append({'video_id':sample['id'],'emotion_label':sample['emotion'],
                     'speech_target_supported':sample['speech_target_supported'],
                     **{k+'_prediction':v for k,v in sample['predictions'].items()}, **summary})
        timeline.extend({'video_id':sample['id'],**r} for r in points)
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'diagnosis.csv',rows,list(rows[0]))
    write_csv(args.output/'timeline.csv',timeline,list(timeline[0]))
    lines=['# 原始录制视频高风险诊断','',f'样本数：{len(rows)}。保留旧评分与旧标签，按原始对齐点等权重拆解风险。','',
           '| 项目 | 结果 |','|---|---:|']
    for key in ['risk_score','face_speech_risk_contribution','face_text_risk_contribution','speech_text_risk_contribution','no_pair_risk_contribution']:
        lines.append(f'| {key}（片段均值） | {sum(r[key] for r in rows)/len(rows):.4f} |')
    lines += ['',f"语音标签空间不支持目标情感的片段：{sum(not r['speech_target_supported'] for r in rows)}。",
              f"最近邻语音标签与实际所在语音窗口不同的对齐点：{sum(r['speech_window_mismatch_points'] for r in rows)}。",
              f"文本片段范围外仍沿用文本标签的对齐点：{sum(r['text_outside_segment_points'] for r in rows)}。",'',
              '## 解释与后续修正','',
              '- 三组风险贡献相加（再加无有效模态对的惩罚）可重建原始风险，已逐视频核对。贡献是算术拆解，不是伪造因果证据。',
              '- 语音记录时间是窗口起点，原最近邻对齐可能提前使用下一个窗口的情感；后续应按区间归属。',
              '- 文本最近邻传播超出实际语句区间；后续使用语句覆盖区间，无覆盖时标记缺失。',
              '- 语音模型不支持 fear/surprise；不能把模型标签空间差异当作伪造。',
              '- 静音、噪声和遮挡的实际影响尚未通过人工或 VAD 核实，不据此断言具体样本的成因。',
              '- diagnosis.csv 含逐视频主情感与各模态对冲突率；timeline.csv 可定位具体异常时刻。',
              '- 新增概率输出不会恢复旧 CSV 丢失的信息，需用新目录重新推理。当前没有修改基线风险或阈值。','']
    (args.output/'diagnosis.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
