"""Audit batch probability outputs and interval-JS coverage against a baseline."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
from evaluate_risk import indexed, write_csv
from interval_consistency import distribution, from_report, read


def summarize(results, baseline):
    old=indexed(baseline,'id'); indexed(results,'id')
    rows=[]; windows=[]; frame_reasons=Counter(); window_reasons=Counter(); probability_counts=Counter()
    total_duration=valid_duration=weighted_js=0.0
    for sample in results:
        key=sample['id']
        base=old.get(key,{})
        row={'video_id':key,'emotion':sample.get('emotion'),'actor':sample.get('actor'),
             'status':sample['status'],'baseline_risk':base.get('risk_score'),
             'current_legacy_risk':sample.get('risk_score'),'js_distance':None,
             'duration':None,'comparable_seconds':None,'coverage':None,'windows':None,'comparable_windows':None,
             'error':sample.get('error','')}
        if sample['status']=='success':
            report=json.loads(Path(sample['report']).read_text(encoding='utf-8'))
            js=from_report(report)
            if js != report['interval_js_comparison']:
                # Tuple labels become lists after JSON round trip.
                if json.loads(json.dumps(js)) != report['interval_js_comparison']:
                    raise ValueError(f'Recomputed JS differs: {key}')
            for modality in ('face','speech','text'):
                source=read(report['artifacts'][modality+'_emotion'])
                for item in source:
                    p=distribution(item)
                    probability_counts[modality+'_rows']+=1
                    probability_counts[modality+('_valid' if p is not None else '_missing')]+=1
            row.update(js_distance=js['mean_js_distance'],duration=js['duration_seconds'],
                       comparable_seconds=js['comparable_duration_seconds'],coverage=js['comparable_duration_fraction'],
                       windows=len(js['windows']),comparable_windows=sum(w['js_distance'] is not None for w in js['windows']))
            total_duration+=row['duration']; valid_duration+=row['comparable_seconds']
            if row['js_distance'] is not None: weighted_js+=row['js_distance']*row['comparable_seconds']
            for w in js['windows']:
                frame_reasons.update(w['face_rejection_counts'])
                reasons=[]
                if w['face_time_coverage']<js['min_face_coverage']: reasons.append('insufficient_face_coverage')
                if w['speech_status']!='comparable': reasons.append('speech_'+w['speech_status'])
                if w['js_distance'] is None and not reasons: reasons.append('no_comparable_face')
                window_reasons.update(reasons)
                windows.append({'video_id':key,'start':w['start'],'end':w['end'],'js_distance':w['js_distance'],
                                'status':w['status'],'face_time_coverage':w['face_time_coverage'],
                                'face_frames':w['face_frames'],'face_comparable_frames':w['face_comparable_frames'],
                                'reasons':';'.join(reasons),'frame_reasons':json.dumps(w['face_rejection_counts'])})
        rows.append(row)
    good=[r for r in rows if r['status']=='success']
    metrics={'planned_results':len(rows),'succeeded':len(good),'failed':len(rows)-len(good),
             'videos_with_js':sum(r['js_distance'] is not None for r in good),
             'videos_coverage_at_least_50pct':sum(r['coverage'] >= .5 for r in good),
             'videos_coverage_at_least_80pct':sum(r['coverage'] >= .8 for r in good),
             'videos_without_js':sum(r['js_distance'] is None for r in good),
             'total_windows':len(windows),'comparable_windows':sum(w['js_distance'] is not None for w in windows),
             'total_seconds':total_duration,'comparable_seconds':valid_duration,
             'duration_coverage':valid_duration/total_duration if total_duration else None,
             'duration_weighted_js':weighted_js/valid_duration if valid_duration else None,
             'face_frame_status_counts':dict(frame_reasons),'window_rejection_counts_nonexclusive':dict(window_reasons),
             'probability_validation':dict(probability_counts),
             'baseline_unmatched_ids':sorted(set(old)-{r['video_id'] for r in rows}),
             'settings':{'min_shared_mass':.8,'min_face_coverage':.5,'threshold_tuned':False}}
    metrics['by_emotion']={}
    for emotion in sorted({r['emotion'] for r in good}):
        group=[r for r in good if r['emotion']==emotion]
        seconds=sum(r['duration'] for r in group)
        metrics['by_emotion'][emotion]={'n':len(group),'videos_with_js':sum(r['js_distance'] is not None for r in group),
                                       'duration_coverage':sum(r['comparable_seconds'] for r in group)/seconds}
    return rows,windows,metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True); p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    rows,windows,m=summarize(json.loads(args.results.read_text(encoding='utf-8')),json.loads(args.baseline.read_text(encoding='utf-8')))
    m['inputs']={name:{'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in [('results',args.results),('baseline',args.baseline)]}
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'comparison.csv',rows,list(rows[0]))
    write_csv(args.output/'windows.csv',windows,list(windows[0]) if windows else ['video_id'])
    (args.output/'metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    coverage_text = '不可计算' if m['duration_coverage'] is None else f"{m['duration_coverage']:.2%}"
    lines=['# 24 段视频概率重跑与 JS 覆盖率审计','',
           f"成功 {m['succeeded']}；失败 {m['failed']}；有 JS 输出的视频 {m['videos_with_js']}；全部拒绝评分 {m['videos_without_js']}。",'',
           f"可比较窗口 {m['comparable_windows']}/{m['total_windows']}；有效时长 {m['comparable_seconds']:.3f}/{m['total_seconds']:.3f} 秒；时长覆盖率 {coverage_text}。",'',
           f"覆盖率至少 50% 的视频：{m['videos_coverage_at_least_50pct']}；至少 80%：{m['videos_coverage_at_least_80pct']}。",'', '## 分情感覆盖','', '| 情感 | 视频数 | 有 JS 的视频 | 时长覆盖率 |','|---|---:|---:|---:|']
    for k,v in m['by_emotion'].items(): lines.append(f"| {k} | {v['n']} | {v['videos_with_js']} | {v['duration_coverage']:.2%} |")
    lines+=['','## 拒绝原因','',f"人脸帧状态计数：{m['face_frame_status_counts']}",
            '',f"窗口拒绝原因（非互斥）：{m['window_rejection_counts_nonexclusive']}",'',
            '## 验证与结论边界','',
            '- 每个成功报告均从原生 CSV 重算 JS 并与嵌入结果核对；三模态完整概率检查有限性、范围与归一化。',
            '- 覆盖率分母为成功处理视频的全部时长；失败样本单列。窗口覆盖不等于所有帧均可靠。',
            '- 原风险与 JS 是不同定义、不同有效样本范围，不将数值下降解释为检测性能提升。',
            '- 空 JS 是证据不足，不是低风险。原标签没有真假或诈骗真值，本轮不计算检测准确率。',
            '- 共同类别概率质量 0.8、人脸覆盖 0.5 均保持固定，未为提高覆盖率调整门槛。',
            '- 在验证真假区分之前，应先检查低覆盖情感和拒绝原因；不能只保留可评分视频汇报效果。','']
    (args.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(m,ensure_ascii=True,indent=2))


if __name__=='__main__': main()
