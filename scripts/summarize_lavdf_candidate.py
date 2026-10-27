"""Compare uncalibrated JS ranking and coverage against independent fake labels."""
import csv
import json
from pathlib import Path
from evaluate_risk import binary_metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'experiments/lavdf_emotion2vec_v2_20261024'
    baseline=json.loads((ROOT/'data/results/lavdf_pilot16_20261024/results.json').read_text(encoding='utf-8'))
    quality=list(csv.DictReader((out/'quality/comparison.csv').open(encoding='utf-8-sig')))
    indexed={r['video_id']:r for r in quality};truth={r['id']:r['is_fake'] for r in baseline}
    rows=[]
    for r in baseline:
        q=indexed[r['id']]
        rows.append({'video_id':r['id'],'is_fake':truth[r['id']],'legacy_risk':r['risk_score'],
                     'baseline_four_js':r['interval_js_comparison']['mean_js_distance'],
                     'baseline_four_coverage':r['interval_js_comparison']['comparable_duration_fraction'],
                     **{k:(float(q[k]) if q[k] else None) for k in ('six_before_js','six_after_js','seven_before_js','seven_after_js')},
                     'seven_after_coverage':float(q['seven_after_valid_seconds'])/float(q['duration'])})
    rankings={}
    for field in ['legacy_risk','baseline_four_js','six_before_js','six_after_js','seven_before_js','seven_after_js']:
        pairs=[(r['is_fake'],r[field]) for r in rows if r[field] is not None]
        metrics=binary_metrics(pairs,.5)
        rankings[field]={k:metrics[k] for k in ('n','positive','negative','roc_auc','pr_auc_average_precision')}
    common=[r for r in rows if r['seven_after_js'] is not None]
    paired={}
    for field in ('legacy_risk','seven_after_js'):
        m=binary_metrics([(r['is_fake'],r[field]) for r in common],.5)
        paired[field]={k:m[k] for k in ('n','positive','negative','roc_auc','pr_auc_average_precision')}
    qm=json.loads((out/'quality/metrics.json').read_text(encoding='utf-8'))
    result={'ranking_only_no_threshold_calibration':rankings,'paired_on_seven_after_available':paired,
            'quality':qm,'limitations':['16 training samples only, correlated originals; no independent generalization claim.',
                                        'JS is face-audio disagreement, not calibrated fake probability.',
                                        'Text is aligned but not fused into JS.',
                                        'VAD only gates saved windows; it does not crop and reinfer speech.',
                                        'Coverage change combines speech model and shared-label-space changes.',
                                        'Different eligible subsets must not be treated as direct performance gains.']}
    (out/'fake_label_comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'fake_label_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# LAV-DF emotion2vec 与语音活动过滤试验','','复用原人脸、文本、音频，仅重跑语音情感和一致性计算。未训练分类器或调整阈值。','','| 分数 | 可评估视频 | 伪造/真实 | ROC-AUC | AP |','|---|---:|---|---:|---:|']
    for k,m in rankings.items():lines.append(f'| {k} | {m["n"]} | {m["positive"]}/{m["negative"]} | {m["roc_auc"]} | {m["pr_auc_average_precision"]} |')
    lines+=['','## 质量与覆盖',f'- 语音窗口 {qm["windows"]} 个，通过质量门控 {qm["eligible_windows"]} 个。']
    for k,m in qm['comparisons'].items():lines.append(f'- {k}：全视频时间覆盖 {m["whole_video_coverage"]:.1%}，有分数视频 {m["videos_with_js"]}/16。')
    lines+=['','## 解释边界','- 本批是训练集流程试验，不能证明跨数据集效果，也不能把覆盖率提高等同于准确率提高。','- JS 仅比较人脸与语音；文本只按时间对齐展示。没有情感真值，不报告情感准确率。','- 不对JS套用旧版0.5阈值，因此不输出新版误报率。排序指标只在有值样本上计算，同样本旧版/新版对照见JSON。','- 下一步检查共同类别过滤和人脸质量；在更大、独立划分的数据上训练小型融合器，并在验证集选择阈值。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'rankings':rankings,'quality':qm['comparisons'],'eligible_windows':qm['eligible_windows'],'windows':qm['windows']},ensure_ascii=True))


if __name__=='__main__':main()
