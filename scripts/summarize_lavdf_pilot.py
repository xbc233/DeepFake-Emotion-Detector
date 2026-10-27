"""Summarize frozen-score pilot without claiming training or three-way JS fusion."""
import csv
import json
from collections import Counter
from pathlib import Path
from evaluate_risk import binary_metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    src=ROOT/'data/results/lavdf_pilot16_20261024'
    out=ROOT/'experiments/lavdf_pilot16_comparison_20261024'
    out.mkdir(parents=True,exist_ok=True)
    results=json.loads((src/'results.json').read_text(encoding='utf-8'))
    samples=json.loads((ROOT/'experiments/lavdf_pilot16_20261024/manifest.json').read_text())['samples']
    byid={r['id']:r for r in results}; rows=[]
    for s in samples:
        r=byid.get(s['id'],{'status':'missing'})
        m=s['metadata'];kind=('audio' if m['modify_audio'] else '')+('video' if m['modify_video'] else '') or 'real'
        row={'video_id':s['id'],'kind':kind,'is_fake':s['is_fake'],'status':r['status'],
             'legacy_risk':r.get('risk_score'),'face_audio_js':None,'js_coverage':0,
             'face_rows':0,'face_known_rows':0,'speech_rows':0,'text_rows':0,'asr_language':'','transcript':''}
        if r['status']=='success':
            report=json.loads(Path(r['report']).read_text(encoding='utf-8'))
            js=report['interval_js_comparison'];row.update(face_audio_js=js['mean_js_distance'],js_coverage=js['comparable_duration_fraction'])
            a=report['artifacts']
            for kindname,key in [('face','face_emotion'),('speech','speech_emotion'),('text','text_emotion')]:
                records=list(csv.DictReader(Path(a[key]).open(encoding='utf-8-sig')))
                row[kindname+'_rows']=len(records)
                if kindname=='face':row['face_known_rows']=sum(x.get('emotion','unknown').lower() not in ('unknown','') for x in records)
            transcript=json.loads(Path(a['transcript']).read_text(encoding='utf-8'))
            row['asr_language']=transcript.get('language','');row['transcript']=transcript.get('text','')
        rows.append(row)
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pairs=[(r['is_fake'],r['legacy_risk']) for r in rows if r['legacy_risk'] is not None]
    metrics={'planned':len(samples),'statuses':dict(Counter(r['status'] for r in rows)),
             'legacy_metrics':binary_metrics(pairs,.5),'js_videos':sum(r['face_audio_js'] is not None for r in rows),
             'js_mean_video_coverage':sum(r['js_coverage'] for r in rows)/len(rows),
             'limitations':['Training-split pilot, not independent evaluation.',
                            'Legacy score and threshold are untrained/unvalidated.',
                            'JS uses face and speech only; text overlaps are evidence, not fused score.',
                            'Baseline uses superb speech model; emotion2vec/VAD rerun not yet performed.',
                            'No fraud labels, no emotion ground truth, no fraud metrics.',
                            'Fixed first-ID sampling is not representative; correlated sources reduce independence.']}
    (out/'summary.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# LAV-DF 16段流程试验','','本次未训练模型、未调阈值。使用已有预训练模型和旧版评分流程。','',f'- 状态：{metrics["statuses"]}。',f'- 旧版阈值0.5混淆矩阵：{metrics["legacy_metrics"]["confusion"]}。',f'- 旧版 ROC-AUC：{metrics["legacy_metrics"]["roc_auc"]}。',f'- 人脸—语音 JS 可计算视频：{metrics["js_videos"]}/16；视频平均覆盖率：{metrics["js_mean_video_coverage"]:.1%}。','','## 逐视频结果','','| ID | 类型 | 原始真假 | 旧版风险 | JS覆盖率 |','|---|---|---:|---:|---:|']
    lines += [f'| {r["video_id"]} | {r["kind"]} | {r["is_fake"]} | {r["legacy_risk"]} | {r["js_coverage"]:.1%} |' for r in rows]
    lines+=['','## 解释限制','','这些是训练集小批流程结果，不是独立测试成绩。文本已提取，但未加入JS距离。未完成emotion2vec与VAD质量门控重跑，不能把当前结果称为新版方案性能。','没有诈骗或情感真值，不能报告诈骗准确率或情感识别准确率。缺失分数不作为真实类。','若误报较高，先检查语音情感类别、共同类别覆盖和时间对齐，再用独立开发集训练融合器；不根据这16段修改阈值。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=True))


if __name__=='__main__':main()
