"""Audit native label compatibility and fixed-grid coverage sensitivity."""
import argparse
from collections import Counter
import json
from pathlib import Path
from alignment import normalize_emotion
from interval_consistency import SHARED, compare, distribution, read
from evaluate_risk import write_csv


def audit(results):
    rows=[]; per_frame=[]; cases=[]; schemas={k:set() for k in ('face','speech','text')}
    for sample in results:
        if sample['status']!='success':
            raise ValueError('Audit requires successful reports; resolve failures first')
        report=json.loads(Path(sample['report']).read_text(encoding='utf-8'))
        sources={k:read(report['artifacts'][k+'_emotion']) for k in schemas}
        cases.append((sample,report,sources))
        for modality,records in sources.items():
            known=[]; correct=0; mass_sum=0; mismatch=0; unsupported=Counter()
            for record in records:
                p=distribution(record)
                if p is None: continue
                schemas[modality].add(tuple(sorted(p)))
                top=max(p,key=p.get); label=normalize_emotion(record['emotion'])
                mismatch+=int(top!=label)
                correct+=int(top==sample['emotion'])
                mass=sum(p.get(k,0) for k in SHARED);mass_sum+=mass;known.append(p)
                if top not in SHARED: unsupported[top]+=1
                if modality=='face':
                    per_frame.append({'video_id':sample['id'],'target':sample['emotion'],'frame':record['frame'],
                                      'prediction':top,'top_probability':p[top],'shared_mass':mass,
                                      'unsupported_mass':1-mass,'target_probability':p.get(sample['emotion']),
                                      'dominant_supported':top in SHARED,'passes_mass_08':mass>=.8})
            n=len(known)
            rows.append({'video_id':sample['id'],'target':sample['emotion'],'modality':modality,
                         'rows':len(records),'probability_rows':n,'label_argmax_mismatches':mismatch,
                         'mean_shared_mass':mass_sum/n if n else None,
                         'observed_label_accuracy':correct/n if n and modality!='text' else None,
                         'target_supported':sample['emotion'] in set().union(*(set(p) for p in known)) if known else False,
                         'unsupported_top_counts':json.dumps(unsupported)})
    grid=[]
    # Descriptive sensitivity only: never select a detection threshold using these originals.
    for mass in (.6,.7,.8,.9):
        for coverage in (.25,.5,.75):
            duration=valid=0;scored=0
            for sample,report,sources in cases:
                result=compare(sources['face'],sources['speech'],sources['text'],report['video']['duration_seconds'],report['run_settings']['frame_fps'],mass,coverage)
                duration+=result['duration_seconds'];valid+=result['comparable_duration_seconds'];scored+=result['mean_js_distance'] is not None
            grid.append({'min_shared_mass':mass,'min_face_coverage':coverage,'videos_with_js':scored,
                         'duration_coverage':valid/duration if duration else None,'baseline_settings':mass==.8 and coverage==.5})
    summary={'videos':len(cases),'schemas':{k:[list(v) for v in sorted(values)] for k,values in schemas.items()},
             'label_argmax_mismatches':sum(r['label_argmax_mismatches'] for r in rows),
             'face_unsupported_dominant_counts':dict(Counter(r['prediction'] for r in per_frame if not r['dominant_supported'])),
             'face_supported_top_but_mass_below_08':sum(r['dominant_supported'] and not r['passes_mass_08'] for r in per_frame),
             'by_target':{}}
    for target in sorted({r['target'] for r in per_frame}):
        group=[r for r in per_frame if r['target']==target]
        summary['by_target'][target]={'frames':len(group),'mean_shared_mass':sum(r['shared_mass'] for r in group)/len(group),
                                     'top_label_counts':dict(Counter(r['prediction'] for r in group))}
    return rows,per_frame,grid,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows,frames,grid,m=audit(json.loads(args.results.read_text(encoding='utf-8')))
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in [('modality_audit',rows),('face_frame_audit',frames),('coverage_sensitivity',grid)]:
        write_csv(args.output/(name+'.csv'),data,list(data[0]))
    (args.output/'metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 情感类别兼容性审计','',f"审计视频 {m['videos']}；标签与概率最大值不一致记录：{m['label_argmax_mismatches']}。",'',
           f"人脸非共同类别主预测计数：{m['face_unsupported_dominant_counts']}。",'',
           f"主预测受支持但共同类别概率质量不足 0.8 的人脸帧：{m['face_supported_top_but_mass_below_08']}。",'',
           '## 固定网格敏感性分析','', '| 共同概率质量门槛 | 人脸覆盖门槛 | 有 JS 视频数 | 时长覆盖率 |','|---|---|---|---|']
    for r in grid:lines.append(f"| {r['min_shared_mass']} | {r['min_face_coverage']} | {r['videos_with_js']} | {r['duration_coverage']:.2%} |")
    lines+=['','## 判断与适配方案','',
            '- 已检查安装的 HSEmotion 类别索引及 softmax 实现，并核验导出标签与概率 argmax 一致性。没有发现导出类别顺序错配的证据；这不证明模型预测正确。',
            '- 人脸原生八类、语音四类、文本六类。语音没有 fear/surprise，不能通过补零或将 fear 强行并入 anger 修复。',
            '- 低覆盖同时来自人脸主预测落在不支持类别，以及受支持主预测仍有较多概率分配给其他类别。概率质量不是经过校准的可靠度。',
            '- 当前方案保留为四类条件比较基线。敏感性表只描述门槛影响，不选取最优值，不修改默认参数，不作为真假检测实验。',
            '- 优先路线：评估能输出相同基本情感集合的语音模型，并先验证六类情感识别质量；选型还需检查本地算力、语言与许可，再接真假数据测试。当前未下载或更换模型。',
            '- 备选路线：限定四类共同情感的研究范围，并将其余情况显式拒绝；这会缩小方法适用范围，不能宣称覆盖全部情感。',
            '- 暂不采用粗粒度正负情绪映射：surprise 的效价不确定，类别合并可能掩盖原本要检测的差异，需要另行标注与验证。',
            '- 情感准确率仅为观察帧/语音窗口对演员整段标签的诊断，不是独立样本测试准确率；文本不使用演员情感作为真值。',
            '- 下一步先进行候选语音模型的小规模对照选型，再重跑统一类别的一致性实验；不建议单纯降低门槛。','']
    (args.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(m,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
