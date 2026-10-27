"""Descriptive fixed-model scores; no fitting or fabrication of human labels."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
from train_lavdf_lightweight import predict,auc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/simulated48_features_20261026'
FEAT=ROOT/'data/results/simulated48_features_20261026'
SYNC=ROOT/'data/results/simulated48_sync_20261026'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def word_errors(reference,hypothesis):
    a=re.findall(r"[a-z0-9]+(?:'[a-z]+)?",reference.lower())
    b=re.findall(r"[a-z0-9]+(?:'[a-z]+)?",hypothesis.lower())
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,d[j]+1,d[j-1]+(x!=y)))
        d=nxt
    return d[-1],len(a)

def main():
    assert read(FEAT/'status.json')['status']=='complete'
    assert read(SYNC/'status.json')['status']=='complete'
    inputs=read(OUT/'inputs.json');assert len(inputs)==48
    sr={r['id']:r for r in read(SYNC/'results.json')}
    initial=ROOT/'data/simulated/pilot40_20261026/reviews/4ddee2d087d2/merged_manifest.json'
    labels={s['id']:s for s in read(initial)['samples']}
    modelpath=ROOT/'experiments/syncnet_test80_20261025/frozen_model.json'
    protocol=read(ROOT/'experiments/syncnet_test80_20261025/frozen_protocol.json')
    assert sha(modelpath)==protocol['model_sha256']
    scripts={s['id']:s['text'] for dataset in ('pilot40_20261026','neutral8_20261026') for s in read(ROOT/'data/simulated'/dataset/'private/generated_manifest.json')['samples']}
    model=read(modelpath);rows=[]
    for s in inputs:
        sid=s['id'];r=read(FEAT/'samples'/sid/'report.json');sync=sr[sid]
        assert sha(s['path'])==s['sha256']==r['source_sha256']==sync['source_sha256']
        assert all(sha(p)==h for p,h in r['artifact_hashes'].items())
        js=r['interval_js_comparison'];label=labels.get(sid,{})
        row=dict(id=sid,source_id=s['source_id'],mean_js_distance=js['mean_js_distance'],js_time_coverage=js['comparable_duration_fraction'],sync_status=sync['status'],sync_model_score=None,sync_alert=None,human_emotion_relation=label.get('human_emotion_relation'),human_semantic_label=label.get('human_semantic_label'),human_annotation_stage='single_initial_pending_followup' if label else 'not_reviewed')
        transcript=read(r['artifacts']['transcript'])['text']
        errors,words=word_errors(scripts[sid],transcript)
        row.update(asr_text=transcript,reference_script=scripts[sid],word_errors=errors,reference_words=words,word_error_rate=errors/words)
        if sync['status']=='success':
            t=sync['sync'];x=np.array([[t['min_distance'],t['zero_distance'],t['confidence'],abs(t['offset_frames'])]])
            score=float(predict(model,x)[0]);row.update(sync_model_score=score,sync_alert=score>=protocol['threshold'])
        rows.append(row)
    eligible=[r for r in rows if r['mean_js_distance'] is not None and r['human_emotion_relation'] in ('consistent','inconsistent')]
    groups={k:[r['mean_js_distance'] for r in eligible if r['human_emotion_relation']==k] for k in ('consistent','inconsistent')}
    summary={'videos':48,'feature_success':48,'source_groups':len({r['source_id'] for r in rows}),'sync_status':dict(Counter(r['sync_status'] for r in rows)),
       'sync_threshold':protocol['threshold'],'sync_alerts':sum(r['sync_alert'] is True for r in rows),'js_available':sum(r['mean_js_distance'] is not None for r in rows),
       'emotion_by_initial_review':{k:{'count':len(v),'mean_js':float(np.mean(v)) if v else None} for k,v in groups.items()},
       'descriptive_emotion_auc':auc(np.array([int(r['human_emotion_relation']=='inconsistent') for r in eligible]),np.array([r['mean_js_distance'] for r in eligible])) if all(groups.values()) else None,
       'asr_word_error_rate_against_script':sum(r['word_errors'] for r in rows)/sum(r['reference_words'] for r in rows),'asr_normalized_exact_match':sum(r['word_errors']==0 for r in rows),
       'fitting_performed':False,'threshold_changed':False,'human_labels_filled_by_model':False,'fraud_semantic_model_evaluated':False,
       'provenance':{'inputs_sha256':sha(OUT/'inputs.json'),'model_sha256':sha(modelpath),'initial_review_sha256':sha(initial),'feature_config_sha256':sha(FEAT/'config.json'),'sync_config_sha256':sha(SYNC/'config.json')}}
    (OUT/'scores.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 48段模拟视频固定特征与评分','', '状态：提取和评分完成；人工定向复核仍待补齐。', '',
     f"48段特征提取成功；同步状态：{summary['sync_status']}；情感JS有值：{summary['js_available']}/48。",
     f"沿用既有同步模型与阈值 {protocol['threshold']}，告警 {summary['sync_alerts']} 段；质量拒绝保留为空，不当作阴性。",'',
     f"ASR相对配音脚本的归一化词错误率：{summary['asr_word_error_rate_against_script']:.4f}；完全匹配：{summary['asr_normalized_exact_match']}/48。忽略大小写和标点；脚本是预期配音内容，不替代人工听辨。",'',
     '## 初审标签探索性对照',json.dumps(summary['emotion_by_initial_review'],ensure_ascii=False),
     f"JS对情感不一致的描述性AUC：{summary['descriptive_emotion_auc']}。仅使用初审关系明确且JS有效的样本，没有按结果调阈值或反转方向。",'',
     '## 解释范围',
     'JS是人脸与语音七类共同情感分布差异，文字情感已提取并按转写时间对齐，但不计入JS。文字情感不是诈骗语义分类，本轮没有训练或验证诈骗语义模型。',
     '同步分数来自LAV-DF开发的固定模型，未经本场景校准，不能解释为诈骗概率。所有48段都是合成配音，没有真实未合成对照，不能计算真假检测AUC、误报率或诈骗检测准确率。',
     '40段原有单人初审保留，其中不确定和争议等待复核；8段新增样本人工标签为空。未以模型输出替代标签。情感一致样本很少、同源相关，描述性AUC不是独立测试结论。',
     '素材来自已有训练来源，研究模拟且多数配音与口型不同步；不能证明情感贡献独立于口型。预设情感条件不作为真值，未做融合模型训练。',
     '所有输入媒体和特征产物哈希已核验；scores.json保存逐视频分数、覆盖和原始初审标签。', '',
     '## 下一步','可以单独建立并冻结诈骗语义基线，分别在已知脚本文本和ASR转写上评分以区分识别误差；仍不能把文本情感当诈骗语义。人工复核回来后另存对照，补样确认后再决定是否训练融合器。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
