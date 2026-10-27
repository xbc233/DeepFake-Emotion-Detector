"""Frozen local zero-shot semantic baseline, without fitting or label substitution."""
import os
for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[k]='1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from evaluate_risk import binary_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/simulated48_semantics_20261026'
SOURCE=ROOT/'experiments/simulated48_features_20261026/scores.json'
LABELS=['a scam or fraudulent request','a legitimate message or safety advice']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    rows=read(SOURCE);assert len(rows)==48 and len({r['id'] for r in rows})==48
    config=dict(model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',candidate_labels=LABELS,hypothesis_template='This text contains {}.',multi_label=False,threshold=0.5,input_sha256=sha(SOURCE),script_sha256=sha(__file__),training=False,selection='Single prespecified prompt; no prompt search or threshold tuning',score_interpretation='Relative zero-shot label score, not calibrated fraud probability')
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    protocol=OUT/'frozen_protocol.json'
    if protocol.exists():assert read(protocol)['fingerprint']==fingerprint
    else:save(protocol,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),fingerprint=fingerprint,**config))
    # Labels are not supplied to the classifier. Exact text duplicates share one inference.
    texts=sorted({r[k] for r in rows for k in ('reference_script','asr_text')})
    cachepath=OUT/'text_scores.json';cache=read(cachepath) if cachepath.exists() else {}
    pending=[t for t in texts if t not in cache]
    if pending:
        import torch
        from text_sentiment import load_classifier
        torch.set_num_threads(4);classifier=load_classifier(config['model'])
        # Preserve identity of actual local model parameters without another weight copy.
        modelhash=hashlib.sha256()
        for name,tensor in sorted(classifier.model.state_dict().items()):
            modelhash.update(name.encode());modelhash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        save(OUT/'model_identity.json',{'state_dict_sha256':modelhash.hexdigest(),'config':classifier.model.config.to_dict(),'tokenizer_class':type(classifier.tokenizer).__name__})
        for i,t in enumerate(pending,1):
            result=classifier(t,candidate_labels=LABELS,hypothesis_template=config['hypothesis_template'],multi_label=False)
            scores=dict(zip(result['labels'],map(float,result['scores'])))
            assert set(scores)==set(LABELS) and all(0<=v<=1 for v in scores.values())
            cache[t]=scores;save(cachepath,cache);print(i,len(pending),'text scored',flush=True)
    output=[]
    for r in rows:
        ps=cache[r['reference_script']][LABELS[0]];pa=cache[r['asr_text']][LABELS[0]]
        output.append(dict(id=r['id'],source_id=r['source_id'],reference_script=r['reference_script'],asr_text=r['asr_text'],script_score=ps,asr_score=pa,script_alert=ps>=.5,asr_alert=pa>=.5,human_semantic_label=r['human_semantic_label'],annotation_stage=r['human_annotation_stage'],sync_status=r['sync_status'],sync_alert=r['sync_alert'],mean_js_distance=r['mean_js_distance']))
    eligible=[r for r in output if r['human_semantic_label'] in ('normal','simulated_scam')]
    metrics={k:binary_metrics([(int(r['human_semantic_label']=='simulated_scam'),r[k+'_score']) for r in eligible],.5) for k in ('script','asr')}
    # Repeated scripts must not be presented as independent observations.
    grouped={}
    for r in eligible:grouped.setdefault(r['reference_script'],[]).append(r)
    unambiguous=[v for v in grouped.values() if len({r['human_semantic_label'] for r in v})==1]
    unique_metrics=binary_metrics([(int(v[0]['human_semantic_label']=='simulated_scam'),v[0]['script_score']) for v in unambiguous],.5)
    summary=dict(videos=48,unique_inputs=len(texts),human_eligible=len(eligible),unreviewed=sum(r['human_semantic_label'] is None for r in output),uncertain=sum(r['human_semantic_label']=='uncertain' for r in output),threshold=.5,metrics=metrics,unique_script_sensitivity=dict(eligible=len(unambiguous),conflicting_scripts_excluded=len(grouped)-len(unambiguous),metrics=unique_metrics),asr_alert_changes=sum(r['asr_alert']!=r['script_alert'] for r in output),max_asr_score_difference=max(abs(r['asr_score']-r['script_score']) for r in output),script_alerts=sum(r['script_alert'] for r in output),asr_alerts=sum(r['asr_alert'] for r in output),fitted=False,calibrated=False,fingerprint=fingerprint)
    save(OUT/'scores.json',output);save(OUT/'summary.json',summary)
    lines=['# 48段模拟视频诈骗语义零样本基线','', '本地既有mDeBERTa零样本分类，推理前冻结两候选标签、提示模板和0.5阈值。没有训练、提示搜索或阈值调优。不是专门训练过的诈骗检测器。', '',
      f"48段视频，{len(texts)}条不同输入文本；36条已知初审语义可评估，4条不确定与8条未审不参与指标。",'',
      '| 输入 | 初审样本数 | TP | FP | FN | TN | AUC |','|---|---:|---:|---:|---:|---:|---:|']
    for k,label in [('script','已知配音脚本'),('asr','自动转写')]:
        v=metrics[k];c=v['confusion'];lines.append(f"|{label}|{v['n']}|{c['TP']}|{c['FP']}|{c['FN']}|{c['TN']}|{v['roc_auc']:.4f}|")
    lines+=['',f"ASR引起告警变化：{summary['asr_alert_changes']}条；最大分数差：{summary['max_asr_score_difference']:.4f}。",
      f"去重脚本敏感性分析：{len(unambiguous)}条语义明确且同文判断不冲突的脚本，AUC={unique_metrics['roc_auc']}；这是敏感性分析，不是独立泛化评估。",'',
      '## 使用边界','模型分数只是两个候选解释的相对分数，不是诈骗发生概率。全部文本是短篇模拟脚本，初审存在分歧且复核未完成；同源/同文重复、样本量小，不报告普通视频独立采样置信区间或真实诈骗识别能力。',
      '人类不确定标签未转成阴性，新增8条未补假标签。文字情感与诈骗语义分开。同步告警不等于诈骗，情感差异也不作诈骗标签。此处把各通道分数并列保存，不虚构联合真值或训练融合器。',
      'scores.json保留每段脚本/ASR评分、初审语义和同步/情感分数，便于后续复核返回后另存对照；text_scores.json复用重复文本，避免重复推理。','',
      '## 下一步','先整理语义误报/漏报与初审分歧，再固定融合规则用于演示并清楚区分合成线索与诈骗语义。若要训练融合模型和证明泛化，需要更多独立来源、独立脚本及保留测试，不能用本批反复调参后宣称独立测试。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
