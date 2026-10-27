"""Same available subset, fixed logistic baselines; rejection remains abstention."""
import json
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from train_lavdf_lightweight import ROOT,QUALITY,EMOTION,fit,predict,metrics,auc

BATCH=ROOT/'data/results/syncnet400_20261025'
OUT=ROOT/'experiments/syncnet400_comparison_20261025'
SYNC=['sync_min_distance','sync_zero_distance','sync_confidence','sync_abs_offset']
EXTRA=['sync_track_fraction','sync_support_fraction','sync_tracks']


def main():
    assert json.loads((BATCH/'status.json').read_text())['status']=='complete'
    results=json.loads((BATCH/'results.json').read_text());assert len(results)==400
    old=json.loads((ROOT/'experiments/lavdf_lightweight_20261025/features.json').read_text())
    original={r['id']:r for r in old};assert len(original)==400 and {r['id'] for r in results}==set(original)
    rows=[];coverage={}
    for split in ('train','dev'):
        selected=[r for r in results if r['split']==split]
        coverage[split]={'planned':len(selected),'status':dict(Counter(r['status'] for r in selected)),
                         'by_label':{str(y):dict(Counter(r['status'] for r in selected if original[r['id']]['y']==y)) for y in (0,1)}}
    for r in results:
        if r['status']!='success':continue
        prior=original[r['id']];assert prior['split']==r['split'] and prior['source_id']==r['source_id']
        f=dict(prior['features']);s=r['sync'];duration=r['total_frames']/25
        f.update(sync_min_distance=s['min_distance'],sync_zero_distance=s['zero_distance'],sync_confidence=s['confidence'],sync_abs_offset=abs(s['offset_frames']),
                 sync_track_fraction=r['retained_track_seconds']/duration,sync_support_fraction=s['support_anchor_seconds']/duration,sync_tracks=len(r['segments']))
        rows.append({**prior,'features':f})
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    assert not {r['source_id'] for r in train}&{r['source_id'] for r in dev}
    yt=np.array([r['y'] for r in train]);yv=np.array([r['y'] for r in dev]);assert len(set(yt))==len(set(yv))==2
    OUT.mkdir(exist_ok=True);summary={'coverage':coverage,'threshold':.5,'test_used':False,'models':{},'note':'Same successful subset for all models. Rejections are abstentions, not negative predictions. No hyperparameter search.'}
    scores={}
    for name,fields in [('quality',QUALITY+EXTRA),('emotion',EMOTION),('sync',SYNC),('sync_quality',SYNC+QUALITY+EXTRA),('fusion',SYNC+QUALITY+EXTRA+EMOTION)]:
        def array(items):return np.array([[r['features'][k] if r['features'][k] is not None else np.nan for k in fields] for r in items])
        xt,xv=array(train),array(dev);model=fit(xt,yt);model['features']=fields
        pt,pv=predict(model,xt),predict(model,xv);scores[name]=pv
        text=json.dumps(model,indent=2,allow_nan=False);(OUT/f'{name}_model.json').write_text(text)
        assert np.allclose(predict(json.loads(text),xv),pv,atol=1e-12,rtol=0)
        summary['models'][name]={'train':metrics(yt,pt),'dev':metrics(yv,pv)}
    groups=sorted({r['source_id'] for r in dev});indices=[np.array([i for i,r in enumerate(dev) if r['source_id']==g]) for g in groups]
    rng=np.random.default_rng(20261025);diffs={k:[] for k in ('fusion_minus_sync_quality','sync_quality_minus_quality')}
    for _ in range(2000):
        ix=np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
        if len(set(yv[ix]))<2:continue
        diffs['fusion_minus_sync_quality'].append(auc(yv[ix],scores['fusion'][ix])-auc(yv[ix],scores['sync_quality'][ix]))
        diffs['sync_quality_minus_quality'].append(auc(yv[ix],scores['sync_quality'][ix])-auc(yv[ix],scores['quality'][ix]))
    summary['source_group_bootstrap']={k:{'replicates':len(v),'ci95':list(map(float,np.quantile(v,[.025,.975])))} for k,v in diffs.items()}
    summary['provenance']={'batch_config_sha256':hashlib.sha256((BATCH/'config.json').read_bytes()).hexdigest(),'results_sha256':hashlib.sha256((BATCH/'results.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (OUT/'features.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (OUT/'dev_predictions.json').write_text(json.dumps([{'id':r['id'],'source_id':r['source_id'],'y':r['y'],**{k:float(v[i]) for k,v in scores.items()}} for i,r in enumerate(dev)],indent=2))
    lines=['# 400段同步扩展与固定基线对照','','训练集拟合，验证集比较；测试集未使用。所有模型采用相同有效子集，固定L2逻辑回归、训练集填补与标准化、固定阈值0.5。未搜索超参数。','','## 覆盖',json.dumps(coverage,ensure_ascii=False),'','## 验证结果','| 模型 | AUC | AP | 平衡准确率 | 误报率 |','|---|---:|---:|---:|---:|']
    for name,m in summary['models'].items():
        d=m['dev'];lines.append(f"| {name} | {d['roc_auc']:.4f} | {d['pr_auc_average_precision']:.4f} | {d['balanced_accuracy']:.4f} | {d['false_positive_rate']['value']:.4f} |")
    lines+=['','## 增益不确定性',json.dumps(summary['source_group_bootstrap'],ensure_ascii=False),'','## 限制','质量拒绝不属于分类正确或错误，不得隐去覆盖率。数据来源相关，bootstrap按来源组抽样；人物身份独立性未证实。验证集已经参与方法探索，此处不是最终测试结论。文本未参与融合，无诈骗真值，不证明诈骗检测效果。不得用不同可用样本的历史AUC直接宣称改进。模型分数未经校准。','', '各模型JSON保存参数及特征顺序，已重载验证预测一致性；metrics.json保存完整指标，dev_predictions.json保存逐视频分数。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'coverage':coverage,'dev_auc':{k:v['dev']['roc_auc'] for k,v in summary['models'].items()}}))


if __name__=='__main__':main()
