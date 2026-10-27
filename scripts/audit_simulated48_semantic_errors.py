"""Evidence-based error triage; preserves scores and all human labels."""
import json
import hashlib
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/simulated48_semantic_errors_20261026'
BASE=ROOT/'experiments/simulated48_semantics_20261026'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    OUT.mkdir(exist_ok=True)
    rows=read(BASE/'scores.json')
    audit=read(ROOT/'data/simulated/pilot40_20261026/reviews/4ddee2d087d2/annotation_audit.json')
    disagreement=set(audit['semantic_disagreements'])
    source=read(ROOT/'data/simulated/pilot40_20261026/private/generated_manifest.json')
    scenarios={s['id']:s['scenario'] for s in source['samples']}
    groups=defaultdict(list);errors=[]
    for r in rows:
        if r['human_semantic_label'] not in ('normal','simulated_scam'):continue
        truth=r['human_semantic_label']=='simulated_scam'
        if r['asr_alert']==truth:continue
        # Triage categories are not causal proof and never replace human truth.
        category=('pending_label_design_disagreement' if r['id'] in disagreement else
                  'asr_changed_correct_script_decision' if r['script_alert']==truth else
                  'error_already_present_on_clean_script')
        item={**r,'scenario':scenarios[r['id']],'outcome':'FP' if r['asr_alert'] else 'FN','triage':category}
        errors.append(item);groups[category].append(r['id'])
    assert len(errors)==19 and len({e['id'] for e in errors})==19
    same_text=defaultdict(list)
    for r in rows:
        if r['human_semantic_label'] in ('normal','simulated_scam'):same_text[r['reference_script']].append(r)
    conflicts=[{'text':text,'samples':[{'id':r['id'],'human_label':r['human_semantic_label']} for r in values]} for text,values in same_text.items() if len({r['human_semantic_label'] for r in values})>1]
    summary={'error_count':len(errors),'counts':dict(Counter(e['triage'] for e in errors)),'ids':dict(groups),'unique_error_scripts':len({e['reference_script'] for e in errors}),'same_script_label_conflicts':conflicts,'scores_sha256':hashlib.sha256((BASE/'scores.json').read_bytes()).hexdigest(),'labels_changed':False,'prompt_or_threshold_changed':False}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'error_cases.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    # A review inventory only: current scripts are explicitly barred from a new blind test.
    inventory=[{'id':r['id'],'source_group':r['source_id'],'script_sha256':hashlib.sha256(r['reference_script'].encode()).hexdigest(),'text':r['reference_script'],'human_semantic_label':r['human_semantic_label'],'usage':'development_only_already_inspected','new_test_eligible':False} for r in rows]
    (OUT/'development_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 诈骗语义基线错误诊断','', '复用已冻结的输出，未重新推理、改提示、调阈值或修改人工标签。', '',
     '| 排查类别 | 数量 | 编号 |','|---|---:|---|']
    names={'pending_label_design_disagreement':'初审与设计分歧待复核','asr_changed_correct_script_decision':'原文本判对，转写后判错','error_already_present_on_clean_script':'原文本上已存在的误判'}
    for k,ids in groups.items():lines.append(f"|{names[k]}|{len(ids)}|{', '.join(ids)}|")
    lines+=['', '以上是互斥的排查优先级，不是已证实的根本原因。设计标签不自动高于人工标签。', '',
     '## 能从现有结果确认什么',
     '- S027/S028把 support staff 转为 Support Steph，模型从告警变为不告警。只能说这次输入变化跨过了固定阈值，不能据此泛化所有转写错误的影响。',
     '- S013/S014的投资风险提醒被误报，而S015/S016的保证收益和私人钱包转账请求被漏报。模型对行为和上下文的区分在这些案例上失败，单纯提升转写精度不能解决。',
     '- 验证码索取、紧急亲属转账、银行卡安全码索取和安全账户转账等脚本存在漏报。这里仅根据案例描述失败现象，没有证明模型内部依赖了哪些词。',
     '- S039/S040文本相同，但初审语义判断不同。原视频语气或理解差异可能影响判断，仍需复核；不能强行以多数或制作意图覆盖。', '',
     '## 下一版数据与训练方案',
     '1. 标签定义：对文本中的索取敏感信息、诱导绕开正规渠道付款、冒充身份要求行动等语义进行判断；与反诈提醒、新闻转述、合法业务和上下文不足分开。只有疑似语义，不宣称法律事实或真实事件。',
     '2. 当前48段及其同文改写全部作为已查看的开发素材；development_inventory.json登记文本与源组，禁止再包装成全新测试。',
     '3. 新建独立文本集，优先补充合法金融/客服提醒、反诈教育、引用诈骗话术的正常内容、真正提出转账/验证码要求的模拟话术，以及需要上下文的难例。每类都覆盖不同措辞，避免只靠private、money等关键词。',
     '4. 启动规模建议：60个独立场景/模板族，每族普通和风险各一条，共120条。按模板族预先划分36/12/12组训练/验证/测试（72/24/24条）；这只是试点预算，不是充分样本量保证。脚本自动生成标签只标为设计标签，未复核前不能充当最终人工测试真值。',
     '5. 家族相同、同文、近重复、翻译及改写必须处于同一划分；同源视频也不能跨集。固定随机种子、保留原文来源/许可/语言/版本/标注状态。测试文本在训练规则确定后才由独立提供者准备或封存，开发者不逐条检查。',
     '6. 有可用训练标签后，先训练TF-IDF加逻辑回归轻量基线；仅用训练集拟合词表和参数，验证集选择阈值。与现有零样本基线对照，按模板族评估，不按重复视频数量夸大证据。',
     '7. 另做ASR稳健性：同一已知文本的干净版与实际转写版配对，比较告警变化；不能把人为改错后的文本视为独立新场景。', '',
     '## 本阶段完成范围','19条误判已归档，初审分歧和转写影响已单列，现有素材已登记为开发集。未取得新的独立文本集、未训练下一版模型、未补人工标签。缺少独立语义数据和可靠测试标签仍是训练/评测缺口；FakeAVCeleb补深伪证据，不能补这个语义缺口。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
