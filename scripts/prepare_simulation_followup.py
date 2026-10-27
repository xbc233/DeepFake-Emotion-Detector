"""Build a separate blinded follow-up package without changing initial labels."""
import json
import random
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/simulated/pilot40_20261026'
REVIEW = OUT / 'reviews/4ddee2d087d2'
ZH = {
 'S006':'请在官方应用中查看账户状态。验证码请保留在你自己的设备上。',
 'S009':'奖品领取规则已在活动页面公布。获奖者无需支付任何领奖费用。',
 'S010':'奖品领取规则已在活动页面公布。获奖者无需支付任何领奖费用。',
 'S013':'这项投资可能造成亏损。请先阅读风险说明，并比较公开列出的费用。',
 'S017':'面试通过公司门户网站进行。培训所需设备由雇主提供。',
 'S022':'这是一则虚构的 Cedar 家庭通知。遇到异常请求，请通过你已保存的联系方式核实。',
 'S025':'请拨打账单上印出的客服号码。工作人员可以说明正常的恢复流程。',
 'S026':'请拨打账单上印出的客服号码。工作人员可以说明正常的恢复流程。',
 'S030':'请在承运商网站上查询你的包裹单号。配送变更会显示在订单记录中。',
 'S033':'正式通知可以通过独立渠道核实。请拨打公开的办公电话查询案件信息。',
 'S034':'正式通知可以通过独立渠道核实。请拨打公开的办公电话查询案件信息。',
 'S037':'这次购买请使用交易平台的结算功能。付款和退款都在平台内完成。',
 'S039':'这个优惠马上到期。离开交易平台，向我的私人钱包付款，就能更快发货。',
}

def main():
    audit = json.loads((REVIEW/'annotation_audit.json').read_text(encoding='utf-8'))
    ids = set(audit['semantic_disagreements'])
    for values in audit['uncertain_ids'].values():
        ids.update(values)
    m = json.loads((REVIEW/'merged_manifest.json').read_text(encoding='utf-8'))
    samples = [{'id':s['id'], 'text':s['text']+'\n中文参考：'+ZH[s['id']],
                'video':'videos/'+s['id']+'.mp4', 'audio':'audio/'+s['id']+'.wav'}
               for s in m['samples'] if s['id'] in ids]
    assert len(samples)==13 and ids==set(ZH)
    random.Random(20261027).shuffle(samples)
    html=(OUT/'review.html').read_text(encoding='utf-8')
    html,n=re.subn(r'const samples=.*?;let index=0;',lambda _: 'const samples='+json.dumps(samples,ensure_ascii=False)+';let index=0;',html,count=1)
    assert n==1
    for before,after in [('模拟视频单人初审','模拟视频定向复核'),('simulation40_review_v1','simulation40_followup_4ddee2d087d2_v1'),
                         ('single_reviewer_initial','targeted_followup'),('simulation40_initial_review.json','simulation40_followup_review.json'),
                         ('/40 已保存','/13 已保存'),('Math.min(39,index+1)','Math.min(12,index+1)'),('当前初审','当前复核'),('导出初审','导出复核')]:
        html=html.replace(before,after)
    html=html.replace('<h2>独立判断</h2>','<h2>重新判断</h2><p>保持自己的标注者编号；不同人员请使用不同编号。本次未显示旧答案与制作预设。中文只辅助理解，声音情感仍需听英文。口型不同步不代表情感不一致；两侧都平静可以一致，情绪不明确可以不确定。文本语义请结合整句话，不能仅凭“验证码”“付款”等词判断。备注选填，不填写也可以保存和导出。</p>')
    html=html.replace('请填写所有判断，不确定时选择不确定','请填写所有判断；备注选填，允许不确定')
    html=html.replace("stage:'targeted_followup',records", "stage:'targeted_followup',parent_review_sha256:'4ddee2d087d2ce126af392c61bc7987f134091ef301d7f56312278f85f0623d0',records")
    (OUT/'review_followup.html').write_text(html,encoding='utf-8')
    with zipfile.ZipFile(OUT/'review_followup13.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(OUT/'review_followup.html','review_followup.html')
        for s in samples:
            for k in ['video','audio']:
                z.write(OUT/s[k],s[k])
        z.writestr('使用说明.txt','完整解压后用 Edge 或 Chrome 打开 review_followup.html。共13条，中文辅助理解，不提供旧答案。每条填写判断并保存，备注选填；完成导出 simulation40_followup_review.json 发回。可以保留不确定，不要求改变原判断。使用自己的固定编号。文件仅供课题内部研究审核。')
    (REVIEW/'followup_selection.json').write_text(json.dumps({'ids':sorted(ids),'count':len(ids),'selection':'uncertain semantics or emotion relation, or semantic disagreement with design','blinded_to_previous_labels':True,'representative_sample':False},indent=2),encoding='utf-8')
    print('Created review_followup.html and review_followup13.zip; 13 samples; initial labels unchanged.')

if __name__=='__main__':
    main()
