"""Offline blinded single-review form; intended condition is not exposed."""
import json
import random
from prepare_simulated40 import OUT


def main():
    m=json.loads((OUT/'private/generated_manifest.json').read_text(encoding='utf-8'))
    samples=[{'id':s['id'],'text':s['text'],'video':'videos/'+s['id']+'.mp4','audio':'audio/'+s['id']+'.wav'} for s in m['samples']]
    random.Random(20261026).shuffle(samples)
    template=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>模拟视频单人初审</title>
<style>body{font:16px system-ui;max-width:1000px;margin:28px auto;padding:0 20px;background:#f4f6f9;color:#182433}section{background:white;padding:20px;margin:16px 0;border-radius:12px}button,select,input,textarea{font:inherit;padding:8px;margin:5px}video{width:280px;max-height:340px}label{display:block;margin:12px 0}textarea{width:90%}small{color:#536171}.row{display:flex;gap:24px;flex-wrap:wrap}</style>
<h1>模拟视频单人初审</h1><p>全部为研究模拟，人物原声已替换；配音不代表画面人物的真实言论。本轮不评价真实诈骗事件。</p>
<p>先分别检查文字、音频、静音画面，再看组合。情感分歧可以是自然表达，请允许“不确定”。未展示预设情感关系，禁止将生成意图直接当作答案。</p>
<section>标注者编号 <input id="reviewer" value="reviewer_1"> <button id="prev">上一条</button><select id="pick"></select><button id="next">下一条</button><b id="count"></b></section>
<section><h2 id="sid"></h2><p id="text"></p><audio id="audio" controls></audio><div class="row"><div><h3>静音画面</h3><video id="silent" muted controls playsinline></video></div><div><h3>组合视频</h3><video id="combined" controls playsinline></video></div></div></section>
<section id="form"><h2>独立判断</h2>
<label>文本语义 <select data-key="semantics"><option value="">请选择</option><option value="normal">正常内容</option><option value="simulated_scam">含模拟诈骗风险语义</option><option value="uncertain">不确定</option></select></label>
<label>画面主要情感 <select data-key="visual_emotion" class="emotion"></select></label>
<label>声音主要情感 <select data-key="audio_emotion" class="emotion"></select></label>
<label>文字主要情感 <select data-key="text_emotion" class="emotion"></select></label>
<label>声音与表情关系 <select data-key="emotion_relation"><option value="">请选择</option><option value="consistent">大致一致</option><option value="inconsistent">明显不一致</option><option value="uncertain">不确定／不宜比较</option></select></label>
<label>口型与语音同步 <select data-key="lip_sync"><option value="">请选择</option><option value="aligned">大致同步</option><option value="misaligned">明显不同步</option><option value="uncertain">不确定</option></select></label>
<label>素材是否可用 <select data-key="quality"><option value="">请选择</option><option value="usable">画面及声音可判断</option><option value="unusable">质量不足</option><option value="uncertain">不确定</option></select></label>
<label>备注（可记录异常时间与理由）<textarea data-key="notes" rows="3"></textarea></label>
<button id="save">保存当前初审</button><button id="export">导出初审JSON</button><p id="message"></p><small>单人初审不是多人一致性真值；导出文件不会自动上传。请保留“不确定”，不要强行凑齐四种条件。</small></section>
<script>const samples=__DATA__;let index=0;let records={};const key='simulation40_review_v1';try{records=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const pick=document.getElementById('pick');samples.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.id;pick.appendChild(o)});
document.querySelectorAll('.emotion').forEach(e=>{[['','请选择'],['neutral','平静／中性'],['happy','开心／兴奋'],['sad','悲伤'],['angry','愤怒'],['fear','恐惧／紧张'],['surprise','惊讶'],['disgust','厌恶'],['mixed','混合'],['uncertain','无法判断']].forEach(([v,t])=>{let o=document.createElement('option');o.value=v;o.textContent=t;e.appendChild(o)})});
function show(){const s=samples[index];pick.value=index;document.getElementById('sid').textContent=s.id;document.getElementById('text').textContent=s.text;for(const id of ['silent','combined']){let v=document.getElementById(id);v.pause();v.src=s.video}document.getElementById('silent').muted=true;document.getElementById('audio').src=s.audio;document.querySelectorAll('[data-key]').forEach(e=>e.value=(records[s.id]||{})[e.dataset.key]||'');document.getElementById('count').textContent=Object.keys(records).length+'/40 已保存';document.getElementById('message').textContent='';}
document.getElementById('silent').addEventListener('volumechange',e=>{if(!e.target.muted)e.target.muted=true});
pick.onchange=()=>{index=Number(pick.value);show()};document.getElementById('prev').onclick=()=>{index=Math.max(0,index-1);show()};document.getElementById('next').onclick=()=>{index=Math.min(39,index+1);show()};
document.getElementById('save').onclick=()=>{let r={id:samples[index].id,reviewer:document.getElementById('reviewer').value.trim(),reviewed_at:new Date().toISOString()};if(!r.reviewer){alert('请填写标注者编号');return}for(const e of document.querySelectorAll('[data-key]')){if(e.dataset.key!=='notes'&&!e.value){alert('请填写所有判断，不确定时选择不确定');return}r[e.dataset.key]=e.value}records[r.id]=r;try{localStorage.setItem(key,JSON.stringify(records))}catch(e){}document.getElementById('count').textContent=Object.keys(records).length+'/40 已保存';document.getElementById('message').textContent='当前初审已保存；请导出JSON留存。'};
document.getElementById('export').onclick=()=>{const payload={dataset:'pilot40_20261026',stage:'single_reviewer_initial',records:Object.values(records)};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download='simulation40_initial_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};show();</script></html>'''
    (OUT/'review.html').write_text(template.replace('__DATA__',json.dumps(samples,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    print(OUT/'review.html')


if __name__=='__main__':main()
