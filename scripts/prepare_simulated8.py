"""New-source neutral TTS pilot; screening scores are not human truth."""
import csv
import hashlib
import html
import json
from pathlib import Path
from prepare_simulated40 import ROOT

OUT=ROOT/'data/simulated/neutral8_20261026'
PAIRS=[
 ('refund','Your refund returns through the original payment method.','Pay a release fee to my private account for your refund.'),
 ('account','Check your account in the official app. Keep your code private.','Send me your verification code now to prevent your account closing.'),
 ('employment','Your employer provides training equipment. No deposit is required.','Pay a training deposit to my private account before starting work.'),
 ('sale','Pay through the marketplace checkout. Keep the transaction on the platform.','Leave the marketplace and pay my private wallet to receive your order.')]
ZH=[('退款将通过原支付方式退回。','请向我的私人账户支付一笔解冻费，以领取退款。'),
 ('在官方应用中查看账户。不要泄露你的验证码。','现在把验证码发给我，以免你的账户被关闭。'),
 ('雇主提供培训设备，无需支付押金。','开始工作前，请向我的私人账户支付一笔培训押金。'),
 ('通过交易平台结算付款，将交易保留在平台内。','离开交易平台，向我的私人钱包付款，才能收到你的订单商品。')]

def main():
    old=json.loads((ROOT/'data/simulated/pilot40_20261026/private/manifest.json').read_text(encoding='utf-8'))
    excluded={s['source_id'] for s in old['sources']}
    m=json.loads((ROOT/'experiments/lavdf_expansion480_20261025/manifest.json').read_text())
    candidates=[]
    for s in m['samples']:
        if s['split']!='train' or s['is_fake'] or s['id'] in excluded or s['actual_duration_seconds']<6:continue
        with (ROOT/'data/results/lavdf_features400_20261025/samples'/s['id']/'face.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        fraction=sum(r['emotion']=='neutral' for r in rows)/len(rows)
        candidates.append((fraction,s))
    candidates.sort(key=lambda x:(-x[0],x[1]['id']))
    assert len(candidates)>=4
    for d in ['originals','audio_raw','audio','videos','private','reviews']:(OUT/d).mkdir(parents=True,exist_ok=True)
    sources=[];samples=[]
    for i,((fraction,s),(scenario,normal,scam)) in enumerate(zip(candidates[:4],PAIRS),1):
        gid=f'N{i:02d}';assert hashlib.sha256(Path(s['path']).read_bytes()).hexdigest()==s['sha256']
        sources.append(dict(group_id=gid,source_id=s['id'],source_path=s['path'],sha256=s['sha256'],duration=s['actual_duration_seconds'],screening_emotion='neutral',screening_fraction=fraction,screening_is_model_prediction_not_truth=True,original_copy=str(OUT/'originals'/f'{gid}.mp4')))
        for j,(semantic,text) in enumerate([('normal',normal),('simulated_scam',scam)]):
            sid=f'N{2*i-1+j:03d}'
            samples.append(dict(id=sid,group_id=gid,scenario=scenario,text=text,text_zh=ZH[i-1][j],language='en',intended_semantics=semantic,intended_emotion_relation='intended_match',prosody_preset='default_neutral_candidate',human_semantic_label=None,human_emotion_relation=None,annotation_status='pending_single_reviewer',tts_voice='Microsoft Zira Desktop',tts_pitch='0%',tts_rate='0%',ssml=f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><voice name="Microsoft Zira Desktop">{html.escape(text)}</voice></speak>',raw_audio=str(OUT/'audio_raw'/f'{sid}.wav'),audio=str(OUT/'audio'/f'{sid}.wav'),video=str(OUT/'videos'/f'{sid}.mp4'),audio_replaced=True,audio_synthesized=True,face_swapped=False,lip_animation_applied=False,lip_sync_ground_truth=None,real_scam_event=False,synthetic_scenario=True,split='development_pilot_not_independent_test'))
    result=dict(status='candidate_design',dataset='neutral8_20261026',sources=sources,samples=samples,license_scope=old['license_scope'],limitations=old['limitations'][:1]+['All 8 are synthesized speech remixes; neutral screening is not verified emotion.','New source files relative to pilot40, not verified distinct identities; already-used training sources.','Shorter scripts than pilot40 introduce a content/duration change; not a controlled performance comparison.'])
    path=OUT/'private/manifest.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print([(s['source_id'],s['screening_fraction']) for s in sources])

if __name__=='__main__':main()
