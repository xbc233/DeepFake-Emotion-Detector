"""Prepare controlled research remix candidates; intended labels are not truth."""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/simulated/pilot40_20261026'
PAIRS=[
('refund','Your refund follows the original payment route. Track its progress through your official order page.',
 'Your refund is blocked. Pay a release fee through my private payment link right now.'),
('account','Check your account status in the official app. Your verification code stays on your device.',
 'Your account will close today. Send me the verification code so I can protect it.'),
('prize','The prize rules are published on the event page. Winners never pay a collection fee.',
 'You won the grand prize. Send a collection fee to my private account before claiming.'),
('investment','This investment can lose money. Read the risk statement and compare the published fees first.',
 'This investment guarantees your money will double tomorrow. Transfer your savings to my private wallet.'),
('employment','The interview takes place through the company portal. Training equipment is supplied by the employer.',
 'You have been hired immediately. Pay a training deposit to my private account before starting.'),
('family','This is the fictional Cedar family notice. Confirm unusual requests using the contact you already saved.',
 'I am your relative using a new number. Send emergency money now without calling back.'),
('support','Use the support number printed on your statement. Staff can explain the normal recovery process.',
 'I am support staff. Install my remote access tool and open your bank account now.'),
('delivery','Check your parcel number on the carrier website. Delivery changes appear in your order history.',
 'Your parcel is held. Give me your card number and security code to release it.'),
('authority','An official notice can be verified independently. Contact the published office number for case information.',
 'I am the investigator on your case. Move your money to my safety account immediately.'),
('sale','Use the marketplace checkout for this purchase. Your payment and refund remain inside the platform.',
 'This discount expires now. Leave the marketplace and pay my private wallet for faster delivery.')]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in ('originals','audio_raw','audio','videos','private','reviews'): (OUT/folder).mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'experiments/lavdf_expansion480_20261025/manifest.json').read_text())
    candidates=[]
    for s in manifest['samples']:
        if s['split']!='train' or s['is_fake'] or s['actual_duration_seconds']<8.5:continue
        face=ROOT/'data/results/lavdf_features400_20261025/samples'/s['id']/'face.csv'
        with face.open(encoding='utf-8-sig') as f:c=Counter(r['emotion'] for r in csv.DictReader(f))
        dominant,count=c.most_common(1)[0]
        if dominant in ('neutral','happiness'):candidates.append((s,dominant,count/sum(c.values())))
    candidates=sorted(candidates,key=lambda x:(-x[0]['actual_duration_seconds'],x[0]['id']))[:10]
    assert len(candidates)==10
    rows=[];sources=[];number=0
    for i,((s,dominant,fraction),(scenario,normal,scam)) in enumerate(zip(candidates,PAIRS),1):
        gid=f'G{i:02d}';path=Path(s['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==s['sha256']
        sources.append({'group_id':gid,'source_id':s['id'],'source_path':str(path),'sha256':s['sha256'],
                        'duration':s['actual_duration_seconds'],'screening_emotion':dominant,'screening_fraction':fraction,
                        'screening_is_model_prediction_not_truth':True,'original_copy':str(OUT/'originals'/f'{gid}.mp4')})
        matching='bright' if dominant=='happiness' else 'calm'
        for semantic,text in [('normal',normal),('simulated_scam',scam)]:
            for relation in ('intended_match','intended_mismatch'):
                number+=1;sid=f'S{number:03d}'
                style=matching if relation=='intended_match' else ('calm' if matching=='bright' else 'bright')
                pitch,rate=('+12%','+10%') if style=='bright' else ('-5%','-5%')
                ssml=f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><voice name="Microsoft Zira Desktop"><prosody pitch="{pitch}" rate="{rate}">{html.escape(text)}</prosody></voice></speak>'
                rows.append({'id':sid,'group_id':gid,'scenario':scenario,'text':text,'language':'en',
                    'intended_semantics':semantic,'intended_emotion_relation':relation,'prosody_preset':style,
                    'human_semantic_label':None,'human_emotion_relation':None,'annotation_status':'pending_single_reviewer',
                    'tts_voice':'Microsoft Zira Desktop','tts_pitch':pitch,'tts_rate':rate,'ssml':ssml,
                    'raw_audio':str(OUT/'audio_raw'/f'{sid}.wav'),'audio':str(OUT/'audio'/f'{sid}.wav'),
                    'video':str(OUT/'videos'/f'{sid}.mp4'),'audio_replaced':True,'audio_synthesized':True,
                    'face_swapped':False,'lip_animation_applied':False,'lip_sync_ground_truth':None,
                    'real_scam_event':False,'synthetic_scenario':True,'split':'pilot_only_do_not_claim_generalization'})
    result={'status':'candidate_design','sources':sources,'samples':rows,
            'license_scope':'Local non-commercial research preparation; LAV-DF CC BY-NC and additional dataset terms recorded separately; third-party likeness/media permissions not independently cleared; no redistribution approval implied.',
            'limitations':['Prosody is intended calm/bright, not verified emotion. Matched/mismatched labels require human review.',
                           'All 40 are audio-replaced candidates, not an authentic-vs-fake benchmark.',
                           'All same-source variants must remain grouped. This pilot reuses training sources and is not independent evaluation.',
                           'Native English TTS, no voice cloning. No real contact/payment endpoints. All videos carry simulation banner.']}
    p=OUT/'private/manifest.json'
    if p.exists():raise FileExistsError('Keep previous simulation design; do not overwrite')
    p.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('prepared',len(rows),'candidates from',len(sources),'training sources')


if __name__=='__main__':main()
