"""按既有来源组制作中文银行意图试验集；翻译、质检、拆分留痕。"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from banking77_data import normalize, sha256, write_json

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / 'data/banking77-zh-v2'
SOURCE = LAB / 'data/banking77-v1'
MT_REVISION = '408d9bc410a388e1d9aef112a2daba955b945255'
MT_PATH = Path.home()/'.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-zh/snapshots'/MT_REVISION
QUOTAS = {'train':10,'development':2,'calibration':2,'test':5}


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def write_rows(path, values):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))


def select():
    DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'selection.jsonl').exists():return rows(DATA/'selection.jsonl')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    selected=[]
    for split,quota in QUOTAS.items():
        assert sha256(SOURCE/f'{split}.jsonl')==manifest['files'][f'{split}.jsonl']['sha256']
        by_label=defaultdict(list)
        for r in rows(SOURCE/f'{split}.jsonl'):by_label[r['label']].append(r)
        for label,values in sorted(by_label.items()):
            seen=set();count=0
            # 固定种子散列排序，不依据模型是否答对选择数据。
            for r in sorted(values,key=lambda r:hashlib.sha256(('zh-v2-20260922:'+r['id']).encode()).hexdigest()):
                if r['group_id'] in seen:continue
                selected.append({**r,'source_id':r['id'],'text_en':r['state']})
                seen.add(r['group_id']);count+=1
                if count==quota:break
            assert count==quota
    write_rows(DATA/'selection.jsonl',selected)
    write_json(DATA/'selection-protocol.json',{'source_manifest_sha256':sha256(SOURCE/'manifest.json'),'quotas_per_class':QUOTAS,'seed':'zh-v2-20260922','selection':'fixed hash order, unique source groups; no accuracy-based sampling','translation_model':'Helsinki-NLP/opus-mt-en-zh','translation_revision':MT_REVISION,'translation_license':'Apache-2.0','source_license':'CC BY 4.0','evaluation_only':'All MInDS-14 zh-CN records and BANKING77 official test stay evaluation-only; translations retain source split and group.'})
    return selected


def translate(limit=None):
    import torch
    from transformers import MarianMTModel,MarianTokenizer
    selected=select()
    cache=DATA/'translations.jsonl'
    done={r['source_id']:r for r in rows(cache)} if cache.exists() else {}
    pending=[r for r in selected if r['source_id'] not in done]
    if limit:pending=pending[:limit]
    if not pending:return
    torch.set_num_threads(4);torch.manual_seed(20260922)
    tokenizer=MarianTokenizer.from_pretrained(MT_PATH,local_files_only=True)
    model=MarianMTModel.from_pretrained(MT_PATH,local_files_only=True).to('mps').eval()
    started=time.perf_counter()
    with cache.open('a') as stream,torch.inference_mode():
        for off in range(0,len(pending),16):
            batch=pending[off:off+16]
            # 本模型没有语言代码 token；保留用户话语，不额外加入标签提示。
            inputs=tokenizer([r['text_en'] for r in batch],return_tensors='pt',padding=True,truncation=False)
            assert inputs.input_ids.shape[1]<512
            output=model.generate(**{k:v.to('mps') for k,v in inputs.items()},num_beams=4,max_new_tokens=160,do_sample=False)
            translations=tokenizer.batch_decode(output,skip_special_tokens=True)
            for r,t,ids in zip(batch,translations,output):
                if tokenizer.eos_token_id not in ids.tolist():raise ValueError('翻译达到长度上限，停止质检')
                entry={**r,'text_zh':t,'translation_model':'Helsinki-NLP/opus-mt-en-zh','translation_revision':MT_REVISION,'human_reviewed':False}
                stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
            stream.flush()
            print(json.dumps({'phase':'translation','completed':len(done)+min(off+16,len(pending)),'total':len(selected),'seconds':time.perf_counter()-started},ensure_ascii=False),flush=True)
    write_json(DATA/'translation-timing.json',{'last_batch_run_seconds':time.perf_counter()-started,'device':'mps','batch':16,'beams':4})


def clean(text):return re.sub(r'[^\w\u3400-\u9fff]','',text.casefold())


def finalize():
    selected=select();translated=rows(DATA/'translations.jsonl')
    assert len(translated)==len(selected) and len({r['source_id'] for r in translated})==len(selected)
    amendments=json.loads((DATA/'translation-review.json').read_text()) if (DATA/'translation-review.json').exists() else {}
    filtered=[];excluded=[]
    for r in translated:
        correction=amendments.get(r['source_id'])
        if correction:
            if correction.get('exclude'):
                excluded.append({'source_id':r['source_id'],'reason':correction['reason']});continue
            r={**r,'text_zh':correction.get('text_zh',r['text_zh']),'review':'Codex source/translation semantic review','human_reviewed':False}
        if not re.search('[\u3400-\u9fff]',r['text_zh']):
            excluded.append({'source_id':r['source_id'],'reason':'译文无汉字'});continue
        filtered.append(r)
    # 冻结外部评测优先，跨划分精确重合及字符三元组 Jaccard >= .85 的候选从低优先级侧移除。
    external=rows(LAB/'data/minds14-zh-v1/evaluation.jsonl')
    known=[{'key':r['id'],'split':'external','label':None,'text':clean(r['text_zh'])} for r in external]
    accepted=[];rank={'test':0,'calibration':1,'development':2,'train':3}
    for r in sorted(filtered,key=lambda r:(rank[r['split']],r['source_id'])):
        text=clean(r['text_zh']);grams={text[i:i+3] for i in range(len(text)-2)};match=None
        for previous in known:
            other=previous['text']
            if text==other:match=previous;break
            if min(len(text),len(other))<12 or min(len(text),len(other))/max(len(text),len(other))<.85:continue
            other_grams={other[i:i+3] for i in range(len(other)-2)}
            if len(grams&other_grams)/max(1,len(grams|other_grams))>=.85:match=previous;break
        if match:
            excluded.append({'source_id':r['source_id'],'reason':'Chinese exact/near duplicate','matches':match['key'],'match_split':match['split']});continue
        accepted.append(r);known.append({'key':r['source_id'],'split':r['split'],'label':r['label'],'text':text})
    output=defaultdict(list)
    for r in accepted:
        output[r['split']].append({**r,'id':r['source_id']+':zh','state':r['text_zh'],'language':'zh','data_kind':'machine_translation','local_role':r['split']})
    # 英文回放只取通过质检的训练来源；同一原文与中文版本永远同组。
    for split in ('train','development'):
        counts=Counter()
        quota=5 if split=='train' else 2
        for r in sorted(output[split],key=lambda r:r['source_id']):
            if counts[r['label']]>=quota:continue
            output[split].append({**r,'id':r['source_id']+':en','state':r['text_en'],'language':'en','data_kind':'original_english'})
            counts[r['label']]+=1
    seen=set();files={}
    for split in QUOTAS:
        values=output[split];groups={r['group_id'] for r in values}
        assert not(seen&groups);seen|=groups
        write_rows(DATA/f'{split}.jsonl',values)
        files[f'{split}.jsonl']={'records':len(values),'groups':len(groups),'languages':dict(Counter(r['language'] for r in values)),'classes':len({r['label'] for r in values}),'sha256':sha256(DATA/f'{split}.jsonl')}
    write_rows(DATA/'excluded.jsonl',excluded)
    (DATA/'labels.json').write_text((SOURCE/'labels.json').read_text())
    write_json(DATA/'manifest.json',{'dataset':'BANKING77 Chinese incremental pilot v2','source':'PolyAI BANKING77; 77 original labels','license':'CC BY 4.0','translation':json.loads((DATA/'selection-protocol.json').read_text()),'files':files,'exclusions':len(excluded),'reviewed_records':len(amendments),'cross_split_group_overlap':0,'split_policy':'Inherited BANKING77 source split and group; Chinese duplicates removed before training. Test never used for selection/calibration.','external_test':'All 502 MInDS14 zh-CN records remain external; 492 contain Han characters.','limitations':['Chinese text is machine translation plus recorded Codex corrections, not native customer logs.','No claim of human review or semantic near-duplicate exhaustiveness.','Small pilot; 8k cap retained but these are short single-turn customer requests.'],'code_sha256':sha256(Path(__file__))})
    print(json.dumps(files,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','translate','finalize']);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.stage=='translate':translate(a.limit)
    elif a.stage=='finalize':finalize()
    else:print(len(select()))
