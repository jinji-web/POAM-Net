#!/usr/bin/env python3
"""Source-domain pretrain -> target-domain supervised fine-tuning.

Target fractions are fractions of target TRAIN ONLY, never its held-out test set.
For a scratch baseline, pass no --source-checkpoint.
"""
import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from poam.utils import json_records, read_config, write_json
from poam.engine import run_train, run_evaluate

p=argparse.ArgumentParser()
p.add_argument('--target-config',required=True)
p.add_argument('--source-checkpoint',help='Source-domain best.pt, with compatible variant/backbones')
p.add_argument('--target-fractions',type=float,nargs='+',default=[.01,.05,.10,.25,1.])
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
original=read_config(a.target_config)
records=list(json_records(original['data']['train']))
res=[]
for fraction in a.target_fractions:
    if fraction<=0 or fraction>1:raise ValueError('Fractions must be in (0,1]')
    rows=records if fraction==1 else train_test_split(records,train_size=fraction,
        random_state=a.seed,stratify=[x['label'] for x in records])[0]
    dest=Path(a.output)/f'target_{fraction:.2f}'
    dest.mkdir(parents=True,exist_ok=True)
    partial_path=dest/'target_train_subset.jsonl'
    with open(partial_path,'w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    cfg=read_config(a.target_config)
    cfg['data']['train']=str(partial_path)
    model_path=run_train(cfg,dest/'training',seed=a.seed,resume=a.source_checkpoint)
    metrics,_=run_evaluate(model_path,output_dir=dest/'test')
    res.append({'fraction':fraction,'target_train_posts':len(rows),
                'seed':a.seed,'source_pretrained':bool(a.source_checkpoint),
                'test_macro_f1':metrics['macro_f1'], 'test_accuracy':metrics['accuracy']})
    write_json(res,Path(a.output)/'transfer_summary.json')
print(res)
