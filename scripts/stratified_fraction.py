#!/usr/bin/env python3
"""Make a target-domain train subset WITHOUT touching validation/test records."""
import argparse, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from poam.utils import json_records
p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
p.add_argument('--fraction',required=True,type=float)
p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
if not (0<a.fraction<=1):p.error('fraction must be (0,1]')
rows=list(json_records(a.input))
subset=rows if a.fraction==1 else train_test_split(rows,train_size=a.fraction,
    stratify=[x['label'] for x in rows],random_state=a.seed)[0]
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with open(a.output,'w',encoding='utf-8') as f:
    for row in subset:f.write(json.dumps(row,ensure_ascii=False)+'\n')
print('Target-label fraction',a.fraction,'samples',len(subset))
