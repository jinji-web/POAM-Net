#!/usr/bin/env python3
"""Table 11 post/comment correspondence diagnostic.
Only SHUFFLES comments among records in supplied single split, never across splits.
Random mapping is label-blind; original post labels and images are unchanged.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from poam.utils import json_records, write_json

p=argparse.ArgumentParser()
p.add_argument('--input',required=True,help='ONE split (for example held-out test.jsonl)')
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
rows=list(json_records(a.input));n=len(rows)
if n<2:raise ValueError('Need >=2 posts')
rng=np.random.default_rng(a.seed)
# Rotate a uniformly shuffled post permutation to prevent mapping each record to itself.
order=rng.permutation(n)
source=np.empty(n,dtype=int)
for k,i in enumerate(order):source[i]=order[(k+1)%n]
assert all(source[i]!=i for i in range(n))
result=[]
for i,r in enumerate(rows):
    x=dict(r);x['comments']=rows[int(source[i])].get('comments',[])
    result.append(x)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with open(a.output,'w',encoding='utf-8') as f:
    for x in result:f.write(json.dumps(x,ensure_ascii=False)+'\n')
write_json({'seed':a.seed,'n_posts':n,'source_split':a.input,
            'meaning':'Held-out post/comment shuffle; labels not inspected'},a.output+'.shuffle.json')
print('Shuffled',n,'comment sets without touching labels')
