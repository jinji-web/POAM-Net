#!/usr/bin/env python3
"""Deterministically replace comment slots with attack corpus without using labels.

Attack corpus is a researcher-supplied JSONL/CSV collection of independently
sourced comments. This code does NOT fabricate evidence, provenance, or scores.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from poam.utils import json_records, write_json

p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
p.add_argument('--attack-corpus',required=True,help='One text per line; NOT model-generated from test labels')
p.add_argument('--fraction',type=float,default=.2)
p.add_argument('--kind',choices=['irrelevant','on_topic','repeated_consensus','coordinated'],required=True)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--max-comments',type=int,default=20)
a=p.parse_args()
if not (0 <= a.fraction <= 1):p.error('fraction must be within [0,1]')
corpus=[s.strip() for s in open(a.attack_corpus,encoding='utf-8') if s.strip()]
if not corpus:raise ValueError('Nonempty externally curated attack corpus required')
rows=list(json_records(a.input));results=[]
for item in rows:
    out=dict(item)
    comments=[({'text':c} if isinstance(c,str) else dict(c)) for c in item.get('comments',[])][:a.max_comments]
    n=min(a.max_comments, max(1, round(a.max_comments*a.fraction))) if a.fraction>0 else 0
    local_seed=int(hashlib.sha256(f'{a.seed}:{item["id"]}'.encode()).hexdigest()[:12],16)
    rng=random.Random(local_seed)
    if a.kind in ('repeated_consensus','coordinated'):
        selected=[rng.choice(corpus)]*n
    else:
        selected=[rng.choice(corpus) for _ in range(n)]
    for j,comment in enumerate(selected):
        # Replacements of existing positions; insertions only when fewer than n original slots.
        if j<len(comments):comments[j]={'text':comment}
        else:comments.append({'text':comment})
    out['comments']=comments
    results.append(out)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with open(a.output,'w',encoding='utf-8') as f:
    for x in results:f.write(json.dumps(x,ensure_ascii=False)+'\n')
write_json({'kind':a.kind,'attack_fraction':a.fraction,'max_comments':a.max_comments,
            'seed':a.seed, 'n_posts':len(results), 'attack_corpus':a.attack_corpus,
            'note':'Input labels never used to select attack text. No claims of real bots.'},a.output+'.attack.json')
print(f'Wrote {len(results)} perturbed post records -> {a.output}')
