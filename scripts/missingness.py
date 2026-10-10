#!/usr/bin/env python3
"""Controlled masking test sets and observational comment-count slicing."""
import argparse
import json
from pathlib import Path
from poam.utils import json_records

p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
p.add_argument('--mode',required=True,choices=['remove_comments','remove_image','short_text'])
p.add_argument('--short-max-words',type=int,default=10)
a=p.parse_args()
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with open(a.output,'w',encoding='utf-8') as f:
    for item in json_records(a.input):
        item=dict(item)
        if a.mode=='remove_comments':item['comments']=[]
        elif a.mode=='remove_image':item['image']=''
        elif a.mode=='short_text':item['text']=' '.join(item['text'].split()[:a.short_max_words])
        f.write(json.dumps(item,ensure_ascii=False)+'\n')
print('For --mode remove_image, set data.allow_missing_images: true in the evaluation checkpoint config.')
