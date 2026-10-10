#!/usr/bin/env python3
"""Construct a fixed, timestamp-valid population for every publication-relative horizon.

Missing timestamps do not magically become early comments; the cohort is reported.
"""
import argparse
import json
from pathlib import Path
from poam.data import parse_timestamp, comment_list
from poam.utils import json_records, write_json

p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
p.add_argument('--require-timed-comment',action='store_true')
a=p.parse_args()
rows=list(json_records(a.input));eligible=[]
for row in rows:
    if parse_timestamp(row.get('post_ts')) is None:continue
    if a.require_timed_comment and not any(parse_timestamp(x.get('created_ts')) is not None
                                           for x in comment_list(row.get('comments'))):continue
    eligible.append(row)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with open(a.output,'w',encoding='utf-8') as f:
    for x in eligible:f.write(json.dumps(x,ensure_ascii=False)+'\n')
write_json({'total_posts':len(rows),'eligible_posts':len(eligible),
            'eligible_fraction':len(eligible)/max(len(rows),1),
            'require_timed_comment':a.require_timed_comment},a.output+'.metadata.json')
print(f'Fixed cohort: {len(eligible)}/{len(rows)} posts')
