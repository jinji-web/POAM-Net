#!/usr/bin/env python3
"""Evaluate test predictions on natural comment-count strata and declared data subgroups.
Results are calculated from model predictions; no manuscript numbers are embedded.
"""
import argparse
import json
import numpy as np
from poam.metrics import binary_metrics
from poam.utils import json_records, write_json

p=argparse.ArgumentParser()
p.add_argument('--predictions',required=True)
p.add_argument('--records',required=True,help='Matching unaltered test JSONL')
p.add_argument('--output',required=True)
p.add_argument('--short-max-words',type=int,
   help='A threshold fixed using TRAIN-only corpus statistics, never tuned on test labels')
p.add_argument('--low-sim-threshold',type=float,
   help='A cosine threshold fixed with training data; requires saved tv_similarity in predictions')
a=p.parse_args()
raw={str(x['id']):x for x in json_records(a.records)}
rows=list(json_records(a.predictions))
if not rows or any(str(x['id']) not in raw for x in rows):raise ValueError('Predictions need corresponding source records')
maskgroups={
 'no_comments':lambda x:x['comment_count']==0,
 '1_2_comments':lambda x:1<=x['comment_count']<=2,
 '3_5_comments':lambda x:3<=x['comment_count']<=5,
 '6_10_comments':lambda x:6<=x['comment_count']<=10,
 'more_than_10_comments':lambda x:x['comment_count']>10,
}
if a.short_max_words is not None:
    maskgroups['short_text']=lambda x:len(raw[str(x['id'])]['text'].split())<=a.short_max_words
if a.low_sim_threshold is not None:
    if any('tv_similarity' not in row for row in rows):raise ValueError('tv_similarity not saved')
    maskgroups['low_text_image_similarity']=lambda x:x['tv_similarity']<=a.low_sim_threshold
out=[]
for name,func in maskgroups.items():
    sub=[x for x in rows if func(x)]
    if not sub:continue
    metrics=binary_metrics([x['label'] for x in sub],[x['probability'] for x in sub])
    out.append({'group':name,'posts':len(sub),'fraction':len(sub)/len(rows),'metrics':metrics})
write_json(out,a.output)
print(json.dumps([{'group':x['group'],'posts':x['posts'],
                   'macro_f1':x['metrics']['macro_f1']} for x in out],indent=2))
