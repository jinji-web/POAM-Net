#!/usr/bin/env python3
"""Compare paired clean vs attacked records using their saved per-post predictions."""
import argparse
import json
import numpy as np
from poam.metrics import binary_metrics
from poam.utils import json_records, write_json
p=argparse.ArgumentParser()
p.add_argument('--clean',required=True)
p.add_argument('--attacked',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
c={x['id']:x for x in json_records(a.clean)}
d={x['id']:x for x in json_records(a.attacked)}
if set(c)!=set(d):raise ValueError('Clean and attacked prediction ID sets differ')
ids=sorted(c)
if any(c[i]['label']!=d[i]['label'] for i in ids):raise ValueError('Labels modified')
before=np.array([c[x]['probability']>=.5 for x in ids]);after=np.array([d[x]['probability']>=.5 for x in ids])
r={'n':len(ids),'prediction_flip_rate':float(np.mean(before!=after)),
   'clean_metrics':binary_metrics([c[x]['label'] for x in ids],[c[x]['probability'] for x in ids]),
   'attacked_metrics':binary_metrics([d[x]['label'] for x in ids],[d[x]['probability'] for x in ids])}
write_json(r,a.output);print(json.dumps({'flip_rate':r['prediction_flip_rate']},indent=2))
