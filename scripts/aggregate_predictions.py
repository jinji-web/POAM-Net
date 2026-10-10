#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
from poam.metrics import binary_metrics
from poam.stats import paired_seed_ci
from poam.utils import json_records, write_json

p=argparse.ArgumentParser()
p.add_argument('--runs', nargs='+', required=True,help='Metric JSON files from independently trained seed runs')
p.add_argument('--output',required=True)
a=p.parse_args()
rows=[json.load(open(path,encoding='utf-8')) for path in a.runs]
fields=['macro_f1','accuracy','roc_auc','ece','brier']
output={'n_seeds':len(rows),'seeds':[x['seed'] for x in rows]}
for k in fields:
    numbers=[x[k] for x in rows if x.get(k) is not None]
    if numbers:
        output[k]={'mean':float(np.mean(numbers)),
                   'std':float(np.std(numbers,ddof=1)) if len(numbers)>1 else None}
write_json(output,a.output)
print(json.dumps(output,indent=2))
