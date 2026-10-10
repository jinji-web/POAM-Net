#!/usr/bin/env python3
import argparse
from pathlib import Path
from poam.engine import run_evaluate
from poam.utils import write_json
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--cohort',required=True,help='File produced by temporal_cohort.py')
p.add_argument('--output',required=True)
p.add_argument('--horizons',type=float,nargs='+',default=[1,6,24])
a=p.parse_args()
results=[]
for hours in a.horizons+[None]:
    out=Path(a.output)/('all' if hours is None else f'{hours:g}h')
    metrics,_=run_evaluate(a.checkpoint,path=a.cohort,output_dir=out,horizon_hours=hours)
    results.append({'window':'all' if hours is None else f'{hours:g}h',
                    'n':metrics['n'], 'macro_f1':metrics['macro_f1'],
                    'accuracy':metrics['accuracy']})
write_json(results,Path(a.output)/'windows.json')
print(results)
