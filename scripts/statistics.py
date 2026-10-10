#!/usr/bin/env python3
"""Evaluate paired predictions; never infer paired intervals from aggregated means ± SD."""
import argparse
import json
from poam.utils import json_records, write_json
from poam.stats import paired_bootstrap_post, paired_seed_ci

p=argparse.ArgumentParser()
p.add_argument('--poam', required=True, help='predictions.jsonl of POAM-Net')
p.add_argument('--baseline', required=True, help='predictions.jsonl of SAME held-out posts')
p.add_argument('--output', required=True)
p.add_argument('--n-bootstrap', type=int, default=2000)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--groups', help='Optional JSON mapping post ID -> event cluster ID')
args=p.parse_args()
a={x['id']:x for x in json_records(args.poam)}
b={x['id']:x for x in json_records(args.baseline)}
if set(a)!=set(b):raise SystemExit(f'Different post IDs! A={len(a)}, B={len(b)}')
ids=sorted(a)
if any(a[x]['label'] != b[x]['label'] for x in ids):
    raise SystemExit('Different labels for same IDs')
groups=None
if args.groups:
    grouping=json.load(open(args.groups,encoding='utf-8'))
    if any(x not in grouping for x in ids):
        raise SystemExit('Unassigned event ids; cannot claim event-cluster inference')
    groups=[grouping[x] for x in ids]
r=paired_bootstrap_post([a[x]['label'] for x in ids],
        [a[x]['probability'] for x in ids], [b[x]['probability'] for x in ids],
        args.n_bootstrap,args.seed,groups=groups)
write_json(r,args.output)
print(json.dumps(r,indent=2))
