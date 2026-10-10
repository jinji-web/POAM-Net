#!/usr/bin/env python3
import argparse
import json
from poam.engine import run_evaluate
p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--split', default='test', choices=['train','val','test'])
p.add_argument('--path', help='Optional transformed JSONL; labels still required')
p.add_argument('--output', required=True)
p.add_argument('--horizon-hours', type=float)
p.add_argument('--ece-bins', type=int, default=15)
p.add_argument('--device', default='auto')
p.add_argument('--allow-missing-images', action='store_true',
               help='Required for controlled missing-image evaluation')
args = p.parse_args()
m, _ = run_evaluate(args.checkpoint, args.split, args.path, args.output,
                    args.horizon_hours, args.device, args.ece_bins,
                    allow_missing_images=True if args.allow_missing_images else None)
print(json.dumps(m, indent=2))
