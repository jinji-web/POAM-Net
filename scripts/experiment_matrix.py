#!/usr/bin/env python3
"""Runs actual experiments from data, NOT transcription of manuscript scores."""
import argparse
from pathlib import Path
import numpy as np

from poam.engine import run_train, run_evaluate
from poam.utils import read_config, write_json

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--output', required=True)
p.add_argument('--seeds', type=int, nargs='+')
p.add_argument('--variants', nargs='+', default=['full','no_tid','no_ra','no_cod','no_af','no_opinion',
                                                    'comment_mean','comment_attention','concat_mlp',
                                                    'gated_fusion','product_only','relevance_only'])
args = p.parse_args()
base = read_config(args.config)
seeds = args.seeds or base['train'].get('seeds', [42,43,44,45,46])
results = []
for variant in args.variants:
    for seed in seeds:
        cfg = read_config(args.config)
        cfg['model']['variant'] = variant
        dest = Path(args.output)/variant/str(seed)
        ckpt = run_train(cfg, dest, seed=seed)
        metric, rows = run_evaluate(ckpt, output_dir=dest/'test')
        results.append({'variant':variant,'seed':seed,'accuracy':metric['accuracy'],
                        'macro_f1':metric['macro_f1'], 'roc_auc':metric['roc_auc']})
        write_json(results, Path(args.output)/'all_runs.json')
summary = []
for variant in args.variants:
    selected = [x for x in results if x['variant'] == variant]
    for key in ('accuracy','macro_f1','roc_auc'):
        numbers = np.array([x[key] for x in selected if x[key] is not None])
        summary.append({'variant':variant, 'metric':key, 'n_seeds':len(numbers),
                        'mean':float(numbers.mean()),
                        'std':float(numbers.std(ddof=1)) if len(numbers)>1 else None})
write_json(summary, Path(args.output)/'summary.json')
