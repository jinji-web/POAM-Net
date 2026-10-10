#!/usr/bin/env python3
import argparse
from poam.engine import run_train
from poam.utils import read_config

p = argparse.ArgumentParser(description='Train POAM-Net with validation-only model selection')
p.add_argument('--config', required=True)
p.add_argument('--output', required=True)
p.add_argument('--variant', choices=['full','no_tid','no_ra','no_cod','no_af','no_opinion','comment_mean',
    'comment_attention','concat_mlp','gated_fusion','product_only','relevance_only','text_only','image_only'])
p.add_argument('--seed', type=int)
p.add_argument('--resume', help='Fine-tune weights from a local trained checkpoint (matching architecture)')
p.add_argument('--batch-size', type=int)
p.add_argument('--grad-accumulation', type=int)
args = p.parse_args()
cfg = read_config(args.config)
if args.variant: cfg['model']['variant'] = args.variant
if args.batch_size: cfg['train']['batch_size'] = args.batch_size
if args.grad_accumulation: cfg['train']['grad_accumulation'] = args.grad_accumulation
ckpt = run_train(cfg, args.output, seed=args.seed, resume=args.resume)
print('Best checkpoint:', ckpt)
