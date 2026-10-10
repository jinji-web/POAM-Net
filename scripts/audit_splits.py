#!/usr/bin/env python3
"""Screen post-level splits for repeated IDs, duplicate normalized text and pHash reuse.

This is a diagnostic, not a proof of independence. Perceptual hashes are computed
from actual images, not from filename assumptions. Near-text screening needs
manual review or a separately documented MinHash/embedding algorithm.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct

from poam.data import normalize_text, parse_timestamp
from poam.utils import json_records, write_json


def phash(path):
    with Image.open(path) as im:
        x=np.asarray(im.convert('L').resize((32,32)),dtype=np.float64)
    y=dct(dct(x,axis=0,norm='ortho'),axis=1,norm='ortho')[:8,:8]
    vals=y.ravel()[1:]
    return ''.join('1' if z>np.median(vals) else '0' for z in vals)


def image_path(root, s):
    path=Path(s or '')
    return path if path.is_absolute() else Path(root)/path

p=argparse.ArgumentParser()
p.add_argument('--train',required=True)
p.add_argument('--val',required=True)
p.add_argument('--test',required=True)
p.add_argument('--image-root',default='.')
p.add_argument('--output',required=True)
a=p.parse_args()
splits={'train':list(json_records(a.train)),'val':list(json_records(a.val)),'test':list(json_records(a.test))}
metrics={'counts':{x:len(y) for x,y in splits.items()},'metadata':{},'pairwise':{}}
for name,rows in splits.items():
    metrics['metadata'][name]={
      'post_timestamps':sum(parse_timestamp(x.get('post_ts')) is not None for x in rows),
      'posts_with_timestamped_comments':sum(any(parse_timestamp(c.get('created_ts')) is not None
            for c in x.get('comments',[]) if isinstance(c,dict)) for x in rows),
      'posts_with_zero_comments':sum(len(x.get('comments',[]))==0 for x in rows)}
features={}
for name,rows in splits.items():
    subset=[]
    for r in rows:
        raw=normalize_text(r.get('text','')).lower()
        txt=hashlib.sha256(raw.encode('utf-8')).hexdigest() if raw else None
        pth=image_path(a.image_root,r.get('image',''))
        img=phash(pth) if pth.is_file() else None
        subset.append((str(r['id']),txt,img))
    features[name]=subset
for name1,name2 in [('train','val'),('train','test'),('val','test')]:
    ids1={i for i,t,p in features[name1]};ids2={i for i,t,p in features[name2]}
    txt1={t for i,t,p in features[name1] if t};txt2={t for i,t,p in features[name2] if t}
    img1={p for i,t,p in features[name1] if p};img2={p for i,t,p in features[name2] if p}
    metrics['pairwise'][f'{name1}:{name2}']={
      'duplicate_ids':len(ids1 & ids2),
      'exact_normalized_text_hash_overlap':len(txt1 & txt2),
      'identical_perceptual_hash_overlap':len(img1 & img2),
      'warning':'Hash match counts are distinct hash counts, not percentage of test posts.'}
write_json(metrics,a.output)
if any(v['duplicate_ids'] for v in metrics['pairwise'].values()):
    raise SystemExit('WARNING: post IDs overlap across training, validation, and test partitions')
print(json.dumps(metrics,indent=2))
