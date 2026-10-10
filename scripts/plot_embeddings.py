#!/usr/bin/env python3
"""Sample real test embeddings for t-SNE, not an invented point cloud.
Comparison is post-content (pre discrepancy) vs opinion-fused representations.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from poam.engine import load_checkpoint, get_tokenizer, make_loader, model_input
from poam.data import move_to

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
p.add_argument('--max-posts',type=int,default=1200)
p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
model,cfg,device,_=load_checkpoint(a.checkpoint)
loader=make_loader(cfg,cfg['data']['test'],get_tokenizer(cfg),training=False)
model.eval();latents=[];targets=[]
with torch.no_grad():
    for batch in loader:
        if sum(len(z) for z in targets)>=a.max_posts:break
        batch=move_to(batch,device)
        output=model(**model_input(batch))
        if not ('content' in output and 'opinion' in output):
            raise ValueError('Unimodal variants do not provide dual branch features')
        latents.append(torch.cat([output['content'],output['opinion']],dim=1).cpu().float().numpy())
        targets.append(batch['labels'].cpu().int().numpy())
X=np.concatenate(latents)[:a.max_posts];y=np.concatenate(targets)[:a.max_posts]
if len(X)<5:raise ValueError('Insufficient real test posts for t-SNE')
embedding=TSNE(n_components=2,random_state=a.seed,perplexity=min(30.,(len(X)-1)/3)).fit_transform(X)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out.with_suffix('.npz'),embeddings=embedding,labels=y)
fig,ax=plt.subplots(figsize=(6,5))
for label,name in [(0,'Real'),(1,'Fake')]:
    points=embedding[y==label]
    ax.scatter(points[:,0],points[:,1],s=12,alpha=.58,label=name)
ax.set(xticks=[],yticks=[],xlabel='t-SNE dimension 1',ylabel='t-SNE dimension 2')
ax.legend();fig.tight_layout();fig.savefig(out,dpi=200);plt.close(fig)
print('Created',out,'from',len(X),'held-out post representations')
