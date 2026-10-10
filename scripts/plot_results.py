#!/usr/bin/env python3
"""Visualizations built from saved observed prediction files, not manuscript table numbers."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc
from poam.utils import json_records

p=argparse.ArgumentParser()
p.add_argument('--predictions',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
rows=list(json_records(a.predictions))
if not rows:raise ValueError('Empty predictions file')
y=np.asarray([r['label'] for r in rows]);pr=np.asarray([r['probability'] for r in rows]);pred=(pr>=.5).astype(int)
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay(confusion_matrix(y,pred,labels=[0,1]),display_labels=['real','fake']).plot(ax=ax,colorbar=False)
fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=210);plt.close(fig)
if len(np.unique(y))==2:
    fpr,tpr,_=roc_curve(y,pr)
    fig,ax=plt.subplots(figsize=(5,4))
    ax.plot(fpr,tpr,label=f'AUC = {auc(fpr,tpr):.3f}')
    ax.plot([0,1],[0,1],linestyle='--',alpha=.5)
    ax.set(xlabel='False positive rate',ylabel='True positive rate')
    ax.legend();fig.tight_layout();fig.savefig(out/'roc.png',dpi=210);plt.close(fig)
# Reliability plot, binary-class probability of fake vs actual fake proportion
edges=np.linspace(0,1,11);xs=[];ys=[]
for i in range(10):
    sel=(pr>=edges[i])&(pr<(edges[i+1]) if i<9 else pr<=edges[i+1])
    if sel.any():xs.append(pr[sel].mean());ys.append(y[sel].mean())
fig,ax=plt.subplots(figsize=(5,4));ax.plot([0,1],[0,1],linestyle='--',alpha=.5)
ax.plot(xs,ys,'o-');ax.set(xlabel='Predicted P(fake)',ylabel='Observed fake fraction')
fig.tight_layout();fig.savefig(out/'calibration.png',dpi=210);plt.close(fig)
print(f'Created plots in {out}')
