import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, roc_auc_score, confusion_matrix, brier_score_loss)


def expected_calibration_error(y, prob, bins=15):
    y = np.asarray(y).astype(int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= .5).astype(int)
    confidence = np.maximum(p, 1-p)
    boundaries = np.linspace(0, 1, bins+1)
    result = 0.
    for i in range(bins):
        in_bin = ((confidence >= boundaries[i]) &
                  (confidence < boundaries[i+1] if i < bins-1 else confidence <= boundaries[i+1]))
        if in_bin.any():
            result += in_bin.mean()*abs((pred[in_bin] == y[in_bin]).mean()-confidence[in_bin].mean())
    return float(result)


def binary_metrics(y, prob, threshold=.5, bins=15):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if len(y) == 0:
        raise ValueError('Cannot score empty test set')
    pred = (prob >= threshold).astype(int)
    per_class = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        'n': int(len(y)), 'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)),
        'precision_real': float(per_class[0][0]), 'precision_fake': float(per_class[0][1]),
        'recall_real': float(per_class[1][0]), 'recall_fake': float(per_class[1][1]),
        'f1_real': float(per_class[2][0]), 'f1_fake': float(per_class[2][1]),
        'support_real': int(per_class[3][0]), 'support_fake': int(per_class[3][1]),
        'roc_auc': float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
        'brier': float(brier_score_loss(y, prob)),
        'ece': expected_calibration_error(y, prob, bins=bins),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        'threshold': float(threshold), 'ece_bins': int(bins),
    }
