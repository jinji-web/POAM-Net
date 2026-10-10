"""Uncertainty units are explicit: paired posts vs paired independent training seeds."""

import numpy as np
from sklearn.metrics import f1_score


def paired_bootstrap_post(y, p_a, p_b, n_boot=2000, seed=42, threshold=.5,
                          groups=None):
    y, a, b = map(np.asarray, (y, p_a, p_b))
    if not (len(y) == len(a) == len(b)):
        raise ValueError('Two models must predict the exact same test posts in the same order')
    rng = np.random.default_rng(seed)
    pred_a = (a >= threshold).astype(int)
    pred_b = (b >= threshold).astype(int)
    def difference(idx):
        return 100*(f1_score(y[idx], pred_a[idx], average='macro', labels=[0,1], zero_division=0)
                    -f1_score(y[idx], pred_b[idx], average='macro', labels=[0,1], zero_division=0))
    values = np.empty(n_boot, dtype=float)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError('Group id count mismatch')
        unique = np.unique(groups)
        index_map = {g: np.flatnonzero(groups == g) for g in unique}
        for i in range(n_boot):
            idx = np.concatenate([index_map[g] for g in rng.choice(unique, len(unique), replace=True)])
            values[i] = difference(idx)
        unit = 'event_cluster'
    else:
        for i in range(n_boot):
            values[i] = difference(rng.integers(len(y), size=len(y)))
        unit = 'post'
    return {'unit': unit, 'n_boot': n_boot, 'difference_points': difference(np.arange(len(y))),
            'ci95_percentile': np.percentile(values, [2.5, 97.5]).tolist(),
            'bootstrap_mean_points': float(values.mean())}


def paired_seed_ci(model_a, model_b):
    import scipy.stats as st
    a = np.asarray(model_a, dtype=float)
    b = np.asarray(model_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError('Require >=2 paired seeds evaluated on the same split')
    d = a-b
    half = st.t.ppf(.975, len(d)-1) * d.std(ddof=1)/np.sqrt(len(d))
    return {'unit': 'seed', 'n_seeds': int(len(d)), 'paired_differences_points':d.tolist(),
            'mean_difference_points':float(d.mean()),
            'ci95_t_approx': [float(d.mean()-half), float(d.mean()+half)]}
