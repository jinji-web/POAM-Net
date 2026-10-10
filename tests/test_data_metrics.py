import pytest
from poam.data import select_comments, parse_timestamp
from poam.metrics import binary_metrics
from poam.stats import paired_bootstrap_post


def test_temporal_censor_excludes_late_or_undated():
    post={'id':'p','post_ts':'2025-01-01T00:00:00+00:00', 'comments':[
      {'text':'late','created_ts':'2025-01-01T07:00:00Z'},
      {'text':'early','created_ts':'2025-01-01T00:30:00Z'},
      {'text':'unknown'}]}
    assert select_comments(post,horizon_hours=1)==['early']
    assert len(select_comments(post,horizon_hours=None))==3


def test_untimed_post_raises_on_temporal():
    with pytest.raises(ValueError):select_comments({'id':'x','comments':[]},horizon_hours=1)


def test_metrics_and_paired_ci():
    y=[0,0,1,1,0,1]
    a=[.1,.2,.9,.8,.4,.7]
    b=[.6,.2,.9,.1,.4,.7]
    assert binary_metrics(y,a)['macro_f1']==1
    x=paired_bootstrap_post(y,a,b,n_boot=60,seed=42)
    assert x['unit']=='post' and x['difference_points']>0
