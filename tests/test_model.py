"""Fast CPU architecture smoke tests with deliberately tiny offline encoder stubs.
These are SOFTWARE TESTS, not fake-news benchmark results.
"""
import types
import torch
import torch.nn as nn
import pytest

from poam.model import POAMNet


class DummyText(nn.Module):
    def __init__(self,hidden=24):
        super().__init__()
        self.config=types.SimpleNamespace(hidden_size=hidden)
        self.embed=nn.Embedding(128,hidden)
    def forward(self,input_ids,attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.embed(input_ids))


class DummyImage(nn.Module):
    def __init__(self,hidden=16):
        super().__init__()
        self.config=types.SimpleNamespace(hidden_size=hidden)
        self.net=nn.Linear(3,hidden)
    def forward(self,pixel_values):
        patches=pixel_values.mean(dim=(-2,-1))
        return types.SimpleNamespace(last_hidden_state=self.net(patches).unsqueeze(1).expand(-1,4,-1))


def make_inputs():
    B=2;M=3
    return {
     'input_ids':torch.randint(0,128,(B,8)),
     'attention_mask':torch.tensor([[1]*7+[0],[1]*8]),
     'pixel_values':torch.rand(B,3,16,16),
     'image_present':torch.ones(B,dtype=torch.bool),
     'comment_input_ids':torch.randint(0,128,(B,M,5)),
     'comment_attention_mask':torch.ones(B,M,5,dtype=torch.long),
     'comment_mask':torch.tensor([[1,1,0],[0,0,0]],dtype=torch.bool),
    }


@pytest.mark.parametrize('variant', sorted(POAMNet.VALID_VARIANTS))
def test_variant_has_finite_gradient(variant):
    model=POAMNet({'feature_dim':16,'heads':4,'dropout':0.,'variant':variant},DummyText(),DummyImage())
    result=model(**make_inputs())
    assert result['logits'].shape==(2,)
    assert torch.isfinite(result['logits']).all()
    result['logits'].sum().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    if 'comment_weights' in result:
        assert torch.allclose(result['comment_weights'][0].sum(),torch.tensor(1.),atol=1e-6) if variant!='no_opinion' else True
        assert result['comment_weights'][1].sum()==0


def test_masked_comments_do_not_change_output():
    model=POAMNet({'feature_dim':16,'heads':4,'dropout':0.,'variant':'full'},DummyText(),DummyImage())
    model.eval()
    x=make_inputs()
    with torch.no_grad():a=model(**x)['logits']
    x['comment_input_ids'][:,2,:]=torch.randint(0,128,(2,5))
    x['comment_input_ids'][1,:,:]=torch.randint(0,128,(3,5))
    with torch.no_grad():b=model(**x)['logits']
    assert torch.allclose(a,b,atol=1e-6)
