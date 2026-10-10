"""Integration smoke test: image file + JSONL -> batch -> forward -> predictions."""
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

from poam.data import NewsDataset, BatchCollator
from poam.model import POAMNet
from poam.engine import predict
from poam.metrics import binary_metrics
from test_model import DummyText, DummyImage


class TinyTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        ids=[]; masks=[]
        for text in texts:
            size=min(len(text.split())+2,max_length)
            ids.append([10]*size+[0]*(max_length-size))
            masks.append([1]*size+[0]*(max_length-size))
        return {'input_ids':torch.tensor(ids), 'attention_mask':torch.tensor(masks)}


def test_end_to_end_without_external_models(tmp_path):
    path=tmp_path/'images';path.mkdir()
    Image.fromarray(np.ones((32,32,3),dtype=np.uint8)*150).save(path/'x.jpg')
    rows=[{'id':'real-post','text':'A real example post','label':0,'image':'x.jpg','comments':[]},
          {'id':'fake-post','text':'An alternate example','label':1,'image':'x.jpg',
           'comments':[{'text':'First observed comment'}]}]
    data=tmp_path/'test.jsonl'
    data.write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf-8')
    tok=TinyTokenizer()
    ds=NewsDataset(data,tok,image_root=path,max_news_len=8,max_comment_len=6,
                   max_comments=3,image_size=32)
    dl=DataLoader(ds,batch_size=2,collate_fn=BatchCollator(tok,8,6,3))
    model=POAMNet({'feature_dim':16,'heads':4,'dropout':0.,'variant':'full'},DummyText(),DummyImage())
    outs=predict(model,dl,torch.device('cpu'),amp=False)
    assert len(outs)==2
    assert outs[0]['comment_count']==0 and outs[1]['comment_count']==1
    assert all(np.isfinite(x['probability']) for x in outs)
    assert binary_metrics([x['label'] for x in outs],
                          [x['probability'] for x in outs])['n']==2
