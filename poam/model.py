"""Core manuscript-informed POAM-Net (Equations 7–41).

Backbone outputs use Hugging Face XLM-RoBERTa/Swin.
Implementation details that the manuscript does not pin down (projection activation,
MLP hidden size, residual-dropout placement) are explicitly disclosed in README.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_dim, output_dim, hidden_dim=512, dropout=0.2):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))


def masked_mean(sequence, mask, dim=1):
    weights = mask.to(sequence.dtype).unsqueeze(-1)
    return (sequence * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1, eps=1e-8)


class POAMNet(nn.Module):
    VALID_VARIANTS = {
        'full', 'no_tid', 'no_ra', 'no_cod', 'no_af', 'no_opinion',
        'comment_mean', 'comment_attention', 'concat_mlp',
        'gated_fusion', 'product_only', 'relevance_only',
        'text_only', 'image_only'
    }

    def __init__(self, cfg, text_encoder=None, image_encoder=None):
        super().__init__()
        self.cfg = cfg = dict(cfg)
        self.variant = cfg.get('variant', 'full')
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(f'Unknown variant {self.variant!r}; choices {sorted(self.VALID_VARIANTS)}')
        d = int(cfg.get('feature_dim', 256))
        self.d = d
        heads = int(cfg.get('heads', 8))
        if d % heads:
            raise ValueError('feature_dim must be divisible by heads')
        dropout = float(cfg.get('dropout', .2))
        if text_encoder is None or image_encoder is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError('pip install -r requirements.txt (transformers needed for production)') from exc
            if text_encoder is None:
                text_encoder = AutoModel.from_pretrained(cfg['text_checkpoint'])
            if image_encoder is None:
                image_encoder = AutoModel.from_pretrained(cfg['image_checkpoint'])
        self.text_encoder = text_encoder  # shared between news text and comments
        self.image_encoder = image_encoder
        if cfg.get('grad_checkpointing', False):
            for x in (self.text_encoder, self.image_encoder):
                if hasattr(x, 'gradient_checkpointing_enable'):
                    x.gradient_checkpointing_enable()
        tdim = int(text_encoder.config.hidden_size)
        vdim = int(image_encoder.config.hidden_size)
        self.text_project = nn.Linear(tdim, d)
        self.image_project = nn.Linear(vdim, d)
        self.comment_project = nn.Linear(tdim, d)
        self.text_seq_project = nn.Linear(tdim, d)
        self.image_seq_project = nn.Linear(vdim, d)
        self.text_queries_image = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.image_queries_text = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_norm_text = nn.LayerNorm(d)
        self.cross_norm_image = nn.LayerNorm(d)
        self.tv_net = mlp(2*d + d + d + 1, d, 2*d, dropout)
        self.content_net = mlp(5*d, d, 2*d, dropout)
        self.comment_score = nn.Sequential(nn.Linear(d+2, d), nn.Tanh(), nn.Linear(d, 1))
        self.standard_att = nn.Sequential(nn.Linear(2*d, d), nn.Tanh(), nn.Linear(d, 1))
        self.null_opinion = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.null_opinion, std=.02)
        self.co_content = nn.Linear(d, d)
        self.co_opinion = nn.Linear(d, d)
        self.co_net = mlp(4*d+1, d, 2*d, dropout)
        self.fusion_gate = nn.Linear(4*d, 4)
        self.fusion_projection = nn.ModuleList([nn.Linear(d, d) for _ in range(4)])
        self.ffn = mlp(d, d, 2*d, dropout)
        self.fusion_norm = nn.LayerNorm(d)
        self.final_head = nn.Linear(d, 1)
        self.simple_head = mlp(3*d, 1, 2*d, dropout)
        self.alternative_head = mlp(4*d, 1, 2*d, dropout)
        self.gated_alt = nn.Linear(4*d, 4)
        self.unimodal_head = nn.Linear(d, 1)
        self.max_comment_encode_batch = int(cfg.get('comment_chunk_size', 16))

    def _encode_comments(self, input_ids, attention_mask, is_valid):
        B, M, L = input_ids.shape
        valid_indices = is_valid.reshape(-1).nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return torch.zeros(B, M, self.d, device=input_ids.device,
                               dtype=self.text_project.weight.dtype)
        ids = input_ids.reshape(B*M, L)
        masks = attention_mask.reshape(B*M, L)
        pieces = []
        indices = []
        for chunk in valid_indices.split(self.max_comment_encode_batch):
            hidden = self.text_encoder(input_ids=ids[chunk], attention_mask=masks[chunk]).last_hidden_state
            pieces.append(self.comment_project(hidden[:, 0, :]))
            indices.append(chunk)
        values = torch.cat(pieces)
        result = values.new_zeros((B*M, self.d))
        result = result.index_copy(0, torch.cat(indices), values)
        return result.reshape(B, M, self.d)

    def forward(self, input_ids, attention_mask, pixel_values,
                comment_input_ids, comment_attention_mask, comment_mask,
                image_present=None, **unused):
        B = input_ids.size(0)
        device = input_ids.device
        if image_present is None:
            image_present = torch.ones(B, dtype=torch.bool, device=device)
        t = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        v = self.image_encoder(pixel_values=pixel_values).last_hidden_state
        tcls = self.text_project(t[:, 0])
        vcls = self.image_project(v.mean(dim=1)) * image_present[:, None].to(v.dtype)
        if self.variant == 'text_only':
            logits = self.unimodal_head(tcls).squeeze(-1)
            return {'logits': logits, 'probabilities': torch.sigmoid(logits)}
        if self.variant == 'image_only':
            logits = self.unimodal_head(vcls).squeeze(-1)
            return {'logits': logits, 'probabilities': torch.sigmoid(logits)}
        tseq = self.text_seq_project(t)
        vseq = self.image_seq_project(v)
        t2v, _ = self.text_queries_image(tseq, vseq, vseq, need_weights=False)
        v2t, _ = self.image_queries_text(vseq, tseq, tseq,
                                        key_padding_mask=~attention_mask.bool(), need_weights=False)
        t2v = self.cross_norm_text(t2v + tseq)
        v2t = self.cross_norm_image(v2t + vseq)
        interacted = torch.cat([masked_mean(t2v, attention_mask.bool()), v2t.mean(1)], dim=-1)
        tv_similarity = cosine(tcls, vcls)
        tv = self.tv_net(torch.cat([interacted, (tcls-vcls).abs(),
                                    tcls*vcls, tv_similarity[:, None]], dim=-1))
        if self.variant == 'no_tid':
            tv = torch.zeros_like(tv)
        content = self.content_net(torch.cat([tcls, vcls, interacted, tv], dim=-1))
        if self.variant == 'no_opinion':
            opinions = torch.zeros_like(content)
            attention = torch.zeros_like(comment_mask, dtype=content.dtype)
        else:
            q = self._encode_comments(comment_input_ids, comment_attention_mask, comment_mask)
            denominator = comment_mask.sum(dim=1).clamp_min(1)[:, None]
            centroid = (q*comment_mask[:, :, None]).sum(dim=1)/denominator
            r = cosine(q, content[:, None, :])
            a = cosine(q, centroid[:, None, :])
            if self.variant == 'comment_mean' or self.variant == 'no_ra':
                scores = torch.zeros_like(r)
            elif self.variant == 'comment_attention':
                scores = self.standard_att(torch.cat([q, content[:, None, :].expand_as(q)], -1)).squeeze(-1)
            else:
                if self.variant == 'relevance_only':
                    a = torch.zeros_like(a)
                scores = self.comment_score(torch.cat([q, r[:, :, None], a[:, :, None]], dim=-1)).squeeze(-1)
            # padded comment positions must not influence attention or centroids
            safe_scores = scores.masked_fill(~comment_mask, -1e4)
            weights = torch.softmax(safe_scores, dim=-1)
            attention = weights * comment_mask.float()
            attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-8)
            opinions = (attention[:, :, None] * q).sum(1)
            opinions = torch.where(comment_mask.any(1)[:, None], opinions,
                                   self.null_opinion[None].expand_as(opinions))
        if self.variant in ('comment_mean', 'comment_attention'):
            logits = self.simple_head(torch.cat([tcls, vcls, opinions], dim=-1)).squeeze(-1)
            return {'logits': logits, 'probabilities': torch.sigmoid(logits),
                    'comment_weights': attention, 'content': content, 'opinion': opinions}
        zc, zo = self.co_content(content), self.co_opinion(opinions)
        co_similarity = cosine(zc, zo)
        co = self.co_net(torch.cat([zc, zo, (zc-zo).abs(), zc*zo,
                                    co_similarity[:, None]], dim=-1))
        if self.variant == 'product_only':
            co = zc*zo
        if self.variant in ('no_cod', 'no_opinion'):
            co = torch.zeros_like(co)
        branches = [content, opinions, tv, co]
        combined = torch.cat(branches, dim=-1)
        if self.variant == 'concat_mlp' or self.variant == 'no_af':
            logits = self.alternative_head(combined).squeeze(-1)
            gate = None
        else:
            if self.variant == 'gated_fusion':
                gate = torch.softmax(self.gated_alt(combined), dim=-1)
            else:
                gate = torch.softmax(self.fusion_gate(combined), dim=-1)
            fused = sum(gate[:, i, None] * self.fusion_projection[i](x)
                        for i, x in enumerate(branches))
            final = self.fusion_norm(fused + self.ffn(fused))
            logits = self.final_head(final).squeeze(-1)
        return {'logits': logits, 'probabilities': torch.sigmoid(logits),
                'comment_weights': attention, 'fusion_gate': gate,
                'content': content, 'opinion': opinions,
                'tv_similarity': tv_similarity, 'co_similarity': co_similarity}
