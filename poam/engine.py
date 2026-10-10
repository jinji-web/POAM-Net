import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import NewsDataset, BatchCollator, move_to
from .metrics import binary_metrics
from .model import POAMNet
from .utils import device_from_string, write_json, number_parameters, seed_all


def get_tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg['model']['text_checkpoint'], use_fast=True)


def make_loader(cfg, path, tokenizer, training=False, horizon=None, batch_size=None):
    data_cfg = cfg['data']
    ds = NewsDataset(path, tokenizer,
        image_root=data_cfg.get('image_root', '.'),
        max_news_len=data_cfg.get('max_news_len', 256),
        max_comment_len=data_cfg.get('max_comment_len', 64),
        max_comments=data_cfg.get('max_comments', 20),
        image_size=data_cfg.get('image_size', 224),
        training=training, horizon_hours=horizon,
        comment_order=data_cfg.get('comment_order', 'earliest'),
        allow_missing_images=data_cfg.get('allow_missing_images', False))
    collate = BatchCollator(tokenizer, data_cfg.get('max_news_len', 256),
                            data_cfg.get('max_comment_len', 64),
                            data_cfg.get('max_comments', 20))
    return DataLoader(ds,
        batch_size=batch_size or cfg['train'].get('batch_size', 16),
        shuffle=training, num_workers=cfg['train'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available(), collate_fn=collate)


def create_model(cfg, device):
    model = POAMNet(cfg['model'])
    model.to(device)
    for p in model.text_encoder.parameters():
        p.requires_grad = not cfg['train'].get('freeze_text', False)
    for p in model.image_encoder.parameters():
        p.requires_grad = not cfg['train'].get('freeze_vision', False)
    return model


def make_optimizer(model, train_cfg):
    heads, encoders = [], []
    text_ids = {id(x) for x in model.text_encoder.parameters()}
    image_ids = {id(x) for x in model.image_encoder.parameters()}
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if id(p) in text_ids or id(p) in image_ids:
            encoders.append(p)
        else:
            heads.append(p)
    groups = []
    if encoders: groups.append({'params':encoders, 'lr':float(train_cfg['encoder_lr'])})
    if heads: groups.append({'params':heads, 'lr':float(train_cfg['head_lr'])})
    if not groups: raise ValueError('No trainable parameters')
    return torch.optim.AdamW(groups, weight_decay=float(train_cfg.get('weight_decay', .01)))


def model_input(batch):
    return {k: batch[k] for k in ('input_ids', 'attention_mask', 'pixel_values',
                                 'image_present', 'comment_input_ids',
                                 'comment_attention_mask', 'comment_mask')}


@torch.no_grad()
def predict(model, loader, device, amp=False):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc='predict', leave=False):
        batch = move_to(batch, device)
        with torch.autocast(device_type=device.type, enabled=(amp and device.type == 'cuda')):
            outputs = model(**model_input(batch))
        probs = outputs['probabilities'].detach().float().cpu().tolist()
        labels = batch['labels'].detach().int().cpu().tolist()
        counts = batch['comment_counts'].detach().int().cpu().tolist()
        originals = batch['original_comment_counts'].detach().int().cpu().tolist()
        for i, rowid in enumerate(batch['ids']):
            row = {'id':rowid, 'label':labels[i], 'probability':probs[i],
                   'prediction':int(probs[i] >= 0.5),
                   'comment_count':counts[i], 'original_comment_count':originals[i]}
            if 'tv_similarity' in outputs:
                row['tv_similarity'] = float(outputs['tv_similarity'][i])
            if 'co_similarity' in outputs:
                row['co_similarity'] = float(outputs['co_similarity'][i])
            if 'fusion_gate' in outputs and outputs['fusion_gate'] is not None:
                row['fusion_gate'] = outputs['fusion_gate'][i].detach().float().cpu().tolist()
            if 'comment_weights' in outputs:
                row['comment_weights'] = outputs['comment_weights'][i, :counts[i]].detach().float().cpu().tolist()
            rows.append(row)
    return rows


def evaluate_rows(rows, threshold=.5, ece_bins=15):
    return binary_metrics([r['label'] for r in rows], [r['probability'] for r in rows],
                          threshold=threshold, bins=ece_bins)


def run_train(cfg, output_dir, seed=None, resume=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg['train']
    seed = int(seed if seed is not None else train_cfg.get('seed', 42))
    seed_all(seed)
    device = device_from_string(train_cfg.get('device', 'auto'))
    tokenizer = get_tokenizer(cfg)
    train_loader = make_loader(cfg, cfg['data']['train'], tokenizer, training=True)
    val_loader = make_loader(cfg, cfg['data']['val'], tokenizer, training=False)
    model = create_model(cfg, device)
    if resume:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=True)
    optimizer = make_optimizer(model, train_cfg)
    epochs = int(train_cfg.get('epochs', 15))
    accumulate = int(train_cfg.get('grad_accumulation', 1))
    total_steps = max(1, math.ceil(len(train_loader) / accumulate) * epochs)
    warmup_steps = int(total_steps * train_cfg.get('warmup_ratio', .1))
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = bool(train_cfg.get('use_amp', True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    pos_weight = None
    cw = train_cfg.get('class_weights')
    if cw is not None:
        # Positive-weight ratio for BCE = w_fake / w_real; report raw weights in saved config
        pos_weight = torch.tensor(float(cw[1])/float(cw[0]), device=device)
    best, bad = float('-inf'), 0
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs+1):
        model.train()
        epoch_losses = []
        steps_in_epoch = len(train_loader)
        for step, batch in enumerate(tqdm(train_loader, desc=f'train {epoch}/{epochs}'), 1):
            batch = move_to(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**model_input(batch))
                loss = F.binary_cross_entropy_with_logits(outputs['logits'], batch['labels'],
                                                          pos_weight=pos_weight)
                lam = float(train_cfg.get('regularization_lambda', 0.))
                if lam:
                    # AdamW weight decay already regularizes weights; optional explicit L2 is disabled by default.
                    reg = sum((p*p).sum() for p in model.parameters() if p.requires_grad)
                    loss = loss + lam * reg
                # For final partial accumulation window, preserve intended gradient scale
                window = accumulate if step <= steps_in_epoch - (steps_in_epoch % accumulate) else (steps_in_epoch % accumulate or accumulate)
                scaled = loss / window
            scaler.scale(scaled).backward()
            epoch_losses.append(float(loss.detach().cpu()))
            if step % accumulate == 0 or step == steps_in_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get('gradient_clip', 1)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        val_rows = predict(model, val_loader, device, use_amp)
        metric = evaluate_rows(val_rows, threshold=float(train_cfg.get('threshold', .5)))
        current = {'epoch':epoch, 'seed':seed, 'loss':float(np.mean(epoch_losses)),
                   'val_macro_f1':metric['macro_f1'], 'val_accuracy':metric['accuracy']}
        history.append(current)
        print(current, flush=True)
        if metric['macro_f1'] > best:
            best, bad = metric['macro_f1'], 0
            torch.save({'model':model.state_dict(), 'config':cfg, 'seed':seed,
                        'epoch':epoch, 'val_metrics':metric}, output_dir/'best.pt')
            write_json(metric, output_dir/'best_validation_metrics.json')
        else:
            bad += 1
            if bad >= int(train_cfg.get('early_stopping', 3)):
                break
    write_json(history, output_dir/'history.json')
    write_json({'seed':seed, 'best_val_macro_f1':best, 'parameters':number_parameters(model),
                'best_checkpoint':str(output_dir/'best.pt')}, output_dir/'training_summary.json')
    write_json(cfg, output_dir/'config_resolved.json')
    return str(output_dir/'best.pt')


def load_checkpoint(checkpoint_path, device='auto'):
    device = device_from_string(device)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    cfg = state['config']
    model = create_model(cfg, device)
    model.load_state_dict(state['model'], strict=True)
    return model, cfg, device, state


def run_evaluate(checkpoint_path, split='test', path=None, output_dir=None,
                 horizon_hours=None, device='auto', ece_bins=15, allow_missing_images=None):
    model, cfg, device, state = load_checkpoint(checkpoint_path, device)
    if allow_missing_images is not None:
        cfg['data']['allow_missing_images'] = allow_missing_images
    tokenizer = get_tokenizer(cfg)
    loader = make_loader(cfg, path or cfg['data'][split], tokenizer, training=False,
                         horizon=horizon_hours)
    rows = predict(model, loader, device, bool(cfg['train'].get('use_amp', True)))
    m = evaluate_rows(rows, float(cfg['train'].get('threshold', .5)), ece_bins)
    m.update({'seed':state['seed'], 'checkpoint_epoch':state['epoch'],
              'model_variant':cfg['model']['variant'], 'split':split,
              'horizon_hours':horizon_hours})
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out/'predictions.jsonl', 'w', encoding='utf-8') as f:
            import json
            for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
        write_json(m, out/'metrics.json')
    return m, rows
