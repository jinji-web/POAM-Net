"""Data contract: post-level JSONL records and strict optional temporal censoring.

Each row: {"id":str, "text":str, "image":str, "label":0|1,
           "post_ts":optional ISO/epoch,
           "comments":[{"text":str,"created_ts":optional ISO/epoch}, ...],
           "source_id":optional str, "event_id":optional str}
0=real and 1=fake, as stated in the manuscript. Input order is preserved.
"""
import datetime as dt
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .utils import json_records


def parse_timestamp(ts):
    if ts in (None, ''):
        return None
    if isinstance(ts, (float, int)):
        return float(ts)
    s = str(ts).strip()
    try:
        return float(s)
    except ValueError:
        pass
    try:
        datetime = dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
        if datetime.tzinfo is None:
            raise ValueError('Naive timestamps are forbidden; add a timezone')
        return datetime.timestamp()
    except ValueError as err:
        raise ValueError(f'Unsupported datetime {ts!r}') from err


def normalize_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'https?://\S+', ' [URL] ', str(text or ''))).strip()


def comment_list(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [({'text': str(x)} if not isinstance(x, dict) else x) for x in raw]
    raise TypeError('comments must be a list of text strings or dictionaries')


def select_comments(item, max_comments=20, horizon_hours=None, order='earliest'):
    rows = comment_list(item.get('comments'))
    if horizon_hours is not None:
        start = parse_timestamp(item.get('post_ts'))
        if start is None:
            raise ValueError(f"Post {item.get('id')} lacks post_ts in strict temporal evaluation")
        retained = []
        for c in rows:
            when = parse_timestamp(c.get('created_ts'))
            if when is None:  # unknown timestamps are ineligible, not assumed early
                continue
            if start <= when <= start + float(horizon_hours) * 3600:
                retained.append(c)
        rows = retained
    if order == 'earliest':
        # If all timestamps are known, stable-sort chronologically; otherwise preserve archive order.
        parsed = [parse_timestamp(x.get('created_ts')) for x in rows]
        if all(x is not None for x in parsed):
            rows = [c for _, c in sorted(zip(parsed, rows), key=lambda x: x[0])]
    elif order != 'archive':
        raise ValueError(f'unknown comment order: {order}')
    return [normalize_text(c.get('text')) for c in rows if normalize_text(c.get('text'))][:max_comments]


class NewsDataset(Dataset):
    def __init__(self, path, tokenizer, image_root='', max_news_len=256, max_comment_len=64,
                 max_comments=20, image_size=224, training=False, horizon_hours=None,
                 comment_order='earliest', allow_missing_images=False):
        self.items = list(json_records(path))
        self.tok = tokenizer
        self.root = Path(image_root or '.')
        self.max_news_len = int(max_news_len)
        self.max_comment_len = int(max_comment_len)
        self.max_comments = int(max_comments)
        self.horizon = horizon_hours
        self.order = comment_order
        self.allow_missing_images = allow_missing_images
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size) if training else T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip() if training else T.Lambda(lambda im: im),
            T.ColorJitter(brightness=0.10, contrast=0.10) if training else T.Lambda(lambda im: im),
            T.ToTensor(), normalize,
        ])
        ids = [str(x['id']) for x in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate post ids in {path}; each post must be unique')
        for x in self.items:
            if x.get('label') not in (0, 1):
                raise ValueError(f"{x.get('id')} has invalid label; expected 0=real,1=fake")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        image_path = Path(row.get('image') or '')
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image_available = image_path.is_file()
        if not image_available and not self.allow_missing_images:
            raise FileNotFoundError(f"{row['id']}: image not found: {image_path}")
        with Image.open(image_path) if image_available else Image.new('RGB', (224, 224)) as im:
            image = self.transform(im.convert('RGB'))
        comments = select_comments(row, self.max_comments, self.horizon, self.order)
        return {'id': str(row['id']), 'text': normalize_text(row.get('text')),
                'comments': comments, 'comment_count': len(comments), 'image': image,
                'image_present': int(image_available), 'label': int(row['label']),
                'original_comment_count': len(comment_list(row.get('comments')))}


class BatchCollator:
    def __init__(self, tokenizer, news_len=256, comment_len=64, max_comments=20):
        self.tok = tokenizer
        self.news_len = news_len
        self.comment_len = comment_len
        self.max_comments = max_comments

    def __call__(self, rows):
        B, M = len(rows), self.max_comments
        enc = self.tok([r['text'] for r in rows], padding='max_length', truncation=True,
                       max_length=self.news_len, return_tensors='pt')
        flat = [txt for r in rows for txt in (r['comments'][:M] + [''] * (M-len(r['comments'][:M])))]
        # Tokenizer work is bounded by B*M; embeddings of absent comments are never encoded by the network.
        comments = self.tok(flat, padding='max_length', truncation=True,
                            max_length=self.comment_len, return_tensors='pt')
        comment_mask = torch.zeros(B, M, dtype=torch.bool)
        for i, r in enumerate(rows):
            comment_mask[i, :min(len(r['comments']), M)] = True
        return {
            'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'],
            'pixel_values': torch.stack([r['image'] for r in rows]),
            'image_present': torch.tensor([r['image_present'] for r in rows], dtype=torch.bool),
            'comment_input_ids': comments['input_ids'].reshape(B, M, -1),
            'comment_attention_mask': comments['attention_mask'].reshape(B, M, -1),
            'comment_mask': comment_mask,
            'labels': torch.tensor([r['label'] for r in rows], dtype=torch.float32),
            'ids': [r['id'] for r in rows],
            'comment_counts': torch.tensor([r['comment_count'] for r in rows]),
            'original_comment_counts': torch.tensor([r['original_comment_count'] for r in rows]),
        }


def move_to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def check_split_ids(paths):
    sets = [set(str(x['id']) for x in json_records(p)) for p in paths]
    counts = {}
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            counts[f'{i}-{j}'] = len(sets[i] & sets[j])
    if any(counts.values()):
        raise ValueError(f'Post IDs shared across partitions: {counts}')
    return counts
