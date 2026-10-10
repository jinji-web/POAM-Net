import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def json_records(path):
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as err:
                    raise ValueError(f'{path}:{lineno} invalid JSON: {err}') from err


def number_parameters(model):
    return {
        'all': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def device_from_string(device='auto'):
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def set_trainability(model, freeze_text=False, freeze_vision=False):
    for p in model.text_encoder.parameters():
        p.requires_grad = not freeze_text
    for p in model.image_encoder.parameters():
        p.requires_grad = not freeze_vision


def config_diff(actual, expected):
    return {k: {'actual': actual.get(k), 'expected': v}
            for k, v in expected.items() if actual.get(k) != v}
