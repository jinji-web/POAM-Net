#!/usr/bin/env python3
"""Adapt arbitrary post-level CSV/TSV/JSONL to the documented common POAM JSONL schema.

Fakeddit official TSVs: supply --train, --val, --test, --id-column, --label-column 2_way_label,
--text-column clean_title (or your chosen field) and --image-column. Supply separately
exported comments through --comments. The dataset versions do not share one universal
image/comment export format, so field mapping is explicit and no placeholder comments are created.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split
from poam.utils import json_records


def records(path):
    path = Path(path)
    if path.suffix.lower() == '.jsonl':
        yield from json_records(path)
    elif path.suffix.lower() == '.json':
        obj = json.load(open(path, encoding='utf-8'))
        if not isinstance(obj, list):
            raise ValueError('JSON file must contain an array of post records')
        yield from obj
    else:
        with open(path, encoding='utf-8', newline='') as f:
            yield from csv.DictReader(f, delimiter='\t' if path.suffix.lower() == '.tsv' else ',')


def get(row, field, required=False, default=None):
    x = row.get(field, default) if field else default
    if required and (x is None or str(x).strip() == ''):
        raise ValueError(f'Missing required field {field}; available={list(row)[:25]}')
    return x


def join_comments(path, id_field, text_field, time_field):
    table = defaultdict(list)
    if path:
        for row in records(path):
            pid = str(get(row, id_field, required=True))
            txt = get(row, text_field, default='')
            if txt:
                c = {'text':str(txt)}
                stamp = get(row, time_field)
                if stamp is not None and str(stamp).strip():c['created_ts'] = stamp
                table[pid].append(c)
    return table


def convert(path, args, comments):
    result = []
    for row in records(path):
        pid = str(get(row, args.id_column, required=True))
        raw_label = str(get(row, args.label_column, required=True))
        if raw_label not in args.label_map:
            raise ValueError(f'Label {raw_label!r} has no declared conversion; expected {args.label_map}')
        val = int(args.label_map[raw_label])
        new = {'id':pid, 'text':str(get(row,args.text_column,default='') or ''),
               'image':str(get(row,args.image_column, required=True)),
               'label':val, 'comments':comments.get(pid, [])}
        for key,col in [('post_ts',args.post_time_column),('source_id',args.source_column),
                        ('event_id',args.event_column)]:
            x = get(row, col)
            if x is not None and str(x).strip(): new[key]=x
        result.append(new)
    if len(set(x['id'] for x in result)) != len(result):
        raise ValueError(f'{path}: duplicate post IDs; deduplicate before splitting')
    return result


def write(rows,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        for item in rows:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    print(f'{path}: {len(rows)} posts, {sum(len(x["comments"]) for x in rows)} comments')

p=argparse.ArgumentParser(description='Prepare official post splits or Weibo stratified 70/10/20')
p.add_argument('--train', help='Fakeddit train post table')
p.add_argument('--val', help='Fakeddit validation post table')
p.add_argument('--test', help='Fakeddit test post table')
p.add_argument('--all', help='Weibo single post table, stratify 70/10/20 using seed 42')
p.add_argument('--output', required=True, help='output folder for train.jsonl, val.jsonl, test.jsonl')
p.add_argument('--comments', help='Optional separate comment CSV/TSV/JSONL joined by post ID')
p.add_argument('--id-column', required=True)
p.add_argument('--label-column', required=True)
p.add_argument('--text-column', required=True)
p.add_argument('--image-column', required=True)
p.add_argument('--post-time-column')
p.add_argument('--source-column')
p.add_argument('--event-column')
p.add_argument('--comment-id-column', default='post_id')
p.add_argument('--comment-text-column', default='text')
p.add_argument('--comment-time-column', default='created_ts')
p.add_argument('--label-map', default='0:0,1:1',
    help='EXPLICIT raw:target mapping, where target 0=REAL and 1=FAKE; e.g. real:0,fake:1')
p.add_argument('--seed', type=int, default=42)
a=p.parse_args()
a.label_map = dict(x.split(':',1) for x in a.label_map.split(','))
if set(a.label_map.values()) - {'0','1'}:
    p.error('Mapped label values must be 0=real or 1=fake')
if a.all and any((a.train,a.val,a.test)):
    p.error('Choose either --all or three explicit split files')
if not a.all and not all((a.train,a.val,a.test)):
    p.error('Supply --all or --train/--val/--test')
comments=join_comments(a.comments,a.comment_id_column,a.comment_text_column,a.comment_time_column)
out=Path(a.output)
if a.all:
    all_rows=convert(a.all,a,comments)
    y=[x['label'] for x in all_rows]
    rest,test=train_test_split(all_rows, test_size=.2, random_state=a.seed, stratify=y)
    train,val=train_test_split(rest, test_size=.125, random_state=a.seed,
                               stratify=[x['label'] for x in rest])
else:
    train=convert(a.train,a,comments)
    val=convert(a.val,a,comments)
    test=convert(a.test,a,comments)
if (set(x['id'] for x in train) & set(x['id'] for x in val)
    or set(x['id'] for x in train) & set(x['id'] for x in test)
    or set(x['id'] for x in val) & set(x['id'] for x in test)):
    raise ValueError('The provided splits overlap by post ID')
for name,rows in [('train',train),('val',val),('test',test)]:write(rows,out/f'{name}.jsonl')
