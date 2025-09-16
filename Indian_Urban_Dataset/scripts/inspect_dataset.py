#!/usr/bin/env python3
import csv
from pathlib import Path

root = Path('.').resolve()
ann = root / 'annotations'
train_csv = ann / 'train.csv'
val_csv = ann / 'val.csv'
test_csv = ann / 'test.csv'

counts = {'train': {}, 'val': {}, 'test': {}}
splits = [('train', train_csv), ('val', val_csv)]
if test_csv.exists():
    splits.append(('test', test_csv))
for split, csv_path in splits:
    with csv_path.open('r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p, idx, name = line.strip().split(',', 2)
            counts[split][name] = counts[split].get(name, 0) + 1

print('Train classes:', len(counts['train']), 'images:', sum(counts['train'].values()))
print('Val classes:', len(counts['val']), 'images:', sum(counts['val'].values()))
if counts['test']:
    print('Test classes:', len(counts['test']), 'images:', sum(counts['test'].values()))

# Top-10 classes by count in train
top_train = sorted(counts['train'].items(), key=lambda x: -x[1])[:10]
print('\nTop-10 train classes:')
for cls, n in top_train:
    print(f'{cls}: {n}')

# Check overlap in class names between splits
train_only = set(counts['train']) - set(counts['val'])
val_only = set(counts['val']) - set(counts['train'])
print('\nClasses only in train:', len(train_only))
print('Classes only in val:', len(val_only))
if counts['test']:
    test_only = set(counts['test']) - set(counts['train']) - set(counts['val'])
    print('Classes only in test:', len(test_only))
