#!/usr/bin/env python3
import os
import json
import argparse
import yaml

def collect_txt_files(labels_root):
    files = []
    for root, _dirs, fns in os.walk(labels_root):
        for f in fns:
            if f.endswith('.txt'):
                files.append(os.path.join(root, f))
    return files

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', required=True, help='root of labels directory (recursively searched)')
    p.add_argument('--data_yaml', required=True, help='path to data.yaml containing names list')
    p.add_argument('--out', required=True, help='output JSON path')
    args = p.parse_args()

    with open(args.data_yaml) as f:
        data = yaml.safe_load(f)
    names = data.get('names')
    # names may be a dict {id:name} or list; normalize to list ordered by id
    if isinstance(names, dict):
        class_list = [names[i] for i in sorted(names.keys())]
    else:
        class_list = list(names)

    freq = {c: 0 for c in class_list}

    for txt in collect_txt_files(args.labels):
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls_id = int(parts[0])
                except Exception:
                    continue
                if 0 <= cls_id < len(class_list):
                    freq[class_list[cls_id]] += 1

    with open(args.out, 'w') as f:
        json.dump(freq, f, indent=2)
    print('[OK] Wrote', args.out)

if __name__ == '__main__':
    main()
