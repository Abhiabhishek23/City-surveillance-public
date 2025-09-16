#!/usr/bin/env python3
"""Dynamic Layer: aggregate per-zone metrics from attribute_layer outputs.
Writes JSON summaries to dynamic_layer/snapshot/.
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
ATTR = ROOT / 'attribute_layer'
OUT = ROOT / 'dynamic_layer' / 'snapshot'
OUT.mkdir(parents=True, exist_ok=True)

def main():
    metrics = defaultdict(lambda: defaultdict(int))
    for jf in ATTR.glob('*.json'):
        data = json.loads(jf.read_text())
        zone = 'global'  # if zones per detection were computed, use that here
        for it in data.get('items', []):
            obj = it.get('object'); status = it.get('status')
            if obj == 'person': metrics[zone]['people'] += 1
            if obj in ('car','bus','truck','motorbike','bicycle'): metrics[zone]['vehicles'] += 1
            if status == 'illegal': metrics[zone]['illegal'] += 1
            if obj in ('hawker','tent_pandal','idol_statue'): metrics[zone]['encroachments'] += 1
            if obj in ('garbage_pile','open_fire','smoke'): metrics[zone]['pollution'] += 1
    out = [{'zone': z, **vals} for z, vals in metrics.items()]
    (OUT/'snapshot.json').write_text(json.dumps(out, indent=2))
    print('[DONE] Wrote dynamic snapshot to', OUT/'snapshot.json')

if __name__ == '__main__':
    main()
