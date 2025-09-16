#!/usr/bin/env python3
import os, json
ROOT=os.path.dirname(os.path.dirname(__file__))
log_path=os.path.join(ROOT,'logs','client_frame_log.jsonl')
if not os.path.exists(log_path):
    print('No client_frame_log.jsonl found')
    raise SystemExit(1)

stats={}
with open(log_path) as f:
    for line in f:
        try:
            obj=json.loads(line)
        except Exception:
            continue
        cam=obj.get('camera_id')
        stats.setdefault(cam, {'frames':0,'detections':0,'by_class':{}})
        stats[cam]['frames']+=1
        dets=obj.get('detections',[])
        stats[cam]['detections']+=len(dets)
        for d in dets:
            cl=d.get('class')
            stats[cam]['by_class'][cl]=stats[cam]['by_class'].get(cl,0)+1

import pprint
pprint.pprint(stats)
