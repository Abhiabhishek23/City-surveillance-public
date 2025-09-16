#!/usr/bin/env python3
import requests, os, sys
ROOT=os.path.dirname(os.path.abspath(__file__))
base='http://127.0.0.1:8000'
try:
    r=requests.get(base+'/alerts', timeout=5)
    print('GET /alerts', r.status_code)
    data=r.json()
    if data:
        img_url=data[0].get('image_url')
        print('First alert image_url:', img_url)
        if img_url:
            rr=requests.get(img_url, timeout=5)
            print('GET image', rr.status_code, 'content-length', len(rr.content))
    else:
        print('No alerts present')
except Exception as e:
    print('Error:', e)
    sys.exit(1)
