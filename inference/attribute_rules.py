#!/usr/bin/env python3
"""Attribute Layer: tag detections with status based on zones/permits.
Input: base_layer/*.json
Zones: config/zones.geojson (polygons: no_hawker, no_parking, no_construction, footpath, riverbank, etc.)
Output: attribute_layer/*.json
"""
import json
from pathlib import Path
import argparse
from shapely.geometry import Point, Polygon, shape
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'base_layer'
OUT = ROOT / 'attribute_layer'
OUT.mkdir(exist_ok=True)

def load_zones(zones_path: Path):
    if not zones_path.exists():
        return None
    gdf = gpd.read_file(zones_path)
    return gdf

def classify(det, zones_gdf):
    """Return status and notes. Uses pixel coords unless lat/lon present.
    For now, we treat all detections as points by bbox center.
    """
    cls = det.get('object')
    xyxy = det.get('bbox_xyxy')
    status = 'legal'; notes = ''
    if not zones_gdf is None and xyxy:
        x1,y1,x2,y2 = xyxy
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        pt = Point(cx, cy)
        # simple rule examples; enhance as needed
        def inside(zone_kind):
            z = zones_gdf[zones_gdf['kind']==zone_kind]
            return any(shape(geom).contains(pt) for geom in z.geometry)

        if cls in ('hawker','stall_cart'):
            if inside('no_hawker'):
                status, notes = 'illegal', 'Hawker in No-Hawker Zone'
            else:
                status, notes = 'conditional', 'Hawker requires license'
        elif cls in ('car','bus','truck','motorbike','bicycle'):
            if inside('footpath'):
                status, notes = 'illegal', 'Vehicle on footpath'
            elif inside('no_parking'):
                status, notes = 'illegal', 'Vehicle in No-Parking Zone'
        elif cls in ('idol_statue','tent_pandal'):
            if inside('no_construction'):
                status, notes = 'illegal', 'Structure in No-Construction Zone'
            else:
                status, notes = 'conditional', 'Requires permit'
        elif cls in ('garbage_pile','open_fire','smoke'):
            if inside('riverbank'):
                status, notes = 'illegal', f'{cls} near riverbank'
            else:
                status, notes = 'conditional', f'{cls} requires review'

    return status, notes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zones', default=str(ROOT/'config/zones.geojson'))
    args = ap.parse_args()
    zones_gdf = load_zones(Path(args.zones))
    for jf in BASE.glob('*.json'):
        data = json.loads(jf.read_text())
        out = {'source': data.get('image') or data.get('video'), 'items': []}
        for det in data.get('detections', []):
            status, notes = classify(det, zones_gdf)
            out['items'].append({**det, 'status': status, 'notes': notes})
        (OUT / jf.name).write_text(json.dumps(out, indent=2))
    print('[DONE] Wrote attribute_layer JSON for', len(list(BASE.glob('*.json'))), 'items')

if __name__ == '__main__':
    main()
