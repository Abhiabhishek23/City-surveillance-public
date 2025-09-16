#!/usr/bin/env python3
"""
Build Zones Utility

Merges city-provided GIS data (GeoJSON or Shapefiles via GeoPandas if installed) into
config/zones.geojson with standardized properties expected by the inference pipeline.

Usage examples:
  # From one or more input GeoJSON files, map categories to zone_type
  python tools/build_zones.py \
    --input data/geo/no_hawker_wards.geojson data/geo/silence_zones.geojson \
    --map "name:regex(no hawker)->zone_type:no_hawker_zone" \
    --map "name:regex(silence)->zone_type:silence_zone" \
    --output config/zones.geojson

Notes:
- If GeoPandas is available, you may pass Shapefile paths as inputs (it will auto-read).
- The --map arguments support a simple mapping mini-language to derive `zone_type` from properties.
- This script is intentionally simple; customize for your GIS attribute schema as needed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

try:
    import geopandas as gpd  # type: ignore
    HAVE_GPD = True
except Exception:
    HAVE_GPD = False


def read_geo(path: str) -> Dict[str, Any]:
    if path.lower().endswith(('.geojson', '.json')):
        with open(path, 'r') as f:
            return json.load(f)
    if HAVE_GPD:
        gdf = gpd.read_file(path)
        return json.loads(gdf.to_json())
    raise SystemExit(f"Unsupported input format without GeoPandas: {path}")


def apply_maps(props: Dict[str, Any], maps: List[str]) -> Dict[str, Any]:
    out = dict(props)
    for rule in maps:
        # Format: "field:regex(<pattern>)->zone_type:<value>"
        m = re.match(r"^(?P<field>[^:]+):regex\((?P<pat>.+)\)->zone_type:(?P<val>.+)$", rule)
        if not m:
            continue
        fld, pat, val = m.group('field'), m.group('pat'), m.group('val')
        v = str(props.get(fld, ''))
        if re.search(pat, v, flags=re.I):
            out['zone_type'] = val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', nargs='+', required=True, help='One or more GeoJSON/Shapefile inputs')
    ap.add_argument('--map', dest='maps', action='append', default=[], help='Mapping rule to set zone_type from a property via regex')
    ap.add_argument('--camera-id', default=None, help='Optional camera_id to stamp on all features')
    ap.add_argument('--output', default='config/zones.geojson', help='Output GeoJSON path')
    args = ap.parse_args()

    all_feats: List[Dict[str, Any]] = []
    for p in args.input:
        gj = read_geo(p)
        for feat in gj.get('features', []):
            if not feat.get('geometry'):
                continue
            props = feat.get('properties', {}) or {}
            props = apply_maps(props, args.maps)
            if 'zone_type' not in props:
                # Skip features without a resolved zone_type
                continue
            if args.camera_id:
                props['camera_id'] = args.camera_id
            all_feats.append({'type': 'Feature', 'properties': props, 'geometry': feat['geometry']})

    out = {'type': 'FeatureCollection', 'features': all_feats}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f)
    print(f"Wrote {args.output} with {len(all_feats)} features")


if __name__ == '__main__':
    main()
