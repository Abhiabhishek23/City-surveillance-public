#!/usr/bin/env python3
"""
Map object detections to Legal/Illegal classes using GIS zones and simple rules.

Input: CSV with columns [image, lon, lat, class_name, (optional) permit_status]
Zones: dataset/zones.geojson with features having properties.name in:
    river_buffer, road_footpath, vending_zone, festival_zone, natural_area

Output: CSV with added columns [zone, permanence, top_class]
"""
import argparse
import json
import os
import pandas as pd
from shapely.geometry import Point, shape
import json

LEGAL = {
    'Permanent_Legal', 'Temporary_Legal', 'Natural_Area'
}


def decide(permanence: str, zone: str, permit: str) -> str:
    if zone == 'natural_area':
        return 'Natural_Area'
    if zone in ('river_buffer', 'road_footpath'):
        return 'Permanent_Illegal' if permanence == 'permanent' else 'Temporary_Illegal'
    if zone in ('vending_zone', 'festival_zone'):
        if permit == 'approved':
            return 'Temporary_Legal' if permanence == 'temporary' else 'Permanent_Legal'
        return 'Temporary_Illegal' if permanence == 'temporary' else 'Permanent_Illegal'
    # outside all zones
    if permanence == 'permanent':
        return 'Permanent_Legal' if permit == 'approved' else 'Permanent_Illegal'
    return 'Temporary_Illegal'


def permanence_for_object(cls: str) -> str:
    permanent = {
        'permanent_shop_house','temple_shrine','ghat_steps','kiosk_cabin'
    }
    temporary = {
        'pandal_tent','stage_platform','vendor_cart_thela','food_stall',
        'barricade_fence','portable_toilet','dustbin_dump','garbage_heap',
        'sand_heap','open_fire_stove'
    }
    if cls in permanent:
        return 'permanent'
    if cls in temporary:
        return 'temporary'
    # vehicles/boats/open areas default
    return 'temporary'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detections', required=True, help='CSV with columns: image, lon, lat, class_name, permit_status(optional)')
    ap.add_argument('--zones', default='dataset/zones.geojson')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.detections)
    # Load zones using plain GeoJSON parsing
    with open(args.zones, 'r') as f:
        gj = json.load(f)
    zones = []  # list of (zone_name, shapely_geom)
    feats = gj.get('features', [])
    for feat in feats:
        props = feat.get('properties', {})
        name = props.get('name') or props.get('zone') or 'none'
        geom = shape(feat.get('geometry'))
        zones.append((name, geom))

    zone_col = []
    perm_col = []
    top_col = []
    for _, r in df.iterrows():
        pt = Point(float(r['lon']), float(r['lat']))
        zname = 'none'
        for name, geom in zones:
            if geom.contains(pt):
                zname = name
                break
        zone_col.append(zname)
        cls = r['class_name']
        # Background classes are Natural_Area regardless of zone
        if cls in ('open_ghat','open_road'):
            perm_col.append('temporary')
            top_col.append('Natural_Area')
            continue
        perm = permanence_for_object(cls)
        perm_col.append(perm)
        permit = str(r.get('permit_status', 'unknown'))
        top_col.append(decide(perm, zname, permit))

    out_df = df.copy()
    out_df['zone'] = zone_col
    out_df['permanence'] = perm_col
    out_df['top_class'] = top_col
    out_df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")


if __name__ == '__main__':
    main()
