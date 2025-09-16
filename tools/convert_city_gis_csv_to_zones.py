#!/usr/bin/env python3
"""
Convert City GIS CSV (zone listings) to zones.geojson FeatureCollection.

Input CSV columns (as provided):
- Zone, Category, Object_Name, Feature_Type, Approx_Lat_Long, Attributes / Rules & Regulations

Geometry handling:
- If Approx_Lat_Long contains a latitude/longitude like "23.1845° N, 75.7651° E",
  we create a small rectangle buffer around that point (default 100 m for Polygon/Point, 150 m for Polyline).
- Rows with unknown or various locations are skipped (no geometry derivable).

Properties:
- zone_level: red|orange|yellow|green (from Zone)
- name: Object_Name
- category: Category
- feature_type: Feature_Type
- zone_type: optional semantic label derived from category/name (basic mapping); defaults empty
- notes: free text from Attributes / Rules & Regulations

Usage:
  python tools/convert_city_gis_csv_to_zones.py \
    --input "one final list comprising all of them - one final list comprising all of them(maintain no....csv" \
    --output config/zones.geojson
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple


def parse_latlon_cell(cell: str) -> Optional[Tuple[float, float]]:
    """Parse strings like '23.1845° N, 75.7651° E' or '23.14° N, 75.81° E'.
    Returns (lat, lon) in decimal degrees or None if not found.
    """
    if not cell or not isinstance(cell, str):
        return None
    s = cell.strip()
    # Remove surrounding parentheses and 'e.g.' prefix
    s = re.sub(r"^\(e\.g\.,?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\(|\)$", "", s)
    s = s.replace("e.g.", "")
    # Normalize separators
    s = s.replace(";", ",")

    # Regex capturing 'number ° [N|S], number ° [E|W]'
    m = re.search(r"([0-9]+\.?[0-9]*)\s*°?\s*([NS])\s*,\s*([0-9]+\.?[0-9]*)\s*°?\s*([EW])", s, re.IGNORECASE)
    if not m:
        # Try plain decimals separated by comma
        m2 = re.search(r"([0-9]+\.?[0-9]*)\s*,\s*([0-9]+\.?[0-9]*)", s)
        if m2:
            lat = float(m2.group(1))
            lon = float(m2.group(2))
            return lat, lon
        return None
    lat = float(m.group(1))
    ns = m.group(2).upper()
    lon = float(m.group(3))
    ew = m.group(4).upper()
    if ns == 'S':
        lat = -lat
    if ew == 'W':
        lon = -lon
    return lat, lon


def meters_to_degree_offsets(lat_deg: float, meters: float) -> Tuple[float, float]:
    """Approximate conversion from meters to degree deltas at given latitude."""
    # 1 deg latitude ~= 111_320 m
    dlat = meters / 111_320.0
    # 1 deg longitude ~= 111_320 * cos(lat)
    lat_rad = math.radians(lat_deg)
    denom = 111_320.0 * max(0.2, math.cos(lat_rad))  # avoid divide by near poles
    dlon = meters / denom
    return dlat, dlon


def rect_polygon_around(lat: float, lon: float, half_height_m: float, half_width_m: float) -> Dict:
    dlat, dlon_w = meters_to_degree_offsets(lat, half_width_m)
    dlat_h, dlon_h = meters_to_degree_offsets(lat, half_height_m)
    # Use dlat_h for north/south, dlon_w for east/west
    lat_min = lat - dlat_h
    lat_max = lat + dlat_h
    lon_min = lon - dlon_w
    lon_max = lon + dlon_w
    # GeoJSON Polygon coordinates are [ [ [lon,lat], ... closed ] ]
    ring = [
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def normalize_zone_level(z: str) -> Optional[str]:
    if not z:
        return None
    z = z.strip().lower()
    if 'red' in z:
        return 'red'
    if 'orange' in z:
        return 'orange'
    if 'yellow' in z:
        return 'yellow'
    if 'green' in z:
        return 'green'
    return None


def infer_zone_type(category: str, name: str) -> Optional[str]:
    cat = (category or '').strip().lower()
    nm = (name or '').strip().lower()
    # Simple heuristics; zone_level already drives most policy
    if 'route' in cat or 'routes' in cat or 'marg' in nm or 'bridge' in nm:
        return 'procession_route' if 'shahi' in nm else 'corridor'
    if 'ghat' in cat or 'ghat' in nm:
        return 'ghat_area'
    if 'temple' in cat or 'temple' in nm:
        return 'temple_precinct'
    if 'accommodation' in cat or 'tent' in nm or 'akhara' in nm:
        return 'camp_housing'
    if 'transport' in cat and 'parking' in nm:
        return 'satellite_parking'
    if 'media' in cat:
        return 'media_centre'
    if 'vip' in cat:
        return 'vip_secure'
    if 'admin' in cat or 'security' in cat:
        return 'admin_security'
    if 'environmental' in cat or 'waste' in nm or 'stp' in nm:
        return 'infra_environment'
    if 'commercial' in cat or 'market' in nm or 'bazaar' in nm:
        return 'market_zone'
    if 'emergency' in cat:
        return 'emergency_services'
    return None


def convert_csv_to_geojson(csv_path: str, out_path: str) -> Dict:
    feats: List[Dict] = []
    total = 0
    created = 0
    skipped = 0

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            zone = row.get('Zone', '')
            category = row.get('Category', '')
            name = row.get('Object_Name', '')
            ftype = row.get('Feature_Type', '')
            loc = row.get('Approx_Lat_Long', '')
            notes = row.get('Attributes / Rules & Regulations', '')

            level = normalize_zone_level(zone) or ''
            if not level:
                skipped += 1
                continue

            latlon = parse_latlon_cell(loc)
            if not latlon:
                # No concrete location; skip
                skipped += 1
                continue
            lat, lon = latlon

            # Buffer sizes based on feature type
            ftype_l = (ftype or '').strip().lower()
            if 'polyline' in ftype_l or 'line' in ftype_l:
                geom = rect_polygon_around(lat, lon, half_height_m=150, half_width_m=150)
            else:
                # Point or Polygon with single coordinate -> use 100m envelope
                geom = rect_polygon_around(lat, lon, half_height_m=100, half_width_m=100)

            props = {
                'zone_level': level,
                'zone_type': infer_zone_type(category, name),
                'name': name,
                'category': category,
                'feature_type': ftype,
                'notes': notes,
            }
            feats.append({
                'type': 'Feature',
                'properties': props,
                'geometry': geom,
            })
            created += 1

    fc = {'type': 'FeatureCollection', 'features': feats}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

    print(f"Processed rows: {total}, features created: {created}, skipped: {skipped}")
    return fc


def main():
    ap = argparse.ArgumentParser(description='Convert City GIS CSV to zones.geojson')
    ap.add_argument('--input', required=True, help='Path to the CSV file')
    ap.add_argument('--output', default='config/zones.geojson', help='Output GeoJSON path')
    args = ap.parse_args()

    convert_csv_to_geojson(args.input, args.output)


if __name__ == '__main__':
    main()
