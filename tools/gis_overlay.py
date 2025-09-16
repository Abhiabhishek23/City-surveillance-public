#!/usr/bin/env python3
"""
Lightweight GIS overlay utilities using shapely and pyproj to map detections to zone polygons.

Functions:
- load_zones(geojson_path): returns dict of named polygons (projected CRS)
- image_to_world(cx_px, cy_px, image_w, image_h, homography): maps image coords to world lon/lat using a 3x3 homography or simple affine.
- classify_zone(lon, lat, zones): returns zone type string: river_buffer | road_footpath | vending_zone | festival_zone | none

Dependencies (add to requirements if missing): shapely>=2.0, pyproj>=3.5
"""
from typing import Dict, Tuple
import json

from shapely.geometry import shape, Point
from shapely.ops import transform as shp_transform
from pyproj import Transformer


class Zones:
    def __init__(self, crs_epsg: int, polygons: Dict[str, list]):
        self.crs_epsg = crs_epsg
        self._polys = {name: shape(geom) for name, geom in polygons.items()}
        # Prepare transformer from WGS84 to target CRS if needed
        self._to_local = Transformer.from_crs(4326, crs_epsg, always_xy=True)

    def contains(self, lon: float, lat: float) -> str:
        x, y = self._to_local.transform(lon, lat)
        pt = Point(x, y)
        for name, poly in self._polys.items():
            if poly.contains(pt):
                return name
        return 'none'


def load_zones(geojson_path: str, crs_epsg: int = 4326) -> Zones:
    with open(geojson_path, 'r') as f:
        gj = json.load(f)
    # Expect FeatureCollection with properties.name indicating zone type
    polygons = {}
    for feat in gj['features']:
        name = feat.get('properties', {}).get('name') or feat.get('properties', {}).get('zone')
        if not name:
            continue
        polygons[name] = feat['geometry']
    return Zones(crs_epsg=crs_epsg, polygons=polygons)


def image_to_world(cx_px: float, cy_px: float, image_w: int, image_h: int, homography) -> Tuple[float, float]:
    """Map image pixel center to lon/lat using homography (placeholder).
    For real deployments, calibrate per camera. Here, pass-through returns input assuming inputs are lon/lat already.
    """
    # TODO: implement actual mapping when calibration known
    return cx_px, cy_px


def classify_zone(lon: float, lat: float, zones: Zones) -> str:
    return zones.contains(lon, lat)
