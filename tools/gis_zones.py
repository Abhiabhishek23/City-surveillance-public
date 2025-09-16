#!/usr/bin/env python3
import os

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    raise SystemExit("Please install: pip install geopandas shapely")

# Lazy-loaded layers
_ROADS = None
_WATER = None
_RELIGIOUS = None


def _load_layers():
    global _ROADS, _WATER, _RELIGIOUS
    if _ROADS is None and os.path.exists("zones/roads.shp"):
        _ROADS = gpd.read_file("zones/roads.shp")
    if _WATER is None and os.path.exists("zones/water.shp"):
        _WATER = gpd.read_file("zones/water.shp")
    if _RELIGIOUS is None and os.path.exists("zones/religious.shp"):
        _RELIGIOUS = gpd.read_file("zones/religious.shp")


def get_zone(lat: float, lon: float) -> str:
    """Return a coarse zone for the given point using OSM-derived layers."""
    _load_layers()
    p = Point(lon, lat)

    if _WATER is not None and _WATER.contains(p).any():
        return "riverbed"
    if _ROADS is not None and _ROADS.contains(p).any():
        return "road"
    if _RELIGIOUS is not None and _RELIGIOUS.contains(p).any():
        return "ghat"  # simplified placeholder
    return "unknown"
