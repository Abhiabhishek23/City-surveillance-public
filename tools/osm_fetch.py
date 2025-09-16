#!/usr/bin/env python3
import os

try:
    import osmnx as ox
    import geopandas as gpd
except Exception as e:
    raise SystemExit("Please install: pip install osmnx geopandas")


def fetch_osm_data(city: str = "Ujjain, India"):
    print(f"Fetching OSM data for {city}...")

    os.makedirs("zones", exist_ok=True)

    roads = ox.graph_from_place(city, network_type="drive")
    gdf_roads = ox.graph_to_gdfs(roads, nodes=False, edges=True)
    gdf_roads.to_file("zones/roads.shp")

    water = ox.geometries_from_place(city, tags={"natural": "water"})
    if not water.empty:
        water.to_file("zones/water.shp")

    parks = ox.geometries_from_place(city, tags={"leisure": "park"})
    if not parks.empty:
        parks.to_file("zones/parks.shp")

    temples = ox.geometries_from_place(city, tags={"amenity": "place_of_worship"})
    if not temples.empty:
        temples.to_file("zones/religious.shp")

    print("Saved to zones/ directory")


if __name__ == "__main__":
    fetch_osm_data("Ujjain, India")
    fetch_osm_data("Indore, India")
