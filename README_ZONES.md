# GIS Zones for Legal/Conditional Policies

This document explains how to define `config/zones.geojson` to encode where certain activities are permitted or prohibited, powering the Attribute Layer checks during inference.

Key concepts
- Each zone is a GeoJSON Feature with a `zone_type` that matches keys used in `config/attribute_rules.json` under each class's `zones` map.
- Optionally set `camera_id` to limit a zone to a specific camera; omit to apply to all cameras.
- Coordinates: by default these polygons are in pixel space of the camera frames (x,y in pixels). If you prefer real-world coordinates, provide a homography in `config/camera_calibration.json` for that `camera_id`.
- You can also encode the Mela zoning level with `zone_level` property: one of `red`, `orange`, `yellow`, `green`. The rules file supports per-class overrides by level (e.g., hawkers are illegal in `red`).

Schema
- See `config/zones.schema.json` for the formal schema.

Example
- See `config/zones.example.geojson` which includes:
  - `no_hawker_zone` (affects `hawker` → illegal)
  - `silence_zone` (affects `loudspeaker` → illegal)
   - `zone_level` examples: one red (core ghats/pedestrian-only), one orange (managed area)

Calibration (optional)
- `config/camera_calibration.json` supports per-camera homography matrices to map pixel points into the coordinate system used by your zones file. Default is identity.

Workflow to populate zones.geojson
1) Identify policy areas per use case:
   - Encroachment-Free Corridors, vending zones, silence zones, no-pandal zones, approved construction areas, etc.
   - High-level Mela levels: `red` (core ghats, processions), `orange` (sectors/camps), `yellow` (buffer/logistics), `green` (outer ring/city).
2) For per-camera pixel zones:
   - Draw polygons in pixels (e.g., via a simple annotation tool or any image editor that outputs polygon vertices).
   - Add features with `properties.zone_type` = one of: `no_hawker_zone`, `vending_zone`, `silence_zone`, `no_pandal_zone`, `permitted_pandal_zone`, `no_construction_zone`, `approved_construction_zone`, etc.
   - Set `properties.camera_id` to the camera this polygon belongs to.
   - Optionally set `properties.zone_level` to one of the Mela levels to apply general rules even when a specific `zone_type` isn’t present.
3) For geo zones (citywide):
   - Start from official shapefiles/GeoJSON. Use a spatial ETL (see `tools/build_zones.py`) to clip/label zones and export a merged `config/zones.geojson` with only needed properties.
   - Provide homographies for cameras in `config/camera_calibration.json` so detections map into the same CRS.

Validation
- Use a GeoJSON linter or JSON schema validator with `config/zones.schema.json`.

Runtime behavior
- During inference, each detection center point is tested against polygons of matching `camera_id` (or global). If a class has `status=conditional`, its `zones` policy determines if the detection is legal or illegal in that zone.
