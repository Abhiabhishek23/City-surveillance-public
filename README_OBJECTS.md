# Object-first Dataset (Phase 1)

Train YOLO on visible object/structure categories; decide Legal/Illegal later using GIS rules.

Classes (21):
0 permanent_shop_house, 1 temple_shrine, 2 ghat_steps, 3 pandal_tent, 4 stage_platform,
5 vendor_cart_thela, 6 food_stall, 7 kiosk_cabin, 8 vehicle_car, 9 vehicle_bike,
10 vehicle_auto, 11 vehicle_truck_bus, 12 boat, 13 barricade_fence, 14 portable_toilet,
15 dustbin_dump, 16 garbage_heap, 17 sand_heap, 18 open_fire_stove, 19 open_ghat, 20 open_road

Train with `dataset/objects_data.yaml`.

Why this design:
- Legality depends on location/permits; detectors should learn appearance only.
- Post-process detections with zone polygons and simple rules.

How to label:
- Use the Annotation UI. Set Label Mode = Objects, choose the class, draw boxes, save to JSONL.
- Generate YOLO labels using the Objects button (writes to `dataset/labels/train`).
- Train YOLO and iterate.

Map to Legal/Illegal later:
- Use `tools/object_legality_overlay.py` with your `dataset/zones.geojson`.
- Rules examples:
  - permanent_shop_house/temple_shrine/ghat_steps inside river_buffer → Permanent_Illegal; else with permit → Permanent_Legal
  - pandal_tent/stage_platform/kiosk_cabin in festival zone + permit → Temporary_Legal; on road/ghat/riverbed → Temporary_Illegal
  - vendor_cart_thela/food_stall in vending_zone → Temporary_Legal; on road/ghat → Temporary_Illegal
  - vehicles in designated parking → Legal; on ghats/roads/footpaths → Temporary_Illegal
  - garbage_heap/sand_heap/open_fire_stove in protected zones → Temporary_Illegal

Search queries (examples):
- "pandal tent festival India", "street vendor cart India", "food stall fair India"
- "illegal stalls riverbed India", "unauthorized temple on ghat", "sand mining riverbed India"
- "garbage dumping on ghat", "festival barricades crowd control", "portable toilets festival India"

Tip: Start with 300–500 images across classes; grow with hard examples.
