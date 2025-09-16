# rule_engine.py
# Static rule engine mapping detections to LEGAL/ILLEGAL (coarse)

LEGAL_CLASSES = {
    "vehicle_car",
    "vehicle_bike",
    "vehicle_auto",
    "vehicle_truck_bus",
    "vehicle_tractor",
    "emergency_vehicle",
    "pedestrian_walking",
    "pedestrian_queue",
    "pandal_tent",
    "stage_platform",
    "idol_statue",
    "flag_banner",
    "vendor_cart_thela",
    "food_stall",
    "kiosk_cabin",
    "barricade_fence",
    "portable_toilet",
    "dustbin_dump",
    "garbage_heap",
    "water_tank_tap",
    "bus_shelter_signage",
    "cctv_tower_drone_station",
}

ILLEGAL_CLASSES = {
    "beggar_squatter",
    "religious_marker",
    "shop_house",
    "open_fire_stove",
    "sand_heap",
    "sewage_pipe_drain",
    "boat",
    "hoarding_poster_banner",
}


def classify_detection(class_name: str) -> str:
    if class_name in LEGAL_CLASSES:
        return "LEGAL"
    if class_name in ILLEGAL_CLASSES:
        return "ILLEGAL"
    return "UNKNOWN"
