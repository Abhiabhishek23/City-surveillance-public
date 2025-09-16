# rule_engine_zones.py
# Context-aware rule engine (zone + permit) for Mahakumbh encroachment

from typing import Dict

ZONES = {
    "road",
    "footpath",
    "ghat",
    "riverbed",
    "floodplain",
    "hawking_zone",
    "licensed_market",
    "eco_sensitive",
    "heritage_protected",
}

LEGAL_ZONES = {
    "vehicle_car": {"road"},
    "vehicle_bike": {"road"},
    "vehicle_auto": {"road"},
    "vehicle_truck_bus": {"road"},
    "vehicle_tractor": {"road"},
    "emergency_vehicle": {"road"},
    "pedestrian_walking": {"road", "footpath", "ghat"},
    "pedestrian_queue": {"ghat", "temple"},
    "pandal_tent": {"licensed_market", "ghat"},
    "stage_platform": {"licensed_market", "ghat"},
    "idol_statue": {"ghat"},
    "flag_banner": {"licensed_market", "hawking_zone"},
    "vendor_cart_thela": {"hawking_zone"},
    "food_stall": {"hawking_zone", "licensed_market"},
    "kiosk_cabin": {"licensed_market"},
    "barricade_fence": {"road", "ghat"},
    "portable_toilet": {"ghat", "licensed_market"},
    "dustbin_dump": {"ghat", "licensed_market", "hawking_zone"},
    "garbage_heap": {"licensed_market", "hawking_zone"},
    "water_tank_tap": {"ghat", "licensed_market"},
    "bus_shelter_signage": {"road"},
    "cctv_tower_drone_station": {"ghat", "licensed_market"},
    "boat": {"river"},
}


REQUIRES_LICENSE = {"pandal_tent", "vendor_cart_thela", "food_stall", "kiosk_cabin"}


def classify_detection(class_name: str, context: Dict) -> str:
    zone = context.get("zone", None)
    licensed = context.get("licensed", False)

    if class_name in ("open_ghat", "open_road"):
        return "LEGAL"  # background

    allowed = LEGAL_ZONES.get(class_name)
    if not allowed:
        return "UNKNOWN"

    if zone in allowed:
        if class_name in REQUIRES_LICENSE and not licensed:
            return "ILLEGAL"
        return "LEGAL"

    return "ILLEGAL"
