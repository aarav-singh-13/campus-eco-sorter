"""Campus bin locations and lookup helpers (no web framework code).

data/bins.json lists LOCATIONS, each with its physical bins (blue / green / ewaste / community).
Which waste categories a bin takes comes from data/categories.json ("bin" field per category),
so the blue/green rules live in one place.
"""

import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
BINS_FILE = DATA / "bins.json"
CATEGORIES_FILE = DATA / "categories.json"


def _bin_types():
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        cfg = json.load(f)
    types = {k: dict(v) for k, v in cfg["bin_types"].items()}
    for t in types.values():
        t.setdefault("accepts", [])
    for cat_id, cat in cfg["categories"].items():
        if cat.get("bin") in types and cat_id not in types[cat["bin"]]["accepts"]:
            types[cat["bin"]]["accepts"].append(cat_id)
    return types


def load_bins():
    """Locations with each bin's name/colour/accepted categories filled in."""
    types = _bin_types()
    with open(BINS_FILE, encoding="utf-8") as f:
        locations = json.load(f)["locations"]
    out = []
    for loc in locations:
        loc = dict(loc)
        bins = []
        for b in loc["bins"]:
            t = types[b["type"]]
            bins.append({**b, "name": t["name"], "label": t["label"], "color": t["color"], "accepts": t["accepts"]})
        loc["bins"] = bins
        loc["accepts"] = sorted({c for b in bins for c in b["accepts"]})
        loc["verified_any"] = any(b["verified"] for b in bins)
        out.append(loc)
    return out


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in metres between two GPS points."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bins_for(category=None, lat=None, lon=None, bins=None):
    """Locations with a bin for `category` (all if None).

    Each result gets `match_verified`: whether a bin it has for this category is verified.
    Sorted nearest first when a location is given, otherwise verified locations first.
    """
    locations = load_bins() if bins is None else bins
    result = []
    for loc in locations:
        if category is not None and category not in loc["accepts"]:
            continue
        loc = dict(loc)
        if category is None:
            loc["match_verified"] = loc["verified_any"]
        else:
            loc["match_verified"] = any(b["verified"] and category in b["accepts"] for b in loc["bins"])
        result.append(loc)

    if lat is not None and lon is not None:
        for loc in result:
            loc["distance_m"] = round(haversine_m(lat, lon, loc["lat"], loc["lon"]))
        result.sort(key=lambda l: l["distance_m"])
    else:
        result.sort(key=lambda l: not l["match_verified"])  # stable: keeps file order within groups
    return result
