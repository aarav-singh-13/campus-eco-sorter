"""Campus bin / collection-point data and lookup helpers (no web framework code)."""

import json
import math
from pathlib import Path

BINS_FILE = Path(__file__).resolve().parent / "data" / "bins.json"


def load_bins():
    with open(BINS_FILE, encoding="utf-8") as f:
        return json.load(f)["bins"]


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in metres between two GPS points."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bins_for(category=None, lat=None, lon=None, bins=None):
    """Bins that accept `category` (all bins if None), nearest first when a location is given."""
    bins = load_bins() if bins is None else bins
    result = [dict(b) for b in bins if category is None or category in b["accepts"]]

    if lat is not None and lon is not None:
        for b in result:
            b["distance_m"] = round(haversine_m(lat, lon, b["lat"], b["lon"]))
        result.sort(key=lambda b: b["distance_m"])
    return result
