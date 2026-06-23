import json
from pathlib import Path

import pandas as pd

from services._legacy_loader import load_module

_mod = load_module("weekly_dispatch.py", "legacy_weekly")

WAREHOUSES = _mod.WAREHOUSES
DAYS = _mod.DAYS
PILOT_DAY_ZIPS = _mod.PILOT_DAY_ZIPS
LOCAL_ASSETS_FILE = _mod.LOCAL_ASSETS_FILE
normalize_plate = _mod.normalize_plate
extract_coordinates = _mod.extract_coordinates
read_orders_excel = _mod.read_orders_excel
detect_day_from_name = _mod.detect_day_from_name
load_routes_from_local_zips = _mod.load_routes_from_local_zips
read_assets = _mod.read_assets
send_orders_and_create_route = _mod.send_orders_and_create_route


def routes_to_summary(routes):
    """Route list for the UI without full order rows (keeps responses small)."""
    out = []
    for r in routes:
        orders_df = r["orders"]
        out.append(
            {
                "day": r["day"],
                "route_name": r["route_name"],
                "source_zip": r["source_zip"],
                "stops": len(orders_df),
                "total_tonnage": round(float(orders_df["TONNAGE"].fillna(0).sum()), 2),
                "total_amount": round(float(orders_df["AMOUNT"].fillna(0).sum()), 2),
            }
        )
    return out


def _records_for_json(df):
    """JSON-safe records; NaN/Inf become null (valid JSON)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def route_orders_to_json(route):
    return _records_for_json(route["orders"])


def routes_from_session(data):
    routes = []
    for r in data:
        orders_df = pd.DataFrame(r["orders"])
        routes.append(
            {
                "day": r["day"],
                "route_name": r["route_name"],
                "source_zip": r["source_zip"],
                "orders": orders_df,
            }
        )
    return routes


def load_weekly_data(base_dir: Path | None = None):
    base_dir = base_dir or Path(__file__).resolve().parent.parent
    routes = load_routes_from_local_zips(base_dir)
    if not routes:
        expected = ", ".join(PILOT_DAY_ZIPS.values())
        raise ValueError(f"No valid route templates found. Expected local zip files: {expected}")
    assets_path = base_dir / LOCAL_ASSETS_FILE
    if not assets_path.exists():
        raise ValueError(f"Assets file not found: {LOCAL_ASSETS_FILE}")
    assets_df = read_assets(assets_path)
    return routes, assets_df


def find_route_in_loaded(routes, day, route_name):
    for r in routes:
        if r["day"] == day and r["route_name"] == route_name:
            return r
    raise ValueError("Selected route not found.")


def assets_to_json(assets_df):
    return _records_for_json(assets_df)


find_route = find_route_in_loaded
