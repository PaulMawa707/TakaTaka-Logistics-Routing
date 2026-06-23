from services._legacy_loader import load_module

_mod = load_module("test.py", "legacy_optimized")

WAREHOUSES = _mod.WAREHOUSES
normalize_plate = _mod.normalize_plate
extract_truck_number_from_text = _mod.extract_truck_number_from_text
extract_coordinates = _mod.extract_coordinates
read_asset_id_from_excel = _mod.read_asset_id_from_excel
read_excel_to_df = _mod.read_excel_to_df
send_orders_and_create_route = _mod.send_orders_and_create_route
process_multiple_excels = _mod.process_multiple_excels
