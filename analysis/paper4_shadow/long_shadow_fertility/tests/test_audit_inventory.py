"""The audit inventory should find the known paper4_shadow assets."""
from analysis.paper4_shadow.long_shadow_fertility.audit_inventory import inventory

def test_inventory_finds_campop():
    inv = inventory()
    assert any("campop_england_annual.csv" in str(p) for p in inv["data_csvs"])

def test_inventory_finds_modera_country_panel():
    inv = inventory()
    assert any("modera_country_monthly" in str(p).lower() for p in inv["data_parquets"])

def test_inventory_finds_volcanic_script():
    inv = inventory()
    assert any("volcanic_event_study.py" in str(p) for p in inv["scripts"])
