"""Catalog the paper4_shadow assets relevant to the Long Shadow on Fertility
extension. Writes AUDIT.md and returns the inventory as a dict.
"""
from __future__ import annotations
from pathlib import Path
from typing import TypedDict

ROOT = Path("/Volumes/BIGDATA/HYDE35")
P4S = ROOT / "analysis" / "paper4_shadow"
DATA = ROOT / "analysis" / "data"


class Inventory(TypedDict):
    scripts: list[Path]
    data_csvs: list[Path]
    data_parquets: list[Path]
    figures: list[Path]


SCRIPT_KEYWORDS = (
    "campop", "volcanic", "rolling", "long_shadow",
    "structural", "preventive_positive", "tambora", "sigl",
)
DATA_KEYWORDS = (
    "campop", "modera", "cru_country", "country_climate",
    "country_seasonality", "volcanic", "hyde_era5",
)


def inventory() -> Inventory:
    inv: Inventory = {"scripts": [], "data_csvs": [], "data_parquets": [], "figures": []}
    for p in sorted(P4S.glob("*.py")):
        if any(kw in p.name.lower() for kw in SCRIPT_KEYWORDS):
            inv["scripts"].append(p)
    for p in sorted(DATA.rglob("*.csv")):
        if any(kw in p.name.lower() for kw in DATA_KEYWORDS):
            inv["data_csvs"].append(p)
    for p in sorted(DATA.rglob("*.parquet")):
        if any(kw in p.name.lower() for kw in DATA_KEYWORDS):
            inv["data_parquets"].append(p)
    for p in sorted((ROOT / "analysis" / "figures").rglob("*.p*")):
        if any(kw in p.name.lower() for kw in ("rolling_malthusian", "volcanic", "campop")):
            inv["figures"].append(p)
    return inv


def write_audit_md(inv: Inventory, out: Path) -> None:
    lines = ["# paper4_shadow asset inventory (relevant to Long Shadow on Fertility)\n"]
    for section, items in (("Scripts", inv["scripts"]),
                            ("Data CSVs", inv["data_csvs"]),
                            ("Data parquets", inv["data_parquets"]),
                            ("Figures", inv["figures"])):
        lines.append(f"\n## {section} ({len(items)})\n")
        for p in items:
            lines.append(f"- `{p.relative_to(ROOT)}`")
    out.write_text("\n".join(lines))


def main() -> None:
    inv = inventory()
    out = Path(__file__).parent / "AUDIT.md"
    write_audit_md(inv, out)
    print(f"wrote {out} ({sum(len(v) for v in inv.values())} items)")


if __name__ == "__main__":
    main()
