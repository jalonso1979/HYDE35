"""Orchestrator for the pre-1500 volcanic appendix exercises.

Runs the three appendix-only scripts in sequence:
  1. build_allen_pre1500_panel  — parse Allen-Nuffield xls files into
     analysis/data/allen_silver_wheat_prices_1259_1914.parquet
  2. sigl_volcanic_pre1500      — century-rate population regression 100-1500 CE
  3. volcanic_price_event_study — Allen + Harper event studies at major eruptions
  4. volcanic_price_continuous  — continuous-VSSI distributed-lag price regression

Usage
-----
    python -m analysis.paper4_shadow.run_pre1500_volcanic
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path

SCRIPTS = [
    "build_allen_pre1500_panel",
    "build_conflict_panel",
    "sigl_volcanic_pre1500",
    "volcanic_price_event_study",
    "volcanic_price_continuous",
    "variance_decomposition",
    "volcanic_price_robust",
    "volcanic_price_regional",
]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    for name in SCRIPTS:
        print("\n" + "=" * 72)
        print(f"  {name}")
        print("=" * 72)
        mod = importlib.import_module(name)
        mod.main()


if __name__ == "__main__":
    main()
