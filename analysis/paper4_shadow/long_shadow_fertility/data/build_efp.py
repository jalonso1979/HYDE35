"""Princeton European Fertility Project — province x decade Coale indices.

Source: OPR archive at https://opr.princeton.edu/archive/pefp/
Permissive design: if no cached CSVs exist, raise a clear error so the
controller can request manual download. Builder is callable from the
end-to-end driver, which catches the error and skips EFP-dependent figures.
"""
from __future__ import annotations
from pathlib import Path
import urllib.error
import urllib.request
import pandas as pd

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
           "efp_province_decade.parquet")
CACHE = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/_efp_raw")
CATALOG_URL = "https://opr.princeton.edu/archive/pefp/"


def _try_fetch_catalog() -> str:
    try:
        with urllib.request.urlopen(CATALOG_URL, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not reach {CATALOG_URL}. Download the EFP province-decade "
            f"files manually and place them in {CACHE}/ as one CSV per country "
            f"with columns: country, province, decade, If, Ig, Im (and optional Ih). "
            f"Then re-run this builder."
        ) from exc


def build_efp(write: bool = False) -> pd.DataFrame:
    if CACHE.exists() and any(CACHE.glob("*.csv")):
        parts = [pd.read_csv(p) for p in CACHE.glob("*.csv")]
        df = pd.concat(parts, ignore_index=True)
    else:
        _try_fetch_catalog()
        raise NotImplementedError(
            "OPR archive reached but parsing not implemented. "
            f"Provide cached CSVs in {CACHE}/ instead."
        )

    required = {"country", "province", "decade", "If"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"EFP CSV missing required columns: {missing}")

    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    try:
        df = build_efp(write=True)
        print(f"wrote {OUT}: {len(df)} rows")
    except (RuntimeError, NotImplementedError) as exc:
        print(f"BLOCKED: {exc}")
