"""Build annual England CBR/CDR series from CamPOP 26-parish family-reconstitution
microdata (UKDS Study SN 854465), 1538-1851.

Source: /Users/jalonso/Downloads/26FamReconsIDSWithFertility/INDIVIDUAL.csv
(363 MB, 3.7M IDS-format rows).

We aggregate baptism events (= births, in W-S convention) and funeral events
(= deaths) by year across the 26 parishes, then divide by the BoE Millennium
W-S England population denominator to obtain annual aggregate CBR and CDR per
1000 per year. The result is the parish-level aggregate analogue of the
canonical W-S 1981 published series (which we cannot retrieve directly without
UKDS SN 4491 registration). The 26-parish sample is the same one Wrigley
et al (1997) used for family reconstitution; CamPop and SN 4491 are
overlapping sources.

Output: analysis/data/wrigley_schofield/campop_england_annual.csv
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
CAMPOP = Path("/Users/jalonso/Downloads/26FamReconsIDSWithFertility/INDIVIDUAL.csv")


def _annual_event_counts(event_type: str) -> pd.Series:
    """Sum events of given type by year, across all 26 parishes."""
    counts: dict[int, int] = {}
    for chunk in pd.read_csv(CAMPOP, chunksize=500_000,
                              usecols=["type", "year", "day", "month"],
                              encoding="latin-1"):
        sub = chunk[chunk["type"] == event_type].copy()
        sub["year"] = pd.to_numeric(sub["year"], errors="coerce")
        sub = sub.dropna(subset=["year"])
        sub["year"] = sub["year"].astype(int)
        for y, n in sub.groupby("year").size().items():
            counts[y] = counts.get(y, 0) + int(n)
    return pd.Series(counts, name=event_type).sort_index()


def main() -> None:
    print("Counting baptism events (≈ births) ...")
    baptisms = _annual_event_counts("BAPTISM_DATE")
    print(f"  total baptisms: {baptisms.sum():,}; years {baptisms.index.min()}-{baptisms.index.max()}")

    print("\nCounting funeral events (≈ deaths) ...")
    funerals = _annual_event_counts("FUNERAL_DATE")
    print(f"  total funerals: {funerals.sum():,}; years {funerals.index.min()}-{funerals.index.max()}")

    annual = pd.DataFrame({"baptisms": baptisms, "funerals": funerals}).fillna(0).astype(int)
    annual = annual.reset_index().rename(columns={"index": "year"})
    annual = annual[(annual["year"] >= 1538) & (annual["year"] <= 1851)].copy()

    # Population denominator: BoE Millennium W-S England series (already in repo)
    pop_csv = DATA / "wrigley_schofield" / "ws_england_annual_partial_population_only.csv"
    pop = pd.read_csv(pop_csv)
    pop_col = [c for c in pop.columns if c.lower() == "population"][0]
    pop = pop[["year", pop_col]].rename(columns={pop_col: "population"})
    annual = annual.merge(pop, on="year", how="inner")

    # The 26 parishes cover ~3-5% of England; scale baptism/funeral counts by
    # the relative parish-to-national population ratio. We do NOT scale here:
    # we report raw 26-parish RATES per 1000 of national population, which
    # under-states the absolute CBR/CDR by a factor of ~20-30x but PRESERVES
    # year-to-year variation, which is what the regression identifies.
    # Equivalently, baptisms_per_1000nat = baptisms / pop * 1000.
    # The country fixed effect in the prevpos regression absorbs the level
    # offset across data sources.
    annual["cbr_26par"] = 1000.0 * annual["baptisms"] / annual["population"]
    annual["cdr_26par"] = 1000.0 * annual["funerals"] / annual["population"]
    annual["source"] = "CamPOP 26 family reconstitutions (UKDS SN 854465); pop from BoE Millennium"

    out = DATA / "wrigley_schofield" / "campop_england_annual.csv"
    annual.to_csv(out, index=False)
    print(f"\nSaved {out}: {len(annual)} rows, years {annual.year.min()}-{annual.year.max()}")
    print(f"  CBR (26-parish per 1000 nat pop) mean: {annual['cbr_26par'].mean():.3f}")
    print(f"  CDR (26-parish per 1000 nat pop) mean: {annual['cdr_26par'].mean():.3f}")

    # Anchor sanity: known events
    if 1349 in annual["year"].values:
        print(f"  1349 (post-Black-Death year) CDR: {annual[annual.year==1349]['cdr_26par'].iloc[0]:.3f}")
    if 1665 in annual["year"].values:
        print(f"  1665 (Plague of London) CDR: {annual[annual.year==1665]['cdr_26par'].iloc[0]:.3f}")
    if 1741 in annual["year"].values:
        print(f"  1741 (typhus famine) CDR: {annual[annual.year==1741]['cdr_26par'].iloc[0]:.3f}")
    print(f"  CDR/CBR ratio mean (should be < 1 in growing pop): "
          f"{(annual['cdr_26par']/annual['cbr_26par']).mean():.3f}")


if __name__ == "__main__":
    main()
