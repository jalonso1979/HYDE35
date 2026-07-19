"""Build a country-annual climate panel 1421-2008 from ModE-RA + CRU.

ModE-RA monthly temperature/precipitation anomalies (relative to 1901-2000)
anchored to the CRU 1901-1950 climatology give absolute annual T/P per country.

ERA5 is no longer used. An earlier version spliced the country-level ERA5 rebuild
for 1950-2025 and applied a per-country ERA5-vs-ModE-RA bias offset to the pre-1950
ModE-RA series. That offset is a per-country constant, which is absorbed exactly by
the within-country fixed effects used in every downstream regression (verified: the
headline joint-VAR and single-equation coefficients are identical with and without
it), and the volatility regressor sigma_v^T is a within-window standard deviation and
therefore invariant to a constant offset. The panel is therefore ModE-RA+CRU only,
ending at ModE-RA's 2008 horizon; CRU 1901-1950 remains the absolute-level anchor.

Output: analysis/data/country_climate_1421_2025.parquet
    (legacy filename retained for downstream compatibility; coverage is 1421-2008)
    iso3, year, t_c, t_c_anom_1971_2000, p_mm, p_mm_anom_1971_2000, source
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"


def main() -> None:
    # ModE-RA + CRU absolute annual climate (from the seasonality build)
    seas = pd.read_parquet(DATA / "country_seasonality_1421_2008.parquet")
    panel = (
        seas[["iso3", "year", "t_mean", "p_annual"]]
        .rename(columns={"t_mean": "t_c", "p_annual": "p_mm"})
        .copy()
    )
    panel["source"] = "modera_cru"
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Anomaly relative to 1971-2000 within each country
    ref = (
        panel[panel["year"].between(1971, 2000)]
        .groupby("iso3", as_index=False)
        .agg(t_ref=("t_c", "mean"), p_ref=("p_mm", "mean"))
    )
    panel = panel.merge(ref, on="iso3", how="left")
    panel["t_c_anom_1971_2000"] = panel["t_c"] - panel["t_ref"]
    panel["p_mm_anom_1971_2000"] = panel["p_mm"] - panel["p_ref"]
    panel = panel.drop(columns=["t_ref", "p_ref"])

    out = DATA / "country_climate_1421_2025.parquet"
    panel.to_parquet(out, index=False)
    print(
        f"Wrote {out} ({len(panel):,} rows, {panel['iso3'].nunique()} countries, "
        f"{panel['year'].min()}-{panel['year'].max()}) — ModE-RA+CRU, no ERA5",
        flush=True,
    )


if __name__ == "__main__":
    main()
