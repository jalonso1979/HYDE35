"""Writer for the ERA5 vs ModE-RA country-level calibration parquets.

Reconstructs the (previously uncommitted) computation behind
``era5_modera_calibration_monthly.parquet`` and
``era5_modera_calibration_annual.parquet``, whose semantics are pinned by
their reader ``robustness_v2.py``:

- ERA5 absolute monthly country temperature (``era5_country_monthly.parquet``,
  area-weighted t2m_c) is compared against absolute ModE-RA temperature,
  formed as ModE-RA monthly anomaly (wrt 1901-2000) plus the CRU TS 1901-1950
  country-month climatology, over the 1950-2008 overlap.
- Monthly file: per-country t_bias (mean ERA5 - ModE-RA_abs), t_rmse,
  t_corr_m (Pearson over months), n (months).
- Annual file: calendar-year means of the same two series; corr_anom equals
  corr_abs (correlation is demeaning-invariant — the original file carries
  both, byte-identical), bias_ann, n (years).

The original artifacts (frozen 2026-05-14) were computed from the partial
ERA5 archive (post-1967 months mostly Jan-Apr; 49 countries truncated
1966-78); this writer recomputes them from the complete 1950-2025 panel.
``modera_era5_bias_1950_2008.parquet`` (the earlier region-vs-country
estimate read by robustness.py) is deliberately NOT rewritten — it is
documented as a frozen back-compat artifact.
"""
from __future__ import annotations

from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
OVERLAP = (1950, 2008)


def main() -> None:
    era5 = pd.read_parquet(DATA / "era5_country_monthly.parquet")
    modera = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")

    modera = modera.merge(clim[["iso3", "month", "tmp_c_clim"]],
                          on=["iso3", "month"])
    modera["t_modera_abs"] = modera["t_anom_c"] + modera["tmp_c_clim"]

    m = era5.merge(modera[["iso3", "year", "month", "t_modera_abs"]],
                   on=["iso3", "year", "month"])
    m = m[m["year"].between(*OVERLAP)].dropna(subset=["t2m_c", "t_modera_abs"])

    rows_m, rows_a = [], []
    for iso, sub in m.groupby("iso3"):
        d = sub["t2m_c"] - sub["t_modera_abs"]
        corr = (np.corrcoef(sub["t2m_c"], sub["t_modera_abs"])[0, 1]
                if len(sub) >= 24 else np.nan)
        rows_m.append({"iso3": iso, "t_bias": float(d.mean()),
                       "t_rmse": float(np.sqrt((d ** 2).mean())),
                       "t_corr_m": float(corr), "n": float(len(sub))})
        ann = sub.groupby("year")[["t2m_c", "t_modera_abs"]].mean()
        ann = ann[ann.index.to_series().between(*OVERLAP)]
        ca = (np.corrcoef(ann["t2m_c"], ann["t_modera_abs"])[0, 1]
              if len(ann) >= 10 else np.nan)
        rows_a.append({"iso3": iso, "corr_anom": float(ca),
                       "corr_abs": float(ca),
                       "bias_ann": float((ann["t2m_c"]
                                          - ann["t_modera_abs"]).mean()),
                       "n": float(len(ann))})

    out_m = pd.DataFrame(rows_m)
    out_a = pd.DataFrame(rows_a)
    out_m.to_parquet(DATA / "era5_modera_calibration_monthly.parquet",
                     index=False)
    out_a.to_parquet(DATA / "era5_modera_calibration_annual.parquet",
                     index=False)
    print(f"monthly: {len(out_m)} countries, median |bias| = "
          f"{out_m['t_bias'].abs().median():.2f} C, median t_corr = "
          f"{out_m['t_corr_m'].median():.3f}, median n = {out_m['n'].median():.0f}")
    print(f"annual:  median corr_anom = {out_a['corr_anom'].median():.3f}, "
          f"median n = {out_a['n'].median():.0f}")


if __name__ == "__main__":
    main()
