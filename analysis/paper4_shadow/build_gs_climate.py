"""Climate-defined growing-season volatility / anomaly measures for long_shadow.

Builds four parquet files (this task: the first two):
  - country_seasonality_gs_preindustrial.parquet  (1 row / iso3, 3 weightings)
  - country_climate_gs_1421_2025.parquet          (iso3 × year, 3 weightings)

GS mask is fixed (1421-1750 climatology): m ∈ GS_i iff 5≤T_clim≤30 AND P_clim≥30mm.

Spec: docs/superpowers/specs/2026-05-19-growing-season-volatility-design.md
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

PRE_WINDOW = (1421, 1750)
T_MIN, T_MAX = 5.0, 30.0
P_MIN = 30.0  # mm / month

COUNTRY_WEIGHTINGS = {
    "area":  "modera_country_monthly.parquet",
    "pop":   "modera_country_monthly_popw.parquet",
    "cropw": "modera_country_monthly_cropw.parquet",
}

SUBNAT_WEIGHTINGS = {
    "area": "modera_subnational_monthly.parquet",
    "pop":  "modera_subnational_monthly_popw.parquet",
}


def _absolute_levels(mod: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Add CRU 1901-1950 climatology to anomalies to recover absolute T, P.

    Works for both country (entity_col='iso3') and sub-national
    (entity_col='sub_id') inputs because sub-national source files already
    carry iso3 per row, so the CRU merge keys remain (iso3, month).
    """
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0)
    return df


def _gs_mask(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Return one row per (entity, month) with in_gs flag from 1421-1750 climatology."""
    pre = df[df["year"].between(*PRE_WINDOW)]
    clim = pre.groupby([entity_col, "month"], as_index=False).agg(
        t_clim=("t_abs", "mean"), p_clim=("p_abs", "mean"))
    clim["in_gs"] = ((clim["t_clim"] >= T_MIN) & (clim["t_clim"] <= T_MAX) &
                     (clim["p_clim"] >= P_MIN))
    return clim[[entity_col, "month", "in_gs"]]


def _cross_section(df: pd.DataFrame, mask: pd.DataFrame, entity_col: str,
                   weighting: str) -> pd.DataFrame:
    """Per-entity GS-restricted volatility + climatology, 1421-1750."""
    pre = df[df["year"].between(*PRE_WINDOW)].merge(mask, on=[entity_col, "month"])

    gs = pre[pre["in_gs"]]
    gs_yr = gs.groupby([entity_col, "year"], as_index=False).agg(
        t_gs=("t_abs", "mean"), p_gs=("p_abs", "mean"))
    cs_gs = gs_yr.groupby(entity_col, as_index=False).agg(
        **{f"T_gs_mean_pre1750_{weighting}":     ("t_gs", "mean"),
           f"P_gs_mean_pre1750_{weighting}":     ("p_gs", "mean"),
           f"sigma_v_T_gs_pre1750_{weighting}":  ("t_gs", "std"),
           f"sigma_v_P_gs_pre1750_{weighting}":  ("p_gs", "std")})

    nongs = pre[~pre["in_gs"]]
    nongs_yr = nongs.groupby([entity_col, "year"], as_index=False).agg(
        t_nongs=("t_abs", "mean"))
    cs_nongs = nongs_yr.groupby(entity_col, as_index=False).agg(
        **{f"sigma_v_T_nongs_pre1750_{weighting}": ("t_nongs", "std")})

    counts = (mask.groupby(entity_col, as_index=False)
                  .agg(**{f"n_gs_months_{weighting}": ("in_gs", "sum")}))
    months_list = (mask[mask["in_gs"]]
                    .groupby(entity_col)["month"]
                    .apply(lambda s: ",".join(str(m) for m in sorted(s)))
                    .rename(f"gs_months_mask_{weighting}")
                    .reset_index())

    out = counts.merge(months_list, on=entity_col, how="left").merge(
        cs_gs, on=entity_col, how="left").merge(
        cs_nongs, on=entity_col, how="left")
    out[f"gs_months_mask_{weighting}"] = out[f"gs_months_mask_{weighting}"].fillna("")
    return out


def _annual_panel(df: pd.DataFrame, mask: pd.DataFrame, entity_col: str,
                  weighting: str) -> pd.DataFrame:
    """Per (entity, year): GS-mean T, P, and anomalies vs 1421-1750 GS climatology."""
    merged = df.merge(mask, on=[entity_col, "month"])
    gs = merged[merged["in_gs"]]
    yr = gs.groupby([entity_col, "year"], as_index=False).agg(
        **{f"t_gs_mean_{weighting}": ("t_abs", "mean"),
           f"p_gs_mean_{weighting}": ("p_abs", "mean")})
    pre = yr[yr["year"].between(*PRE_WINDOW)]
    clim = pre.groupby(entity_col, as_index=False).agg(
        **{f"t_gs_clim_{weighting}": (f"t_gs_mean_{weighting}", "mean"),
           f"p_gs_clim_{weighting}": (f"p_gs_mean_{weighting}", "mean")})
    yr = yr.merge(clim, on=entity_col, how="left")
    yr[f"t_gs_anom_{weighting}"] = (yr[f"t_gs_mean_{weighting}"]
                                      - yr[f"t_gs_clim_{weighting}"])
    yr[f"p_gs_anom_{weighting}"] = (yr[f"p_gs_mean_{weighting}"]
                                      - yr[f"p_gs_clim_{weighting}"])
    return yr.drop(columns=[f"t_gs_clim_{weighting}", f"p_gs_clim_{weighting}"])


def _build_country() -> None:
    cs_pieces: list[pd.DataFrame] = []
    panel_pieces: list[pd.DataFrame] = []
    for w, fname in COUNTRY_WEIGHTINGS.items():
        print(f"[country/{w}] reading {fname}", flush=True)
        mod = pd.read_parquet(DATA / fname)
        abs_df = _absolute_levels(mod, entity_col="iso3")
        mask = _gs_mask(abs_df, entity_col="iso3")
        cs_pieces.append(_cross_section(abs_df, mask, "iso3", w))
        panel_pieces.append(_annual_panel(abs_df, mask, "iso3", w))

    cs = cs_pieces[0]
    for piece in cs_pieces[1:]:
        cs = cs.merge(piece, on="iso3", how="outer")
    for w in COUNTRY_WEIGHTINGS:
        cs[f"n_gs_months_{w}"] = cs[f"n_gs_months_{w}"].fillna(0).astype(int)
        cs[f"gs_months_mask_{w}"] = cs[f"gs_months_mask_{w}"].fillna("")
    cs_out = DATA / "country_seasonality_gs_preindustrial.parquet"
    cs.to_parquet(cs_out, index=False)
    print(f"[country] wrote {cs_out} ({len(cs)} countries)")

    panel = panel_pieces[0]
    for piece in panel_pieces[1:]:
        panel = panel.merge(piece, on=["iso3", "year"], how="outer")
    # Carry n_gs_months_* into the panel so downstream regression scripts can
    # filter empty-GS rows without a secondary merge against the cross-section.
    count_cols = ["iso3"] + [f"n_gs_months_{w}" for w in COUNTRY_WEIGHTINGS]
    panel = panel.merge(cs[count_cols], on="iso3", how="left")
    panel_out = DATA / "country_climate_gs_1421_2025.parquet"
    panel.to_parquet(panel_out, index=False)
    print(f"[country] wrote {panel_out} ({len(panel):,} rows, "
          f"{panel['iso3'].nunique()} countries)")

    print("\n[country/diagnostic] empty-GS (cropw):")
    empty = cs[cs["n_gs_months_cropw"] == 0]["iso3"].tolist()
    print(f"  N={len(empty)}: {empty}")
    print("[country/diagnostic] short-GS (cropw, 1-3 months):")
    short = cs[cs["n_gs_months_cropw"].between(1, 3)]["iso3"].tolist()
    print(f"  N={len(short)}: {short}")


def _build_subnational() -> None:
    cs_pieces: list[pd.DataFrame] = []
    panel_pieces: list[pd.DataFrame] = []
    for w, fname in SUBNAT_WEIGHTINGS.items():
        print(f"[subnat/{w}] reading {fname}", flush=True)
        mod = pd.read_parquet(DATA / fname)
        abs_df = _absolute_levels(mod, entity_col="sub_id")
        mask = _gs_mask(abs_df, entity_col="sub_id")
        cs_pieces.append(_cross_section(abs_df, mask, "sub_id", w))
        panel_pieces.append(_annual_panel(abs_df, mask, "sub_id", w))

    cs = cs_pieces[0]
    for piece in cs_pieces[1:]:
        cs = cs.merge(piece, on="sub_id", how="outer")
    for w in SUBNAT_WEIGHTINGS:
        cs[f"n_gs_months_{w}"] = cs[f"n_gs_months_{w}"].fillna(0).astype(int)
        cs[f"gs_months_mask_{w}"] = cs[f"gs_months_mask_{w}"].fillna("")
    # Carry iso3 forward for downstream regression merges
    iso_map = (pd.read_parquet(DATA / SUBNAT_WEIGHTINGS["pop"],
                                columns=["sub_id", "iso3"])
                 .drop_duplicates("sub_id"))
    cs = cs.merge(iso_map, on="sub_id", how="left")
    cs_out = DATA / "subnational_seasonality_gs_preindustrial.parquet"
    cs.to_parquet(cs_out, index=False)
    print(f"[subnat] wrote {cs_out} ({len(cs)} units)")

    panel = panel_pieces[0]
    for piece in panel_pieces[1:]:
        panel = panel.merge(piece, on=["sub_id", "year"], how="outer")
    # Carry iso3 and n_gs_months_* into the panel so downstream regressions
    # can filter empty-GS rows without a secondary merge.
    count_cols = ["sub_id"] + [f"n_gs_months_{w}" for w in SUBNAT_WEIGHTINGS]
    panel = panel.merge(cs[count_cols + ["iso3"]], on="sub_id", how="left")
    panel_out = DATA / "subnational_climate_gs_1421_2025.parquet"
    panel.to_parquet(panel_out, index=False)
    print(f"[subnat] wrote {panel_out} ({len(panel):,} rows, "
          f"{panel['sub_id'].nunique()} units)")

    print("\n[subnat/diagnostic] empty-GS (pop) sub-units:")
    empty = cs[cs["n_gs_months_pop"] == 0]
    print(f"  N={len(empty)} sub-units (in {empty['iso3'].nunique()} countries)")
    print("[subnat/diagnostic] short-GS (pop, 1-3 months):")
    short = cs[cs["n_gs_months_pop"].between(1, 3)]
    print(f"  N={len(short)} sub-units (in {short['iso3'].nunique()} countries)")


def main() -> None:
    _build_country()
    _build_subnational()


if __name__ == "__main__":
    main()
