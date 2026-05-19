# analysis/paper5_horserace/build_horserace_panel.py
from pathlib import Path
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
OUT = ROOT / "analysis/data/deep_determinants_horserace.parquet"


def _climate_bundle_pre1750() -> pd.DataFrame:
    """Climate-bundle substrates: T̄, P̄, σᵥᵀ, σᵥᴾ over 1421-1750.

    Absolute T̄ and absolute annual P̄ are taken from country_seasonality_preindustrial.parquet
    (which adds back the CRU 1901-1950 climatology to the ModE-RA anomalies, giving
    physically meaningful absolute values).

    σᵥᵀ and σᵥᴾ are computed as the standard deviation of annual realisations within
    the 1421-1750 window from country_climate_1421_2025.parquet. (Standard deviation
    is invariant to whether the underlying series is in level or anomaly form.)

    Returns per-country DataFrame with columns:
        iso3, t_mean_pre1750, p_mean_pre1750,
              sigma_v_T_pre1750, sigma_v_P_pre1750
    """
    seas = pd.read_parquet(ROOT / "analysis/data/country_seasonality_preindustrial.parquet")
    seas = seas[["iso3", "t_mean_preind", "p_annual_preind"]].rename(columns={
        "t_mean_preind": "t_mean_pre1750",
        "p_annual_preind": "p_mean_pre1750",
    })

    clim = pd.read_parquet(ROOT / "analysis/data/country_climate_1421_2025.parquet")
    sub = clim[(clim["year"] >= 1421) & (clim["year"] <= 1750)]
    vol = sub.groupby("iso3").agg(
        sigma_v_T_pre1750=("t_c", "std"),
        sigma_v_P_pre1750=("p_mm", "std"),
    ).reset_index()

    return seas.merge(vol, on="iso3", how="outer")


def _pathway_dummies() -> pd.DataFrame:
    """K=5 climate cluster dummies from climate_pathways_country (cluster col: climate_cluster)."""
    p = ROOT / "analysis/data/climate_pathways_country.parquet"
    df = pd.read_parquet(p)
    # Column is climate_cluster (integer 0-4)
    label_col = "climate_cluster"
    if label_col not in df.columns:
        # Fallback: find any column with 'cluster' or 'pathway' and 'label'
        label_col = next(
            (c for c in df.columns if c != "iso3"),
            None
        )
        if label_col is None:
            raise ValueError(f"No pathway/cluster column found in {p}. Columns: {list(df.columns)}")
    keep = df[["iso3", label_col]].copy()
    dummies = pd.get_dummies(keep[label_col], prefix="pathway").astype(int)
    result = pd.concat([keep[["iso3"]], dummies], axis=1)

    # Drop singleton pathway dummies (only 1 country) to avoid collinearity
    singleton_cols = [c for c in dummies.columns if dummies[c].sum() <= 1]
    if singleton_cols:
        print(f"  Dropping singleton pathway dummies: {singleton_cols}")
        result = result.drop(columns=singleton_cols)

    return result


def main() -> None:
    print("Loading substrate data layers...")
    climate = _climate_bundle_pre1750()
    pathways = _pathway_dummies()
    geo = pd.read_parquet(ROOT / "analysis/data/deep_determinants_extended.parquet")
    het = pd.read_parquet(ROOT / "analysis/data/deep_determinants/predicted_het_pw_adjusted.parquet")
    state = pd.read_parquet(ROOT / "analysis/data/deep_determinants/state_history_pw.parquet")
    yld = pd.read_parquet(ROOT / "analysis/data/deep_determinants/ancestral_crop_yield.parquet")
    pan = pd.read_parquet(ROOT / "analysis/data/deep_determinants/pandemic_intensity_pre1500.parquet")
    out = pd.read_parquet(ROOT / "analysis/data/deep_determinants/modern_outcomes.parquet")

    print(f"  climate bundle: {len(climate)} | pathways: {len(pathways)} | geo: {len(geo)}")
    print(f"  Het: {len(het)} | state: {len(state)} | yield: {len(yld)} | pandemic: {len(pan)} | outcomes: {len(out)}")

    # Start from climate (covers all countries with climate data), outer-join with geo
    df = climate.merge(geo, on="iso3", how="outer")
    df = df.merge(pathways, on="iso3", how="left")
    df = df.merge(het[["iso3", "H_pred", "H_pred_pwadj"]], on="iso3", how="left")
    df = df.merge(state[["iso3", "state_hist", "state_hist_pwadj"]], on="iso3", how="left")
    df = df.merge(yld[["iso3", "ancestral_yield", "ancestral_yield_log"]], on="iso3", how="left")
    df = df.merge(pan[["iso3", "pandemic_intensity", "n_pandemic_years", "pandemic_intensity_norm"]],
                  on="iso3", how="left")
    df = df.merge(out.drop(columns=["source"], errors="ignore"), on="iso3", how="left")

    # Drop duplicates and filter to valid ISO3 codes
    df = df.drop_duplicates(subset=["iso3"])
    df = df[df["iso3"].str.match(r"^[A-Z]{3}$", na=False)].copy()
    df = df.reset_index(drop=True)

    df.to_parquet(OUT, index=False)

    # Climate-bundle coverage = all 4 climate vars non-null
    climate_cols = ["t_mean_pre1750", "p_mean_pre1750",
                    "sigma_v_T_pre1750", "sigma_v_P_pre1750"]
    full = df.dropna(subset=climate_cols + ["H_pred_pwadj",
                              "ancestral_yield_log", "pandemic_intensity_norm"])
    print(f"\nWrote {OUT}")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with full 4-bundle climate + 3 other substrates: {len(full)}")
    print(f"  log_popd_1500 non-null: {df['log_popd_1500'].notna().sum()}, "
          f"log_popd_2025: {df['log_popd_2025'].notna().sum()}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
