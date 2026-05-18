# analysis/paper5_horserace/build_horserace_panel.py
from pathlib import Path
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
OUT = ROOT / "analysis/data/deep_determinants_horserace.parquet"


def _sigma_v_T_pre1750() -> pd.DataFrame:
    """Per-country std of annual temperature 1421-1750 from country_climate_1421_2025."""
    p = ROOT / "analysis/data/country_climate_1421_2025.parquet"
    df = pd.read_parquet(p)
    # Column is t_c (annual mean temperature in Celsius)
    t_col = "t_c"
    if t_col not in df.columns:
        # Fallback: find any column starting with t_
        candidates = [c for c in df.columns if c.startswith("t_") and c != "t_c_anom_1971_2000"]
        if not candidates:
            raise ValueError(f"No annual-T column in {p}. Columns: {list(df.columns)}")
        t_col = candidates[0]
    sub = df[(df["year"] >= 1421) & (df["year"] <= 1750)]
    out = sub.groupby("iso3")[t_col].std()
    return out.rename("sigma_v_T_pre1750").reset_index()


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
    sigma = _sigma_v_T_pre1750()
    pathways = _pathway_dummies()
    geo = pd.read_parquet(ROOT / "analysis/data/deep_determinants_extended.parquet")
    het = pd.read_parquet(ROOT / "analysis/data/deep_determinants/predicted_het_pw_adjusted.parquet")
    state = pd.read_parquet(ROOT / "analysis/data/deep_determinants/state_history_pw.parquet")
    yld = pd.read_parquet(ROOT / "analysis/data/deep_determinants/ancestral_crop_yield.parquet")
    pan = pd.read_parquet(ROOT / "analysis/data/deep_determinants/pandemic_intensity_pre1500.parquet")
    out = pd.read_parquet(ROOT / "analysis/data/deep_determinants/modern_outcomes.parquet")

    print(f"  sigma_v_T: {len(sigma)} | pathways: {len(pathways)} | geo: {len(geo)}")
    print(f"  Het: {len(het)} | state: {len(state)} | yield: {len(yld)} | pandemic: {len(pan)} | outcomes: {len(out)}")

    # Start from sigma (covers all countries with climate data), outer-join with geo
    df = sigma.merge(geo, on="iso3", how="outer")
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

    full = df.dropna(subset=["sigma_v_T_pre1750", "H_pred_pwadj",
                              "ancestral_yield_log", "pandemic_intensity_norm"])
    print(f"\nWrote {OUT}")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with all 4 substrates non-null: {len(full)}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
