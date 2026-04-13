"""Paper 4 — The Long Shadow of Seasonality: orchestrator.

Loads data and runs all six exercises sequentially.

Usage
-----
    python -m analysis.paper4_shadow.run_all
    python -m analysis.paper4_shadow.run_all --exercises 1 2 3
    python -m analysis.paper4_shadow.run_all --figures-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

BANNER_WIDTH = 72


def _banner(title: str) -> None:
    print("\n" + "=" * BANNER_WIDTH)
    print(f"  {title}")
    print("=" * BANNER_WIDTH + "\n")


def _sub(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict:
    """Load all required parquet files and return as a dict.

    Returns
    -------
    dict with keys:
        extended_panel  — hyde_era5_extended_panel (filtered)
        climate_panel   — climate_panel_0_2025
        era5_panel      — era5_full_panel
        pathways        — paper1_clustered_features (with 'country' alias)
        country_panel   — country_analysis_panel (with derived columns)
    """
    data: dict = {}

    # -- Extended panel (modern, country-level) ----------------------------
    ep = pd.read_parquet(DATA_DIR / "hyde_era5_extended_panel.parquet")
    ep = ep.loc[ep["temperature_c"].notna() & (ep["pop"] > 0)].copy()

    # Ensure temp_anomaly exists
    if "temp_anomaly" not in ep.columns:
        try:
            from analysis.paper3_climate.climate_shocks import (
                build_climate_shock_panel,
            )
            ep = build_climate_shock_panel(
                ep,
                climate_vars=["temperature_c"],
                entity_col="country_id",
            )
        except Exception as exc:
            print(f"[warn] Could not build temp_anomaly: {exc}")

    data["extended_panel"] = ep

    # -- Climate panel (long historical, region-level) ---------------------
    data["climate_panel"] = pd.read_parquet(
        DATA_DIR / "climate_panel_0_2025.parquet",
    )

    # -- ERA5 full panel ---------------------------------------------------
    data["era5_panel"] = pd.read_parquet(DATA_DIR / "era5_full_panel.parquet")

    # -- Pathway assignments -----------------------------------------------
    pw = pd.read_parquet(DATA_DIR / "paper1_clustered_features.parquet")
    # The file uses 'iso3' — create a 'country' alias for merging
    if "country" not in pw.columns and "iso3" in pw.columns:
        pw["country"] = pw["iso3"]
    # Also create a country_id alias so we can merge with extended_panel
    if "country_id" not in pw.columns and "country" in pw.columns:
        pw["country_id"] = pw["country"]
    data["pathways"] = pw

    # -- Country analysis panel (0-1750 CE) --------------------------------
    cp = pd.read_parquet(DATA_DIR / "country_analysis_panel.parquet")
    # Derive columns expected by malthusian_extended
    if "pop_growth" not in cp.columns and "pop_growth_rate" in cp.columns:
        cp["pop_growth"] = cp["pop_growth_rate"]
    if "log_density" not in cp.columns and "popdens_p_km2_mean" in cp.columns:
        cp["log_density"] = np.log(cp["popdens_p_km2_mean"].clip(lower=1e-6))
    if "log_land_labor" not in cp.columns and "land_labor_ratio" in cp.columns:
        cp["log_land_labor"] = np.log(cp["land_labor_ratio"].clip(lower=1e-6))
    data["country_panel"] = cp

    print(f"Data loaded: {', '.join(data.keys())}")
    for k, v in data.items():
        print(f"  {k}: {v.shape[0]:,} rows x {v.shape[1]} cols")

    return data


# ---------------------------------------------------------------------------
# Figure helper
# ---------------------------------------------------------------------------

def _try_figure(func, *args, **kwargs):
    """Call a figure function, swallowing ImportError / AttributeError."""
    try:
        func(*args, **kwargs)
        print(f"  [fig] {func.__name__} done")
    except Exception as exc:
        print(f"  [fig] {func.__name__} skipped: {exc}")


def _import_figures():
    """Try to import figures module; return module or None."""
    try:
        from analysis.paper4_shadow import figures
        return figures
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Exercise 1 — Pathway prediction from climate endowments
# ---------------------------------------------------------------------------

def exercise_1(data: dict, *, figures_only: bool = False) -> pd.DataFrame:
    """Exercise 1: Does climate seasonality predict agricultural pathway?

    Returns seasonality endowments (one row per region).
    """
    _banner("Exercise 1: Pathway Prediction from Climate Endowments")

    from analysis.paper4_shadow.seasonality import build_seasonality_panel
    from analysis.paper4_shadow.pathway_prediction import (
        run_pathway_multinomial,
        compute_marginal_effects,
    )

    # Build seasonality endowments from the long climate panel
    endowments = build_seasonality_panel(
        data["climate_panel"],
        entity_col="region",
        historical_window=50,
        historical_period=(1, 1850),
    )
    print(f"Seasonality endowments: {len(endowments)} regions")

    if not figures_only:
        # Merge endowments with pathway assignments via era5_region
        ep = data["extended_panel"]
        pw = data["pathways"]

        # Build a region-to-pathway mapping through the extended panel
        region_pw = (
            ep[["country_id", "era5_region"]]
            .drop_duplicates()
            .merge(pw[["country_id", "cluster"]], on="country_id", how="inner")
        )
        # Average seasonality per region, then merge
        merged = endowments.merge(
            region_pw.rename(columns={"era5_region": "region"}),
            on="region",
            how="inner",
        )
        # Aggregate to one row per country_id
        merged = (
            merged.groupby(["country_id", "cluster"])
            .agg({"hist_seasonality_proxy": "mean"})
            .reset_index()
            .rename(columns={"hist_seasonality_proxy": "seasonality"})
        )
        merged["pathway"] = merged["cluster"]

        _sub("Multinomial logit: pathway ~ seasonality")
        if merged["seasonality"].notna().sum() >= 10:
            result = run_pathway_multinomial(
                merged.dropna(subset=["seasonality", "pathway"]),
                pathway_col="pathway",
                climate_vars=["seasonality"],
            )
            print(result["summary"])
            me = compute_marginal_effects(
                result["model"], merged, ["seasonality"],
            )
            if not me.empty:
                _sub("Marginal effects")
                print(me.to_string(index=False))
        else:
            print("  [warn] Too few observations for multinomial logit")

    # Figures
    figs = _import_figures()
    if figs:
        _try_figure(getattr(figs, "fig3", lambda: None), data, endowments)

    return endowments


# ---------------------------------------------------------------------------
# Exercise 2 — Dual-channel separation
# ---------------------------------------------------------------------------

def exercise_2(data: dict, *, figures_only: bool = False) -> None:
    """Exercise 2: Seasonality vs. volatility — dual-channel separation."""
    _banner("Exercise 2: Dual-Channel Separation")

    from analysis.paper4_shadow.dual_channel import (
        run_dual_channel_pathway,
        run_dual_channel_malthusian,
    )
    from analysis.paper4_shadow.seasonality import (
        compute_historical_seasonality_proxy,
    )

    if not figures_only:
        climate = data["climate_panel"]
        pw = data["pathways"]

        # Compute seasonality and volatility per region
        proxy = compute_historical_seasonality_proxy(
            climate, window=50, entity_col="region",
        )
        vol = (
            climate.groupby("region")["temperature_c"]
            .std()
            .reset_index()
            .rename(columns={"temperature_c": "volatility"})
        )
        seas = (
            proxy.groupby("region")["seasonality_proxy"]
            .mean()
            .reset_index()
            .rename(columns={"seasonality_proxy": "seasonality"})
        )

        # Map regions to pathways via extended panel
        ep = data["extended_panel"]
        region_pw = (
            ep[["country_id", "era5_region"]]
            .drop_duplicates()
            .merge(pw[["country_id", "cluster"]], on="country_id", how="inner")
            .rename(columns={"era5_region": "region"})
        )
        region_pw = (
            region_pw.groupby("region")["cluster"]
            .first()
            .reset_index()
            .rename(columns={"cluster": "pathway"})
        )

        xs = seas.merge(vol, on="region").merge(region_pw, on="region").dropna()

        if len(xs) >= 10:
            _sub("Stage 1: pathway ~ seasonality + volatility")
            r1 = run_dual_channel_pathway(
                xs,
                pathway_col="pathway",
                seasonality_col="seasonality",
                volatility_col="volatility",
            )
            print(f"  Seasonality coef: {r1['seasonality_coef']:.4f} "
                  f"(p={r1['seasonality_pval']:.4f})")
            print(f"  Volatility  coef: {r1['volatility_coef']:.4f} "
                  f"(p={r1['volatility_pval']:.4f})")
            print(f"  R-squared: {r1['rsquared']:.4f}, N={r1['nobs']}")

            # Stage 2 requires a country-level malthusian_beta
            # Compute it as correlation of pop_growth with log_density per country
            cp = data["country_panel"]
            betas = (
                cp.dropna(subset=["pop_growth", "log_density"])
                .groupby("country")
                .apply(
                    lambda g: g["pop_growth"].corr(g["log_density"])
                    if len(g) > 5 else np.nan,
                    include_groups=False,
                )
                .reset_index(name="malthusian_beta")
            )
            # Map country to region via extended panel
            country_region = (
                ep[["country_id", "era5_region"]]
                .drop_duplicates()
                .rename(columns={"country_id": "country", "era5_region": "region"})
            )
            betas = betas.merge(country_region, on="country", how="inner")
            betas = betas.groupby("region")["malthusian_beta"].mean().reset_index()

            xs2 = xs.merge(betas, on="region").dropna()
            if len(xs2) >= 10:
                _sub("Stage 2: malthusian_beta ~ seasonality + volatility")
                r2 = run_dual_channel_malthusian(
                    xs2,
                    beta_col="malthusian_beta",
                    seasonality_col="seasonality",
                    volatility_col="volatility",
                )
                print(f"  Seasonality coef: {r2['seasonality_coef']:.4f} "
                      f"(p={r2['seasonality_pval']:.4f})")
                print(f"  Volatility  coef: {r2['volatility_coef']:.4f} "
                      f"(p={r2['volatility_pval']:.4f})")
                print(f"  R-squared: {r2['rsquared']:.4f}, N={r2['nobs']}")
            else:
                print("  [warn] Too few observations for stage 2")
        else:
            print("  [warn] Too few observations for dual-channel test")

    figs = _import_figures()
    if figs:
        _try_figure(getattr(figs, "fig4", lambda: None), data)


# ---------------------------------------------------------------------------
# Exercise 3 — Malthusian dynamics by pathway
# ---------------------------------------------------------------------------

def exercise_3(data: dict, *, figures_only: bool = False) -> None:
    """Exercise 3: Malthusian regressions stratified by pathway cluster."""
    _banner("Exercise 3: Malthusian Dynamics by Pathway")

    from analysis.paper4_shadow.malthusian_extended import (
        run_malthusian_by_pathway,
        run_rolling_by_pathway,
    )

    if not figures_only:
        cp = data["country_panel"]
        pw = data["pathways"]

        _sub("Fixed-effects regression by pathway")
        fe_results = run_malthusian_by_pathway(
            cp,
            pw,
            dep_var="pop_growth",
            density_var="log_density",
            land_labor_var="log_land_labor",
            entity_col="country",
            pathway_col="cluster",
        )
        if not fe_results.empty:
            print(fe_results.to_string(index=False))
        else:
            print("  [warn] No results from stratified FE regressions")

        _sub("Rolling-window regressions by pathway")
        roll = run_rolling_by_pathway(
            cp,
            pw,
            dep_var="pop_growth",
            key_var="log_density",
            control_vars=["log_land_labor"],
            entity_col="country",
            pathway_col="cluster",
            window_years=400,
            step_years=100,
        )
        if not roll.empty:
            # Summarise: for each pathway, the range of coefficients
            summary = roll.groupby("pathway").agg(
                n_windows=("coefficient", "size"),
                coef_mean=("coefficient", "mean"),
                coef_min=("coefficient", "min"),
                coef_max=("coefficient", "max"),
            )
            print(summary.to_string())
        else:
            print("  [warn] No results from rolling regressions")

    figs = _import_figures()
    if figs:
        _try_figure(getattr(figs, "fig5", lambda: None), data)
        _try_figure(getattr(figs, "fig6", lambda: None), data)


# ---------------------------------------------------------------------------
# Exercise 4 — Pathway-stratified IRFs
# ---------------------------------------------------------------------------

def exercise_4(data: dict, *, figures_only: bool = False) -> None:
    """Exercise 4: Climate-shock impulse response functions by pathway."""
    _banner("Exercise 4: Pathway-Stratified IRFs")

    from analysis.paper4_shadow.pathway_irfs import (
        run_irfs_by_pathway,
        compare_pathway_irfs,
    )

    if not figures_only:
        ep = data["extended_panel"]
        pw = data["pathways"]

        _sub("Local projections: pop_growth response to temp_anomaly")
        irf_dict = run_irfs_by_pathway(
            ep,
            pw,
            shock_var="temp_anomaly",
            response_var="pop_growth",
            entity_col="country_id",
            pathway_col="cluster",
            max_horizon=10,
            n_lags=2,
        )

        if irf_dict:
            print(f"  IRFs computed for {len(irf_dict)} pathways: "
                  f"{list(irf_dict.keys())}")

            _sub("Comparison at horizon 5")
            comp = compare_pathway_irfs(irf_dict, horizon=5)
            if not comp.empty:
                print(comp.to_string(index=False))
            else:
                print("  [warn] No IRFs available at horizon 5")
        else:
            print("  [warn] No IRFs could be computed")

    figs = _import_figures()
    if figs:
        _try_figure(getattr(figs, "fig7", lambda: None), data)


# ---------------------------------------------------------------------------
# Exercise 5 — Escape mechanisms
# ---------------------------------------------------------------------------

def exercise_5(data: dict, *, figures_only: bool = False) -> None:
    """Exercise 5: Escape-mechanism interactions (intensification/urbanization)."""
    _banner("Exercise 5: Escape Mechanism Interactions")

    from analysis.paper4_shadow.escape_mechanism import (
        run_escape_interactions,
        run_rolling_with_mediator,
    )

    if not figures_only:
        ep = data["extended_panel"]

        _sub("Interaction regressions: shock x mediator")
        results = run_escape_interactions(
            ep,
            shock_var="temp_anomaly",
            outcome_var="pop_growth",
            mediators=["urban_share", "intensification_index"],
            entity_col="country_id",
        )

        for med_name, res in results.items():
            if res is None:
                print(f"  {med_name}: insufficient data")
                continue
            _sub(f"Mediator: {med_name}")
            if "interaction_coef" in res:
                print(f"  Shock coef:       {res['shock_coef']:.6f} "
                      f"(p={res['shock_pval']:.4f})")
                print(f"  Interaction coef: {res['interaction_coef']:.6f} "
                      f"(p={res['interaction_pval']:.4f})")
                print(f"  R-squared: {res['rsquared']:.4f}, N={res['nobs']}")
            else:
                # Joint model — report per-mediator interactions
                print(f"  Shock coef: {res['shock_coef']:.6f} "
                      f"(p={res['shock_pval']:.4f})")
                for k, v in res.items():
                    if k.endswith("_interaction_coef"):
                        med = k.replace("_interaction_coef", "")
                        pval = res.get(f"{med}_interaction_pval", np.nan)
                        print(f"  {med} interaction: {v:.6f} (p={pval:.4f})")
                print(f"  R-squared: {res['rsquared']:.4f}, N={res['nobs']}")

        # Rolling window with intensification
        if "intensification_index" in ep.columns:
            _sub("Rolling interaction: temp_anomaly x intensification_index")
            roll = run_rolling_with_mediator(
                ep,
                shock_var="temp_anomaly",
                outcome_var="pop_growth",
                mediator_var="intensification_index",
                entity_col="country_id",
                window=20,
                step=5,
            )
            if not roll.empty:
                print(f"  {len(roll)} windows computed")
                print(roll[["center_year", "interaction_coef",
                            "interaction_pval"]].to_string(index=False))
            else:
                print("  [warn] Rolling interaction returned no results")

    figs = _import_figures()
    if figs:
        _try_figure(getattr(figs, "fig8", lambda: None), data)


# ---------------------------------------------------------------------------
# Exercise 6 — Long shadow cross-section
# ---------------------------------------------------------------------------

def exercise_6(
    data: dict,
    endowments: pd.DataFrame | None = None,
    *,
    figures_only: bool = False,
) -> None:
    """Exercise 6: Long shadow — do pre-industrial endowments predict modern outcomes?"""
    _banner("Exercise 6: The Long Shadow Cross-Section")

    from analysis.paper4_shadow.long_shadow import (
        build_long_shadow_cross_section,
        run_long_shadow_regressions,
    )

    if endowments is None:
        from analysis.paper4_shadow.seasonality import build_seasonality_panel
        endowments = build_seasonality_panel(
            data["climate_panel"],
            entity_col="region",
            historical_window=50,
            historical_period=(1, 1850),
        )

    # Rename region -> era5_region for the cross-section builder
    seas_endow = endowments.rename(
        columns={
            "region": "era5_region",
            "hist_seasonality_proxy": "seasonality_historical",
        },
    )

    if not figures_only:
        ep = data["extended_panel"]
        pw = data["pathways"]

        _sub("Building cross-section")
        xs = build_long_shadow_cross_section(
            ep,
            seas_endow,
            pw,
            entity_col="country_id",
        )
        print(f"  Cross-section: {len(xs)} countries")

        _sub("OLS: modern outcomes ~ historical seasonality")
        results = run_long_shadow_regressions(
            xs,
            seasonality_col="seasonality_historical",
            pathway_col="cluster",
            include_pathway_fe=True,
        )
        if not results.empty:
            print(results.to_string(index=False))
        else:
            print("  [warn] No results from long-shadow regressions")

    figs = _import_figures()
    if figs:
        _try_figure(getattr(figs, "fig9", lambda: None), data, endowments)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXERCISES = {
    1: exercise_1,
    2: exercise_2,
    3: exercise_3,
    4: exercise_4,
    5: exercise_5,
    6: exercise_6,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper 4 — The Long Shadow of Seasonality: run all exercises",
    )
    parser.add_argument(
        "--exercises",
        nargs="+",
        type=int,
        default=list(EXERCISES.keys()),
        help="Which exercises to run (default: all)",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only generate figures, skip regressions",
    )
    args = parser.parse_args()

    data = load_data()
    endowments = None

    for ex_num in sorted(args.exercises):
        if ex_num not in EXERCISES:
            print(f"[warn] Unknown exercise {ex_num}, skipping")
            continue

        if ex_num == 1:
            endowments = exercise_1(data, figures_only=args.figures_only)
        elif ex_num == 6:
            exercise_6(data, endowments, figures_only=args.figures_only)
        else:
            EXERCISES[ex_num](data, figures_only=args.figures_only)

    _banner("All requested exercises complete")


if __name__ == "__main__":
    main()
