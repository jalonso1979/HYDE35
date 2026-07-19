"""AADR v66 country-level composite index exploration.

Side exploration — outputs are standalone; the main paper is NOT touched.

Steps
-----
1. Load AADR v66 .anno metadata (downloaded from Harvard Dataverse).
2. Map Political Entity -> ISO3 using pycountry + manual overrides.
3. Compute country-level sample-count and temporal metrics.
4. Produce 4 figures + a sanity-check regression table.
5. Build a "deep population structure" composite index for countries >= 10 samples.
6. Save aadr_country_metrics.parquet.

Outputs
-------
  analysis/data/deep_determinants/aadr_country_metrics.parquet
  analysis/figures/paper5_horserace/aadr_coverage_map.pdf
  analysis/figures/paper5_horserace/aadr_time_histograms.pdf
  analysis/figures/paper5_horserace/aadr_correlation_matrix.pdf
  analysis/paper5_horserace/aadr_sanity_check_regression.md  (printed inline)
"""

from __future__ import annotations
import warnings
warnings.simplefilter("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import seaborn as sns
import statsmodels.api as sm
import pycountry

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
DEEP_DET = DATA / "deep_determinants"
RAW_AADR = DEEP_DET / "_raw" / "aadr_v66"
FIGS = ROOT / "analysis" / "figures" / "paper5_horserace"
FIGS.mkdir(parents=True, exist_ok=True)

ANNO_FILE = RAW_AADR / "v66.1240K.aadr.PUB.anno"
NE_SHAPEFILE = "/tmp/ne_shp/ne_110m_admin_0_countries.shp"
PANEL_PATH = DATA / "deep_determinants_horserace.parquet"

SUBSTRATES = [
    "sigma_v_T_pre1750",
    "H_pred_pwadj",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]
OUTCOMES = [
    "log_pop_growth_1950_2025",
    "urban_change_1950_2025",
    "log_gdppc_2015",
    "dt_timing_year",
]
CONTROLS_BASE = [
    "abs_lat",
    "log_area",
    "landlocked",
    "ruggedness_proxy",
    "log_dist_neolithic",
]
# 3 FWER survivors (from robustness_battery.parquet wy_correction check)
FWER_CELLS = [
    ("log_pop_growth_1950_2025", "ancestral_yield_log"),
    ("log_pop_growth_1950_2025", "H_pred_pwadj"),
    ("log_gdppc_2015",           "H_pred_pwadj"),
]


# ============================================================================
# Step 1: Load AADR anno file
# ============================================================================

def load_anno() -> pd.DataFrame:
    """Load the .anno file, keeping only ancient samples (date > 0) that pass QC."""
    print(f"Loading {ANNO_FILE} ...")
    df = pd.read_csv(ANNO_FILE, sep="\t", low_memory=False, encoding="utf-8")
    print(f"  Raw: {len(df):,} samples, {df.shape[1]} columns")

    # Short-name the key columns
    date_col = df.columns[10]   # "Date mean in BP ..."
    assess_col = "ASSESSMENT"
    country_col = "Political Entity"

    df = df.rename(columns={
        date_col: "date_bp",
        country_col: "country_raw",
        assess_col: "assessment",
    })

    df["date_bp"] = pd.to_numeric(df["date_bp"], errors="coerce")

    # Filter: keep ancient samples only (date > 200 BP to exclude modern reference)
    # and passing / provisional QC
    good_assess = {"Pass", "PROVISIONAL_PASS", "MERGE_PASS"}
    mask = (
        (df["date_bp"] > 200) &
        (df["assessment"].isin(good_assess)) &
        (df["country_raw"].notna()) &
        (df["country_raw"] != "..")
    )
    df = df[mask].copy()
    print(f"  After QC filter (ancient + Pass/PP): {len(df):,} samples")
    return df


# ============================================================================
# Step 2: Country name -> ISO3 mapping
# ============================================================================

MANUAL_MAP: dict[str, str] = {
    "Russia":                    "RUS",
    "Turkey":                    "TUR",
    "Bosnia-Herzegovina":        "BIH",
    "Kosovo":                    "XKX",   # not in ISO but used in panel
    "Czech Republic":            "CZE",
    "Czechia":                   "CZE",
    "Slovakia":                  "SVK",
    "Republic of Macedonia":     "MKD",
    "North Macedonia":           "MKD",
    "Moldova":                   "MDA",
    "Tanzania":                  "TZA",
    "South Korea":               "KOR",
    "North Korea":               "PRK",
    "Iran":                      "IRN",
    "Syria":                     "SYR",
    "Bolivia":                   "BOL",
    "Venezuela":                 "VEN",
    "Vietnam":                   "VNM",
    "Laos":                      "LAO",
    "Democratic Republic of the Congo": "COD",
    "Republic of the Congo":     "COG",
    "Ivory Coast":               "CIV",
    "United States":             "USA",
    "USA":                       "USA",
    "England":                   "GBR",
    "Scotland":                  "GBR",
    "Wales":                     "GBR",
    "Crimea":                    "UKR",   # treat as Ukraine
    "Abkhazia":                  "GEO",   # disputed -> Georgia
    "South Ossetia":             "GEO",
    "Transnistria":              "MDA",
    "Taiwan":                    "TWN",
    "Palestine":                 "PSE",
    "East Timor":                "TLS",
    "Cape Verde":                "CPV",
    "Micronesia":                "FSM",
    "Comoros":                   "COM",
    "Reunion":                   "REU",
    "Guadeloupe":                "GLP",
    "Martinique":                "MTQ",
    "French Guiana":             "GUF",
    "Canary Islands":            "ESP",   # treat as Spain
    "Sardinia":                  "ITA",
    "Sicily":                    "ITA",
    "Corsica":                   "FRA",
    "Faroe Islands":             "FRO",
    "Gotland":                   "SWE",
    "Azores":                    "PRT",
}


def build_iso3_map(country_names: pd.Index) -> dict[str, str | None]:
    """Build {country_raw: iso3} mapping using pycountry + manual overrides."""
    # Pre-build pycountry lookup
    pc_map: dict[str, str] = {}
    for c in pycountry.countries:
        pc_map[c.name] = c.alpha_3
        if hasattr(c, "common_name"):
            pc_map[c.common_name] = c.alpha_3
        if hasattr(c, "official_name"):
            pc_map[c.official_name] = c.alpha_3

    mapping: dict[str, str | None] = {}
    unmatched = []
    for name in country_names:
        if name in MANUAL_MAP:
            mapping[name] = MANUAL_MAP[name]
        elif name in pc_map:
            mapping[name] = pc_map[name]
        else:
            # Try fuzzy search as fallback
            try:
                results = pycountry.countries.search_fuzzy(name)
                if results:
                    mapping[name] = results[0].alpha_3
                else:
                    mapping[name] = None
                    unmatched.append(name)
            except LookupError:
                mapping[name] = None
                unmatched.append(name)

    if unmatched:
        print(f"  Unmapped countries ({len(unmatched)}): {unmatched[:20]}")
    return mapping


# ============================================================================
# Step 3: Country-level aggregation
# ============================================================================

def aggregate_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-country metrics from the filtered anno DataFrame."""
    print("Aggregating by country ...")

    # Map country names to ISO3
    unique_names = df["country_raw"].unique()
    iso_map = build_iso3_map(unique_names)
    df = df.copy()
    df["iso3"] = df["country_raw"].map(iso_map)

    # Drop rows with no ISO3
    df = df.dropna(subset=["iso3"])
    print(f"  {len(df):,} samples after ISO3 mapping "
          f"({df['iso3'].nunique()} countries)")

    # Time period bins (BP = years before 1950 CE)
    # post_500bp: 200-500 BP  (roughly 1450-1750 CE)
    # 3000-500bp: 500-3000 BP (Bronze Age through medieval)
    # pre_3000bp: > 3000 BP   (Neolithic, Mesolithic)
    bp = df["date_bp"]

    grp = df.groupby("iso3")
    agg = pd.DataFrame({
        "n_samples_total":    grp.size(),
        "n_samples_pre_3000bp":   grp.apply(lambda g: (g["date_bp"] > 3000).sum(), include_groups=False),
        "n_samples_3000_500bp":   grp.apply(lambda g: ((g["date_bp"] >= 500) & (g["date_bp"] <= 3000)).sum(), include_groups=False),
        "n_samples_post_500bp":   grp.apply(lambda g: (g["date_bp"] < 500).sum(), include_groups=False),
        "mean_sample_age_bp": grp["date_bp"].mean(),
        "max_sample_age_bp":  grp["date_bp"].max(),
        "min_sample_age_bp":  grp["date_bp"].min(),
    }).reset_index()

    agg["temporal_span_bp"] = agg["max_sample_age_bp"] - agg["min_sample_age_bp"]

    # Country name for display
    name_mode = df.groupby("iso3")["country_raw"].agg(
        lambda x: x.mode().iloc[0] if len(x) > 0 else None
    )
    agg["country_name"] = agg["iso3"].map(name_mode)

    agg = agg.sort_values("n_samples_total", ascending=False)
    print(f"  Top 10 countries by sample count:")
    print(agg[["iso3", "country_name", "n_samples_total", "max_sample_age_bp",
               "temporal_span_bp"]].head(10).to_string(index=False))
    return agg


# ============================================================================
# Step 4: Deep population structure composite index
# ============================================================================

def build_composite_index(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Composite 'deep population structure' index for countries with >= 10 samples.

    Index components (each standardised to [0,1] then averaged):
      (i)  log_temporal_span: log(temporal_span_bp + 1) / max  -> time depth
      (ii) log_density: log(n_samples_total + 1) / max -> sampling density
      (iii) transition_coverage: binary presence of samples in 3 key windows
            - Neolithic 7000-5000 BP (0/1)
            - Bronze Age 5000-3000 BP (0/1)
            - Iron Age   3000-2500 BP (0/1)
            mean of three binaries -> 0 to 1
    Index = (i + ii + iii) / 3
    """
    sub = agg[agg["n_samples_total"] >= 10].copy()
    print(f"\nBuilding composite index for {len(sub)} countries (>= 10 samples) ...")

    # (i) log temporal span
    lt = np.log(sub["temporal_span_bp"] + 1)
    sub["comp_temporal"] = lt / lt.max()

    # (ii) log density
    ld = np.log(sub["n_samples_total"] + 1)
    sub["comp_density"] = ld / ld.max()

    # (iii) transition period coverage — need the raw sample-level data;
    # use pre-computed period columns as proxies for presence
    sub["neolithic_flag"]  = (sub["n_samples_pre_3000bp"] > 0).astype(float)  # >3000BP includes Neolithic
    sub["bronze_age_flag"] = (sub["n_samples_3000_500bp"] > 0).astype(float)  # 500-3000 BP spans Bronze/Iron
    # Iron Age 3000-2500 BP is within 3000_500bp bucket — we approximate as n_samples_pre_3000bp indicates
    # depth reaching at least Bronze Age; use max_sample_age proxy for Neolithic
    sub["neolithic_flag_7k"] = (sub["max_sample_age_bp"] >= 7000).astype(float)
    sub["bronze_flag_5k"]    = (sub["max_sample_age_bp"] >= 5000).astype(float)
    sub["iron_flag_3k"]      = (sub["max_sample_age_bp"] >= 3000).astype(float)
    sub["comp_transitions"]  = (
        sub["neolithic_flag_7k"] +
        sub["bronze_flag_5k"] +
        sub["iron_flag_3k"]
    ) / 3.0

    sub["deep_pop_index"] = (
        sub["comp_temporal"] +
        sub["comp_density"] +
        sub["comp_transitions"]
    ) / 3.0

    top5 = sub.nlargest(5, "deep_pop_index")[["iso3", "country_name", "deep_pop_index",
                                               "n_samples_total", "max_sample_age_bp"]]
    bot5 = sub.nsmallest(5, "deep_pop_index")[["iso3", "country_name", "deep_pop_index",
                                                "n_samples_total", "max_sample_age_bp"]]
    print("  Top 5 by composite index:")
    print(top5.to_string(index=False))
    print("  Bottom 5 by composite index:")
    print(bot5.to_string(index=False))

    return sub


# ============================================================================
# Figure 1: Coverage choropleth map
# ============================================================================

def fig_coverage_map(agg: pd.DataFrame) -> None:
    print("\nFigure 1: Coverage map ...")
    world = gpd.read_file(NE_SHAPEFILE)
    # Use ISO_A3_EH which is more complete than ISO_A3 (handles disputed territories)
    world = world.rename(columns={"ISO_A3_EH": "iso3"})

    # Some NE iso3 codes are "-99"; try name match as fallback
    merge = world.merge(agg[["iso3", "n_samples_total"]], on="iso3", how="left")
    merge["log_n"] = np.log1p(merge["n_samples_total"].fillna(0))

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    # Use a perceptually uniform colourmap; grey for zero
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under("lightgrey")
    vmin = 0.01   # just above 0 to trigger 'under' colour for true zeros

    merge.plot(
        column="log_n",
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=np.log1p(agg["n_samples_total"].max()),
        legend=True,
        legend_kwds={
            "label": "log(n samples + 1)",
            "orientation": "horizontal",
            "shrink": 0.6,
            "pad": 0.02,
        },
        missing_kwds={"color": "lightgrey", "label": "No data"},
        linewidth=0.3,
        edgecolor="0.7",
    )
    ax.set_title("AADR v66 — Ancient sample coverage by modern country\n"
                 "(1240K panel, QC-passing, date > 200 BP)",
                 fontsize=11)
    ax.set_axis_off()

    # Annotate a few high-coverage countries
    label_iso = ["GBR", "DEU", "RUS", "CHN", "ISR", "IRN"]
    for iso in label_iso:
        row = agg[agg["iso3"] == iso]
        if row.empty:
            continue
        geo = merge[merge["iso3"] == iso]
        if geo.empty:
            continue
        cx = geo.geometry.centroid.x.values[0]
        cy = geo.geometry.centroid.y.values[0]
        n = int(row["n_samples_total"].values[0])
        ax.annotate(f"{iso}\n{n}", xy=(cx, cy), fontsize=6.5,
                    ha="center", va="center", color="black",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5, lw=0))

    out = FIGS / "aadr_coverage_map.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 2: Time distribution histograms for top-10 countries
# ============================================================================

def fig_time_histograms(df_samples: pd.DataFrame, agg: pd.DataFrame) -> None:
    print("\nFigure 2: Time-period histograms for top-10 countries ...")
    # Get ISO3 in df_samples
    top10_iso = agg.nlargest(10, "n_samples_total")["iso3"].tolist()

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
    axes_flat = axes.flatten()

    for ax, iso in zip(axes_flat, top10_iso):
        sub = df_samples[df_samples["iso3"] == iso]
        name = agg.loc[agg["iso3"] == iso, "country_name"].values[0]
        n = len(sub)
        ax.hist(sub["date_bp"], bins=40, color="#2c7bb6", alpha=0.8,
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"{name}\n(n={n:,})", fontsize=9)
        ax.set_xlabel("Date (BP)", fontsize=8)
        ax.set_ylabel("Samples", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.invert_xaxis()   # older dates to the right

        # Mark key periods with vertical lines
        ymax = ax.get_ylim()[1]
        for bp_mark, lbl, clr in [
            (7000, "Neo", "#e41a1c"),
            (5000, "BA",  "#ff7f00"),
            (3000, "IA",  "#4daf4a"),
            (500,  "Med", "#984ea3"),
        ]:
            ax.axvline(bp_mark, color=clr, lw=0.8, ls="--", alpha=0.7)
            ax.text(bp_mark, ymax * 0.88, lbl,
                    color=clr, fontsize=6, ha="center")

    fig.suptitle("AADR v66 — Sample-age distributions for 10 most-covered countries",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIGS / "aadr_time_histograms.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 3: Correlation matrix
# ============================================================================

def fig_correlation_matrix(agg: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Merge AADR metrics with horserace panel and plot correlation matrix."""
    print("\nFigure 3: Correlation matrix ...")

    merged = panel.merge(
        agg[["iso3", "n_samples_total", "temporal_span_bp",
             "mean_sample_age_bp", "max_sample_age_bp",
             "n_samples_pre_3000bp", "n_samples_3000_500bp"]],
        on="iso3",
        how="left",
    )
    merged["log_n_samples"] = np.log1p(merged["n_samples_total"].fillna(0))

    corr_vars = (
        ["log_n_samples", "temporal_span_bp", "max_sample_age_bp"] +
        SUBSTRATES +
        OUTCOMES
    )
    corr_vars = [v for v in corr_vars if v in merged.columns]

    labels = {
        "log_n_samples":          "log(n_samples+1)",
        "temporal_span_bp":       "Temporal span (BP)",
        "max_sample_age_bp":      "Max age (BP)",
        "sigma_v_T_pre1750":      "σᵥᵀ (climate)",
        "H_pred_pwadj":           "H_pred (het)",
        "ancestral_yield_log":    "Anc. yield (log)",
        "pandemic_intensity_norm":"Pandemic intensity",
        "log_pop_growth_1950_2025":"Log pop growth",
        "urban_change_1950_2025": "Urban change",
        "log_gdppc_2015":         "Log GDPpc",
        "dt_timing_year":         "DT timing",
    }
    corr_mat = merged[corr_vars].rename(columns=labels).corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(
        corr_mat,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )
    ax.set_title("AADR coverage metrics vs pre-industrial channels and modern outcomes\n"
                 "(Pearson r, pairwise complete observations)",
                 fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    out = FIGS / "aadr_correlation_matrix.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")
    return merged


# ============================================================================
# Step 4: Sanity-check regression — FWER cells + AADR density control
# ============================================================================

def run_sanity_regressions(merged: pd.DataFrame) -> str:
    """For 3 FWER cells, add log(n_samples+1) as control. Return markdown table."""
    print("\nSanity-check regressions: 3 FWER cells + AADR density control ...")
    pathway_cols = sorted(c for c in merged.columns
                          if c.startswith("pathway_") and c != "pathway_0")

    rows = []
    for outcome, substrate in FWER_CELLS:
        for include_aadr in [False, True]:
            controls = CONTROLS_BASE + pathway_cols
            if include_aadr:
                controls = controls + ["log_n_samples"]
            needed = [outcome, substrate] + controls
            d = merged.dropna(subset=needed)
            if len(d) < 10:
                rows.append(dict(outcome=outcome, substrate=substrate,
                                 with_aadr=include_aadr, t_stat=np.nan,
                                 p_val=np.nan, n_obs=0))
                continue
            X = sm.add_constant(d[[substrate] + controls])
            res = sm.OLS(d[outcome], X).fit()
            t = float(res.tvalues[substrate])
            p = float(res.pvalues[substrate])
            rows.append(dict(outcome=outcome, substrate=substrate,
                             with_aadr=include_aadr,
                             t_stat=t, p_val=p, n_obs=len(d)))

    df_r = pd.DataFrame(rows)

    # Format as markdown table
    lines = [
        "## Sanity-check: FWER cells + AADR density control",
        "",
        "| Outcome | Substrate | AADR control | |t| | p-value | N |",
        "|---------|-----------|:------------:|-----|---------|---|",
    ]
    for _, r in df_r.iterrows():
        aadr_flag = "yes" if r["with_aadr"] else "no"
        t_str = f"{abs(r['t_stat']):.3f}" if not np.isnan(r["t_stat"]) else "n/a"
        p_str = f"{r['p_val']:.3f}" if not np.isnan(r["p_val"]) else "n/a"
        lines.append(
            f"| {r['outcome']} | {r['substrate']} | {aadr_flag} "
            f"| {t_str} | {p_str} | {int(r['n_obs'])} |"
        )

    md = "\n".join(lines)
    print(md)
    return md


# ============================================================================
# Figure 4: Composite index distribution
# ============================================================================

def fig_composite_index(sub_idx: pd.DataFrame) -> None:
    """Horizontal bar chart of composite index for countries >= 10 samples."""
    print("\nFigure 4: Composite index distribution ...")
    sub = sub_idx.sort_values("deep_pop_index", ascending=True).copy()
    n = len(sub)

    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.22)))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n))
    bars = ax.barh(range(n), sub["deep_pop_index"].values, color=colors,
                   edgecolor="white", linewidth=0.3)

    labels = [f"{row['iso3']} ({int(row['n_samples_total'])})"
              for _, row in sub.iterrows()]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Deep Population Structure Index (0–1)", fontsize=9)
    ax.set_title(
        "AADR v66 — Deep population structure composite index\n"
        "Countries with ≥ 10 QC-passing ancient samples\n"
        "(components: temporal span, sample density, transition-period coverage)",
        fontsize=9,
    )
    ax.axvline(sub["deep_pop_index"].median(), color="steelblue", ls="--",
               lw=1, label=f"Median = {sub['deep_pop_index'].median():.2f}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    out = FIGS / "aadr_composite_index.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Save parquet
# ============================================================================

def save_metrics(agg: pd.DataFrame, sub_idx: pd.DataFrame) -> None:
    """Merge composite index back into agg and save parquet."""
    idx_cols = ["iso3", "deep_pop_index", "comp_temporal", "comp_density",
                "comp_transitions"]
    merged = agg.merge(
        sub_idx[idx_cols] if not sub_idx.empty else pd.DataFrame(columns=idx_cols),
        on="iso3", how="left"
    )
    merged["aadr_version"] = "v66"
    out_path = DEEP_DET / "aadr_country_metrics.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path} ({len(merged)} rows, {len(merged.columns)} cols)")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("AADR v66 — Country-level composite index exploration")
    print("=" * 70)

    # 1. Load
    df_raw = load_anno()

    # 2. Map ISO3
    unique_names = df_raw["country_raw"].unique()
    iso_map = build_iso3_map(unique_names)
    df_raw["iso3"] = df_raw["country_raw"].map(iso_map)
    df_samples = df_raw.dropna(subset=["iso3"]).copy()

    # 3. Aggregate
    agg = aggregate_by_country(df_raw)

    # Coverage stats
    n_ge1  = (agg["n_samples_total"] >= 1).sum()
    n_ge10 = (agg["n_samples_total"] >= 10).sum()
    n_ge50 = (agg["n_samples_total"] >= 50).sum()
    print(f"\nCoverage: {n_ge1} countries >=1 sample, "
          f"{n_ge10} >=10, {n_ge50} >=50")

    # 4. Composite index
    sub_idx = build_composite_index(agg)

    # 5. Load horserace panel
    panel = pd.read_parquet(PANEL_PATH)

    # 6. Figures
    fig_coverage_map(agg)
    fig_time_histograms(df_samples, agg)
    merged = fig_correlation_matrix(agg, panel)
    md_table = run_sanity_regressions(merged)
    if not sub_idx.empty:
        fig_composite_index(sub_idx)

    # 7. Save parquet
    save_metrics(agg, sub_idx)

    print("\n" + "=" * 70)
    print("Done.  AADR v66 exploration complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
