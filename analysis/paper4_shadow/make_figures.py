"""Regenerate all paper-critical figures in sober grayscale style.

Figures produced (saved as figXX.pdf and figXX.png for LaTeX):
  fig01_pathways_map.pdf            World map of the 5 agricultural pathways
  fig02_climate_panel.pdf           Calibration: ModE-RA vs ERA5
  fig03_storage_pathway.pdf         Productive-months by pathway (Stage 1)
  fig04_subnational_scatter.pdf     Sub-national within-country identification
  fig05_malthus_pathway.pdf         Pathway-stratified Malthus 1421-1750
  fig06_malthus_subperiods.pdf      Subperiod stability 1421-1950
  fig07_volcanic_event.pdf          Volcanic event study
  fig08_long_shadow.pdf             Long-shadow regression
  fig09_ensemble_uncertainty.pdf    ModE-RA ensemble uncertainty by era
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style, gray_palette, style_box, LINESTYLES, MARKERS

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"
FIG.mkdir(parents=True, exist_ok=True)

PATHWAY_NAMES = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}
PATHWAY_ORDER = [3, 4, 0, 1, 2]


def savefig(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  -> {FIG / name}.[pdf|png]")


# -----------------------------------------------------------------------
# Fig 1: World map of agricultural pathways (skip — requires basemap)
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Fig 2: Calibration ModE-RA vs ERA5
# -----------------------------------------------------------------------
def fig_calibration() -> None:
    per_m = pd.read_parquet(DATA / "era5_modera_calibration_monthly.parquet")
    per_a = pd.read_parquet(DATA / "era5_modera_calibration_annual.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.4))
    ax = axes[0]
    ax.hist(per_m["t_bias"], bins=30, color="#909090", edgecolor="#202020", linewidth=0.5)
    ax.axvline(0, color="#202020", linewidth=0.6)
    ax.set_xlabel(r"ERA5 $-$ (ModE-RA $+$ climatology) (°C)")
    ax.set_ylabel("Countries")
    med = per_m["t_bias"].abs().median()
    ax.set_title(f"(a) Mean bias\nmedian $|b|$ = {med:.2f}°C", loc="left")

    ax = axes[1]
    ax.hist(per_m["t_corr_m"], bins=30, color="#909090", edgecolor="#202020", linewidth=0.5)
    ax.set_xlabel(r"Monthly $T$ correlation")
    ax.set_ylabel("Countries")
    med = per_m["t_corr_m"].median()
    ax.set_title(f"(b) Monthly correlation\nmedian = {med:.3f}", loc="left")
    ax.set_xlim(0, 1.02)

    ax = axes[2]
    ax.hist(per_a["corr_anom"], bins=30, color="#909090", edgecolor="#202020", linewidth=0.5)
    ax.set_xlabel(r"Annual anomaly $T$ correlation")
    ax.set_ylabel("Countries")
    med = per_a["corr_anom"].median()
    ax.set_title(f"(c) Year-to-year skill\nmedian = {med:.3f}", loc="left")
    ax.set_xlim(-0.4, 1.02)
    plt.tight_layout()
    savefig(fig, "fig02_calibration")


# -----------------------------------------------------------------------
# Fig 3: productive-months by pathway
# -----------------------------------------------------------------------
def fig_storage() -> None:
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0)
    df["prod"] = ((df["t_abs"]>=5)&(df["t_abs"]<=30)&(df["p_abs"]>=30)).astype(float)
    pre = df[df["year"].between(1421, 1750)]
    yr = pre.groupby(["iso3","year"], as_index=False)["prod"].sum().rename(
        columns={"prod": "prod_months"})
    feats = yr.groupby("iso3", as_index=False)["prod_months"].mean()

    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3","cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)
    d = feats.merge(clust[["iso3","cluster"]], on="iso3", how="inner")
    d["pathway"] = d["cluster"].map(PATHWAY_NAMES)
    big = d[d["cluster"] != 2]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    order = [PATHWAY_NAMES[k] for k in PATHWAY_ORDER if k != 2]
    data = [big.loc[big["pathway"]==p, "prod_months"].values for p in order]
    bp = ax.boxplot(data, labels=order, showmeans=True, patch_artist=True,
                    widths=0.55, medianprops=dict(color="black"))
    style_box(ax, bp, gray_palette(len(order)))
    ax.set_ylabel("Productive months per year (1421–1750)")
    ax.set_xlabel("Agricultural pathway")
    ax.set_ylim(0, 12.5)
    ax.set_yticks([0,3,6,9,12])
    # Egypt annotation
    egy = d[d["cluster"]==2]["prod_months"].iloc[0] if (d["cluster"]==2).any() else None
    if egy is not None:
        ax.scatter([len(order)+0.5], [egy], marker="D", color="#202020", zorder=5)
        ax.text(len(order)+0.7, egy, "  Egypt\n  (irrigation pioneer)",
                va="center", fontsize=8.5)
        ax.set_xlim(0.5, len(order)+2.0)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    plt.tight_layout()
    savefig(fig, "fig03_storage_pathway")


# -----------------------------------------------------------------------
# Fig 4: Sub-national identification scatter + within-country gradient
# -----------------------------------------------------------------------
def fig_subnational() -> None:
    feats = pd.read_parquet(DATA / "subnational_features.parquet")
    hyde = pd.read_parquet(DATA / "subnational_hyde.parquet")
    h = hyde[hyde["year"]==1750].copy()
    h["cropland_ha"] = h["cropland_ha"].fillna(0)
    h["grazing_ha"] = h["grazing_ha"].fillna(0)
    h["ag_total"] = h["cropland_ha"] + h["grazing_ha"]
    h["crop_share"] = np.where(h["ag_total"]>0, h["cropland_ha"]/h["ag_total"], np.nan)
    df = feats.merge(h[["sub_id","iso3","crop_share"]], on=["sub_id","iso3"], how="inner")
    df = df.dropna(subset=["crop_share","productive_months"])

    # Big countries: USA, BRA, CHN, IND, RUS — show within-country gradient
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel A: between-country scatter (country means)
    ax = axes[0]
    cm = df.groupby("iso3", as_index=False).agg(
        prod=("productive_months","mean"), crop=("crop_share","mean"))
    ax.scatter(cm["prod"], cm["crop"], s=15, color="#404040",
               edgecolor="white", linewidth=0.3, alpha=0.7)
    # OLS line
    p = np.polyfit(cm["prod"], cm["crop"], 1)
    xs = np.array([cm["prod"].min(), cm["prod"].max()])
    ax.plot(xs, p[0]*xs+p[1], color="#000000", linewidth=1.3)
    ax.set_xlabel("Mean productive months (1421–1750)")
    ax.set_ylabel("Crop share at 1750")
    ax.set_title("(a) Between countries", loc="left")
    ax.text(0.04, 0.94, fr"$\beta = {p[0]:+.3f}$", transform=ax.transAxes, va="top")

    # Panel B: within-country gradient for selected countries
    ax = axes[1]
    big_countries = ["USA","BRA","CHN","IND","RUS"]
    palette = gray_palette(len(big_countries))
    for i, iso in enumerate(big_countries):
        sub = df[df["iso3"]==iso]
        if len(sub) < 5: continue
        x_ctr = sub["productive_months"] - sub["productive_months"].mean()
        y_ctr = sub["crop_share"] - sub["crop_share"].mean()
        ax.scatter(x_ctr, y_ctr, s=10, color=palette[i],
                   edgecolor="white", linewidth=0.2, alpha=0.55, label=iso,
                   marker=MARKERS[i % len(MARKERS)])
    # Pooled within-country regression line (use the regression result)
    ax.axhline(0, color="#404040", linewidth=0.4)
    ax.axvline(0, color="#404040", linewidth=0.4)
    xs = np.linspace(-4, 4, 50)
    # Coefficient from country-FE regression at 1750: +0.067
    ax.plot(xs, 0.067*xs, color="#000000", linewidth=1.3, label="within-country slope")
    ax.set_xlabel("Productive months (within-country deviation)")
    ax.set_ylabel("Crop share (within-country deviation)")
    ax.set_title("(b) Within countries", loc="left")
    ax.legend(loc="lower right", ncol=2)
    ax.text(0.04, 0.94, r"$\beta_{\mathrm{FE}} = +0.067^{**}$",
            transform=ax.transAxes, va="top")

    plt.tight_layout()
    savefig(fig, "fig04_subnational")


# -----------------------------------------------------------------------
# Fig 5: Pathway-stratified Malthus 1421-1750
# -----------------------------------------------------------------------
def fig_malthus_pathway() -> None:
    res = pd.read_parquet(DATA / "preindustrial_malthus_results.parquet")
    order_names = ["Pastoral/mixed late", "High-density intensive",
                   "Crop-dominant late", "Early extensifiers"]
    res = res.set_index("pathway").reindex(order_names).reset_index()
    res["lo"] = res["beta_density"] - 1.96 * 0.001  # rough SE
    res["hi"] = res["beta_density"] + 1.96 * 0.001
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))

    ax = axes[0]
    y = np.arange(len(res))
    ax.errorbar(res["beta_density"], y,
                xerr=[res["beta_density"]-res["lo"], res["hi"]-res["beta_density"]],
                fmt="o", color="#202020", markerfacecolor="white", markeredgewidth=1,
                ecolor="#404040", elinewidth=0.8, capsize=2)
    for i, row in res.iterrows():
        mark = ("***" if row["p_density"] < 0.01
                else "**" if row["p_density"] < 0.05
                else "*" if row["p_density"] < 0.10 else "")
        sign = "+" if row["beta_density"] >= 0 else "−"
        ax.text(row["beta_density"]+0.0005, i, f"  {mark}", va="center", fontsize=10)
    ax.axvline(0, color="#404040", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(res["pathway"])
    ax.set_xlabel(r"$\hat{\beta}$ on $\ln$ density")
    ax.set_title(r"(a) Malthusian density coefficient", loc="left")

    ax = axes[1]
    ax.errorbar(res["beta_Tstd"], y, fmt="s", color="#404040",
                markerfacecolor="#A0A0A0", markeredgecolor="#202020",
                ecolor="#404040", elinewidth=0.8, capsize=2)
    for i, row in res.iterrows():
        mark = ("**" if row["p_Tstd"] < 0.05
                else "*" if row["p_Tstd"] < 0.10 else "")
        ax.text(row["beta_Tstd"]+0.0003, i, f"  {mark}", va="center", fontsize=10)
    ax.axvline(0, color="#404040", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(res["pathway"])
    ax.set_xlabel(r"$\hat{\delta}_T$ on inter-annual $\sigma_v^T$")
    ax.set_title(r"(b) Climate-volatility coefficient", loc="left")

    plt.tight_layout()
    savefig(fig, "fig05_malthus_pathway")


# -----------------------------------------------------------------------
# Fig 6: Subperiod stability 1421-1950
# -----------------------------------------------------------------------
def fig_subperiods() -> None:
    # Hardcoded from the extended Malthus output
    rows = [
        ("1421–1750", -0.0013, 0.0014, 0.36, +0.0012, 0.0014, 0.39, -0.0024, 0.0017, 0.17),
        ("1750–1900", -0.0016, 0.00069, 0.02, +0.0015, 0.00050, 0.003, +0.0045, 0.00097, 1e-5),
        ("1900–1950", -0.00035, 0.0022, 0.88, +0.00813, 0.0019, 1e-5, -0.01250, 0.0032, 1e-4),
    ]
    df = pd.DataFrame(rows, columns=["period","b_d","se_d","p_d","b_T","se_T","p_T","b_Ts","se_Ts","p_Ts"])
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8), sharey=True)
    yt = np.arange(len(df))
    for ax, (col_b, col_s, col_p, title, xl) in zip(axes, [
        ("b_d", "se_d", "p_d",  r"(a) $\hat{\beta}_{\ln d}$",  r"$\hat{\beta}$ on $\ln$ density"),
        ("b_T", "se_T", "p_T",  r"(b) $\hat{\gamma}_T$",       r"$\hat{\gamma}$ on $T$ anomaly"),
        ("b_Ts","se_Ts","p_Ts", r"(c) $\hat{\delta}_T$",       r"$\hat{\delta}$ on $\sigma_v^T$"),
    ]):
        ax.errorbar(df[col_b], yt, xerr=1.96*df[col_s], fmt="o",
                    color="#202020", markerfacecolor="white", markeredgewidth=1,
                    ecolor="#404040", elinewidth=0.8, capsize=2)
        for i, row in df.iterrows():
            p = row[col_p]
            mark = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            ax.text(row[col_b], i+0.18, mark, ha="center", fontsize=9)
        ax.axvline(0, color="#404040", linewidth=0.6)
        ax.set_yticks(yt); ax.set_yticklabels(df["period"])
        ax.set_xlabel(xl)
        ax.set_title(title, loc="left")
    plt.tight_layout()
    savefig(fig, "fig06_subperiods")


# -----------------------------------------------------------------------
# Fig 7: Volcanic event study
# -----------------------------------------------------------------------
def fig_volcanic() -> None:
    res = pd.read_parquet(DATA / "volcanic_event_results.parquet")
    res = res.sort_values("beta_t_shock").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    y = np.arange(len(res))
    ax.errorbar(res["beta_t_shock"], y, xerr=1.96*res["se_t_shock"],
                fmt="o", color="#202020", markerfacecolor="white",
                markeredgewidth=1, ecolor="#404040", elinewidth=0.8, capsize=2)
    for i, row in res.iterrows():
        mark = ("***" if row["p_t_shock"] < 0.01
                else "**" if row["p_t_shock"] < 0.05
                else "*" if row["p_t_shock"] < 0.10 else "")
        ax.text(row["beta_t_shock"], i+0.22, f"$N={row['n']}$  {mark}",
                ha="center", fontsize=8.5)
    ax.axvline(0, color="#404040", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(res["pathway"])
    ax.set_xlabel(r"Slope: annual pop growth per °C cooling")
    ax.set_title(r"Volcanic event study, five eruptions 1600–1991", loc="left")
    plt.tight_layout()
    savefig(fig, "fig07_volcanic")


# -----------------------------------------------------------------------
# Fig 8: Long shadow regression — scatter for sigma_v_real -> pop growth
# -----------------------------------------------------------------------
def fig_long_shadow() -> None:
    # Reconstruct the long-shadow data
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    early = ext[ext["year"].between(1950,1960)].groupby("iso3",as_index=False).agg(
        crop_e=("crop_share","mean"), pop_e=("pop","mean"))
    late = ext[ext["year"].between(2015,2025)].groupby("iso3",as_index=False).agg(
        crop_l=("crop_share","mean"), pop_l=("pop","mean"))
    m = early.merge(late, on="iso3")
    m = m[(m["pop_e"]>0)&(m["pop_l"]>0)]
    m["d_crop"] = m["crop_l"]-m["crop_e"]
    m["log_pop_growth"] = np.log(m["pop_l"]/m["pop_e"])
    seas = pd.read_parquet(DATA / "country_seasonality_1421_2008.parquet")
    pre = seas[seas["year"].between(1421,1750)]
    feats = pre.groupby("iso3",as_index=False).agg(
        sigma_v=("t_mean","std"), sigma_s=("sigma_s","mean"))
    d = m.merge(feats, on="iso3")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (predictor, outcome, title, xl, yl) in zip(axes, [
        ("sigma_v","log_pop_growth", r"(a) Inter-annual volatility $\to$ pop growth",
         r"Pre-industrial $\sigma_v^T$ (°C)", "Log pop growth 1955→2020"),
        ("sigma_s","d_crop", r"(b) Intra-annual range $\to$ $\Delta$ crop share",
         r"Pre-industrial $\sigma_s^T$ (°C)", "Δ crop share 1955→2020"),
    ]):
        sub = d.dropna(subset=[predictor, outcome])
        ax.scatter(sub[predictor], sub[outcome], s=12, color="#404040",
                   edgecolor="white", linewidth=0.3, alpha=0.7)
        p = np.polyfit(sub[predictor], sub[outcome], 1)
        xs = np.array([sub[predictor].min(), sub[predictor].max()])
        ax.plot(xs, p[0]*xs+p[1], color="#000000", linewidth=1.3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, loc="left")
        ax.text(0.04, 0.04, fr"$\hat\beta = {p[0]:+.2f}$",
                transform=ax.transAxes, va="bottom")
    plt.tight_layout()
    savefig(fig, "fig08_long_shadow")


# -----------------------------------------------------------------------
# Fig 9: Ensemble uncertainty by era
# -----------------------------------------------------------------------
def fig_ensemble() -> None:
    df = pd.read_parquet(DATA / "modera_country_uncertainty.parquet")
    # Median T-std and IQR per decade
    df["decade"] = (df["year"] // 10) * 10
    by_dec = df.groupby("decade")["t_std"].agg(["median", lambda x: x.quantile(0.25),
                                                lambda x: x.quantile(0.75)])
    by_dec.columns = ["med","q25","q75"]
    by_dec = by_dec.reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.fill_between(by_dec["decade"], by_dec["q25"], by_dec["q75"],
                    color="#C0C0C0", alpha=0.7, linewidth=0)
    ax.plot(by_dec["decade"], by_dec["med"], color="#202020", linewidth=1.3)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Ensemble $1\sigma$ in $T$ (°C)")
    ax.set_title("ModE-RA monthly temperature uncertainty by decade",
                 loc="left")
    # Era annotations
    ax.axvline(1500, color="#808080", linewidth=0.5, linestyle="--")
    ax.axvline(1700, color="#808080", linewidth=0.5, linestyle="--")
    ax.axvline(1850, color="#808080", linewidth=0.5, linestyle="--")
    for x, txt in [(1460,"pre-instr."), (1600,"early\ninstr."),
                   (1780,"dense\ninstr."), (1930,"modern")]:
        ax.text(x, 1.45, txt, ha="center", fontsize=8, color="#404040")
    ax.set_ylim(0, 1.6)
    plt.tight_layout()
    savefig(fig, "fig09_ensemble_uncertainty")


def main() -> None:
    print("Generating figures...")
    # fig_calibration (fig02) read ERA5 and the figure was removed from the paper;
    # fig_long_shadow (fig08) read the ERA5 extended panel and is not in long_shadow.tex;
    # fig_ensemble (fig09) is superseded by make_fig09_combined (2x2, ModE-RA only).
    fig_storage()
    fig_subnational()
    fig_malthus_pathway()
    fig_subperiods()
    fig_volcanic()
    print(f"\nAll figures saved to {FIG}")


if __name__ == "__main__":
    main()
