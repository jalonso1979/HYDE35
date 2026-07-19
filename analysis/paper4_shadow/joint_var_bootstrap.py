"""Joint VAR robustness exercise (b): wild-cluster bootstrap on the
pathway-stratified VSSI coefficients in the three-equation panel VAR.

The original paper reports twelve pathway × equation slopes (4 informative
pathways × 3 equations) and concludes that "each pathway has exactly one
significant margin".  With 12 tests at the 5% level a casual reader would
expect ~0.6 false discoveries by chance, so the multiple-testing concern is
real.  We address it two ways:

(i)  Wild-cluster bootstrap (Webb 6-point) for each of the 12 slopes,
     re-sampling residuals by country (the clustering unit).  Reports
     bootstrap p-values that are robust to small-cluster asymmetry.
(ii) Permutation inference: re-shuffle the pathway labels across countries
     1,000 times, re-run the pathway-stratified joint VAR each time, and
     compare the observed pattern (number of significant coefficients per
     equation) against the permutation null.

Outputs:
    analysis/data/joint_var_bootstrap_pvalues.parquet
    analysis/figures/paper4_v2/figJ_bootstrap.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}

# Webb (2014) 6-point wild bootstrap auxiliary distribution
WEBB_VALUES = np.array([-np.sqrt(3/2), -1.0, -np.sqrt(1/2),
                          np.sqrt(1/2),  1.0,  np.sqrt(3/2)])

CONTROLS = ["log_pop", "log_crop_share", "log_urban_share",
            "t_bar_dev", "p_bar_dev", "t_sd_dev", "vssi_int"]


def _fit_cluster_se(d: pd.DataFrame, lhs: str, controls: list[str]
                      ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Return (beta_vssi, se_vssi, p_vssi, residuals, fitted)."""
    d = d.dropna(subset=[lhs] + controls + ["iso3"]).copy()
    g = d.groupby("iso3")
    for c in [lhs] + controls:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[controls])
    res = sm.OLS(d[lhs], X).fit(cov_type="cluster",
                                  cov_kwds={"groups": d["iso3"]})
    beta = float(res.params.get("vssi_int", np.nan))
    se = float(res.bse.get("vssi_int", np.nan))
    p = float(res.pvalues.get("vssi_int", np.nan))
    return beta, se, p, res.resid.values, res.fittedvalues.values


def _wild_cluster_bootstrap(d: pd.DataFrame, lhs: str, controls: list[str],
                              n_boot: int = 1999, seed: int = 42) -> dict:
    """Webb 6-point wild-cluster bootstrap-t for the VSSI coefficient.

    For each bootstrap iteration, draw one Webb sign per country, multiply
    that country's residuals by the sign, construct y_b = X β_R + ε_b
    (Mammen restricted bootstrap), re-fit, and collect t-statistics for
    VSSI.  p-value = share of |t_b| >= |t_obs|.
    """
    d = d.dropna(subset=[lhs] + controls + ["iso3"]).copy()
    if len(d) < 30 or d["iso3"].nunique() < 3:
        return None
    g = d.groupby("iso3")
    for c in [lhs] + controls:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[controls].values)
    y = d[lhs].values
    iso = d["iso3"].values

    # Observed t-stat (cluster SE)
    res_full = sm.OLS(y, X).fit(cov_type="cluster",
                                  cov_kwds={"groups": iso})
    obs_t = float(res_full.tvalues[X.shape[1] - 1])  # vssi_int is last col

    # Restricted fit: drop vssi_int (last column)
    X_r = X[:, :-1]
    res_r = sm.OLS(y, X_r).fit()
    resid_r = y - X_r @ res_r.params
    fit_r = X_r @ res_r.params

    rng = np.random.default_rng(seed)
    countries = np.unique(iso)
    iso_to_idx = {c: np.flatnonzero(iso == c) for c in countries}

    t_boots = np.empty(n_boot)
    for b in range(n_boot):
        signs = rng.choice(WEBB_VALUES, size=len(countries))
        eps_b = np.empty_like(resid_r)
        for c, s in zip(countries, signs):
            eps_b[iso_to_idx[c]] = s * resid_r[iso_to_idx[c]]
        y_b = fit_r + eps_b
        try:
            res_b = sm.OLS(y_b, X).fit(cov_type="cluster",
                                         cov_kwds={"groups": iso})
            t_boots[b] = float(res_b.tvalues[X.shape[1] - 1])
        except Exception:
            t_boots[b] = np.nan
    t_boots = t_boots[np.isfinite(t_boots)]
    p_boot = (np.abs(t_boots) >= np.abs(obs_t)).mean()
    beta = float(res_full.params[X.shape[1] - 1])
    se = float(res_full.bse[X.shape[1] - 1])
    return {"beta": beta, "se": se, "t_obs": obs_t,
            "p_asymp": float(res_full.pvalues[X.shape[1] - 1]),
            "p_boot": float(p_boot), "n": int(res_full.nobs),
            "n_boot_eff": int(len(t_boots))}


def _permutation_label_shuffle(panel: pd.DataFrame, controls: list[str],
                                 n_perm: int = 1000, seed: int = 7) -> pd.DataFrame:
    """Permutation null: how often does a random pathway labeling produce
    the observed number of significant coefficients?"""
    rng = np.random.default_rng(seed)
    countries = panel[["iso3", "cluster"]].drop_duplicates()
    labels = countries["cluster"].values
    iso = countries["iso3"].values

    def _count_sig(df: pd.DataFrame) -> dict:
        """Run pathway-stratified joint VAR on df, count significant slopes
        per equation."""
        counts = {"pop": 0, "crop": 0, "urb": 0}
        for cl in sorted(df["cluster"].dropna().unique()):
            sub = df[df["cluster"] == cl]
            if len(sub) < 30 or sub["iso3"].nunique() < 3:
                continue
            for tag, lhs in [("pop", "g_pop_ann"), ("crop", "g_crop_ann"),
                              ("urb", "g_urb_ann")]:
                try:
                    sub2 = sub.dropna(subset=[lhs] + controls + ["iso3"]).copy()
                    g = sub2.groupby("iso3")
                    for c in [lhs] + controls:
                        sub2[c] = sub2[c] - g[c].transform("mean")
                    X = sm.add_constant(sub2[controls])
                    res = sm.OLS(sub2[lhs], X).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": sub2["iso3"]})
                    if res.pvalues.get("vssi_int", 1.0) < 0.05:
                        counts[tag] += 1
                except Exception:
                    pass
        return counts

    obs = _count_sig(panel)
    print(f"  observed (real labels): {obs}")

    perm_counts = []
    for k in range(n_perm):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        shuf_map = dict(zip(iso, shuffled))
        p2 = panel.copy()
        p2["cluster"] = p2["iso3"].map(shuf_map)
        perm_counts.append(_count_sig(p2))
        if (k + 1) % 100 == 0:
            print(f"    permutation {k+1}/{n_perm}", flush=True)
    perm_df = pd.DataFrame(perm_counts)
    perm_df["total"] = perm_df.sum(axis=1)
    return perm_df, obs


def main() -> None:
    print("=== Joint VAR robustness (b): wild-cluster bootstrap + permutation ===\n")

    panel = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    print(f"Panel: {len(panel):,} rows, {panel['iso3'].nunique()} countries, "
          f"{panel['cluster'].nunique()} pathways")

    # Wild-cluster bootstrap on each pathway × equation slope
    print("\n=== Wild-cluster bootstrap (Webb 6-point, n=1999) ===")
    rows = []
    for cl in sorted(panel["cluster"].dropna().unique()):
        sub = panel[panel["cluster"] == cl]
        if len(sub) < 30 or sub["iso3"].nunique() < 3:
            print(f"  Cluster {cl} ({PATHWAY_NAMES.get(cl, '?')}): N<30 or <3 countries, skip")
            continue
        for tag, lhs, eq_name in [
            ("pop",  "g_pop_ann", "Δ log Pop"),
            ("crop", "g_crop_ann", "Δ log CropShare"),
            ("urb",  "g_urb_ann", "Δ log UrbanShare"),
        ]:
            r = _wild_cluster_bootstrap(sub, lhs, CONTROLS, n_boot=1999, seed=42)
            if r is None:
                continue
            label = "***" if r["p_boot"] < 0.01 else "**" if r["p_boot"] < 0.05 \
                else "*" if r["p_boot"] < 0.10 else ""
            label_asymp = "***" if r["p_asymp"] < 0.01 else "**" if r["p_asymp"] < 0.05 \
                else "*" if r["p_asymp"] < 0.10 else ""
            print(f"  {PATHWAY_NAMES.get(cl, '?'):<22s} {eq_name:<18s}: "
                  f"β={r['beta']:+.6f}  asymp p={r['p_asymp']:.3g} {label_asymp:3s}  "
                  f"boot p={r['p_boot']:.3g} {label}")
            rows.append({"pathway": PATHWAY_NAMES.get(cl, "?"),
                          "equation": eq_name,
                          "beta": r["beta"], "se": r["se"],
                          "t_obs": r["t_obs"],
                          "p_asymp": r["p_asymp"],
                          "p_boot": r["p_boot"],
                          "n": r["n"],
                          "n_boot_eff": r["n_boot_eff"]})
    res = pd.DataFrame(rows)
    res.to_parquet(DATA / "joint_var_bootstrap_pvalues.parquet", index=False)
    print(f"\nSaved {DATA/'joint_var_bootstrap_pvalues.parquet'}")

    # Bonferroni-corrected significance at the 5% level (12 tests, but we
    # report at the equation level too)
    print("\n=== Multiple-testing corrections (5% nominal) ===")
    n_tests = len(res)
    print(f"Total tests: {n_tests}")
    bonf = 0.05 / n_tests
    print(f"Bonferroni threshold: p < {bonf:.4f}")
    sig_bonf_asymp = (res["p_asymp"] < bonf).sum()
    sig_bonf_boot = (res["p_boot"] < bonf).sum()
    print(f"  Significant under Bonferroni: asymp={sig_bonf_asymp}, boot={sig_bonf_boot}")
    # BH FDR
    s = res.sort_values("p_boot").reset_index(drop=True)
    m = len(s)
    s["bh_thresh"] = (np.arange(1, m+1) / m) * 0.05
    s["bh_sig"] = s["p_boot"] <= s["bh_thresh"]
    print(f"  Benjamini-Hochberg FDR-significant (boot p): {int(s['bh_sig'].sum())}")
    print("  BH-significant tests:")
    print(s.loc[s["bh_sig"], ["pathway", "equation", "beta", "p_asymp", "p_boot"]]
            .to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Permutation test on the asymmetry pattern
    print("\n=== Permutation: how often does random labeling produce the observed pattern? ===")
    perm_df, obs = _permutation_label_shuffle(panel, CONTROLS, n_perm=300, seed=7)
    perm_df.to_parquet(DATA / "joint_var_permutation_counts.parquet", index=False)
    print(f"\nObserved sig coefficients (real labels): {obs}")
    print(f"Permutation null distribution (n={len(perm_df)}):")
    for eq, name in [("pop", "Δ log Pop"), ("crop", "Δ log CropShare"),
                       ("urb", "Δ log UrbanShare")]:
        med = perm_df[eq].median(); q90 = perm_df[eq].quantile(0.90)
        pval = (perm_df[eq] >= obs[eq]).mean()
        print(f"  {name}: observed={obs[eq]}  perm median={med:.1f}  90%={q90:.1f}  "
              f"P[perm >= obs]={pval:.3g}")

    # Figure: forest plot of asymp vs boot p-values
    if len(res):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
        for ax, eq, color in zip(
                axes, ["Δ log Pop", "Δ log CropShare", "Δ log UrbanShare"],
                ["#202020", "#A02020", "#0072B2"]):
            sub = res[res["equation"] == eq].copy()
            sub = sub.sort_values("beta").reset_index(drop=True)
            y = np.arange(len(sub))
            ax.errorbar(sub["beta"], y, xerr=1.96 * sub["se"], fmt="o",
                        color=color, markerfacecolor="white",
                        markeredgewidth=1, ecolor=color, elinewidth=0.7,
                        capsize=2.5)
            for i, rr in sub.iterrows():
                tag_a = ("***" if rr["p_asymp"] < 0.01 else
                         "**" if rr["p_asymp"] < 0.05 else
                         "*" if rr["p_asymp"] < 0.10 else "ns")
                tag_b = ("***" if rr["p_boot"] < 0.01 else
                         "**" if rr["p_boot"] < 0.05 else
                         "*" if rr["p_boot"] < 0.10 else "ns")
                ax.text(rr["beta"], i + 0.20,
                        f"asymp:{tag_a}  boot:{tag_b}", ha="center",
                        fontsize=8.0)
            ax.axvline(0, color="#404040", linewidth=0.6)
            ax.set_yticks(y); ax.set_yticklabels(sub["pathway"])
            ax.set_xlabel(r"$\beta$ on VSSI per Tg")
            ax.set_title(eq, loc="left", fontsize=10.5)
            ax.grid(alpha=0.3)
        fig.suptitle("Wild-cluster bootstrap vs. asymptotic p-values for the joint VAR pathway slopes",
                     y=1.04, x=0.04, ha="left", fontsize=11.5)
        plt.tight_layout()
        fig.savefig(FIG / "figJ_bootstrap.pdf", bbox_inches="tight")
        fig.savefig(FIG / "figJ_bootstrap.png", bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"\nSaved {FIG/'figJ_bootstrap.pdf'}")


if __name__ == "__main__":
    main()
