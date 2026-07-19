"""Dynamic event-studies on the Roman v2 panel, 86 BCE - 802 CE.

The panel (roman_v2_panel.csv) has 889 years of Mediterranean economic activity
indicators (lead-Z from Greenland ice-core lead, a proxy for Roman-era mining
and smelting; cf. Hong et al. 1994, McConnell et al. 2018), plus pandemic
dummies, war intensity, and climate forcings.

We run three sets of dynamic event studies, parallel in spirit to the Sigl
volcanic IRF in the Long Shadow paper:

(1) Volcanic forcing → lead-Z, leads and lags h ∈ {-5,...,+10} years.
(2) Pandemic events (intensity-weighted) → lead-Z, same leads/lags.
(3) Pandemic-family-specific IRFs (Antonine, Cyprian, Justinianic).

All with parallel-trends tests on pre-event leads.

Output: dynamic IRF tables and figures saved to analysis/figures/roman_jevons/.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "paper4_shadow"))
from figstyle import set_style, gray_palette, LINESTYLES

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
OUT = ROOT / "analysis" / "roman_jevons"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LEADS_LAGS = list(range(-5, 11))   # -5 to +10 years


def _load_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA / "roman_v2_panel.csv")
    df = df.sort_values("year_ce").reset_index(drop=True)
    return df


def dynamic_irf(df: pd.DataFrame, outcome: str, treatment: str,
                  leads_lags: list[int] = LEADS_LAGS,
                  controls: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Project outcome at year t on treatment at years t+h for each h in leads_lags.
    HAC standard errors. Returns IRF dataframe and parallel-trends test."""
    controls = controls or []
    p = df.copy().sort_values("year_ce")
    # Build lead/lag columns (NOTE: shift(-h) gives the future value at offset h)
    cols = []
    for h in leads_lags:
        col = f"{treatment}_h{h:+d}"
        p[col] = p[treatment].shift(-h)
        cols.append(col)
    p = p.dropna(subset=[outcome] + cols + controls)
    if len(p) < 30:
        return pd.DataFrame(), {}
    X = sm.add_constant(p[cols + controls])
    y = p[outcome]
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    rows = []
    for h in leads_lags:
        col = f"{treatment}_h{h:+d}"
        rows.append({"lag": h, "beta": r.params[col], "se": r.bse[col],
                      "p": r.pvalues[col]})
    out_df = pd.DataFrame(rows)
    out_df["ci_lo"] = out_df["beta"] - 1.96 * out_df["se"]
    out_df["ci_hi"] = out_df["beta"] + 1.96 * out_df["se"]

    # Parallel-trends F-test on pre-event leads
    leads = [c for h, c in zip(leads_lags, cols) if h < 0]
    diag = {}
    if leads:
        ft = r.f_test([f"{c} = 0" for c in leads])
        diag["lead_F"] = float(ft.fvalue)
        diag["lead_p"] = float(ft.pvalue)
        diag["n_leads"] = len(leads)
    diag["N"] = int(r.nobs)
    return out_df, diag


def plot_irf(irf: pd.DataFrame, title: str, fname: str,
              ylabel: str = r"$\hat\beta$ on lead-Z") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.errorbar(irf["lag"], irf["beta"], yerr=1.96 * irf["se"],
                fmt="o-", color="#202020", markerfacecolor="white",
                markeredgewidth=1, ecolor="#606060", linewidth=1.0,
                capsize=2)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.axvline(0, color="#606060", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Year relative to event")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def main() -> None:
    df = _load_panel()
    print(f"Roman v2 panel: {len(df)} years, "
          f"{int(df['year_ce'].min())} to {int(df['year_ce'].max())} CE")
    print(f"  lead_z coverage: {df['lead_z'].notna().sum()} / {len(df)}")
    print(f"  volcanic coverage: {df['volcanic'].notna().sum()} / {len(df)}")
    print(f"  pandemic_v2_intensity coverage: {df['pandemic_v2_intensity'].notna().sum()}")
    print()

    # ── (1) Volcanic → lead-Z ──────────────────────────────────────────
    print("=== (1) Dynamic IRF: volcanic forcing → lead-Z ===")
    print("    (volcanic is NEGATIVE when volcanism occurs, so positive β → ")
    print("     cooling reduces activity)")
    irf, diag = dynamic_irf(df, outcome="lead_z", treatment="volcanic")
    print(irf.round(5).to_string(index=False))
    print(f"\n    Parallel-trends F-test (h<0): F = {diag.get('lead_F', float('nan')):.3f}, "
          f"p = {diag.get('lead_p', float('nan')):.4g}  "
          f"({'PASS' if diag.get('lead_p', 0) > 0.10 else 'FAIL'} parallel trends)")
    print(f"    N = {diag.get('N')}")
    irf.to_csv(OUT / "irf_volcanic_lead.csv", index=False)
    plot_irf(irf, "(1) Volcanic forcing → Roman lead-Z (economic activity)",
              "fig_irf_volcanic.pdf",
              ylabel=r"$\hat\beta$ on lead-Z per unit volcanic forcing")

    # ── (2) Pandemic intensity → lead-Z ────────────────────────────────
    print("\n=== (2) Dynamic IRF: pandemic intensity → lead-Z ===")
    irf2, diag2 = dynamic_irf(df, outcome="lead_z", treatment="pandemic_v2_intensity")
    print(irf2.round(5).to_string(index=False))
    print(f"\n    Parallel-trends F-test: F = {diag2.get('lead_F', float('nan')):.3f}, "
          f"p = {diag2.get('lead_p', float('nan')):.4g}  "
          f"({'PASS' if diag2.get('lead_p', 0) > 0.10 else 'FAIL'} parallel trends)")
    print(f"    N = {diag2.get('N')}")
    irf2.to_csv(OUT / "irf_pandemic_lead.csv", index=False)
    plot_irf(irf2, "(2) Pandemic intensity → Roman lead-Z",
              "fig_irf_pandemic.pdf",
              ylabel=r"$\hat\beta$ per unit pandemic intensity")

    # ── (3) Pandemic-family IRFs ───────────────────────────────────────
    fams = ["pandemic_v2_family_antonine", "pandemic_v2_family_cyprian",
             "pandemic_v2_family_justinianic"]
    fam_titles = {"pandemic_v2_family_antonine":   "Antonine (~165–180 CE)",
                   "pandemic_v2_family_cyprian":    "Cyprian (~249–262 CE)",
                   "pandemic_v2_family_justinianic":"Justinianic (~541 onward)"}

    print("\n=== (3) Pandemic-family-specific IRFs ===")
    all_fam = []
    for fam in fams:
        sub_irf, sub_diag = dynamic_irf(df, outcome="lead_z", treatment=fam)
        print(f"\n  {fam_titles[fam]}:")
        print(sub_irf.round(5).to_string(index=False))
        print(f"    Parallel-trends F = {sub_diag.get('lead_F', float('nan')):.3f}, "
              f"p = {sub_diag.get('lead_p', float('nan')):.4g}, N = {sub_diag.get('N')}")
        sub_irf["family"] = fam
        all_fam.append(sub_irf)
    fam_df = pd.concat(all_fam, ignore_index=True)
    fam_df.to_csv(OUT / "irf_pandemic_by_family.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
    for ax, fam in zip(axes, fams):
        sub = fam_df[fam_df["family"] == fam]
        ax.errorbar(sub["lag"], sub["beta"], yerr=1.96 * sub["se"],
                    fmt="o-", color="#202020", markerfacecolor="white",
                    markeredgewidth=1, ecolor="#606060", linewidth=1.0, capsize=2)
        ax.axhline(0, color="#404040", linewidth=0.5)
        ax.axvline(0, color="#606060", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Year relative to event")
        ax.set_title(fam_titles[fam], loc="left", fontsize=10)
    axes[0].set_ylabel(r"$\hat\beta$ on lead-Z")
    plt.tight_layout()
    fig.savefig(FIG / "fig_irf_pandemic_by_family.pdf")
    plt.close(fig)

    print(f"\nSaved IRF tables and figures to {OUT}/")


if __name__ == "__main__":
    main()
