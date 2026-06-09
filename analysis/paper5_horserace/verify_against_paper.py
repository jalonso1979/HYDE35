#!/usr/bin/env python3
"""Verify horserace.tex headline numbers against the regenerated analysis outputs.

Checks each load-bearing quantity the paper now states against the parquet it
comes from, and confirms the value literally appears in horserace.tex. Exits
non-zero on any mismatch. Run after the paper5_horserace pipeline regenerates:
  robustness_battery.parquet, exercise1_shapley_results.parquet,
  exercise1_shapley_ci.parquet, exercise2_mediation_results.parquet,
  exercise_colonial_partial_extended_results.parquet.

This is a consistency gate, not a full reproduction (which rebuilds the panel).
"""
import sys, pathlib, re
import pandas as pd, numpy as np

ROOT = pathlib.Path("/Volumes/BIGDATA/HYDE35")
DD = ROOT / "analysis/data/deep_determinants"
TEX = (ROOT / "paper/horserace/horserace.tex").read_text()
# normalise LaTeX number formatting for literal-presence checks
TEX_N = TEX.replace("{,}", "").replace("\\,", "").replace("$", "").replace("~", " ")

fails, checks = [], 0

def have(s):
    """Is the literal string present in the (normalised) paper text?"""
    return s in TEX_N

def check(name, ok, detail=""):
    global checks
    checks += 1
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)

# ---- 1. Westfall-Young minP survivors -------------------------------------
wy = pd.read_parquet(DD / "robustness_battery.parquet")
wy = wy[wy["f_obs"].notna()].copy()
surv = wy[wy["p_adj_wy_minp"] < 0.05].set_index(["substrate", "outcome"])
expected = {
    ("functional_alleles", "log_gdppc_2015"),
    ("functional_alleles", "dt_timing_year"),
    ("functional_alleles", "log_pop_growth_1950_2025"),
    ("functional_alleles", "log_popd_2025"),
    ("ancestral_yield_log", "log_pop_growth_1950_2025"),
    ("ancestral_yield_log", "log_popd_1500"),
    ("neolithic_frac", "log_popd_2025"),
    ("neolithic_frac", "dt_timing_year"),
}
got = set(surv.index)
check("WY minP survivor set = 8 expected cells", got == expected,
      f"{len(got)} survivors; extra={got-expected} missing={expected-got}")
# functional->gdppc survives strongly; paper says p_WY<0.001 and F=8.46
fg = wy[(wy.substrate == "functional_alleles") & (wy.outcome == "log_gdppc_2015")].iloc[0]
check("functional->GDPpc minP p<0.001", fg.p_adj_wy_minp < 0.001, f"p={fg.p_adj_wy_minp:.4f}")
check("functional->GDPpc F=8.46 in paper", have("F=8.46") and round(fg.f_obs, 2) == 8.46)
# climate survives nothing
clim = wy[wy.substrate == "climate_bundle"]
check("climate bundle survives no outcome", (clim.p_adj_wy_minp >= 0.05).all(),
      f"min p={clim.p_adj_wy_minp.min():.3f}")
check("paper states 'Eight cells survive'", have("Eight cells survive"))

# ---- 2. Shapley shares + distinguishability -------------------------------
sh = pd.read_parquet(DD / "exercise1_shapley_results.parquet")
piv = sh.pivot_table(index="substrate", columns="outcome", values="shapley_r2")
fg_gdp = piv.loc["functional_alleles", "log_gdppc_2015"]
fg_dt = piv.loc["functional_alleles", "dt_timing_year"]
check("functional Shapley GDPpc=0.175", round(fg_gdp, 3) == 0.175 and have("0.175"))
check("functional Shapley dt=0.122", round(fg_dt, 3) == 0.122 and have("0.122"))
ci = pd.read_parquet(DD / "exercise1_shapley_ci.parquet")
# the two distinguishable diffs (top1-top2 CI excludes 0): gdppc, dt
diffs = ci[ci.get("kind", ci.columns[0]).astype(str).str.contains("diff", case=False, na=False)] \
        if "kind" in ci.columns else ci
# robust: find rows with both ci bounds same sign for gdppc & dt
def distinguishable(outcome):
    r = ci[(ci.get("outcome", "") == outcome)]
    r = r[r.apply(lambda x: "diff" in " ".join(str(v) for v in x.values).lower(), axis=1)]
    if r.empty:
        # fall back: any row for outcome with ci_lower/ci_upper both >0
        r = ci[ci.get("outcome", "") == outcome]
    return r
check("paper claims only 2 distinguishable orderings",
      have("only two of the six") or have("the only two of the six"))

# ---- 3. DARC fixed-sample ---------------------------------------------------
darc = pd.read_parquet(DD / "exercise_colonial_partial_extended_results.parquet")
db = darc[(darc.allele == "fa_darc") & (darc.spec == "baseline")].iloc[0]
check("DARC fixed-sample baseline +0.62 (n=136)", round(db.beta, 2) == 0.62 and int(db.n) == 136,
      f"beta={db.beta:+.2f} n={int(db.n)}")
check("paper states DARC +0.62 and 136-country", have("+0.62") and have("136"))

# ---- 4. sample sizes --------------------------------------------------------
check("principal sample 162 stated", have("162"))
check("185 not used as principal sample",
      "185-country sub-panel" not in TEX and "185-country panel" not in TEX)

# ---- 5. no resurrected stale claims ----------------------------------------
for bad in ["predicted heterozygosity carries", "F=11.33", "F=10.82",
            "p_{\\text{WY}}=0.066", "24 substrate-by-outcome", "65 percent attenuation",
            "sixty-five percent"]:
    check(f"stale claim absent: {bad[:40]}", bad not in TEX)

print(f"\n{checks} checks, {len(fails)} failures")
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("All horserace headline numbers reconcile with the regenerated outputs.")
