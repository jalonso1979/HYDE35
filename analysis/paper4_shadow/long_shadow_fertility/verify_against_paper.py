#!/usr/bin/env python3
"""Verify reproduced Long Shadow JSON outputs against the numbers printed in paper.tex.

Reads the headline JSON from the analysis output dir and the paper source, compares
each headline quantity to the value the paper states, and checks that the value
literally appears in paper.tex. Exits non-zero if any headline reproduced value
deviates from the paper's stated value, or if the Claim-1 bootstrap CIs fail to
separate. Run AFTER reproduce_long_shadow.sh completes.

Usage:  python verify_against_paper.py
"""
import json, sys, pathlib

OUT = pathlib.Path("/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility")
# The 2026-05-29 repo reorg moved long_shadow/ -> papers/long_shadow/. Try the
# new location first, fall back to the pre-reorg path.
_FERT = pathlib.Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/"
    "My Drive/Fertility"
)
PAPER = next(
    (p for p in (_FERT / "papers/long_shadow/paper.tex", _FERT / "long_shadow/paper.tex")
     if p.exists()),
    _FERT / "papers/long_shadow/paper.tex",
)

def load(name):
    return json.load(open(OUT / name))

grid = load("phase10_threshold_grid.json")
fevd = load("phase11_regime_fevd.json")
boot = load("phase11_regime_fevd_bootstrap.json")

# normalize LaTeX number formatting: 1{,}789 / 1,789 -> 1789
tex = PAPER.read_text()
tex_norm = tex.replace("{,}", "").replace(",", "")

def in_paper(s):
    return s is None or s in tex_norm

# (label, reproduced, expected, tol, literal-string-expected-in-paper-or-None)
checks = [
    ("Hansen wage c_hat",       grid["log_real_wage"]["c_hat"],            10.14,  0.02,   "10.14"),
    ("Hansen wage p",           grid["log_real_wage"]["sup_wald_pvalue"],  0.002,  0.0005, "0.002"),
    ("Hansen wage N",           grid["log_real_wage"]["n"],                1789,   0,      "1789"),
    ("Hansen wage beta_M",      grid["log_real_wage"]["beta_M"],          -0.060,  0.005,  None),
    ("Hansen wage beta_T",      grid["log_real_wage"]["beta_T"],          -0.388,  0.01,   None),
    ("Hansen CDR p (n.s.)",     grid["log_cdr"]["sup_wald_pvalue"],        0.192,  0.01,   "0.192"),
    ("Hansen GDP c_hat",        grid["log_gdppc"]["c_hat"],                9.94,   0.02,   "9.94"),
    ("Hansen GDP p",            grid["log_gdppc"]["sup_wald_pvalue"],      0.012,  0.001,  "0.012"),
    ("Regime split (meta)",     fevd["meta"]["hansen_threshold"],          9.97,   0.0,    "9.97"),
    ("Core FEVD N_total",       fevd["core_by_regime"]["n_total"],         1440,   0,      None),
    ("Boot mortality Malth pt", boot["malthusian"]["mortality"]["point"],  0.181,  0.01,   None),
    ("Boot mortality Mod pt",   boot["modern"]["mortality"]["point"],      0.016,  0.01,   None),
    ("Boot n_countries Malth",  boot["malthusian"]["n_countries"],         12,     0,      "twelve"),
    ("Boot n_countries Mod",    boot["modern"]["n_countries"],             12,     0,      "twelve"),
]

print(f"{'quantity':28}{'reproduced':>13}{'paper':>11}{'Δok':>6}{'in.tex':>8}")
print("-" * 66)
fails = 0
for label, rep, exp, tol, ps in checks:
    ok = abs(float(rep) - float(exp)) <= tol
    inp = "-" if ps is None else ("yes" if in_paper(ps) else "NO")
    print(f"{label:28}{float(rep):>13.4g}{float(exp):>11.4g}{('ok' if ok else 'FAIL'):>6}{inp:>8}")
    fails += (not ok) + (ps is not None and not in_paper(ps))

# Claim 1: bootstrap mortality CIs must SEPARATE (Malthusian ci_lo > Modern ci_hi)
m_lo = boot["malthusian"]["mortality"]["ci_lo"]
mod_hi = boot["modern"]["mortality"]["ci_hi"]
sep = m_lo > mod_hi
print(f"\nClaim-1 mortality CI separation: Malthusian ci_lo={m_lo:.3f} > "
      f"Modern ci_hi={mod_hi:.3f}  ->  {'SEPARATE (ok)' if sep else 'OVERLAP (FAIL)'}")
fails += (not sep)

print(f"\nMISMATCHES: {fails}")
sys.exit(1 if fails else 0)
