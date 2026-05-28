# Phase 11: Precipitation DL Robustness — Sample-Collapse Diagnosis

**Date:** 2026-05-27

## Mechanism Confirmed

The full control vector includes `disaster_count` and `log_disaster_deaths` (EMDAT-based). These have **only 11.6% non-null coverage**, spanning 1900–2022. All other controls — war/pandemic (57.2%, 1421–2022), heat/drought/volatility (97%+, 1421–2008) — are far broader. Listwise deletion on the full control vector collapses the sample to **N=424, 4 countries, 1903–2008**, discarding the entire pre-1900 panel.

## 3 × 2 Results Grid

| Spec | Shock | N | Countries | Years | cum_β | SE | t |
|------|-------|---|-----------|-------|-------|----|---|
| (a) Full controls | p_growing | 424 | 4 | 1903–2008 | −0.00881 | 0.00200 | −4.41 |
| (a) Full controls | t_growing | 424 | 4 | 1903–2008 | +0.00863 | 0.05483 | +0.16 |
| (b) No disaster controls | p_growing | 978 | 4 | 1544–2008 | −0.00636 | 0.00315 | −2.02 |
| (b) No disaster controls | t_growing | 978 | 4 | 1544–2008 | +0.03948 | 0.07438 | +0.53 |
| (c) Minimal (FE + lags) | p_growing | 1398 | 7 | 1544–2008 | −0.00401 | 0.00408 | −0.98 |
| (c) Minimal (FE + lags) | t_growing | 1398 | 7 | 1544–2008 | −0.01796 | 0.06022 | −0.30 |

## Verdict

**The precipitation effect is a post-1900 artifact, not a robust full-panel result.** The significant t=−4.4 in spec (a) is driven entirely by the modern sample (N=424, 1903–2008) produced by EMDAT listwise deletion. When disaster controls are dropped and the 1544–2008 window is restored (spec b), significance drops to t=−2.0 and is borderline only for FRA/GBR/ITA/SWE. In the full 7-country sample without exogenous controls (spec c), the effect collapses to t=−0.98 — statistically null.

**Temperature's null is robust across all three specs** (t=0.16 / 0.53 / −0.30), confirming that the temperature DL remains flat regardless of sample or control choices.

**Paper implications:** The precipitation DL result cannot be reported as a full-panel finding. Either (i) restrict to the explicit 1900–2008 modern subsample and label accordingly, or (ii) use the no-disaster spec (b) with the caveat of marginal significance and 4-country coverage. The zero temperature result is robust and can be reported without qualification.

## Files

- Script: `analysis/paper4_shadow/scripts/run_phase11_precip_dl_robustness.py`
- Output: `analysis/output/long_shadow_fertility/phase11_precip_dl_robustness.json`
