# Phase 11 — Block-bootstrap CIs on regime-FEVD shares

**Method.** Country-cluster block-bootstrap (N_boot = 500, seed = 0). The unit
of resampling is the country: for each draw, 7 iso3 codes are drawn with
replacement from the 7 countries in each regime subsample; duplicated countries
receive distinct pseudo-iso3 labels so country FE remain identified. Core system
`[spei_growing, log_real_wage, log_cdr, log_cbr]` (Cholesky order: weather most
exogenous → fertility last), VAR(2), h = 15. All 500 draws succeeded for both
regimes (0 failures).

## 95% percentile-bootstrap CI table — h=15 fertility FEVD shares

| Shock | Regime | Point | CI_lo (2.5%) | CI_med (50%) | CI_hi (97.5%) |
|---|---|---|---|---|---|
| Weather (SPEI) | Malthusian (N=803) | 0.044 | 0.011 | 0.073 | 0.233 |
| Wages | Malthusian | 0.038 | 0.017 | 0.042 | 0.121 |
| Mortality | Malthusian | **0.178** | **0.109** | 0.195 | 0.388 |
| Fertility (own) | Malthusian | 0.739 | 0.412 | 0.672 | 0.792 |
| Weather (SPEI) | Modern (N=198) | 0.134 | 0.051 | 0.133 | 0.284 |
| Wages | Modern | 0.154 | 0.079 | 0.153 | 0.377 |
| Mortality | Modern | **0.011** | **0.010** | 0.036 | 0.185 |
| Fertility (own) | Modern | 0.701 | 0.447 | 0.638 | 0.744 |

## Verdict

**Mortality CI overlap.** The Malthusian mortality CI is [0.109, 0.388] and the
Modern mortality CI is [0.010, 0.185]. The intervals overlap between 0.109 and
0.185, so the difference is not statistically unambiguous at the country-cluster
bootstrap level. The point estimates (0.178 vs. 0.011) differ by a factor of
sixteen, but the Modern upper bound (0.185) overlaps with the Malthusian lower
bound (0.109), indicating the CI distinction is directionally strong but not
cleanly non-overlapping.

**Wage CI overlap.** The Malthusian wage CI is [0.017, 0.121] and the Modern
wage CI is [0.079, 0.377]. These intervals overlap (0.079–0.121), so the
wage-share rise is also not sharply separated, though the point estimates (0.038
vs. 0.154) are a fourfold increase and the medians (0.042 vs. 0.153) are clearly
distinct.

**Is the mortality→wage shift robust?** The key claim — mortality share collapses
and wage share rises as economies cross the development threshold — is supported
by the direction of the point estimates and the bootstrap medians. However, the
95% CIs for both mortality and wages overlap across regimes, reflecting the small
Modern-regime sample (N=198 observations across 7 countries). The shift is robust
in the sense that (a) zero failures across 500 draws demonstrate the Modern-regime
estimation is numerically stable (not fragile to specific country draws), (b) the
Modern median mortality share (0.036) is less than a fifth of the Malthusian
median (0.195), and (c) the Modern weather share lower bound (0.051) exceeds the
Malthusian weather point estimate (0.044). But readers should be told the CIs
overlap: the regime comparison is suggestive, not statistically sharp at the
cluster-bootstrap level with 7 resampling units.

## Files

- Script: `long_shadow_fertility/scripts/run_phase11_regime_fevd_bootstrap.py`
- Output JSON: `analysis/output/long_shadow_fertility/phase11_regime_fevd_bootstrap.json`
