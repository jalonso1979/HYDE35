# Phase 10 audit: Hansen threshold country FE

**Audit date:** 2026-05-27
**File audited:** `estimators/threshold_regression.py`

**Result: PASS.**

- `_fit_at_c` (line 28) includes country fixed effects via `pd.get_dummies(df[unit_col], drop_first=True, dtype=float)` in the design matrix.
- No year FE (intentional — needed for climate identification per Phase 7 spec).
- Wild-cluster bootstrap applies Rademacher signs at the country/unit level (line 67), correct for cluster-robust inference.

**Implication:** Phase 9 Hansen threshold result (`c_hat = 9.97`, β_M = −0.089, β_T = −0.421, sup-Wald p = 0.012) is identified from within-country variation, not contaminated by between-country composition. Safe to promote to headline.

**API note for Phase 10 work:** the function returns keys `beta_M`, `beta_T`, `c_hat`, `c_ci_lo`, `c_ci_hi`, `sup_wald_pvalue`, `n`, `lr_path`. The Phase 10 plan's test pseudocode used different key names (`beta_below`, `beta_above`, `lr_ci`, `sup_wald_p`, `n_obs`) — downstream subagents should use the actual keys above, not the plan's nominal names.
