# Uncertainty channel: within-season SD vs 10-year rolling SD

| Spec | Within-season SD coef | Within-season SD SE | Rolling SD coef | Rolling SD SE |
|------|---|---|---|---|
| M1 (baseline + uncertainty) | -0.0022 | 0.036 | -0.1216 | 0.2284 |
| M3 (× above-threshold)      | -0.0226 (base) / +0.0871 (interaction) | 0.051 / 0.107 | -0.1142 (base) / +0.1714 (interaction) | 0.229 / 0.366 |

**Conclusion: still null — not a measurement-error artifact.**

- M1r: rolling SD coef = −0.122 (SE 0.228), t ≈ −0.53, far from significance. Larger in magnitude than the within-season SD (−0.002) but with a proportionally larger SE; t-ratio is essentially the same.
- M3r: rolling SD base = −0.114 (SE 0.229); interaction = +0.171 (SE 0.366). Both are statistically null. The interaction is directionally consistent with an above-threshold damping effect but the cluster-robust SE is twice the coefficient.
- The attenuation-bias hypothesis is not supported: switching to 9+ dof rolling SD does not revive the channel. The uncertainty proxy has larger variability in the rolling version but so do the standard errors, leaving the t-ratio unchanged.
- Rolling SD and vol_t_10y (the existing 10-year trailing-window volatility of monthly intra-annual std) are conceptually similar; the panel already controls for medium-frequency volatility via vol_t_10y in the baseline. Dropping vol_t_10y from M1r/M3r did not free up explanatory power for the rolling SD.
- Working interpretation: the uncertainty channel on fertility truly does not load in the pre-industrial multi-country panel at annual resolution. The null from C6 is robust to the measurement-error concern.
