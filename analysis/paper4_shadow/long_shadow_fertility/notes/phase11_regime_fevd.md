# Phase 11 — State-dependent system LP-FEVD by development regime

**The new headline.** A Cholesky-identified system local-projection FEVD
(Jordà 2005; state-dependent à la Auerbach–Gorodnichenko 2012 / Ramey–Zubairy
2018), estimated *separately by Hansen wage regime*
(`S_ct = 1{log_real_wage_ct > 9.97}`; Malthusian below, Modern above). It
unifies the wage-regime split, the system FEVD dynamic decomposition, and the
catastrophe shocks into a single object. Core system Cholesky order:
`[SPEI, log_real_wage, log_CDR, log_CBR]` (weather most exogenous → fertility
last). Country FE, no year FE, VAR(2). Figure: `fig25_regime_fevd.{pdf,png}`.

## Headline result (core system A, h=15 fertility FEVD shares)

| Shock | Malthusian (N=803) | Modern (N=198) |
|---|---|---|
| **Weather (SPEI)** | **0.044** | **0.134** |
| Wages | 0.038 | 0.154 |
| Mortality (CDR) | 0.178 | 0.011 |
| Fertility (own) | 0.739 | 0.701 |

**The weather/SPEI share of fertility variance rises from 0.044 (Malthusian) to
0.134 (Modern) — roughly a threefold increase (Δ = +0.090).** This is the
elasticity transition rendered dynamically: as economies cross the development
threshold, weather explains a *larger* share of fertility's forecast-error
variance, not a smaller one. The modern-regime weather share is back-loaded,
climbing from ~0.00 at h=0 to 0.13 by h=15, i.e. the climate–fertility link in
modern economies operates at medium-run horizons.

Simultaneously the **mortality channel collapses** (0.178 → 0.011): in the
Malthusian regime, mortality shocks drive a sixth of fertility variance (the
classic high-mortality / replacement-fertility regime), whereas in the Modern
regime mortality is essentially irrelevant to fertility and the action shifts to
**weather and wages** (combined 0.082 → 0.288).

## Temperature variant (A′) — an honest divergence

Substituting raw growing-season temperature for SPEI **reverses** the weather
direction: T-share Malthusian = 0.090, Modern = 0.034. So the "weather share
rises in the Modern regime" claim is *specific to the SPEI drought/moisture
index*, not raw temperature. The two indices tell consistent stories about the
*other* channels (modern wage share rises to ~0.17, modern mortality share
collapses to ~0.03 in both), but the weather component itself is index-dependent.
Reading: in modern economies the surviving climate–fertility margin runs through
*moisture/drought* (agricultural and price channels), whereas raw temperature
loses explanatory share once wages are conditioned on. This nuance should be
stated in the paper rather than buried — the dynamic transition is robust for
SPEI; the temperature variant does not deliver the same monotone rise.

## All-shocks pooled view (extended system B, N=420, h=15)

Cholesky order `[nearby_war, own-war, pandemic, disaster, SPEI, wage, CDR, CBR]`
on the EMDAT-era complete-case sample (1900–2022; `disaster_count` is the binding
data constraint and only exists post-1900, so the extended sample is *not* the
full panel and pre-1900 zero-disaster years are **not** fabricated):

| Shock | Fertility FEVD share (h=15) |
|---|---|
| Disasters (EMDAT count) | **0.353** |
| Nearby-war (spatial spillover) | 0.076 |
| Own-war fatalities | 0.025 |
| Pandemic | 0.027 |
| Weather (SPEI) | 0.008 |
| Wages | 0.033 |
| Mortality | 0.039 |
| Fertility (own) | 0.439 |

In the modern (1900+) sample, **catastrophe shocks jointly carry ~0.48 of
fertility variance**, dominated by disasters (0.35) with a non-trivial spatial
war-spillover (`nearby_war` = 0.076, three times the own-war share of 0.025 —
i.e. a war *next door* matters more for fertility than a country's own recorded
war fatalities in this window). Weather's pooled share is small here (0.008)
because the EMDAT-era pooled sample mixes regimes and is dominated by the
20th-century catastrophe block; the regime-split core system (A) is the cleaner
read on the weather transition.

## Extended-by-regime (C) — partial

- **Malthusian** (N=301 ≥ dof floor 230): feasible. h=15 fertility shares —
  nearby-war **0.21**, SPEI 0.071, disasters 0.065, own-war 0.069, pandemic
  0.063, mortality 0.053, wages 0.024, own 0.446. The large nearby-war share
  (0.21) reinforces that *spatial conflict spillovers* were a first-order
  fertility driver in the pre-modern regime.
- **Modern** (N=119 < dof floor 230): **hit the dof fallback** as anticipated
  (8-var VAR(2) needs ≥230 obs; only 119 modern observations have a complete
  catastrophe block). Recorded as `{"status": "insufficient_dof", "n": 119}`.

## Caveats / concerns

- **Modern core subsample is small (N=198).** A 4-var VAR(2) needs ~9 params/eq,
  so 198 obs is ~22× the parameter budget — feasible and not singular, but the
  modern-regime FEVD shares are estimated on a thin sample and should be read as
  indicative. N_malthusian = 803 is comfortable.
- The headline rise is **SPEI-specific** (see A′ above).
- The extended pooled shares are **EMDAT-era only** (post-1900); they describe
  the modern catastrophe regime, not the full 1421–2022 panel.

Outputs: `analysis/output/long_shadow_fertility/phase11_regime_fevd.json`,
`analysis/figures/long_shadow_fertility/fig25_regime_fevd.{pdf,png}`.
