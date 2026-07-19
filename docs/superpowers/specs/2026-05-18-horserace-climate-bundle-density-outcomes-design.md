---
title: "Horserace v2: climate-bundle substrate + population-density outcomes"
date: 2026-05-18
status: draft
---

# Horserace v2 — climate as a bundle, pop density as outcome

## Motivation

The current paper 5 horserace (`paper/horserace/horserace.tex`, commit `2c8f0a0`) carries two asymmetries that an attentive referee will catch:

1. **Climate is represented by σᵥᵀ alone.** The other three substrates (predicted heterozygosity, ancestral crop yield, pre-1500 pandemic intensity) are *bundled* objects — each summarises multiple primitives into a single index. σᵥᵀ is only the inter-annual temperature SD, ignoring mean T, mean P, and inter-annual precipitation SD. The Matranga (2024) storage-demand mechanism that motivates the σᵥᵀ channel is about all four of these jointly. Treating σᵥᵀ as the sole climate variable understates the climate channel's contribution and is conceptually inconsistent with how the other three substrates are constructed.

2. **The headline outcome (log population growth 1950→2025) is a convergence-window measure, not a Malthusian one.** All four substrates are pre-industrial. The natural prediction from each is about the *level* of pre-industrial productivity, which under Malthusian logic translates to population density at the steady state. The current negative coefficient on ancestral crop yield is "places that were dense in 1950 have less room to grow" — a derived consequence of the underlying density story, not the primary phenomenon.

Both fixes were already discussed and chosen by the user (option **1A + 2A** in the 2026-05-18 brainstorming session).

## Changes

### 1. Climate substrate becomes a 4-element bundle

Replace the single `sigma_v_T_pre1750` substrate with a **climate bundle**:

| symbol | column | source | already in panel? |
|---|---|---|---|
| T̄ pre-industrial | `t_mean_preind` | `country_seasonality_preindustrial.parquet` | no, but built |
| P̄ pre-industrial | `p_annual_preind` | `country_seasonality_preindustrial.parquet` | no, but built |
| σᵥᵀ pre-industrial | `sigma_v_T_pre1750` | already in panel | yes |
| σᵥᴾ pre-industrial | `sigma_v_P_pre1750` (new) | derive from `country_climate_1421_2025.parquet` | no, needs computation |

σᵥᴾ is computed as the standard deviation of annual mean precipitation over the 1421-1750 window, per country, matching the σᵥᵀ definition. (`p_mm` column of the annual climate panel.)

The Shapley decomposition treats the 4-element climate bundle as a single coalition player. When the bundle is "in" subset S, the regression includes all four climate variables; when "out", none of the four. This is the standard *grouped-Shapley* extension and preserves the decomposition identity ∑φ_s = R²(full) − R²(baseline).

Internal sub-decomposition (which of the four climate vars contributes most within the bundle) is reported in a separate sub-table for transparency but does not feature in the headline matrix.

### 2. Outcome vector expands from 4 to 6

Add two population-density outcomes from HYDE 3.5 `gbc2025_7apr_base/txt/popd_c.txt`:

- `log_popd_1500`: log of country population density in year 1500 CE (Malthusian steady-state proxy)
- `log_popd_2025`: log of country population density in year 2025 (cumulative productivity proxy)

These join the four existing outcomes:

- `log_pop_growth_1950_2025` (transitional/convergence-window growth)
- `urban_change_1950_2025`
- `log_gdppc_2015`
- `dt_timing_year`

The headline 4×4 Shapley matrix becomes a 4×6 matrix. The two density outcomes precede the four modern outcomes in the column ordering, reflecting the theoretical chronology.

The HYDE `popd_c.txt` `region` column is a numeric ID that needs ISO3 mapping. The existing `analysis/paper5_horserace/build_modern_outcomes.py` already performs this mapping for `popc_c.txt`; we lift the same logic.

**Asymmetry note (to flag in the paper):** the heterozygosity substrate H_pred_pwadj is *ancestry-weighted to the modern composition*. The density-1500 outcome is *geographic* (the population that physically lived at that location in 1500). For settler colonies (USA, AUS, NZ, BRA, ARG) the modern ancestry adjustment maps to European source populations, but the geographic 1500 density reflects pre-Columbian Indigenous populations. We use raw geographic 1500 density as the primary outcome (since 3 of 4 substrates are geographic) and report Ashraf-Galor's pwadj density-1500 (`ln_pd_1500_pwadj` from their `country.dta`) as a robustness column. We flag the asymmetry explicitly in §3.

### 3. Endogeneity flag for the climate-A correlation

Ancestral crop yield A is constructed via GAEZ projections under pre-Columbian *climate* conditions. Mean T and mean P enter both the climate bundle and (implicitly) the A construction. We pre-emptively note this in §3 and run a robustness column where A is partialled-out against (T̄, P̄) before entering the regression.

## Files modified

### Data builders (`analysis/paper5_horserace/`)

- `build_horserace_panel.py` — merge in 4-element climate bundle and 2 density outcomes; rebuild `analysis/data/deep_determinants_horserace.parquet`.
- new step: compute σᵥᴾ from annual P (`country_climate_1421_2025.parquet`).
- `build_modern_outcomes.py` — add 1500 and 2025 density extraction from HYDE `popd_c.txt` with ISO3 mapping (lift existing `popc_c.txt` logic).

### Analysis kernel (`analysis/paper5_horserace/`)

- `shapley.py` — extend `shapley_r2_decomposition` to accept `substrates` as a list of either column names *or* tuples of column names (the latter denoting a coalition player). Backward-compatible: a list of strings behaves as before.
- `exercise1_shapley.py` — change `SUBSTRATES` from 4 column names to 4 substrate definitions, one of which is a climate-bundle tuple. Add the 2 density outcomes to `OUTCOMES`. Run on the 6-outcome × 4-substrate matrix.
- `exercise2_mediation.py` — same outcome and substrate changes. Mediation now runs against the climate bundle as a coalition player (which is well-defined: the bundle is the predictor, pathway is the mediator).
- `exercise2_subsample_stability.py` — same.
- `robustness_battery.py` and `figA_robustness_battery.py` — same outcome extension, same substrate extension. Westfall-Young correction runs over 24 cells (4 substrates × 6 outcomes), not 16.

### Tables and figures (`analysis/figures/paper5_horserace/`)

- `tab01_descriptives.tex` — add T̄, P̄, σᵥᴾ, log density 1500, log density 2025 rows.
- `tab02_substrate_correlations.tex` — keeps the 4×4 cross-substrate correlation table with σᵥᵀ as the climate bundle's representative (simpler than PC1, recovers the existing layout). The within-climate 4×4 correlation matrix goes in a separate appendix table `tab02b_climate_subcorrelations.tex`.
- `tab03_full_ols.tex` — 6 columns instead of 4; substrate row block has 4 vars under "Climate bundle".
- `tab04_shapley_table.tex` — 4×6 matrix.
- new `tab05_climate_subshapley.tex` — internal Shapley among (T̄, P̄, σᵥᵀ, σᵥᴾ) for each of the 6 outcomes. The baseline for this within-bundle decomposition is the geography control battery plus the three non-climate substrates (H, A, Π); the four climate vars are the "substrates" of this sub-decomposition. So ∑φ_climate_sub = R²(full) − R²(controls + H + A + Π).
- `fig01_substrate_covariance.py` — climate bundle represented by PC1 in the scatter matrix.
- `fig02_substrate_maps.py` — keep σᵥᵀ map; add a 4-panel sub-figure for the climate bundle in an appendix.
- `fig03_shapley_heatmap.py` — heatmap becomes 4 rows × 6 columns.
- `fig04_mediation.py` — extended to 6 outcomes.
- `fig06_subsample_stability.py` — extended to 6 outcomes.
- `figA_robustness_battery.py` — extended to 24 cells.

### Paper text (`paper/horserace/horserace.tex`)

- **Abstract**: rewrite to lead with the climate-bundle + density-outcome framing. The three FWER-surviving findings will likely shift; the abstract is rewritten *after* the empirical re-run, not before.
- **§1 Introduction**: rewrite "Three findings survive" paragraph after empirical re-run. Reframe the contribution paragraph around the 4-substrate × 6-outcome matrix.
- **§2 Theory**: light edit — Prediction 1 references the climate bundle, Prediction 3 (climate-window robustness) becomes a within-bundle question.
- **§3 Data**: new sub-section §3.1.1 "Climate bundle" defining the 4 climate primitives and justifying the bundling. New sub-section §3.2.5 "Population-density outcomes" defining log density 1500 and 2025 with the geographic-vs-pwadj asymmetry note.
- **§4 Shapley**: rewrite to present the 4×6 matrix and the climate sub-Shapley. The three observations at the end of §4 (lines 184-188) will be rewritten after the empirical re-run.
- **§5 Mediation**: light edit — same structure, more outcome columns.
- **§6 Robustness**: extend each leg to 6 outcomes; add new climate-A partialling robustness column.
- **§7 Discussion / §8 Conclusion**: rewrite around new findings post-empirics.

### Tests (`analysis/paper5_horserace/tests/`)

- `test_shapley.py` — add tests for grouped/coalition Shapley: the bundle Shapley equals the sum of marginal contributions when the bundle members are added together; backward compatibility with the original 4-feature decomposition.
- `test_build_horserace_panel.py` — assert presence of climate-bundle columns and density outcomes; coverage of 196 (countries with density 1500) and 196 (density 2025) approximately matches HYDE coverage.

## Empirical outputs

- `analysis/data/deep_determinants/exercise1_shapley_results.parquet`: 6 outcomes × 4 substrates = 24 rows (long form).
- `analysis/data/deep_determinants/exercise1_climate_subshapley.parquet`: 6 outcomes × 4 climate vars = 24 rows (within-bundle).
- `analysis/data/deep_determinants/exercise2_mediation_results.parquet`: 24 cells.
- `analysis/data/deep_determinants/robustness_battery.parquet`: extended.

## Risks and trade-offs

- **Climate Shapley may absorb part of the ancestral-crop-yield contribution.** Both ancestral yield and climate are spatially clustered (warm-temperate cores). If the bundle's contribution to log_pop_growth jumps from 0.016 to, say, 0.10, the dominant story may shift toward "climate explains the same things ancestral yield was explaining". This is honest and may be the right answer; we don't pre-commit to a finding.
- **Density 1500 is noisy.** HYDE 1500 estimates are themselves model outputs with uncertainty bands. The 1500 outcome should be read as "Malthusian-era density up to HYDE's reconstruction error", not a sharp empirical quantity. Robustness: also report log density at 1700, 1800, 1900 to show the result is not a HYDE-1500-specific artefact.
- **The headline narrative may shift substantially.** If, e.g., climate dominates density 1500 and ancestral yield dominates density 2025, the paper's contribution becomes about *which channel matters at which historical moment*. This is a richer story but it forces a re-think of the abstract and introduction. We accept this cost.
- **6 outcomes × 4 substrates = 24 cells under FWER instead of 16.** Westfall-Young correction is harder to pass with more cells. Some findings that survived at 16 cells may not survive at 24.

## Acceptance criteria

1. `analysis/data/deep_determinants_horserace.parquet` includes columns `t_mean_preind`, `p_annual_preind`, `sigma_v_T_pre1750`, `sigma_v_P_pre1750`, `log_popd_1500`, `log_popd_2025`, with ≥185 non-null countries for the 4 climate vars and ≥185 for each density outcome (matches current substrate coverage).
2. `analysis/paper5_horserace/exercise1_shapley.py` runs without error and writes a 24-row results parquet.
3. The grouped-Shapley function passes new tests: (i) backward-compatibility with 4 individual substrates, (ii) coalition Shapley value equals the average marginal R² contribution of adding the bundle across all 24 orderings of the 4 coalition players.
4. `horserace.tex` compiles to PDF with all cross-references resolving, all new tables/figures included, and a coherent narrative around the new 4×6 matrix.
5. 67 existing tests continue to pass; new tests pass.

## Plan and sequence

The dependency graph:

```
build_horserace_panel.py (climate bundle + density outcomes)
        │
        ├── shapley.py (group support)
        │       │
        │       ├── exercise1_shapley.py
        │       │       └── fig03, tab03, tab04
        │       └── exercise1_climate_subshapley.py (new)
        │               └── tab05
        ├── exercise2_mediation.py
        │       └── fig04, tab on mediation
        ├── exercise2_subsample_stability.py
        │       └── fig06
        └── robustness_battery.py
                └── figA, tabs in §6
```

After data and analysis layers compile, the paper text is rewritten in §3 (data), §4 (Shapley), §5 (mediation), §6 (robustness), and finally abstract + §1 (introduction) + §8 (conclusion) once the new empirical findings are settled.

Implementation plan to be written next under `docs/superpowers/plans/2026-05-18-horserace-v2-climate-bundle-density.md`.
