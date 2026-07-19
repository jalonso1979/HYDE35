---
title: "Long-shadow climate measures: climate-defined growing-season volatility and anomalies"
date: 2026-05-19
status: draft
---

# Growing-season volatility and anomalies for `long_shadow`

## Motivation

The headline cross-section in `paper/long_shadow.tex` (§5, "The long shadow on modern outcomes") regresses modern log population growth (1950 → 2025) on pre-industrial inter-annual temperature volatility $\sigma_v^T$, computed as the country-mean standard deviation of *annual-mean* temperature over 1421–1750, with $R^2 = 0.45$. The accompanying Malthusian panel (§3) and joint VAR (§4.1) use interval-mean temperature anomalies, interval-mean precipitation anomalies, and within-interval $T$ volatility.

Three deficiencies of this measurement choice are visible on the page:

1. **Annual aggregation discards the agronomic mechanism.** The long-shadow narrative is that pre-industrial *agricultural-system* selection leaves a demographic imprint. A frost in February and a frost in August are demographically equivalent under annual aggregation, but agronomically antipodal. The annual measure understates the channel the paper claims to identify, and a referee will say so.
2. **Precipitation is doing essentially no work.** In the joint VAR (Table 3), $\bar P_{it}$ anomalies enter at $\pm 10^{-5}$ on every outcome, indistinguishable from zero. Annual mean $P$ is the wrong unit when monsoon-failure clustering is the demographically relevant signal. The current paper does not have a precipitation channel, despite a literature consensus that rice/kharif systems live and die by monsoon $P$ variability.
3. **The latitude robustness is uncomfortable.** $\sigma_v^T$ falls from $-2.44$ to $-1.76$ when $|\mathrm{lat}|$ is added (Table `tab:lat-robust`). Annual mean $T$ is mechanically collinear with latitude; growing-season volatility — especially in monsoon systems whose GS is set by ITCZ position rather than solar latitude — should be substantially less so.

A growing-season-restricted volatility measure addresses all three. The user's question — should it be *crop-specific* — is answered no, because the major-crop attribution for 1421–1750 is itself downstream of the HYDE pathway features the paper uses as outcomes, which would break the climate-primitive-orthogonality the headline result rests on. The cleaner construction is a **climate-defined growing-season mask** that uses only climate primitives and remains orthogonal to HYDE.

## Design

### 1. Variables built

For each country $i$ and each annual year $t \in [1421, 2025]$:

| symbol | column name | construction |
|---|---|---|
| $\sigma_v^{T,\mathrm{GS}}_i$ | `sigma_v_T_gs_pre1750` | std over $t \in [1421, 1750]$ of $\bar T_{i,t \mid m \in \mathrm{GS}_i}$ |
| $\sigma_v^{P,\mathrm{GS}}_i$ | `sigma_v_P_gs_pre1750` | std over $t \in [1421, 1750]$ of $\bar P_{i,t \mid m \in \mathrm{GS}_i}$ (mm/month) |
| $\bar T^{\mathrm{GS}}_i$ | `T_gs_mean_pre1750` | 1421–1750 climatology of GS-mean $T$ |
| $\bar P^{\mathrm{GS}}_i$ | `P_gs_mean_pre1750` | 1421–1750 climatology of GS-mean $P$ (mm/month) |
| $\sigma_v^{T,\overline{\mathrm{GS}}}_i$ | `sigma_v_T_nongs_pre1750` | std of $\bar T_{i,t \mid m \notin \mathrm{GS}_i}$ — **placebo** |
| $n^{\mathrm{GS}}_i$ | `n_gs_months` | count $|\mathrm{GS}_i|$, integer 0–12 |
| $\mathrm{GS}_i$ | `gs_months_mask` | comma-separated list of 1-indexed months in $\mathrm{GS}_i$ |

Annual time-series counterparts (one row per country-year):

| symbol | column name | construction |
|---|---|---|
| $\bar T^{\mathrm{GS}}_{i,t}$ | `t_gs_mean` | mean of monthly $T$ over $m \in \mathrm{GS}_i$ in year $t$ |
| $\bar P^{\mathrm{GS}}_{i,t}$ | `p_gs_mean` | mean of monthly $P$ over $m \in \mathrm{GS}_i$ in year $t$ |
| $\Delta \bar T^{\mathrm{GS}}_{i,t}$ | `t_gs_anom` | $\bar T^{\mathrm{GS}}_{i,t} - \bar T^{\mathrm{GS}}_i$ |
| $\Delta \bar P^{\mathrm{GS}}_{i,t}$ | `p_gs_anom` | $\bar P^{\mathrm{GS}}_{i,t} - \bar P^{\mathrm{GS}}_i$ |

The cross-section measures live in a new `analysis/data/country_seasonality_gs_preindustrial.parquet`. The annual measures live in `analysis/data/country_climate_gs_1421_2025.parquet`. Sub-national analogues live in `analysis/data/subnational_climate_gs_1421_2025.parquet` and `analysis/data/subnational_seasonality_gs_preindustrial.parquet`.

**Weighting-variant column convention.** In every output file, each variable above appears three times for country-level (suffixes `_area`, `_pop`, `_cropw`) and twice for sub-national (suffixes `_area`, `_pop`). The headline column for downstream regressions is `_cropw` for country, `_pop` for sub-national. The column names in the tables above (e.g. `sigma_v_T_gs_pre1750`) are the *headline* variant; the full file contains `sigma_v_T_gs_pre1750_area`, `sigma_v_T_gs_pre1750_pop`, `sigma_v_T_gs_pre1750_cropw` for the cross-section and the same three suffixes on each panel column.

### 2. Growing-season mask

$$\mathrm{GS}_i \;\equiv\; \{\, m \in \{1,\dots,12\} \;:\; 5 \le \bar T^{\mathrm{clim}}_{i,m} \le 30 \;\;\text{AND}\;\; \bar P^{\mathrm{clim}}_{i,m} \ge 30~\mathrm{mm/month}\,\}$$

where $\bar T^{\mathrm{clim}}_{i,m}, \bar P^{\mathrm{clim}}_{i,m}$ are the 1421–1750 mean of calendar month $m$ from the cropland-weighted ModE-RA panel `modera_country_monthly_cropw.parquet`, with absolute levels reconstructed by adding the CRU 1901–1950 climatology (`cru_country_climatology_1901_1950.parquet`) per the procedure already in `analysis/paper4_shadow/cropweight_comparison.py:33-36`.

Two design choices are intentional:

- **Fixed mask, not time-varying.** The mask is computed once over 1421–1750 climatology and held constant. A drifting mask (recomputed annually) would absorb part of the very volatility signal we want to measure — months would enter and leave the mask as climate fluctuated, contaminating $\sigma_v$.
- **Threshold values match the existing productive-month index $\Pi$.** The 5°C/30°C/30mm thresholds are the same ones the paper already uses to compute $\Pi$ (`long_shadow.tex:65`). This is a promotion of $\Pi$ from "how many" to "which exact months", not a new threshold choice.

### 3. Aggregation weighting

Country-level uses **cropland-weighted** ModE-RA (`modera_country_monthly_cropw.parquet`) as headline. Area- and pop-weighted are robustness columns.

Sub-national uses **pop-weighted** (`modera_subnational_monthly_popw.parquet`) as headline; area-weighted is the robustness column. A sub-national cropland-weighted panel does not exist and is not built here — at the sub-national resolution (~3,187 units across 196 countries) the "Egypt = Nile not Sahara" problem is much less severe than at the country level, so cropland weighting buys less. Building it is deferred unless the headline depends on it.

### 4. Edge cases

- **Empty-GS countries** ($n^{\mathrm{GS}}_i = 0$, true deserts: expected to include SAU, LBY, possibly OMN, KWT depending on the cropland-weighted aggregation): excluded from the headline cross-section. The exact list and resulting sample size are reported by `build_gs_climate.py` diagnostic output and pinned in a footnote of §5; the current headline reports $N = 154$ countries with pathway FE, so the empty-GS drop is bounded above by 154 minus the number of empty-GS countries also present in the current sample. A **robustness column re-includes them** by substituting the annual-mean $\sigma_v^T$ (the existing measure) for the undefined GS measure.
- **12-month-GS countries** (continuous tropics): GS-$\sigma_v$ degenerates gracefully to annual $\sigma_v$. No special handling.
- **Short-GS countries** ($1 \le n^{\mathrm{GS}}_i \le 3$, Sahel/high-latitude margins): included in headline; a sensitivity in the appendix shows the coefficient with these countries dropped.

### 5. Where the new measures enter `long_shadow.tex`

#### 5.1 §5 Long Shadow cross-section (headline replacement)

The current headline is $\sigma_v^T$ alone. The new headline is the bivariate $\{\sigma_v^{T,\mathrm{GS}}, \sigma_v^{P,\mathrm{GS}}\}$ specification, with $\sigma_v^{T,\overline{\mathrm{GS}}}$ **side-by-side as the non-GS placebo**:

$$\Delta \log \mathrm{pop}_i^{1950 \to 2025} = \alpha_\tau + \beta_T \,\sigma_v^{T,\mathrm{GS}}_i + \beta_P \,\sigma_v^{P,\mathrm{GS}}_i + \beta_T^{\mathrm{plac}} \,\sigma_v^{T,\overline{\mathrm{GS}}}_i + \gamma |\mathrm{lat}_i| + \varepsilon_i$$

with pathway fixed effects $\alpha_\tau$.

The headline figure (`analysis/figures/cross_section_final.png`, referenced as `\ref{fig:long-shadow-map}` in §5) is regenerated using $\sigma_v^{T,\mathrm{GS}}$ in panel (a). Old $\sigma_v^T$ is retained as a robustness column in the cross-section table.

#### 5.2 §3 Pathway-stratified Malthus regression

In the estimating equation (`long_shadow.tex:171`), the within-interval $T$-volatility and interval-mean climate anomalies are replaced by their GS analogues:

$$\Delta \ln P_{it} = \alpha_i + \beta d_{i,t-1} + \gamma_T \Delta \bar T^{\mathrm{GS}}_{it} + \gamma_P \Delta \bar P^{\mathrm{GS}}_{it} + \delta_T \sigma^{T,\mathrm{GS}}_{i,t} + \delta_P \sigma^{P,\mathrm{GS}}_{i,t} + \varepsilon_{it}$$

where the within-interval $\sigma^{T,\mathrm{GS}}_{i,t}$ is the std of *GS-mean* $T$ over the years inside HYDE interval $[t, t+1)$.

#### 5.3 §4.1 Joint VAR

The three-equation panel VAR (population, cropland-share, urban-share) in §4.1 (Table 3, `tab:jointvar-pooled`) replaces three climate regressors: interval-mean $\bar T_{it}$ → GS-mean $\bar T^{\mathrm{GS}}_{it}$; interval-mean $\bar P_{it}$ → GS-mean $\bar P^{\mathrm{GS}}_{it}$; within-interval $T$-volatility → within-interval GS-mean-$T$ volatility. The pathway-stratified version (`tab:jointvar-stratified`) takes the same substitutions. Lagged levels and VSSI exposure $V_{it}$ are unchanged. The Sigl-Toohey volcanic-shock interpretation is unaffected; the substitution sharpens the contemporaneous climate covariate to its agronomic analogue without touching the volcanic identification.

#### 5.4 Appendix robustness

- **Non-GS placebo** (already promoted to headline per 5.1).
- **Modern-window placebo**: $\sigma_v^{T,\mathrm{GS}}$ recomputed over 1950–2008. Expected to attenuate as the annual analogue does.
- **Weighting sensitivity**: report headline cross-section under area- and pop-weighted aggregations.
- **Empty-GS sensitivity**: report headline with empty-GS countries (a) excluded (headline), (b) re-included with annual-$\sigma_v^T$ fallback.
- **Short-GS sensitivity**: report headline with $n^{\mathrm{GS}} \le 3$ countries dropped.

### 6. Sub-national replication

The country-level builders are mirrored at the sub-unit level using `modera_subnational_monthly_popw.parquet`. The sub-national long-shadow (`subnational_long_shadow.py`) and sub-national Malthus (`subnational_malthus.py`) regressions are rerun with the GS analogues of every climate regressor. Sub-national results are reported in §3 (sub-national replication) and §6.4 (sub-national long-shadow choropleth and regression table).

### 7. Pre-registered predictions

| Prediction | Reasoning | Interpretation if false |
|---|---|---|
| $\lvert\beta(\sigma_v^{T,\mathrm{GS}})\rvert > \lvert\beta(\sigma_v^{T,\mathrm{annual}})\rvert$ | If channel is agronomic, GS-restricted measure aligns with mechanism | Channel is *climate-deep*, not GS-specific; §4 narrative shifts toward "climate left a deep mark" rather than "agriculture mediated climate's mark" |
| $\beta(\sigma_v^{P,\mathrm{GS}}) < 0$ at $p < 0.05$ | Monsoon failure / drought-year clustering is subsistence risk in rain-fed systems | $P$ noise dominates OR storage smooths $P$ shocks better than $T$ shocks — both informative for the structural interpretation |
| $\beta(\sigma_v^{T,\overline{\mathrm{GS}}}) \approx 0$ in the non-GS placebo | Out-of-GS variance has no agricultural relevance | If non-GS placebo is also negative, channel may be institutional/path-dependent rather than agricultural — this would prompt a §4 rewrite |

These predictions are stated upfront in §4 of `long_shadow.tex` before the headline table.

## Files modified

### New data builders (`analysis/paper4_shadow/`)

- **`build_gs_climate.py`** (new). One file builds all four output parquets:
  - Country-level: `country_seasonality_gs_preindustrial.parquet` (cross-section) and `country_climate_gs_1421_2025.parquet` (annual panel).
  - Sub-national: `subnational_seasonality_gs_preindustrial.parquet` and `subnational_climate_gs_1421_2025.parquet`.
  - Three weighting variants for country (area / pop / cropland); two for sub-national (area / pop). Variants live as columns within the same files (suffixes `_area`, `_pop`, `_cropw`) for downstream regression convenience.
  - Console output: GS-month distribution, list of empty-GS countries, list of short-GS countries.

### Regression code changes (`analysis/paper4_shadow/`)

- `long_shadow.py`: replace $\sigma_v^T$ load with $\{\sigma_v^{T,\mathrm{GS}}, \sigma_v^{P,\mathrm{GS}}, \sigma_v^{T,\overline{\mathrm{GS}}}\}$ load; regenerate headline cross-section table and figure.
- `pathway_irfs.py`, `joint_landuse_var.py`, `joint_var_climate_pathways.py`, `joint_var_post1700.py`, `joint_var_bootstrap.py`: replace interval-mean climate with GS-mean climate. Use the same source parquet (`country_climate_gs_1421_2025.parquet`) merged on iso3-year.
- `preindustrial_malthus.py`, `preindustrial_malthus_extended.py`, `malthusian_extended.py`, `malthus_with_conflict_controls.py`: replace within-interval $T$-volatility with GS-mean within-interval $T$-volatility; add the $P$ analogue.
- `subnational_long_shadow.py`, `subnational_malthus.py`: same substitutions, sub-national variants of the GS panel.
- `placebo_period_and_rolling.py`: extend rolling-window estimator to the GS measures; add modern-window placebo for GS measures.
- `latitude_controls.py`: rerun latitude robustness on GS measures.
- `robustness.py`, `robustness_v2.py`: append GS-based rows; retain annual rows for back-compatibility.
- `run_all.py`: insert `build_gs_climate.py` step before the regression cascade.

### Tables and figures

- `analysis/figures/paper4_shadow/`: regenerate cross-section figure (panel a uses $\sigma_v^{T,\mathrm{GS}}$), regenerate latitude-robustness table, regenerate joint VAR table, regenerate Malthus table.
- New table: empty-/short-GS sensitivity.
- New table: weighting sensitivity.

### Paper text (`paper/long_shadow.tex`)

- §1 abstract: update headline $R^2$ if it changes; reframe "inter-annual climate volatility" as "growing-season inter-annual climate volatility".
- §2 ("A new monthly paleo-economic panel"): add a paragraph describing the GS mask construction and edge-case handling; add footnote listing dropped empty-GS countries.
- §3 ("From storage demand to the Malthusian regression"): update the empirical equation (`eq:malthus-main` and the empirical analogue in §3.2) to show GS variables; update Table `tab:malthus` and the pathway-stratified table.
- §4.1 (Joint VAR): update Tables `tab:jointvar-pooled` and `tab:jointvar-stratified` captions and surrounding prose.
- §5 ("The long shadow on modern outcomes"): replace headline narrative and table (`tab:shadow`), add pre-registered predictions box at the top of the section, add non-GS placebo column to the headline table.
- Appendix `app:longshadow-extensions`: add empty-GS sensitivity table, short-GS sensitivity table, weighting sensitivity table, modern-window GS placebo row.

## Risks and open questions

1. **Headline coefficient may shrink, not strengthen.** If annual $\sigma_v^T$ was riding partly on winter-month variance, the GS-restricted measure could yield a smaller absolute coefficient. Reporting is unconditional: the result, whichever direction, is reported as-is in §4 with the implied narrative shift (per the table in §7).
2. **Latitude collinearity may not improve.** The hope is that GS-restricted measures are less correlated with $|\mathrm{lat}|$ than annual ones. If they are equally collinear, the lat robustness is no stronger than before. This is checked early in the implementation; if collinearity is unchanged, the headline replacement still proceeds for theoretical reasons but the §4 framing emphasises mechanism rather than identification.
3. **Sub-national cropland weighting absent.** Sub-national headline uses pop-weighting. If a referee insists on cropland weighting at the sub-unit level, building `modera_subnational_monthly_cropw.parquet` is the deferred extension.
4. **Empty-GS list depends on the threshold and weighting.** A SAU that has 0 cropland-weighted productive months may have 1 area-weighted productive month. The empty-GS list is reported per weighting in the diagnostic output of `build_gs_climate.py`.
5. **Sample-size change in the cross-section may break the comparison with the existing $R^2 = 0.45$ headline.** Both old- and new-spec coefficients are reported on the *intersection* sample (countries present in both, i.e. ~180 non-empty-GS countries), so the comparison is apples-to-apples. A separate column reports the new spec on its full ~196 sample with annual-fallback.

## Out of scope

- Crop-specific GS (assigning a major crop per country and using that crop's calendar). Discussed in the brainstorming session; the major-crop attribution for 1421–1750 is downstream of HYDE pathway features the paper uses as outcomes, which would break the climate-primitive-orthogonality the headline rests on.
- Latitude-band GS (e.g. NH temperate Apr–Sep). Dominated by the climate-defined version on cleanness without compensating advantage.
- Sub-national cropland-weighted ModE-RA aggregation. Deferred per §3.
- Crop-yield or yield-volatility outcomes (these belong in `paper5_horserace`, not `long_shadow`).
- Joint changes to other paper4 climate-shocks analyses (volcanic event studies, Sigl/Toohey impulse responses). The GS measures are added to the headline regression slate; the volcanic-shock side of paper4 retains its current climate vocabulary.
