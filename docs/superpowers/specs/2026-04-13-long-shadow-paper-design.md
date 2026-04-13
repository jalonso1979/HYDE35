# The Long Shadow of Seasonality: Paper Design Spec

**Date:** 2026-04-13
**Target outlet:** Journal of Economic Growth
**Status:** Design approved, pending implementation plan

---

## 1. Title and Framing

**Title:** *The Long Shadow of Seasonality: Climate Endowments, Agricultural Pathways, and Escape from the Malthusian Trap*

**One-sentence summary:** The same climate endowment that determines which agricultural system a society adopts also determines the strength of its Malthusian constraint and the speed of its escape — because seasonality selects the agricultural technology, which determines how quickly human capital accumulation can overtake population growth.

**Three contributions:**

1. **Bridging Matranga and Galor** — First paper to formally connect the climate-driven *entry* into agriculture (Matranga QJE 2024) with the conditions for *escape* from the Malthusian trap (Galor 2011), through the intermediate mechanism of agricultural system selection.

2. **2,000-year calibrated climate panel** — Novel data construction merging four climate sources (PAGES 2k, Berkeley Earth, CRU TS, ERA5) into a continuous regional panel (0-2025 CE), enabling within-region tracking of how climate volatility shaped agricultural dynamics across two millennia.

3. **Dual-channel identification** — Distinguishes intra-annual seasonality (drives system selection, Stage 1) from inter-annual volatility (drives Malthusian intensity, Stage 2), showing they operate through separate causal channels.

---

## 2. Theoretical Framework

### Two-stage semi-formal model (~5 pages)

**Stage 1: Climate -> Agricultural System Selection (extending Matranga)**

Extends Matranga's binary forage-vs-farm choice to a three-way choice among agricultural technologies:

- **Intensive crop** (high storage, high labor): cereals, rice, irrigated — selected when intra-annual seasonality is high
- **Extensive pastoral** (low storage, low labor): grazing, rangeland — selected when seasonality is low
- **Mixed/transitional**: intermediate seasonality

Society chooses technology tau in {I, P, M} to maximize:

    V(tau) = Y_bar(tau, c_bar) - lambda(tau) * sigma_s(c) - kappa(tau)

where Y_bar is mean output, lambda(tau) is vulnerability to seasonality sigma_s, and kappa(tau) is fixed cost. Intensive systems have high kappa but low lambda; pastoral the reverse.

Selection condition: tau* = I iff sigma_s > sigma_s_bar(c_bar)

**Stage 2: Agricultural System -> Malthusian Dynamics -> Escape (extending Galor)**

Given selected system tau*, population dynamics follow:

    Delta ln P_it = alpha_i + beta(tau*) * d_{i,t-1} + gamma(tau*) * sigma_v(c_t) + delta(tau*) * h_it + epsilon_it

where d is density (Malthusian pressure), sigma_v is inter-annual volatility (risk exposure), and h is technology/human capital proxy.

**Testable predictions:**

1. |beta(P)| > |beta(I)| — Malthusian feedback stronger in pastoral than intensive
2. |gamma(P)| > |gamma(I)| — pastoral more exposed to inter-annual shocks
3. d_beta/d_h < 0 — as technology/urbanization increases, Malthusian constraint weakens
4. Escape occurs earlier in intensive pathway due to faster accumulation of h

**Formalization level:** Semi-formal. Derive predictions from structure but no full GE proofs. Consistent with JEG style (cf. Ashraf & Galor 2011).

---

## 3. Data

### Five sources, one integrated panel

| Source | Coverage | Resolution | Variables | Role |
|--------|----------|------------|-----------|------|
| HYDE 3.5 | 10,000 BCE-2025 CE | 197 countries | Population, cropland, grazing, rice, irrigated, urban | Outcomes + pathway classification |
| PAGES 2k | 0-1849 CE | 7 continental -> 25 regions | Temperature reconstructions | Historical seasonality/volatility |
| Berkeley Earth | 1850-1900 | 1deg grid -> 25 regions | Temperature | Calibration bridge |
| CRU TS 4.09 | 1901-1949 | 0.5deg grid -> 25 regions | Temperature + precipitation | Pre-ERA5 climate |
| ERA5 | 1950-2025 | 25 regions | 2m temperature, total precipitation | Modern climate analysis |

### Existing data files

- `era5_full_panel.parquet` — 1,131 region-year records, 25 regions
- `climate_panel_0_2025.parquet` — calibrated chain, 0-2025 CE
- `hyde_era5_extended_panel.parquet` — merged HYDE+ERA5, 1950-2025, ~190 countries
- `country_era5_region_map.parquet` — country-to-region assignment
- `trajectory_features.parquet` — pathway trajectory features
- `paper1_clustered_features.parquet` — pathway cluster assignments

### Two climate measures

1. **Intra-annual seasonality (sigma_s):** From ERA5 monthly data (1950-2025) directly. For historical periods, proxied by inter-decadal range in PAGES 2k. Stage 1 channel.

2. **Inter-annual volatility (sigma_v):** Rolling SD of annual temperature (30-year window). Computed in `climate_shocks.py`. Stage 2 channel.

### Sample restrictions

- Historical analysis (0-1750 CE): 159 countries, max density > 1 person/km2
- Modern analysis (1950-2025): Expanding as ERA5 regions complete (currently 12/25, ~73 countries in Regions 1-8 for country-level analysis)
- Cross-sectional endowments: All 197 countries

---

## 4. Empirical Exercises

### Stage 1: Climate -> System Selection

**Exercise 1: Seasonality predicts pathway assignment**

- Method: Multinomial logit / ordered probit
- DV: Pathway cluster (1-5)
- IVs: Historical seasonality (sigma_s), mean temperature, mean precipitation, geographic controls
- Prediction: Higher seasonality -> intensive pathways
- Data: Cross-section, 159 countries
- Status: **New analysis needed**

**Exercise 2: Dual-channel separation**

- Method: Joint regression of pathway assignment on seasonality AND volatility; then Malthusian coefficient on both
- Prediction: Seasonality predicts pathway; volatility predicts Malthusian intensity within pathway. Orthogonal effects.
- Status: **New analysis needed**

### Stage 2: System -> Malthusian Dynamics -> Escape

**Exercise 3: Pathway-specific Malthusian dynamics**

- Method: FE panel with rolling 200-year windows, by pathway
- Prediction: |beta(pastoral)| >> |beta(intensive)|
- Enhancement: Add inter-annual volatility as regressor
- Existing result: Pastoral 7.7x more climate-sensitive
- Status: **Exists, needs volatility addition**

**Exercise 4: Pathway-stratified climate shock IRFs**

- Method: Jorda local projections, stratified by pathway
- Shocks: Temperature/precipitation anomalies
- Responses: Cropland growth, population growth, crop share
- Prediction: Larger, more persistent IRFs for pastoral pathway
- Status: **New — split existing IRFs by pathway**

**Exercise 5: The escape — declining climate sensitivity**

- Method: Rolling 20-year FE + interaction regressions
- Interaction: shock_t x h_it where h = (a) intensification index, (b) urban share, (c) both
- Prediction: beta_2 < 0 — higher technology/urbanization reduces climate sensitivity
- Existing result: Rolling sensitivity shows Green Revolution sign flip
- Status: **Needs interaction regressions identifying mechanism**

**Exercise 6: Cross-sectional long shadow**

- Method: OLS cross-section
- Question: Does pre-industrial seasonality/volatility predict modern outcomes?
- Enhancement: Decompose into seasonality + volatility; add pathway FE
- Existing result: Pre-industrial temp level predicts modern crop share (p=0.001)
- Status: **Needs seasonality reframing + pathway FE**

---

## 5. Figures and Tables

### Figures (8-10)

| # | Title | Status |
|---|-------|--------|
| 1 | Pathway Typology Map (choropleth) | Exists |
| 2 | Great Divergence in Density (0-1750 CE by pathway) | Exists |
| 3 | Seasonality Predicts Pathway (scatter + multinomial fit) | New |
| 4 | Dual-Channel DAG (conceptual) | New |
| 5 | Rolling Malthusian Coefficient by Pathway | Exists, needs pathway split |
| 6 | Pathway-Stratified IRFs (2x3 grid) | New |
| 7 | The Great Escape (rolling sensitivity by pathway) | Exists, needs pathway split |
| 8 | Escape Mechanism (interaction coefficients) | New |
| 9 | The Long Shadow (seasonality vs. modern outcomes) | Exists, needs reframing |
| 10 | ERA5 Climate Trends (all regions) | Exists |

### Tables (5-6)

| # | Title | Status |
|---|-------|--------|
| 1 | Five Agricultural Pathways | Exists |
| 2 | Seasonality -> Pathway Assignment (multinomial logit) | New |
| 3 | Dual-Channel Regressions | New |
| 4 | Malthusian Coefficient by Pathway (with volatility) | Exists, needs enhancement |
| 5 | Panel FE: Climate -> Ag Outcomes (pre/post Green Rev) | Exists |
| 6 | Escape Mechanisms (interaction regressions) | New |

---

## 6. Paper Outline

| Section | Pages | Key content |
|---------|-------|-------------|
| 1. Introduction | ~5 | Motivation, contribution, preview |
| 2. Theoretical Framework | ~5 | Two-stage model, predictions, Fig 4 |
| 3. Data | ~4 | Five-source panel construction |
| 4. Stage 1: Climate and System Selection | ~6 | Exercises 1-2, Figs 1/3, Tables 1-3 |
| 5. Stage 2: Malthusian Dynamics by Pathway | ~6 | Exercises 3-4, Figs 2/5, Table 4 |
| 6. Climate Shocks in the Modern Period | ~6 | Exercise 4 continued, Fig 6/10, Table 5 |
| 7. The Great Escape | ~5 | Exercise 5, Figs 7/8, Table 6 |
| 8. The Long Shadow | ~3 | Exercise 6, Fig 9 |
| 9. Conclusion | ~2 | Summary, implications |
| **Total** | **~42** | + appendix |

---

## 7. Robustness and Identification Threats

### Threats addressed in main text

1. **HYDE measurement error** — Manski bounds across three HYDE scenarios (baseline/lower/upper). Already computed. All Stage 2 regressions reported on all scenarios.

2. **Seasonality endogeneity** — Pre-industrial (0-1850) seasonality as predetermined instrument. At PAGES 2k scale, anthropogenic climate modification is negligible. Modern seasonality robustness check.

3. **Spatial autocorrelation** — Continent FE in cross-sections. Conley spatial SEs. Show seasonality predicts pathways within continents.

4. **PAGES 2k resolution** — Acknowledge coarseness. Use inter-decadal range as proxy. Show results hold with modern ERA5 seasonality (Matranga's assumption). CRU TS intermediate check.

5. **Pathway classification sensitivity** — Report for K in {3,4,5,6,7}. Core intensive-pastoral split robust at all K >= 2.

6. **Green Revolution endogeneity** — Rolling windows (no hard cutoff). Test timing variation by pathway. Not driven by single region.

7. **Selection into modern sample** — Report included/excluded countries. Test for systematic differences on observables. Expanding coverage mitigates.

### Appendix

- A1: Manski bounds across HYDE scenarios
- A2: Alternative K for pathway clustering
- A3: Conley spatial standard errors
- A4: Modern seasonality as proxy for historical
- A5: ERA5 region coverage robustness
- A6: IV diagnostics (first-stage F, overidentification)

---

## 8. Implementation Dependencies

### Existing code to reuse

| Script | Purpose | Modifications needed |
|--------|---------|---------------------|
| `paper1_escape/clustering.py` | Pathway clustering | None — use as-is |
| `paper2_malthus/regressions.py` | Rolling Malthusian FE | Add volatility regressor |
| `paper3_climate/climate_shocks.py` | Anomaly/volatility construction | Add intra-annual seasonality measure |
| `paper3_climate/local_projections.py` | Jorda LP IRFs | Add pathway stratification |
| `paper3_climate/regime_switching.py` | Sample splitting | Use pathway as regime variable |
| `run_ag_impact_final.py` | Full modern analysis | Extend with interaction regressions |
| `update_era5_panel.py` | ERA5 panel rebuild | Run as more regions complete |
| `build_climate_panel_0_2025.py` | Historical climate panel | Add seasonality computation |
| `build_extended_panel.py` | HYDE+ERA5 merge | Run as more regions complete |

### New code needed

1. **Seasonality computation** — Extract intra-annual seasonality from ERA5 monthly data and construct historical proxy from PAGES 2k
2. **Multinomial logit** — Pathway prediction from climate endowments (Exercise 1)
3. **Dual-channel regressions** — Joint seasonality + volatility models (Exercise 2)
4. **Pathway-stratified IRFs** — Split existing LP code by pathway cluster (Exercise 4)
5. **Interaction regressions** — Shock x intensification/urbanization (Exercise 5)
6. **Seasonality cross-section** — Reframe historical endowments exercise (Exercise 6)
7. **New figures** — Figs 3, 4, 6, 8
8. **Paper LaTeX** — New main.tex for this paper

### Data dependencies

- ERA5 downloads: Currently 60% complete (Regions 1-12 done, 13-25 in progress). Core results use Regions 1-8 (73 countries). Full coverage adds robustness.
- All historical data (HYDE, PAGES 2k, Berkeley, CRU) is complete and processed.

---

## 9. Key Literature to Cite

- Matranga (2024) "The Ant and the Grasshopper" QJE — seasonality -> agriculture adoption
- Galor (2011) "Unified Growth Theory" — Malthusian dynamics and escape framework
- Galor & Weil (2000) "Population, Technology, and Growth" AER — canonical UGT model
- Ashraf & Galor (2011) "Dynamics of Inequality" QJE — Malthusian empirics
- Goldewijk et al. (2017) — HYDE 3.5 dataset
- Hersbach et al. (2020) — ERA5 reanalysis
- Dell, Jones & Olken (2012) "Temperature Shocks and Economic Growth" AEJ:Macro — climate shock methodology
- Jorda (2005) — local projection methodology
- Manski (2003) — partial identification / bounds
- Nunn & Qian (2011) "The Potato's Contribution" QJE — crop composition and population
