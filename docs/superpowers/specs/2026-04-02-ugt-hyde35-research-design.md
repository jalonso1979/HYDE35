# Unified Growth Theory meets HYDE 3.5: Research Design

## Overview

A three-paper research pipeline using HYDE 3.5 historical land-use/population data and ERA5 climate data to test implications of Unified Growth Theory (Galor). The project investigates Malthusian dynamics, agricultural intensification as an escape mechanism, and climate as an exogenous driver/moderator of regime transitions across the full historical span (10,000 BCE -- 2025 CE).

**Research domain:** Economic history + Climate/Environmental science
**Target journals:** QJE, AER, JEG, JPE, Demography, Population and Development Review, Journal of Economic History, Nature Climate Change, PNAS, Review of Economics and Statistics
**Analytical toolkit:** Python (primary), Matlab, Stata; spatial econometrics, causal inference, unsupervised learning

---

## Shared Data Infrastructure

All three papers draw from a unified, analysis-ready data layer built on existing HYDE35 processing pipelines.

### Existing assets

- `hyde35_country_year_mean_std.csv` -- country x year panel with uncertainty
- `hyde35_panel_region_year_by_scenario.csv` -- region x year x scenario
- ERA5 integration at region and ISO3 levels (Parquet + CSV.GZ)
- 16 NetCDF variables x 3 scenarios (base/lower/upper), 220+ ASCII anthromes rasters
- Reference masks: land area, country boundaries (ISO3), biomes, forest cover

### Derived variables to construct

1. **Malthusian variables** -- land-labor ratio (arable land per capita = [cropland + grazing] / population), agricultural output proxy (area-weighted sum: cropland_mha + 0.5 * grazing_mha, where 0.5 reflects lower caloric yield of grazing), intensification index (irrigation_share * [cropland / (cropland + grazing)], capturing both water management and crop-vs-pastoral balance)
2. **Regime indicators** -- binary/continuous measures of Malthusian exit based on structural breaks in population-density vs. land-use relationships
3. **Climate anomaly series** -- ERA5 temperature/precipitation deviations from long-run means, both level and volatility
4. **Trajectory features** -- per region/country: timing of peak extensification, onset of intensification, cropland-to-grazing ratio dynamics, urbanization takeoff
5. **Scenario-derived uncertainty bands** -- cross-scenario standard deviation as a variable (for data-quality heterogeneity analysis and partial identification)

### File structure

```
HYDE35/
├── analysis/
│   ├── data/           # Analysis-ready panels (Parquet + CSV)
│   ├── paper1_escape/  # Paper 1 -- spatial-temporal mapping
│   ├── paper2_malthus/ # Paper 2 -- panel econometrics
│   ├── paper3_climate/ # Paper 3 -- causal climate channel
│   └── shared/         # Common utilities, plotting, variable construction
```

---

## Paper 1: The Geography of the Great Escape

**Research question:** When and where did different regions exit Malthusian constraints, and what agricultural transition pathways distinguished early vs. late escapers?

**Role in pipeline:** Descriptive foundation. Generates stylized facts, typologies, and spatial variation that Papers 2 and 3 exploit econometrically.

### Analytical pipeline

1. **Construct transition trajectories** -- For each country/region, extract time-series features:
   - Peak extensification date (when cropland+grazing area growth rate peaked)
   - Intensification onset (when irrigation share or cropland-per-capita began rising while total area stabilized)
   - Urbanization takeoff (when urban share crossed threshold)
   - Population density inflection points

2. **Cluster analysis** -- Unsupervised classification of transition pathways:
   - K-means or hierarchical clustering on trajectory features
   - Expected typologies: irrigation-led (Mesopotamia, Nile, East Asia), extensive-pastoral (Central Asia, Sub-Saharan Africa), mixed-intensive (Europe), late-extensive (Americas)
   - Validate clusters against historical narrative

3. **Survival analysis** -- Model time-to-Malthusian-exit:
   - Define exit as sustained decoupling of population growth from land expansion (structural break in pop-density vs. arable-land-per-capita relationship)
   - Cox proportional hazards with covariates: initial biogeography, climate zone, crop suitability, proximity to other escapers (spatial diffusion)

4. **Spatial diffusion** -- Did the agricultural transition spread geographically?
   - Spatial lag models to test whether neighboring regions' intensification predicts a region's own transition timing
   - Moran's I for spatial autocorrelation in transition dates

5. **Uncertainty dimension** -- Map where HYDE scenario spread is largest. Hypothesis: data-sparse regions (Sub-Saharan Africa, pre-Columbian Americas) show wider uncertainty, which tells a story about the archaeological/historical record.

### Key outputs

- Global maps of transition timing and pathway type
- Typology table with representative regions per cluster
- Survival curves by pathway type
- Spatial diffusion heat maps

### Target journals

Journal of Economic History, Demography, Population and Development Review, World Development

---

## Paper 2: Malthusian Mechanics

**Research question:** Can we directly estimate the Malthusian feedback loop in the data, and does agricultural intensification measurably weaken it over time?

**Role in pipeline:** Core econometric paper. Uses Paper 1's typologies and transition dates as key variables.

### Analytical pipeline

1. **Core Malthusian regression** -- Panel estimation of the population-land feedback:
   - Dependent variable: population growth rate (decadal)
   - Key regressors: population density (lagged), arable land per capita (lagged), intensification index
   - Fixed effects: country/region + time period
   - Core prediction: negative coefficient on population density (diminishing returns), positive on land-per-capita

2. **Time-varying Malthusian coefficient** -- Test whether the constraint weakens:
   - Rolling-window estimation of the density coefficient across centuries
   - Interaction: `density x intensification_index` -- if intensification weakens the trap, this should be positive
   - Interaction: `density x pathway_type` (from Paper 1) -- do irrigation-led regions show earlier weakening?

3. **Structural break detection** -- When does the Malthusian regime end?
   - Bai-Perron multiple structural break tests on the density-growth relationship, per region
   - Compare detected break dates against Paper 1's transition timing

4. **Identification strategy:**
   - Within-region variation over time (FE absorbs level differences)
   - Instrument for population density using lagged climate shocks (preview of Paper 3)
   - Robustness: Arellano-Bond dynamic panel GMM for short-T bias concerns

5. **Partial identification using HYDE uncertainty:**
   - Run core specification on all three scenarios (base/lower/upper)
   - Construct Manski-style bounds on the Malthusian coefficient
   - If sign and significance hold across all scenarios, result is robust to substantial measurement error
   - Report bounds alongside point estimates -- methodologically novel for this literature

6. **Heterogeneity analysis:**
   - By Paper 1 pathway type (irrigation-led vs. extensive vs. mixed)
   - By continent/macro-region
   - By initial conditions (land abundance, crop suitability)
   - Pre/post intensification onset (using Paper 1 dates as regime cutoffs)

### Key outputs

- Table of Malthusian coefficients by period and region
- Time-series plot of rolling Malthusian coefficient with structural breaks marked
- Bounds estimates across HYDE scenarios
- Heterogeneity decomposition showing which pathways escaped fastest

### Target journals

Journal of Economic Growth, QJE, AER, Explorations in Economic History

---

## Paper 3: Climate Shocks and Agricultural Regime Shifts

**Research question:** Did exogenous climate shocks trigger lasting transitions from extensive to intensive agriculture, or were they absorbed within the existing Malthusian equilibrium? Under what initial conditions do shocks produce regime shifts vs. mean reversion?

**Role in pipeline:** Most causally ambitious. Uses Paper 1's typologies as initial conditions, Paper 2's framework for interpretation.

### Key challenge

ERA5 covers 1950--2025 only. A dual strategy addresses the deep-history gap.

### Analytical pipeline

1. **Modern period (1950--2025) -- causal core:**
   - ERA5 gridded temperature/precipitation at monthly frequency
   - Local projections (Jorda IRFs): trace how a 1-sigma climate shock in year t affects land-use variables over 1--5 decades
   - Dependent variables: cropland share, irrigation share, grazing-to-cropland ratio, intensification index, population growth
   - Climate shocks: temperature/precipitation anomalies relative to 30-year rolling mean, both level and variance

2. **Nonlinear responses -- regime-switching:**
   - Threshold models: does response to climate shocks differ by position in the Malthusian-to-modern transition?
   - Interaction: `climate_shock x pre/post_intensification` (Paper 1 dates)
   - Interaction: `climate_shock x pathway_type`
   - Smooth transition regression (STAR models) for endogenous threshold estimation

3. **Historical period (pre-1950) -- cross-sectional identification:**
   - Exploit cross-sectional climate variation using ERA5 climatology as proxy for historical endowments
   - Koppen climate zones as categorical instruments
   - Test: regions with higher climate volatility -- did they develop more resilient/intensive systems?

4. **Paleoclimate extension (optional, high-impact):**
   - Source paleoclimate reconstructions (PAGES2k, tree-ring chronologies, speleothem records)
   - Even coarse proxies (Medieval Warm Period timing, Little Ice Age severity by region) enable diff-in-diff designs
   - Upgrades the paper from modern climate effects to climate as a driver of the Malthusian transition

5. **Uncertainty strategy -- data quality as heterogeneity:**
   - HYDE scenario spread varies by region and time
   - Hypothesis: climate-shock responses are better estimated in data-rich regions (low spread)
   - Split sample by HYDE uncertainty terciles

6. **Welfare/counterfactual analysis:**
   - Simulate: what if region X had experienced region Y's climate history?
   - Would the agricultural transition have occurred earlier/later?
   - Connects directly to UGT: climate as an exogenous accelerator/retarder

### Key outputs

- Impulse response functions: climate shock to land-use variables, by regime
- Regime-switching maps showing where shocks trigger transitions vs. mean reversion
- Cross-sectional evidence linking climate volatility to agricultural resilience
- Counterfactual simulations of transition timing under alternative climate histories

### Target journals

AER, QJE, Nature Climate Change, PNAS, Review of Economics and Statistics

---

## Sequencing and Dependencies

```
Shared Data Infrastructure
        |
        v
   Paper 1 (Descriptive)
    |           |
    v           v
Paper 2       Paper 3
(Econometric)  (Causal/Climate)
```

### Phases

1. **Phase 1 -- Data infrastructure:** Extend existing notebooks into modular Python pipeline under `analysis/shared/`. Construct all derived variables. Produce analysis-ready Parquet panels with uncertainty bands.
2. **Phase 2 -- Paper 1 exploration:** Clustering, survival analysis, spatial diffusion. Exploratory analysis to let patterns emerge. Output: typology, transition dates, spatial maps.
3. **Phase 3 -- Papers 2 and 3 in parallel:** Paper 2 (panel regressions, structural breaks, partial identification) and Paper 3 (local projections, cross-sectional tests). Each imports Paper 1 outputs as covariates.

### Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| HYDE temporal resolution (decadal) limits dynamic analysis | Use interpolation cautiously; focus on long-run dynamics where decadal is appropriate |
| ERA5 only back to 1950 limits Paper 3 historical reach | Cross-sectional climate strategy for pre-1950; paleoclimate extension as upgrade path |
| Malthusian coefficient may be weak/noisy in sparse regions | Partial identification bounds via HYDE scenarios; focus on data-rich regions for core results |
| Clustering sensitivity to feature selection | Multiple algorithms, validate against narrative, report robustness |
| Reviewer concern about modern borders on historical data | Be transparent, show robustness to regional vs. country aggregation |

### Explicit non-goals

- No new data collection or scraping
- No web applications or dashboards
- No ML prediction models (clustering is descriptive only)
- No paleoclimate data sourcing in v1 (noted as future extension)
