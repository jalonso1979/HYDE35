# Generated panel data — manifest

All files are Apache Parquet. Total ~262 MB. Re-distributable under CC-BY-4.0.

Each file is produced by a single build script (`analysis/shared/*.py` for panels, `analysis/paper4_shadow/*.py` for analysis outputs).

## Climate panels

| File | Rows | Description | Producer |
|---|---:|---|---|
| `modera_country_monthly.parquet` | 1,382,976 | ModE-RA monthly T, P anomalies aggregated to 196 HYDE countries, 1421–2008 | `build_modera_panel.py` |
| `modera_country_coverage.parquet` | 196 | Country-level cos-lat weight sums from ModE-RA aggregation | `build_modera_panel.py` |
| `modera_country_uncertainty.parquet` | 1,382,976 | ModE-RA ensemble σ, min, max per country-month | `build_modera_uncertainty.py` |
| `modera_subnational_monthly.parquet` | 22,487,472 | ModE-RA monthly T, P anomalies on 3,187 HYDE sub-units, 1421–2008 | `build_subnational.py` |
| `cru_country_monthly_1901_1950.parquet` | 117,600 | CRU TS 4.09 monthly T, P aggregated to countries 1901–1950 | `build_cru_climatology.py` |
| `cru_country_climatology_1901_1950.parquet` | 2,352 | Country × calendar-month climatology from CRU TS 1901–1950 | `build_cru_climatology.py` |
| `era5_country_monthly.parquet` | 178,752 | ERA5 monthly fields aggregated to HYDE countries, 1950–2025 (complete archive; 1.0° tiles ≤1966/67, 0.25° after) | `build_era5_country_monthly.py` (or `build_era5_compact.py` finalize) |
| `country_climate_1421_2025.parquet` | 119,112 | Calibrated annual T, P panel — ModE-RA + CRU only, 1421–2008 (ERA5 splice removed in commit 6afdb3a) | `build_calibrated_annual.py` |
| `country_seasonality_1421_2008.parquet` | 115,248 | Country-year seasonality measures (σₛᵀ, σᵥᵀ, productive months, GDD, monsoon, etc.) | `build_seasonality.py` |
| `country_seasonality_preindustrial.parquet` | 196 | Pre-industrial (1421–1750) means of the above per country | `build_seasonality.py` |

## HYDE panels (sub-national + country)

| File | Rows | Description | Producer |
|---|---:|---|---|
| `subnational_hyde.parquet` | 431,053 | Sub-national HYDE pop, cropland, grazing — all available timesteps | `build_subnational.py` |
| `subnational_features.parquet` | 3,187 | Pre-industrial sub-national climate features for Stage 1 sub-national | `stage1_subnational.py` |
| `paper1_clustered_features.parquet` | 157 | K-means pathway labels and clustering features (from companion paper) | external |

## Stage 1 outputs

| File | Description | Producer |
|---|---|---|
| `stage1_country_features.parquet` | Country-level Stage 1 features merged with pathway labels | `stage1_country.py` |
| `stage1_storage_results.parquet` | ANOVA F-statistics for candidate storage-demand measures | `stage1_storage_index.py` |
| `pathway_silhouette.parquet` | Silhouette score by K ∈ {2,…,8} | `pathway_validation.py` |
| `pathway_stability.parquet` | Country-level bootstrap stability of K=5 partition | `pathway_validation.py` |

## Stage 2 (Malthusian) outputs

| File | Description | Producer |
|---|---|---|
| `preindustrial_malthus_panel.parquet` | Country-interval Malthus panel 1421–1750 with annual climate | `preindustrial_malthus.py` |
| `preindustrial_malthus_results.parquet` | Pathway-stratified coefficients 1421–1750 | `preindustrial_malthus.py` |
| `preindustrial_malthus_panel_extended.parquet` | Country-interval panel 1421–1950 | `preindustrial_malthus_extended.py` |
| `preindustrial_malthus_extended_results.parquet` | Subperiod and pathway coefficients 1421–1950 | `preindustrial_malthus_extended.py` |
| `subnational_malthus_panel.parquet` | Sub-national interval panel 1500–1950 (84,907 cells) | `subnational_malthus.py` |
| `robustness_malthus_subperiods.parquet` | Subperiod stability checks | `robustness.py` |

## Volcanic event study

| File | Description | Producer |
|---|---|---|
| `volcanic_event_panel.parquet` | 5 eruptions × 154 countries panel | `volcanic_event_study.py` |
| `volcanic_event_results.parquet` | Pathway-stratified slope estimates | `volcanic_event_study.py` |
| `volcanic_placebo_results.parquet` | 1,000 placebo draws of 5-year events | `volcanic_placebo.py` |
| `tambora_individual_placebo.parquet` | 1,000 single-year placebos for Tambora alone | `tambora_individual.py` |

## Long-shadow outputs

| File | Description | Producer |
|---|---|---|
| `long_shadow_rerun.parquet` | Cross-country long-shadow coefficients with new climate measures | `long_shadow_rerun.py` |
| `subnational_long_shadow_panel.parquet` | Sub-national long-shadow features + modern outcomes | `subnational_long_shadow.py` |
| `subnational_long_shadow_results.parquet` | With/without country FE coefficient table | `subnational_long_shadow.py` |

## Robustness

| File | Description | Producer |
|---|---|---|
| `era5_modera_calibration_monthly.parquet` | Per-country monthly bias / RMSE / correlation, 1950–2008. Frozen 2026-05-14 artifact computed from the then-partial ERA5 panel; its writer was never committed. Read (not written) by `robustness_v2.py`. | frozen artifact |
| `era5_modera_calibration_annual.parquet` | Per-country annual anomaly correlation (detrended). Same frozen provenance as the monthly file. | frozen artifact |
| `modera_era5_bias_1950_2008.parquet` | Earlier (region-vs-country) bias estimate; kept for back-compat. `build_calibrated_annual.py` no longer writes it (ERA5 removed). Still read by `robustness.py`. | frozen artifact |
| `manski_bounds.parquet` | Coefficient estimates under HYDE base/lower/upper scenarios | `manski_bounds.py` |
| `documentary_validation.parquet` | 5-famine climate + pop change comparison | `documentary_validation.py` |

## Legacy / not directly used by the paper

The following files were produced by earlier iterations or by the companion paper-1 pipeline. They remain in the package for completeness. Caveats: `hyde_era5_extended_panel.parquet` and `era5_full_panel.parquet` ARE still loaded by current scripts (`deep_determinants.py`, `long_shadow_rerun.py`, `run_all.py`, paper5 `exercise_extended_controls.py`) — the extended panel for its HYDE/centroid metadata columns only (its climate columns are stale, built from the partial ERA5 archive; builders deleted in commit 0b91837). `hyde_era5_full_panel.parquet` (1950–1967) feeds `fig11_iron_laws.py`/`fig12_climate_channel.py` for the superseded `paper/main.tex`:

`climate_panel_0_2025.parquet`, `country_analysis_panel.parquet`, `country_climate_stats.parquet`, `era5_full_panel.parquet`, `hyde_era5_full_panel.parquet`, `hyde_era5_extended_panel.parquet`, `hyde_modern_panel.parquet`, `region_analysis_panel.parquet`, `trajectory_features.parquet`.

Deleted 2026-07-19 (orphaned prototypes from the partial-archive era, zero code references): `country_era5_region_map.parquet`, `era5_climate_endowments.parquet`, `era5_climate_shocks.parquet`, `era5_regional_annual.parquet`, `hyde_era5_merged_panel.parquet`.

## Raw data sources

See top-level `REPRODUCE.md` section 1 for download instructions for HYDE 3.5, ModE-RA, ERA5, and CRU TS 4.09. Total raw download: ~95 GB without ERA5; the full ERA5 hourly archive (needed only for `long_shadow_fertility` and the derived products in `era5_derived/`) is an additional ~1.5 TB (complete since 2026-07-17).
