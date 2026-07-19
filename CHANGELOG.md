# Changelog

All notable changes to this replication package are documented in this file.

## [Unreleased] — 2026-07-18

### ERA5: archive completed, readers fixed, panels rebuilt, compact products added
- The raw ERA5 download **completed 2026-07-17**: 22,800/22,800 files (25 regions × 1950–2025 × 12 months, hourly t2m + tp, 1.5 TB; `ERA5/_bulk_download.log`). The archive is mixed-container (8,092 CDS zip files from May, 14,708 plain merged netCDF4 from the July bulk completion) and mixed-resolution (1.0° through 1966/67, 0.25° after).
- **Reader fix (load-bearing):** every ERA5 reader was zip-only and silently skipped the 14,708 plain-netCDF months. `build_era5_country_monthly.py`, `build_era5_country_annual_v2.py`, and `era5_region_country_map.py` now handle both containers (leading-magic detection; `zipfile.is_zipfile` false-positives on some HDF5 files). `build_era5_country_panel.py` (v1) was ported off the corrupt `_extracted/` cache, whose 1968+ years held only the last-extracted month.
- **Fertility panels rebuilt** from the complete archive: `era5_country_annual_v2.parquet` 126 → 532 rows (1950–2025, true 1961–1990 baseline), `era5_country_annual.parquet` de-corrupted, `country_climate_spliced.parquet` 4,116 → 7,175 rows (1421–2025; ERA5 splices all of 1950–2025 on the full 59-year overlap), `england_climate_annual_1421_2008.parquet` now includes a real ERA5 2009–2025 splice (17 rows). Figures fig18v2/fig20/fig20v2 re-rendered; matched ERA5↔ModE-RA growing-season correlations 0.90–0.99 over 1950–2008. All 15 ERA5-related fertility tests pass.
- `era5_country_monthly.parquet` rebuilt: 78,421 → 178,752 rows, all 196 countries at full 912-month coverage; new values reproduce the old builder exactly on the previously-complete ≤1966 months (max |Δ| = 0).
- **New: `analysis/shared/build_era5_compact.py`** — one-sweep compaction (prepass → sweep → finalize) producing gridded `ERA5_derived/cell_daily` (51 GB, quantized per-cell daily stats incl. true hourly Tmin/Tmax, rolling 1/3/6 h precip maxima, wet/heat/frost hours) and `cell_monthly` (6.8 GB, incl. consecutive-hour heat/frost spells and wet-hour intensity), plus a ~0.9 GB laptop bundle in `analysis/data/era5_derived/`: `era5_country_daily.parquet` (5.44 M rows; area-, pop-2000- and cropland-2000-weighted), `era5_country_day_tbins.parquet` (hourly temperature-exposure bins in 3 °C steps), `era5_country_monthly_v2.parquet` (superset of the monthly schema), and the cells/weights reference tables.
- Documentation de-staled: README, REPRODUCE, Makefile download note, analysis/data README, INGESTION_README (status banner), SPEI-note addendum, module docstrings. `ERA5_derived/` gitignored.

### 2026-07-19 follow-on: extensions, revivals, calibration recompute, cleanup
- **Fertility ERA5 chain extended to all 12 panel countries** (added NOR, DNK, FIN, ISL, CHE capitals + bboxes to `era5_region_country_map.py`; all in region 11). `era5_country_annual_v2.parquet` and `era5_country_annual.parquet` now 912 rows each; `country_climate_spliced.parquet` 7,260 rows with ERA5 for 76 years in every country (splice boundaries continuous, max 1949→1950 jump 0.61 °C). Figures fig18v2/fig20/fig20v2 re-rendered.
- **main.tex panel stack revived**: restored `update_era5_panel.py` (ported off the deleted `_extracted/` cache to the raw monthlies, parallelized), `build_extended_panel.py`, `build_modern_panel.py` (window extended 1950–1967 → 1950–2025), `merge_modern_endpoint_panel.py`, and `run_ag_impact_final.py` from commit `0b91837~1`. Rebuilt: `era5_full_panel.parquet` (1,900 region-years, 25×76), `hyde_era5_extended_panel.parquet` (climate columns now 100% non-null, were 77%), `hyde_era5_full_panel.parquet` (3,654 → 15,428 rows, 1950–2025), `long_shadow_modern_endpoint_panel.parquet`; fig11/fig12 re-rendered. **Full-sample re-estimation** (`ag_impact_full_sample_2026-07-19.log`): Law 1 holds (β=−0.0068, p=0.0001), Law 3 strengthens (+0.0027, p<0.001), **Law 2's "warming paradox" reverses sign** (temp → pop growth β=−0.0050, p<0.0001) — main.tex carries a dated revision note.
- **Calibration parquets recomputed** from the complete archive by the new committed writer `analysis/paper4_shadow/build_era5_modera_calibration.py` (the originals were frozen artifacts with no committed writer): median |bias| 0.61 → **0.52 °C**, median monthly t-corr 0.971 → **0.977**, median annual anomaly corr 0.938 → **0.832** (the old annual figure was inflated by the seasonally-truncated partial sample). `make check` re-pinned; both cover letters updated. `modera_era5_bias_1950_2008.parquet` left frozen (documented back-compat artifact).
- **Space reclaimed (~300 GB)**: deleted the corrupt `_extracted/` caches (260 GB, 1,904 dirs), test tiles `region=99–102`, `region=6` `_raw_zips` quarantine, and 5 orphaned partial-era parquets (`era5_regional_annual`, `era5_climate_shocks`, `era5_climate_endowments`, `hyde_era5_merged_panel`, `country_era5_region_map`). Raw archive untouched (22,800 files verified).
- New `make manifest` target (`analysis/make_manifest.py`); MANIFEST.md regenerated.

### 2026-07-19 (later): iron-laws revision on the truly full panel; stale region filter removed
- **Stale filter found and removed**: `run_ag_impact_final.py` still carried `era5_region.isin(range(1, 9))` from the partial-download era, silently cutting the "full-sample" re-estimation to 73 of 197 countries. All 25 regions have complete climate coverage since 2026-07-17; the filter is gone and the log (`ag_impact_full_sample_2026-07-19.log`) is regenerated on 14,972 obs / 197 countries.
- **Full-panel results**: Law 1 holds (temp → ag land growth β=−0.0054, p<0.0001, N=14,359). **Law 2's "warming paradox" reversal is confirmed and stronger** (temp → pop growth β=−0.0055, p<0.0001; +0.0024 in 1950–1969 vs −0.0050 in 1970–2025; rolling windows positive only in the earliest window, strongly negative in 2000–2019/2005–2024). Law 3 holds but weaker (temp × crop share +0.0025, p=0.015). **Law 4's 7.7× pathway gradient collapses**: all four pathways negative within 2.7× of each other, ordering not preserved, interactions jointly insignificant (Wald χ²(3)=1.43, p=0.70; new committed section 1b of `run_ag_impact_final.py`).
- **Figures de-staled**: `fig11_iron_laws.py` and `fig12_climate_channel.py` no longer hardcode 1950–1967 spec values — they load the same panel as `run_ag_impact_final.py` and compute every annotated estimate in-script with the same FE estimator. fig12 panel A now shows the 1950–1969 vs 1970–2025 regime fits; panel B schematic shows both channels negative.
- **main.tex revised and recompiled (28 pp)**: abstract, contributions, iron-laws section (Laws 1–4 with full-sample numbers and explicit reporting of the two reversals), Discussion subsection retitled "The Transient Warming–Population Response", conclusion, fig11/fig12 captions. The dated 2026-07-19 revision note is resolved and removed.

## [1.0.0] — 2026-05-14

First public release. Corresponds to the May 2026 draft of the manuscript.

### Data pipeline
- ModE-RA paleo-reanalysis (1421–2008) area-weighted to 196 HYDE countries via the 5-arcminute `iso_cr` raster.
- ModE-RA also aggregated to 3,187 HYDE sub-national units (`sub_iso_cr`).
- CRU TS 4.09 (1901–1950) aggregated to country-month and used as the absolute-temperature climatology.
- ERA5 reanalysis (1950–2025) rebuilt at country-month resolution from raw 0.25° zipped NetCDFs; replaces earlier 25-macro-region panel.
- Calibrated annual country panel 1421–2025 with per-country bias correction against ERA5 in 1950–2008 overlap.
- ModE-RA ensemble standard-deviation, min, and max fields aggregated to country-month for uncertainty quantification.

### Analyses
- **Stage 1.** Country-level multinomial logit (raw seasonality not significant) and storage-demand reformulation (F=6.31, p<0.001).
- **Sub-national Stage 1.** Within-country FE: β=+0.067 (p=0.019) for crop share; β=+0.66 (p=0.0002) for log density. Two-way clustering by country and 10° latitude band tightens to p=0.0001.
- **Pre-industrial Malthus.** Country-level 1421–1750 with annual climate; extended to 1421–1950 with subperiod stability and the demographic-transition signature.
- **Sub-national Malthus.** 84,907 cells × 3,146 sub-units × 196 countries with sub-unit FE.
- **Volcanic event study.** Five eruptions (Huaynaputina, Parker, Tambora, Krakatoa, Pinatubo); high-density-intensive × T_shock interaction p=0.032.
- **Volcanic placebos.** 1,000-draw five-year placebo (25th percentile) and 1,000-draw Tambora-alone placebo (65th percentile).
- **Long shadow.** Pre-industrial σᵥ → modern population growth, β=−2.48, p<10⁻¹⁵, R²=0.45 with pathway FE. Sub-national within-country replication finds smaller geographic component (β=−0.31, p=0.043).
- **Robustness.** Calibration vs country-level ERA5 (median monthly r=0.97, annual anomaly r=0.94); ModE-RA ensemble uncertainty; documentary validation of five famines; HYDE Manski bounds (scale-invariant by construction); pathway clustering silhouette and bootstrap stability.

### Paper
- 27-page manuscript with sans-serif Helvetica typography, grayscale figures, running footer with page-of-page, working-paper header strip on title page, sober drop-cap on the introduction.
- Two cover-letter drafts (JEG, QJE).

### Infrastructure
- `Makefile`, `requirements.txt`, `LICENSE` (MIT + CC-BY-4.0 split), `CITATION.cff`.
- `README.md`, `REPRODUCE.md`, `analysis/README.md`, `analysis/data/README.md`, `MANIFEST.md`.

### Known limitations
- HYDE 3.5 has decadal resolution at best for the 1500–1950 panel; no annual demographic data pre-1950.
- ModE-RA quality varies by region and era; pre-1700 outside Europe is largely model prior.
- HYDE published uncertainty scenarios (lower/upper) are uniform 80%/120% scalings, so the Manski bounds we compute are tight by construction.
- Individual eruptions are not separately identified; the pooled five-event volcanic study is suggestive triangulation, not standalone identification.

## Earlier internal drafts

Earlier versions of the paper, predating this replication package, used:
- A region-level (25 macro-region) ERA5 panel that incorrectly suggested ModE-RA had poor calibration (median r=0.29).
- Proxy seasonality (rolling standard deviation of decadally-averaged PAGES 2k reconstructions) for pre-industrial climate.
- A pathway typology defended only by external companion paper.

All three issues are resolved in 1.0.0.
