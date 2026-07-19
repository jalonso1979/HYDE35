# Code structure

Two directories:

- `shared/` — data-build pipeline. Each module produces a parquet file in `analysis/data/`.
- `paper4_shadow/` — analyses for the "Long Shadow of Seasonality" paper. Each module produces a parquet of results and/or a figure in `analysis/figures/paper4_v2/`.

## Build pipeline (`analysis/shared/`)

Run in this order; later steps depend on earlier outputs.

| Step | Module | Output | Wall time |
|---:|---|---|---:|
| 1 | `build_modera_panel.py` | `modera_country_monthly.parquet` | ~5 min |
| 2 | `build_cru_climatology.py` | `cru_country_climatology_1901_1950.parquet` | ~3 min |
| 3 | `build_seasonality.py` | `country_seasonality_1421_2008.parquet` | ~1 min |
| 4 | `build_era5_country_monthly.py` | `era5_country_monthly.parquet` | ~5 h serial (manual/optional — not wired into `make`; `build_era5_compact.py` produces the same panel plus the full derived-product set in ~3 h at 6 workers) |
| 5 | `build_calibrated_annual.py` | `country_climate_1421_2025.parquet` | ~1 min |
| 6 | `build_modera_uncertainty.py` | `modera_country_uncertainty.parquet` | ~3 min |
| 7 | `build_subnational.py` | `modera_subnational_monthly.parquet`, `subnational_hyde.parquet` | ~5 min |

Auxiliary:

- `era5_bulk_downloader.py` — for the initial ERA5 download from CDS. Not part of the build chain. Download completed 2026-07-17 (22,800/22,800 files).
- `era5_downloader.py` — single-region downloader (older, kept for back-compat).
- `build_era5_compact.py` — one-sweep ERA5 compaction: cell-daily/-monthly grids (`ERA5_derived/`), country-day panel, hourly temperature-exposure bins, and the laptop bundle in `analysis/data/era5_derived/`. Run `prepass` → `sweep` → `finalize`.
- `loaders.py`, `masks.py`, `variables.py`, `uncertainty.py`, `plotting.py`, `config.py` — utility modules used by the companion paper-1 pipeline. Not loaded by paper 4 scripts.

## Analysis pipeline (`analysis/paper4_shadow/`)

Independent modules; can be run in any order once panels exist.

### Stage 1 (selection)
- `stage1_country.py` — country-level multinomial logit; the negative-result version.
- `stage1_storage_index.py` — composite Matranga-style measure; the positive Stage 1 result.
- `stage1_subnational.py` — within-country causal identification (the headline).

### Stage 2 (Malthusian dynamics)
- `preindustrial_malthus.py` — country-level 1421–1750 with annual climate.
- `preindustrial_malthus_extended.py` — extended to 1421–1950, subperiod breakdown.
- `subnational_malthus.py` — 84k-cell sub-national panel with sub-unit FE.

### Volcanic event study
- `volcanic_event_study.py` — pooled five-eruption pathway-stratified test.
- `volcanic_placebo.py` — 1,000-draw five-year placebo.
- `tambora_individual.py` — single-eruption placebo, honest "not separately identified" finding.

### Long shadow
- `long_shadow_rerun.py` — country-level long shadow with new climate measures.
- `subnational_long_shadow.py` — within-country (geographic) channel.

### Robustness & validation
- `robustness_v2.py` — ModE-RA vs country-level ERA5 calibration.
- `documentary_validation.py` — five-famine climate-vs-demographic-response check.
- `pathway_validation.py` — silhouette, bootstrap stability, ARI across K.
- `manski_bounds.py` — HYDE base/lower/upper sensitivity (uniform scaling → tight bounds).
- `spatial_se_robustness.py` — two-way clustering (country, latitude band).

### Figure production
- `figstyle.py` — matplotlib style helper (sans-serif, grayscale, restrained).
- `make_figures.py` — figures 02–09 in the paper.
- `make_fig1_map.py` — figure 01 (world map of pathways).

## Conventions

- Random seeds: `random_state=42` for `KMeans`, `np.random.default_rng(42)` for `numpy` generators.
- Standard errors: country-clustered by default. Two-way clustering (country × 10° latitude band) reported as a robustness check in `spatial_se_robustness.py`.
- All scripts can be invoked as `python -m analysis.<package>.<module>` from the repo root.
- Output paths are anchored at `ROOT / "analysis" / "data"`; figures at `ROOT / "analysis" / "figures" / "paper4_v2"`.

## Re-running a single analysis

```bash
# Example: re-run the volcanic placebo
python -m analysis.paper4_shadow.volcanic_placebo
```

Output appears in `analysis/data/volcanic_placebo_results.parquet`. The runtime for this specific script is ~6 minutes (1,000 placebo draws).
