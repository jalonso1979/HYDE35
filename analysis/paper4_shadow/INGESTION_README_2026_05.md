# Paper D modern-endpoint data ingestion (2026-05-20)

> **STATUS UPDATE 2026-07-18.** The ERA5 bulk download COMPLETED 2026-07-17
> (22,800/22,800 files, 25 regions × 1950–2025, 0 failed; see
> `ERA5/_bulk_download.log`). Notes below describing the download as running,
> region 25 as partial, or a 49-country "coverage cliff" are historical. Two
> caveats for anyone acting on this file: (1) the completion download wrote
> plain merged netCDF4 files (not CDS zip containers) — readers were fixed
> 2026-07-18 to handle both formats; (2) `../update_era5_panel.py` and
> `merge_modern_endpoint_panel.py` were deleted in commit 0b91837 (2026-06-09)
> — restore from `git show 0b91837~1:<path>` if the modern-endpoint work
> resumes. `era5_country_monthly.parquet` has been rebuilt from the complete
> archive.

Scaffolding for ERA5 + PRIMAP-hist modern endpoint of the long-shadow analyses. Built 2026-05-20 as part of the SETI data sweep. **No analysis code yet — drafting deferred per portfolio plan until the long-shadow headline settles.**

## What was added this session

| File | Role |
|---|---|
| `ingest_primap_hist_v27.py` | Reshape PRIMAP-hist v2.7 (58k rows × 282 cols wide format) into long-format parquet at `analysis/data/primap_hist_v27_long.parquet` (16M rows). Headline CO2/CH4/N2O global aggregate written separately. |
| `merge_modern_endpoint_panel.py` | Join PRIMAP country×year emissions to ERA5 country×year climate covariates → `analysis/data/long_shadow_modern_endpoint_panel.parquet`. |

## Existing assets used

- `../update_era5_panel.py` — builds `analysis/data/era5_full_panel.parquet` from `/Volumes/BIGDATA/HYDE35/ERA5/region={1..25}/year={1950..2025}/_extracted/`. **Rebuilt 2026-05-20 20:59 (this session)**: 1,844 region-year rows (regions 1–24 complete 1950–2025, region 25 partial 1950–1969). Previous April 17 build had 1,195 rows. **Bulk downloader still running** for region 25 1970+ (≈675 of 22,800 files outstanding); a further rebuild will pick up the remainder when complete.
- `analysis/data/era5_country_monthly.parquet` — country-monthly ERA5 panel (78,421 rows, 196 countries, 1950–2025) with `t2m_c` and `tp_mm`. Originally built 2026-05-14 by `../shared/build_era5_country_monthly.py`. **Re-ran 2026-05-20 21:38 in this session — produced identical panel (no-op)**: the May 14 build had already captured all raw-zip-archive files present on disk; the additional `_extracted/` subfolder content from the ongoing downloader has not yet added new top-level `era5_*.nc` archives.

  **Coverage cliff identified.** 147 countries have full 1950–2025 coverage (76 years each); **49 countries are stuck at partial coverage** — 32 at 17 years, 11 at 18 years, 6 at 29 years. These 49 align with region 25 (currently downloading 1950–1969 of 1950–2025). **Re-run `build_era5_country_monthly.py` after the bulk downloader finishes region 25** to lift these 49 countries to full coverage; the merge will then expand the modern-endpoint panel from 12,088 to ~14,900 country-year rows.

## Headline numbers (2026-05-20)

PRIMAP global emissions trajectory by ISO3, after merge:

| Year | n_countries | CO2 total kt | CH4 total kt |
|---|---|---|---|
| 1750 | 212 | 348 | 61,401 |
| 1800 | 212 | 30,638 | 73,299 |
| 1850 | 212 | 225,007 | 97,367 |
| 1900 | 212 | 2,949,637 | 142,944 |
| 1950 | 212 | 10,651,500 | 265,361 |
| 2000 | 212 | 51,289,950 | 578,125 |
| 2020 | 212 | 80,865,280 | 697,266 |

ERA5 country×year coverage: 12,088 rows, 196 countries, 1950–2025.

## Pending — to be wired when drafting begins

1. **Rebuild ERA5 panel** once region 25 download completes (`../update_era5_panel.py`). The current `era5_full_panel.parquet` was built 2026-04-17 with partial coverage.
2. **Decide headline outcome variable** — Paper D is pre-headline per the portfolio memory. Once a long-shadow → modern outcome mapping is chosen (development, urbanization, emissions intensity, etc.), wire the relevant column into `merge_modern_endpoint_panel.py` as a left-side variable.
3. **HYDE 3.5 grid-to-country aggregation** — the long-shadow analyses currently work at HYDE's 5-arcmin grid; the modern endpoint joins are at the country level (PRIMAP ISO3). Decide whether to (a) aggregate HYDE to country×year for compatibility with PRIMAP, or (b) downscale PRIMAP to grid using cropland-area weights. Defer until headline is chosen.
4. **Hotelling-Hartwick structural anchor** — **held until SETI Paper F advances past v0.1** per project memory. Do not introduce until SETI stage-2 (structural depletable-resource solve + Bern multi-reservoir CO2) is implemented.

## Quick reference — SETI data assets used here

| Asset | Path |
|---|---|
| PRIMAP-hist v2.7 raw | `/Users/jalonso/.../Pandemics/SETI/Data/emissions_ipcc/Guetschow_et_al_2025a-PRIMAP-hist_v2.7_final_no_rounding_22-Aug-2025.csv` |
| PRIMAP long-format parquet | `/Volumes/BIGDATA/HYDE35/analysis/data/primap_hist_v27_long.parquet` |
| Modern endpoint merged panel | `/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_modern_endpoint_panel.parquet` |
| ERA5 country-monthly | `/Volumes/BIGDATA/HYDE35/analysis/data/era5_country_monthly.parquet` |
| ERA5 raw extracted | `/Volumes/BIGDATA/HYDE35/ERA5/region={1..25}/year={1950..2025}/_extracted/` |
