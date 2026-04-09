# Historical Temperature Reconstruction Integration

**Date**: 2026-04-09
**Goal**: Extend the ERA5 climate panel from 1950–2025 backward to 0 CE using three complementary datasets, enabling climate-agriculture regressions over the full HYDE 3.5 temporal range.

## Data Sources

| Layer | Dataset | Period | Spatial Resolution | Variables |
|-------|---------|--------|--------------------|-----------|
| C | PAGES 2k Consortium | 0–1849 CE | 7 continental regions | Temperature anomalies (vs 1961-1990) |
| B | Berkeley Earth | 1850–1900 | 1° gridded, monthly | Temperature anomalies (vs 1951-1980) |
| A | CRU TS 4.x | 1901–1949 | 0.5° gridded, monthly | Temperature (°C) + Precipitation (mm) |
| — | ERA5 (existing) | 1950–2025 | 25 ERA5 regions | Temperature (°C) + Precipitation (mm) |

## PAGES 2k → ERA5 Region Mapping

| PAGES 2k Region | ERA5 Regions |
|-----------------|-------------|
| Arctic | 1 |
| North America | 2, 3, 4 |
| South America | 5, 6 |
| Europe | 12, 13, 14, 15 |
| Asia | 9, 10, 11, 16, 17, 18, 19 |
| Australasia | 20, 21, 22 |
| Africa | 7, 8, 23, 24, 25 |

All ERA5 regions within a PAGES 2k continent receive the same anomaly value pre-1850. This is appropriate given HYDE's centennial resolution in this period.

## Harmonization (Splice Calibration)

Working backward from ERA5 as the anchor:

1. **CRU TS ↔ ERA5** (overlap: 1950–1967, 18 years): CRU TS provides absolute temperatures. Compute per-region offset = mean(ERA5) − mean(CRU) over the overlap. Apply offset to CRU 1901–1949.

2. **Berkeley Earth ↔ CRU TS** (overlap: 1901–1930, 30 years): Berkeley Earth anomalies are relative to 1951–1980. Convert to absolute using: T_abs = anomaly + CRU_climatology(1951–1980 or nearest available). Verify with 30-year overlap offsets.

3. **PAGES 2k ↔ Berkeley Earth** (overlap: 1850–1900, 50 years): PAGES 2k anomalies are relative to 1961–1990. Convert to absolute using Berkeley Earth's regional 1961–1990 mean (which itself is calibrated to CRU/ERA5). Calibrate residual offset over the 50-year overlap.

## Output

**File**: `analysis/data/climate_panel_0_2025.parquet`

| Column | Type | Description |
|--------|------|-------------|
| region | int | ERA5 region 1–25 |
| year | int | 0–2025 |
| temperature_c | float | Absolute temperature (°C), harmonized |
| precipitation_mm | float | Precipitation — only available 1901+ (NaN before) |
| source | str | "pages2k", "berkeley", "cru_ts", "era5" |
| source_resolution | str | "continental", "1deg", "0.5deg", "regional" |

## Precipitation

No backfill before 1901. Proxy-based precipitation reconstructions are unreliable and would contaminate regressions. Analyses using pre-1901 data rely on temperature only.

## Implementation Steps

1. Download PAGES 2k CSV from NOAA Paleoclimatology
2. Download Berkeley Earth land-only gridded NetCDF
3. Download CRU TS tmp and pre variables
4. Aggregate Berkeley Earth and CRU TS to 25 ERA5 regions (area-weighted bounding-box means)
5. Map PAGES 2k 7 regions → 25 ERA5 regions
6. Run calibration chain: ERA5 ← CRU ← Berkeley ← PAGES2k
7. Save unified climate panel
8. Rebuild extended HYDE+climate country panel
9. Re-run agricultural impact analysis with deep historical data

## Downstream Integration

The unified climate panel replaces `era5_full_panel.parquet` as input to `build_extended_panel.py`. The extended HYDE+ERA5 country panel gains climate data for all HYDE time steps from 0 CE onward. Country-level regressions can now test climate-agriculture relationships over 2000 years.
