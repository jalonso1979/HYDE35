# How to reproduce the paper

This walk-through assumes a fresh machine with **Python 3.10+**, **git**, and **LaTeX (TeXLive 2023+)**. Total wall time on an M-series Mac: ~7 hours after downloads (including the KK10 cross-validation panel build, ~30 minutes of NetCDF aggregation).

## 0. Clone and install

```bash
git clone https://github.com/<user>/long-shadow-seasonality.git
cd long-shadow-seasonality
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # installs the project as `analysis`
```

Verify the import works:

```bash
python -c "from analysis.shared import build_modera_panel; print('ok')"
```

## 1. Download the raw data

The pipeline expects raw inputs in fixed locations under `/Volumes/BIGDATA/` (or set `HYDE35_ROOT` and `MODERA_ROOT` environment variables). Total disk: ~460 GB.

### HYDE 3.5 sub-national rasters and CSVs

```bash
# Country code raster (5-arcmin)
mkdir -p $HYDE35_ROOT/general_files/general_files
curl -L 'https://landuse.sites.uu.nl/...' -o $HYDE35_ROOT/general_files/general_files/iso_cr.asc
curl -L 'https://landuse.sites.uu.nl/...' -o $HYDE35_ROOT/general_files/general_files/sub_iso_cr.asc

# Three scenarios (baseline, lower, upper)
for scen in base lower upper; do
  mkdir -p $HYDE35_ROOT/gbc2025_7apr_$scen
  # subpopulation, cropland, grazing, irrigation, rice
  for var in subpop sub_hiscrop sub_hispast sub_hisirri sub_hisrice; do
    curl -L "https://landuse.sites.uu.nl/.../${var}_4apr2025.csv" \
         -o $HYDE35_ROOT/gbc2025_7apr_$scen/${var}_4apr2025.csv
  done
done
```

### ModE-RA paleo-reanalysis

```bash
mkdir -p $MODERA_ROOT
# Ensemble-mean monthly anomaly fields (1421-2008)
curl -L 'https://www.wdc-climate.de/...' -o $MODERA_ROOT/ModE-RA_ensanom_1-28.tar
tar -xf $MODERA_ROOT/ModE-RA_ensanom_1-28.tar -C $MODERA_ROOT/extracted/

# Optional: the 140-file member tar for ensemble-spread analysis
curl -L 'https://www.wdc-climate.de/...' -o $MODERA_ROOT/ModE-RA_memanom_1-140.tar
```

Only the four ensemble-statistic files are needed for the headline analyses:

- `ModE-RA_ensmean_temp2_anom_wrt_1901-2000_1421-2008_mon.nc`
- `ModE-RA_ensmean_totprec_anom_wrt_1901-2000_1421-2008_mon.nc`
- `ModE-RA_ensstd_temp2_anom_wrt_1901-2000_1421-2008_mon.nc`
- `ModE-RA_ensstd_totprec_anom_wrt_1901-2000_1421-2008_mon.nc`

### ERA5 reanalysis — not required

ERA5 has been removed from this project. It was previously used to calibrate ModE-RA in the 1950–2008 overlap and to extend the panel past 2008, but the calibration was a per-country additive bias that is absorbed exactly by the within-country fixed effects in every regression (verified: headline coefficients identical with and without it), so it is not load-bearing. Absolute levels are now anchored to the CRU 1901–1950 climatology and the panel ends at ModE-RA's 2008 horizon. The ~366 GB ERA5 download is no longer needed.

### KK10 land-use reconstruction (cross-validation, §4.5)

The KK10 raw NetCDF (17.3 GB) is the population-independent land-use
reconstruction used to cross-validate the area-orthogonality finding.
Download from PANGAEA:

```bash
mkdir -p $HYDE35_ROOT/../Pandemics/Data/KK10
curl -L 'https://doi.pangaea.de/10.1594/PANGAEA.871369?format=textfile' \
     -o $HYDE35_ROOT/../Pandemics/Data/KK10/KK10.nc
```

If you place KK10.nc elsewhere, edit the `KK10_NC` constant at the top of
`analysis/paper4_shadow/build_kk10_country_panel.py`.

### CRU TS 4.09

```bash
mkdir -p $HYDE35_ROOT/climate_reconstructions/cru_ts
for decade in 1901 1911 1921 1931 1941; do
  curl -L "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/.../cru_ts4.09.${decade}.$((decade+9)).tmp.dat.nc.gz" \
       -o $HYDE35_ROOT/climate_reconstructions/cru_ts/cru_ts4.09.${decade}.$((decade+9)).tmp.dat.nc.gz
  # ... same for .pre.dat (precipitation) ...
done
gunzip $HYDE35_ROOT/climate_reconstructions/cru_ts/*.gz
```

## 2. Build the country panels

```bash
# Country-monthly ModE-RA (1421-2008): ~5 min
python -m analysis.shared.build_modera_panel

# CRU TS 1901-1950 country-month climatology: ~3 min
python -m analysis.shared.build_cru_climatology

# Country-year seasonality measures: ~1 min
python -m analysis.shared.build_seasonality

# Calibrated annual panel, ModE-RA + CRU only (1421-2008): ~1 min
python -m analysis.shared.build_calibrated_annual

# ModE-RA ensemble uncertainty: ~3 min
python -m analysis.shared.build_modera_uncertainty
```

## 3. Build the sub-national panels

```bash
# Sub-national monthly ModE-RA (3,187 units, 1421-2008): ~3 min
python -m analysis.shared.build_subnational
```

## 4. Run the analyses

```bash
# Stage 1
python -m analysis.paper4_shadow.stage1_country
python -m analysis.paper4_shadow.stage1_storage_index
python -m analysis.paper4_shadow.stage1_subnational

# Stage 2 (Malthus)
python -m analysis.paper4_shadow.preindustrial_malthus
python -m analysis.paper4_shadow.preindustrial_malthus_extended
python -m analysis.paper4_shadow.subnational_malthus

# Volcanic event study
python -m analysis.paper4_shadow.volcanic_event_study
python -m analysis.paper4_shadow.tambora_individual
python -m analysis.paper4_shadow.volcanic_placebo

# Long shadow
python -m analysis.paper4_shadow.long_shadow_rerun
python -m analysis.paper4_shadow.subnational_long_shadow

# Robustness
python -m analysis.paper4_shadow.robustness_v2
python -m analysis.paper4_shadow.documentary_validation
python -m analysis.paper4_shadow.pathway_validation
python -m analysis.paper4_shadow.manski_bounds
python -m analysis.paper4_shadow.spatial_se_robustness

# Joint reduced-form panel system (§3 headline) + calibrated 3-eq model (App. I)
python -m analysis.paper4_shadow.joint_landuse_var            # builds joint_landuse_var_panel + results + figJ
python -m analysis.paper4_shadow.joint_var_bootstrap          # Webb wild-cluster bootstrap (App. L.2)
python -m analysis.paper4_shadow.joint_var_post1700           # post-1700 restriction (App. L.3)
python -m analysis.paper4_shadow.joint_var_climate_pathways   # climate-only re-clustering (App. L.1)
python -m analysis.paper4_shadow.structural_3eq               # calibrated 3-eq model + counterfactuals (App. I)

# Boserup honest null + KK10 cross-validation (§4.5)
python -m analysis.paper4_shadow.build_kk10_country_panel       # ~25 min, one-time
python -m analysis.paper4_shadow.boserup_robustness             # non-overlapping windows + Americas drop
python -m analysis.paper4_shadow.boserup_cropland_area          # alternative outcome decomposition
python -m analysis.paper4_shadow.area_margin_diagnostic         # pop-control orthogonality
python -m analysis.paper4_shadow.kk10_orthogonality             # HYDE vs KK10 area comparison
python -m analysis.paper4_shadow.kk10_pathway_heterogeneity     # 4-pathway KK10 stratification
python -m analysis.paper4_shadow.channel_substitution           # pathway-level (φ^P, φ^L) scatter
python -m analysis.paper4_shadow.country_substitution           # country-level Deming regression
```

## 5. Regenerate the figures

```bash
python analysis/paper4_shadow/make_figures.py
python analysis/paper4_shadow/make_fig1_map.py
```

Figures are written to `analysis/figures/paper4_v2/` as `.pdf` and `.png`.

## 6. Compile the paper

```bash
cd paper
pdflatex long_shadow.tex
pdflatex long_shadow.tex          # second pass for cross-references
# cover letters
pdflatex cover_letter_jeg.tex     # legacy Journal of Economic Growth
pdflatex cover_letter_qje.tex     # legacy Quarterly Journal of Economics
pdflatex cover_letter_restud.tex  # Review of Economic Studies
pdflatex cover_letter_aejmacro.tex # American Economic Journal: Macroeconomics
```

Output: `paper/long_shadow.pdf` (~49 pages).

## 7. Verify

After the full pipeline finishes, the headline numbers below should appear in the paper and match within 0.001 (random seeds are pinned):

| Coefficient | Expected value |
|---|---|
| Sub-national Stage 1, crop_share ~ productive_months, country FE | $\hat\beta = +0.067$, $p = 0.019$ |
| Sub-national Stage 1, log_density ~ productive_months, country FE | $\hat\beta = +0.658$, $p = 0.0002$ |
| Pre-ind. Malthus 1421–1750, pastoral β_density (country-clustered) | $-0.0059$, $p = 0.010$ |
| Joint VAR, crop-dominant late demographic margin | $-9.65 \cdot 10^{-5}$, $p < 10^{-5}$ |
| Joint VAR, pastoral/mixed late demographic margin | $-5.55 \cdot 10^{-5}$, $p < 10^{-5}$ |
| Long shadow, σᵥ → log pop growth, pathway FE | $-2.484$, $p < 10^{-15}$ |
| KK10 area, 21-c core post-1700 (Boserupian cross-validation) | $+1.07 \cdot 10^{-5}$, $p = 0.005$ |
| KK10 area, 21-c core post-1700 + Δlog P control | $+1.08 \cdot 10^{-5}$, $p = 0.005$ |
| KK10 area, high-density intensive post-1700 | $+4.29 \cdot 10^{-5}$, $p = 0.047$ |
| Country-level Deming substitution slope, bootstrap median | $+0.0045$ (CI $[-0.003, +0.19]$) |
| Tambora individual placebo, 65th percentile | $\hat\beta = -0.0031$, $p = 0.68$ |

## Troubleshooting

- **Memory pressure during sub-national aggregation.** Default peak is ~5 GB. If your machine has <8 GB, edit `build_subnational.py` to process ModE-RA in chunks of 1000 months instead of loading all 7056 at once.
- **h5netcdf import errors.** ModE-RA raw files are netCDFs. Install both `h5netcdf` and `h5py` (already in `requirements.txt`).
- **Figures look wrong.** Make sure `matplotlib` is using Helvetica or Arial fonts. On Linux, `apt-get install ttf-mscorefonts-installer` provides Arial; on macOS it should work out of the box.

## When in doubt

The cleanest verification is to compute the calibration triptych:

```bash
python -m analysis.paper4_shadow.robustness_v2
```

Expected output:

```
Median |bias|:                0.61 °C
Median monthly t-corr:        0.971
Median annual anomaly corr:   0.938
```

If those numbers reproduce, the panel-build pipeline is correct.
