# The Long Shadow of Seasonality — replication package

**Climate endowments, agricultural pathways, and escape from the Malthusian trap.**

Jorge Alonso Ortiz · `jorge.alonso@icloud.com` · Draft: May 2026

[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)]()
[![License](https://img.shields.io/badge/license-MIT%2BCC--BY-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

---

## What's in this package

This package reproduces every figure, table, and reported coefficient in **"The Long Shadow of Seasonality"** from raw climate and historical land-use data.

- **`paper/`** — LaTeX source and compiled PDF of the manuscript (27 pages), plus cover-letter drafts.
- **`analysis/shared/`** — data-build pipeline that turns raw NetCDF and CSV inputs into country-level and sub-national parquet panels.
- **`analysis/paper4_shadow/`** — analysis scripts that produce all figures, tables, and numerical results.
- **`analysis/data/`** — intermediate and final parquet outputs (262 MB, redistributable).
- **`analysis/figures/paper4_v2/`** — paper figures as PDF and PNG.
- **`docs/superpowers/specs/`** — design notes from the project's planning phase.

## Quick start

Reproduce everything in one command (requires Python 3.10+, ~30 GB free for downloads, ~6 hours wall time):

```bash
make all
```

Or step-by-step:

```bash
# 1. Install Python dependencies
pip install -r requirements.txt
pip install -e .

# 2. Download raw data (see data/README.md for sources, ~366 GB total)
make download-data

# 3. Build the country and sub-national climate panels
make build-panels

# 4. Run all analyses (figures, tables, regression outputs)
make analysis

# 5. Compile the paper
make paper
```

See **`REPRODUCE.md`** for a complete walk-through.

## Key results

The paper documents five findings, each with a self-contained analysis script:

| Finding | Script | Outputs |
|---|---|---|
| Stage 1: productive-months index predicts pathway (F=6.31, p<0.001) | `paper4_shadow/stage1_storage_index.py` | `stage1_storage_results.parquet`, `fig03_storage_pathway.pdf` |
| Sub-national within-country causal identification (β=+0.067, p=0.019) | `paper4_shadow/stage1_subnational.py` | `subnational_features.parquet`, `fig04_subnational.pdf` |
| Pre-industrial Malthus with annual climate, 1421–1950 | `paper4_shadow/preindustrial_malthus_extended.py` | `preindustrial_malthus_extended_results.parquet`, `fig06_subperiods.pdf` |
| Volcanic event study, 5 eruptions (interaction p=0.032) | `paper4_shadow/volcanic_event_study.py` | `volcanic_event_results.parquet`, `fig07_volcanic.pdf` |
| Long shadow: σᵥ → modern pop growth (β=−2.48, R²=0.45) | `paper4_shadow/long_shadow_rerun.py` | `long_shadow_rerun.parquet`, `fig08_long_shadow.pdf` |

Robustness:

| Check | Script |
|---|---|
| ModE-RA ensemble uncertainty (median 1σ = 0.61°C) | `shared/build_modera_uncertainty.py` |
| Pathway clustering: silhouette, bootstrap stability | `paper4_shadow/pathway_validation.py` |
| Tambora individual placebo (real at 65th percentile) | `paper4_shadow/tambora_individual.py` |
| 1000-draw volcanic placebo (real at 25th percentile) | `paper4_shadow/volcanic_placebo.py` |
| HYDE Manski bounds across base/lower/upper scenarios | `paper4_shadow/manski_bounds.py` |
| Two-way clustered (country, latitude band) SEs | `paper4_shadow/spatial_se_robustness.py` |

## Data sources

| Source | Coverage | Size | URL |
|---|---|---|---|
| HYDE 3.5 (sub-national, 5-arcmin) | 10,000 BCE – 2030 | ~5 GB | https://landuse.sites.uu.nl/hyde-project/ |
| ModE-RA paleo-reanalysis | 1421 – 2008 monthly | ~90 GB | https://www.wdc-climate.de/ui/entry?acronym=ModE-RA |
| CRU TS 4.09 | 1901 – 2024 monthly | ~5 GB | https://crudata.uea.ac.uk/cru/data/hrg/ |
| Berkeley Earth (optional) | 1850 – present | ~2 GB | http://berkeleyearth.org/data/ |
| PAGES 2k Consortium reconstructions | 0 – 2000 CE | <100 MB | https://www.ncei.noaa.gov/access/paleo-search/study/26872 |

See **`data/README.md`** for one-line download commands and checksums.

## Reproducibility notes

- Random seeds are pinned wherever K-means clustering or bootstrap resampling is used (`random_state=42` and `np.random.default_rng(42)`).
- The Python environment is described by `pyproject.toml` and `requirements.txt`. The pipeline has been verified on macOS 14 (arm64) with Python 3.13 and on Linux Ubuntu 22.04 with Python 3.11.
- Total wall time for a fresh build (after downloads): ~3 hours on an M-series Mac. The longest single step is the sub-national ModE-RA aggregation.
- Memory: peak ~5 GB during the sub-national aggregation.

## License

Code and documentation: MIT (see `LICENSE`).
Generated data files in `analysis/data/`: CC-BY-4.0 (re-share with attribution).

Raw upstream data are governed by their own licenses (HYDE: CC-BY-4.0; ModE-RA: CC-BY-4.0; CRU TS: Open Government Licence).

## How to cite

If you use this package, please cite both the paper and the dataset DOI:

```bibtex
@unpublished{alonsoortiz2026shadow,
  title  = {The Long Shadow of Seasonality: Climate Endowments, Agricultural Pathways, and Escape from the Malthusian Trap},
  author = {Alonso Ortiz, Jorge},
  year   = {2026},
  note   = {Working paper. Replication package: \url{https://osf.io/XXXX}}
}
```

## Contact

Issues, errors, or suggestions: open a GitHub issue or email `jorge.alonso@icloud.com`.
