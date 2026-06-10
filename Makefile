# Makefile for the Long Shadow of Seasonality replication package.
# Usage:
#   make all         — full pipeline (~7 hours after downloads)
#   make panels      — country + sub-national panels only
#   make kk10        — build the KK10 country panel (requires KK10.nc, ~17GB)
#   make analysis    — analyses + figures (assumes panels exist)
#   make robustness  — Boserup robustness, area orthogonality, KK10 cross-validation
#   make paper       — recompile the manuscript and cover letters
#   make clean       — remove generated parquets and figures
#   make distclean   — also remove the compiled PDF

PYTHON ?= python3
ROOT   := $(CURDIR)
DATA   := analysis/data
FIG    := analysis/figures/paper4_v2

# Default to the bigdata mount; override with HYDE35_ROOT / MODERA_ROOT.
export HYDE35_ROOT ?= /Volumes/BIGDATA/HYDE35
export MODERA_ROOT ?= /Volumes/BIGDATA/MODERA

.PHONY: all panels kk10 analysis robustness paper clean distclean check download-data

all: panels kk10 analysis robustness paper

# ── Panel-build pipeline ────────────────────────────────────────────────────
panels: \
	$(DATA)/modera_country_monthly.parquet \
	$(DATA)/cru_country_climatology_1901_1950.parquet \
	$(DATA)/country_seasonality_1421_2008.parquet \
	$(DATA)/country_climate_1421_2025.parquet \
	$(DATA)/modera_country_uncertainty.parquet \
	$(DATA)/modera_subnational_monthly.parquet \
	$(DATA)/modera_country_monthly_popw.parquet

$(DATA)/modera_country_monthly.parquet:
	$(PYTHON) -m analysis.shared.build_modera_panel

$(DATA)/cru_country_climatology_1901_1950.parquet:
	$(PYTHON) -m analysis.shared.build_cru_climatology

$(DATA)/country_seasonality_1421_2008.parquet: \
		$(DATA)/modera_country_monthly.parquet \
		$(DATA)/cru_country_climatology_1901_1950.parquet
	$(PYTHON) -m analysis.shared.build_seasonality

# country_climate_1421_2025.parquet is ModE-RA + CRU only (1421-2008); ERA5 removed.
$(DATA)/country_climate_1421_2025.parquet: \
		$(DATA)/country_seasonality_1421_2008.parquet
	$(PYTHON) -m analysis.shared.build_calibrated_annual

$(DATA)/modera_country_uncertainty.parquet:
	$(PYTHON) -m analysis.shared.build_modera_uncertainty

$(DATA)/modera_subnational_monthly.parquet:
	$(PYTHON) -m analysis.shared.build_subnational

$(DATA)/modera_country_monthly_popw.parquet: \
		$(DATA)/modera_country_monthly.parquet
	$(PYTHON) -m analysis.shared.build_popweighted_panels

# ── Analyses (figures + parquet outputs) ────────────────────────────────────
analysis: panels
	$(PYTHON) -m analysis.paper4_shadow.stage1_country
	$(PYTHON) -m analysis.paper4_shadow.stage1_storage_index
	$(PYTHON) -m analysis.paper4_shadow.stage1_subnational
	$(PYTHON) -m analysis.paper4_shadow.preindustrial_malthus
	$(PYTHON) -m analysis.paper4_shadow.preindustrial_malthus_extended
	$(PYTHON) -m analysis.paper4_shadow.subnational_malthus
	$(PYTHON) -m analysis.paper4_shadow.volcanic_event_study
	$(PYTHON) -m analysis.paper4_shadow.tambora_individual
	$(PYTHON) -m analysis.paper4_shadow.volcanic_placebo
	$(PYTHON) -m analysis.paper4_shadow.long_shadow_rerun
	$(PYTHON) -m analysis.paper4_shadow.subnational_long_shadow
	$(PYTHON) -m analysis.paper4_shadow.robustness_v2
	$(PYTHON) -m analysis.paper4_shadow.documentary_validation
	$(PYTHON) -m analysis.paper4_shadow.pathway_validation
	$(PYTHON) -m analysis.paper4_shadow.manski_bounds
	$(PYTHON) -m analysis.paper4_shadow.spatial_se_robustness
	$(PYTHON) -m analysis.paper4_shadow.sigl_volcanic_forcing
	$(PYTHON) -m analysis.paper4_shadow.allen_wage_malthus
	$(PYTHON) -m analysis.paper4_shadow.allen_wage_malthus_citybuffer
	$(PYTHON) -m analysis.paper4_shadow.latitude_controls
	$(PYTHON) -m analysis.paper4_shadow.gaez_style_suitability
	$(PYTHON) -m analysis.paper4_shadow.popweight_comparison
	$(PYTHON) -m analysis.paper4_shadow.popweight_subnational
	$(PYTHON) -m analysis.paper4_shadow.deep_determinants
	$(PYTHON) -m analysis.paper4_shadow.deep_determinants_extended
	$(PYTHON) -m analysis.shared.build_cropland_weighted_panels
	$(PYTHON) -m analysis.paper4_shadow.cropweight_comparison
	$(PYTHON) -m analysis.paper4_shadow.joint_landuse_var
	$(PYTHON) -m analysis.paper4_shadow.joint_var_bootstrap
	$(PYTHON) -m analysis.paper4_shadow.joint_var_post1700
	$(PYTHON) -m analysis.paper4_shadow.joint_var_climate_pathways
	$(PYTHON) -m analysis.paper4_shadow.structural_3eq
	$(PYTHON) -m analysis.paper4_shadow.structural_calibration
	$(PYTHON) -m analysis.paper4_shadow.structural_simulate
	$(PYTHON) -m analysis.paper4_shadow.structural_counterfactuals
	$(PYTHON) -m analysis.paper4_shadow.welfare_counterfactuals
	$(PYTHON) -m analysis.paper4_shadow.sigl_event_study_dynamic
	$(PYTHON) -m analysis.paper4_shadow.long_shadow_extra_outcomes
	$(PYTHON) -m analysis.paper4_shadow.placebo_period_and_rolling
	$(PYTHON)   analysis/paper4_shadow/make_figures.py
	$(PYTHON)   analysis/paper4_shadow/make_fig1_map.py

# ── KK10 country aggregation (population-independent cross-validation) ──────
# Requires KK10.nc at the path inside the script (17.3 GB NetCDF, PANGAEA
# doi:10.1594/PANGAEA.871369; override KK10_NC env if not at the default).
kk10: $(DATA)/kk10_country_panel.parquet

$(DATA)/kk10_country_panel.parquet:
	$(PYTHON) -m analysis.paper4_shadow.build_kk10_country_panel

# ── Joint reduced-form panel system (§3 headline) ───────────────────────────
# Builds joint_landuse_var_panel.parquet + joint_landuse_var_results.parquet +
# figJ. Also run inside `analysis`; this rule lets `make robustness` build it
# standalone from a clean tree.
$(DATA)/joint_landuse_var_panel.parquet: panels
	$(PYTHON) -m analysis.paper4_shadow.joint_landuse_var

# ── Robustness exercises (Boserup honest null + KK10 cross-validation) ──────
robustness: kk10 $(DATA)/joint_landuse_var_panel.parquet
	$(PYTHON) -m analysis.paper4_shadow.boserup_robustness
	$(PYTHON) -m analysis.paper4_shadow.boserup_cropland_area
	$(PYTHON) -m analysis.paper4_shadow.area_margin_diagnostic
	$(PYTHON) -m analysis.paper4_shadow.kk10_orthogonality
	$(PYTHON) -m analysis.paper4_shadow.kk10_pathway_heterogeneity
	$(PYTHON) -m analysis.paper4_shadow.channel_substitution
	$(PYTHON) -m analysis.paper4_shadow.country_substitution

# ── Paper compilation ───────────────────────────────────────────────────────
paper: paper/long_shadow.pdf \
       paper/cover_letter_jeg.pdf paper/cover_letter_qje.pdf \
       paper/cover_letter_restud.pdf paper/cover_letter_aejmacro.pdf

paper/long_shadow.pdf: paper/long_shadow.tex
	cd paper && pdflatex -interaction=nonstopmode long_shadow.tex
	cd paper && pdflatex -interaction=nonstopmode long_shadow.tex

paper/cover_letter_jeg.pdf: paper/cover_letter_jeg.tex
	cd paper && pdflatex -interaction=nonstopmode cover_letter_jeg.tex

paper/cover_letter_qje.pdf: paper/cover_letter_qje.tex
	cd paper && pdflatex -interaction=nonstopmode cover_letter_qje.tex

paper/cover_letter_restud.pdf: paper/cover_letter_restud.tex
	cd paper && pdflatex -interaction=nonstopmode cover_letter_restud.tex

paper/cover_letter_aejmacro.pdf: paper/cover_letter_aejmacro.tex
	cd paper && pdflatex -interaction=nonstopmode cover_letter_aejmacro.tex

# ── Diagnostics ─────────────────────────────────────────────────────────────
check:
	$(PYTHON) -m analysis.paper4_shadow.robustness_v2 | tail -10
	@echo
	@echo "Expected: median |bias| 0.61°C, monthly t-corr 0.971, annual anomaly corr 0.938"

# ── Cleaning ────────────────────────────────────────────────────────────────
clean:
	rm -f $(DATA)/*.parquet
	rm -f $(FIG)/*.pdf $(FIG)/*.png
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.bbl paper/*.blg

distclean: clean
	rm -f paper/long_shadow.pdf paper/cover_letter_*.pdf

# ── Raw-data download (placeholder; see REPRODUCE.md) ───────────────────────
download-data:
	@echo "See REPRODUCE.md section 1 for one-line download commands."
	@echo "Total ~95 GB across HYDE 3.5, ModE-RA, CRU TS (ERA5 no longer required)."
