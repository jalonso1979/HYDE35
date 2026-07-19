#!/usr/bin/env bash
# reproduce_long_shadow.sh — end-to-end reproduction of the Long Shadow (P5) pipeline.
#
#   ./reproduce_long_shadow.sh                 # core pipeline (~1.5 hr), NO bootstrap
#   ./reproduce_long_shadow.sh --with-bootstrap# core + ~60-min regime-FEVD bootstrap
#   ./reproduce_long_shadow.sh --bootstrap-only# only the ~60-min bootstrap
#
# Run from anywhere; the script cd's to the BIGDATA root. Phases generate their
# figures internally and copy them to the paper figures/ folder. Module names
# verified 2026-05-28 (phase13). NOTE: phase11_precip_dl_robustness.json has no
# generator module in-repo (orphan robustness artifact) and is intentionally NOT
# regenerated here.
set -euo pipefail
ROOT=/Volumes/BIGDATA/HYDE35
PKG=analysis.paper4_shadow.long_shadow_fertility
LSDIR="${ROOT}/analysis/paper4_shadow/long_shadow_fertility"
OUTDIR="${ROOT}/analysis/output/long_shadow_fertility"
mkdir -p "$OUTDIR"
LOG="${OUTDIR}/reproduce_$(date +%Y%m%d_%H%M%S).log"
cd "$ROOT"

run(){ echo "=== [$(date +%H:%M:%S)] $1 ===" | tee -a "$LOG"; python -m "$2" 2>&1 | tee -a "$LOG"; }
script(){ echo "=== [$(date +%H:%M:%S)] $1 ===" | tee -a "$LOG"; python "$2" 2>&1 | tee -a "$LOG"; }

WITH_BOOT=0; BOOT_ONLY=0
for a in "$@"; do case "$a" in
  --with-bootstrap) WITH_BOOT=1;;
  --bootstrap-only) BOOT_ONLY=1;;
  *) echo "unknown arg: $a"; exit 2;;
esac; done

if [ "$BOOT_ONLY" -eq 0 ]; then
  # ---- Core pipeline (builds panels, runs estimators, emits figures) ----
  run "Phase 1 England"             ${PKG}.run_phase1_england
  run "Phase 2 multi-country"       ${PKG}.run_phase2_multi_country
  run "Phase 3 lags+mortality+wage" ${PKG}.run_phase3_lags_mortality_wages
  run "Phase 4 pooled IV"           ${PKG}.run_phase4_pooled_iv
  run "Phase 5 gapfix+expansion+SUR" ${PKG}.run_phase5_gapfix_expansion_sur
  # ---- Phase 10: thresholds + diagnostics ----
  run "Phase 10 threshold grid"     ${PKG}.scripts.run_phase10_threshold_grid
  run "Phase 10 country thresholds" ${PKG}.scripts.run_phase10_country_thresholds
  run "Phase 10 boundary diagnostic" ${PKG}.scripts.run_phase10_country_boundary_diagnostic
  run "Phase 10 uncertainty DL"     ${PKG}.scripts.run_phase10_uncertainty_dl
  run "Phase 10 uncertainty DL rolling" ${PKG}.scripts.run_phase10_uncertainty_dl_rolling
  # ---- Phase 10.5: SPEI + precip climate robustness ----
  run "Phase 10.5 Hansen SPEI"      ${PKG}.scripts.run_phase10p5_hansen_spei
  run "Phase 10.5 DL SPEI"          ${PKG}.scripts.run_phase10p5_dl_spei
  run "Phase 10.5 Hansen precip"    ${PKG}.scripts.run_phase10p5_hansen_precip
  run "Phase 10.5 DL precip"        ${PKG}.scripts.run_phase10p5_dl_precip
  # ---- Phase 11 headline + Phase 12 robustness ----
  run "Phase 11 regime FEVD (HEADLINE)" ${PKG}.scripts.run_phase11_regime_fevd
  run "Phase 12 drop FIN/ISL"       ${PKG}.scripts.run_phase12_drop_finisl_robustness
  # ---- Tables (read the JSON written above) ----
  script "Table 6 threshold grid"   "${LSDIR}/tables/tab6_threshold_grid.py"
  script "Table 6v2 shock decomp"   "${LSDIR}/tables/tab6v2_shock_decomposition.py"
  script "Table 7 uncertainty DL"   "${LSDIR}/scripts/tab7_uncertainty_dl.py"
fi

if [ "$WITH_BOOT" -eq 1 ] || [ "$BOOT_ONLY" -eq 1 ]; then
  run "Phase 11 regime-FEVD bootstrap (~60 min, N=500)" ${PKG}.scripts.run_phase11_regime_fevd_bootstrap
fi

echo "=== [$(date +%H:%M:%S)] DONE. Log: $LOG ===" | tee -a "$LOG"
