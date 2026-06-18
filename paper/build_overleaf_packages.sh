#!/usr/bin/env bash
#
# build_overleaf_packages.sh
#
# Builds two self-contained Overleaf upload packages:
#   overleaf/long_shadow.zip  -> long_shadow.tex  + figures/ (8 PDFs)
#   overleaf/horserace.zip    -> horserace.tex    + figures/ (7 PDFs + 12 table .tex)
#
# Each package is flattened: the .tex sits at the zip root and every referenced
# asset is copied into a figures/ subfolder. The relative ../analysis/... paths
# in the source are rewritten to figures/... in the *staged copy only* -- the
# repo source .tex files are never modified.
#
# Bibliographies are embedded (\begin{thebibliography}), so no .bib/.bbl or
# BibTeX pass is required. All packages used are standard CTAN.
#
# Usage:  bash build_overleaf_packages.sh
# Output: paper/overleaf/{long_shadow,horserace}.zip
#
set -euo pipefail

# Resolve repo paths relative to this script (paper/ dir).
PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PAPER_DIR/.." && pwd)"
FIG4="$REPO_ROOT/analysis/figures/paper4_v2"
FIG5="$REPO_ROOT/analysis/figures/paper5_horserace"

OUT_DIR="$PAPER_DIR/overleaf"
BUILD_DIR="$PAPER_DIR/overleaf/.build"

# Test-compile each staged package if a LaTeX engine is available.
LATEX_OK=0
if command -v latexmk >/dev/null 2>&1; then LATEX_OK=1; fi

rm -rf "$BUILD_DIR"
mkdir -p "$OUT_DIR" "$BUILD_DIR"

# ---- helpers ---------------------------------------------------------------

# copy_assets <dest_figures_dir> <src_dir> <file...>
copy_assets() {
  local dest="$1"; shift
  local src="$1"; shift
  local f
  for f in "$@"; do
    if [[ ! -f "$src/$f" ]]; then
      echo "ERROR: missing asset $src/$f" >&2
      exit 1
    fi
    cp "$src/$f" "$dest/$f"
  done
}

# test_compile <stage_dir> <texname>
test_compile() {
  local stage="$1" tex="$2"
  if [[ "$LATEX_OK" -ne 1 ]]; then
    echo "  (latexmk not found -- skipping test compile; verify on Overleaf)"
    return 0
  fi
  echo "  test-compiling $tex ..."
  ( cd "$stage" && latexmk -pdf -interaction=nonstopmode -halt-on-error \
      "$tex" >/dev/null 2>compile.log ) || {
        echo "ERROR: test compile failed for $tex. Tail of log:" >&2
        tail -30 "$stage/compile.log" >&2
        exit 1
      }
  echo "  OK: $tex compiled (see $stage/${tex%.tex}.pdf)"
  # Drop build artifacts so they do not end up in the zip.
  ( cd "$stage" && latexmk -c "$tex" >/dev/null 2>&1 || true )
  rm -f "$stage"/*.pdf "$stage"/compile.log
}

# make_zip <stage_dir> <zipname>
make_zip() {
  local stage="$1" zipname="$2"
  rm -f "$OUT_DIR/$zipname"
  ( cd "$stage" && zip -r -q "$OUT_DIR/$zipname" . \
      -x '*.aux' '*.log' '*.out' '*.bbl' '*.blg' '*.fls' '*.fdb_latexmk' '*.synctex.gz' )
  echo "  wrote $OUT_DIR/$zipname"
}

# ---- long_shadow -----------------------------------------------------------

echo "[1/2] long_shadow"
LS_STAGE="$BUILD_DIR/long_shadow"
mkdir -p "$LS_STAGE/figures"

LS_FIGS=(
  fig03_storage_pathway.pdf
  fig04_subnational.pdf
  fig09_ensemble_uncertainty.pdf
  figA2_volcanic_price_event.pdf
  figJ_joint_landuse.pdf
  figK_channel_substitution.pdf
  figK_country_substitution.pdf
  figXY_long_shadow_choropleth.pdf
)
copy_assets "$LS_STAGE/figures" "$FIG4" "${LS_FIGS[@]}"

# Rewrite ../analysis/figures/paper4_v2/ -> figures/ in the staged copy only.
sed 's#\.\./analysis/figures/paper4_v2/#figures/#g' \
  "$PAPER_DIR/long_shadow.tex" > "$LS_STAGE/long_shadow.tex"

# Guard: no stale ../ paths should remain.
if grep -q '\.\./analysis' "$LS_STAGE/long_shadow.tex"; then
  echo "ERROR: unrewritten ../analysis path remains in long_shadow.tex" >&2
  grep -n '\.\./analysis' "$LS_STAGE/long_shadow.tex" >&2
  exit 1
fi

test_compile "$LS_STAGE" "long_shadow.tex"
make_zip "$LS_STAGE" "long_shadow.zip"

# ---- horserace -------------------------------------------------------------

echo "[2/2] horserace"
HR_STAGE="$BUILD_DIR/horserace"
mkdir -p "$HR_STAGE/figures"

HR_FIGS=(
  fig01_substrate_covariance_bw.pdf
  fig02_substrate_maps_bw.pdf
  fig03_shapley_heatmap_bw.pdf
  fig04_mediation_bw.pdf
  fig05_pathway_source_robustness_bw.pdf
  fig06_subsample_stability_bw.pdf
  figA_robustness_battery_bw.pdf
)
HR_TABS=(
  tab01_descriptives.tex
  tab02_substrate_correlations.tex
  tab02b_climate_subcorrelations.tex
  tab03_full_ols.tex
  tab04_shapley_table.tex
  tab05_mediation_table.tex
  tab06_climate_subshapley.tex
  tab07_functional_subshapley.tex
  tab_gs_subshapley.tex
  tab_h_pred_evolution.tex
  tab_h_pred_evolution_pca.tex
  tab_h_pred_commonality.tex
)
copy_assets "$HR_STAGE/figures" "$FIG5" "${HR_FIGS[@]}"
copy_assets "$HR_STAGE/figures" "$FIG5" "${HR_TABS[@]}"

# Rewrite ../../analysis/figures/paper5_horserace/ -> figures/ in staged copy.
sed 's#\.\./\.\./analysis/figures/paper5_horserace/#figures/#g' \
  "$PAPER_DIR/horserace/horserace.tex" > "$HR_STAGE/horserace.tex"

if grep -q '\.\./\.\./analysis' "$HR_STAGE/horserace.tex"; then
  echo "ERROR: unrewritten ../../analysis path remains in horserace.tex" >&2
  grep -n '\.\./\.\./analysis' "$HR_STAGE/horserace.tex" >&2
  exit 1
fi

test_compile "$HR_STAGE" "horserace.tex"
make_zip "$HR_STAGE" "horserace.zip"

# ---- done ------------------------------------------------------------------

rm -rf "$BUILD_DIR"
echo ""
echo "Done. Packages in $OUT_DIR:"
ls -lh "$OUT_DIR"/*.zip
