# Long Shadow on Fertility — Phase 1 (England pilot)

Subpackage of `paper4_shadow/` that extends the Long Shadow framework with
a **fertility outcome** (log CBR) and the **regime-change** econometric
framing (rolling-window elasticity, smooth-transition regression with log
GDP/cap as transition variable, volcanic event studies).

Phase 1 scope: **England only**, 1541–2020. Phase 2 (separate plan) adds
France, Sweden, Italy, Princeton EFP.

Entry point: `python -m analysis.paper4_shadow.long_shadow_fertility.run_phase1_england`

Spec + plan live in the sibling Fertility repository under
`Fertility/docs/superpowers/`.

## Phase 1 status: complete (2026-05-22)

**Deliverables:**
- Data outputs in `/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/`:
  - `england_fertility_annual_1541_2020.parquet` (CamPOP 1541-1837 + HFD 1938-2022, 1838-1937 gap)
  - `england_climate_annual_1421_2008.parquet` (ModE-RA cropland-weighted growing-season + winter T/P)
  - `maddison_england_annual_1500_2022.parquet` (Maddison GBR GDP per capita)
  - `england_panel_1541_2020.parquet` (unified panel with eruption flags)
- Figures in `/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/`:
  - `fig1_rolling_elasticity_england.{pdf,png}`
  - `fig2_smooth_transition_england.{pdf,png}` (two-panel: STR + raw within-era OLS)
  - `fig3_volcanic_eventstudy_england.{pdf,png}` (Huaynaputina / Tambora / Pinatubo)
- Phase 1 memo: sibling Fertility repo at `docs/long_shadow_fertility_phase1_memo.md`

**Test suite:** 36 passing (`python -m pytest analysis/paper4_shadow/long_shadow_fertility/tests/`).

**Key findings:** within-era OLS β_M=+0.098 (Malthus 1700-1837), β_T=-0.042 (Modern 1938-2008). Smooth-transition identification is challenged by the 1838-1937 industrial gap — addressed by Phase 2.

**Phase 2 (multi-country):** follow-up plan to be written in Fertility repo at `docs/superpowers/plans/YYYY-MM-DD-long-shadow-fertility-phase2-multicountry.md` after user review.
