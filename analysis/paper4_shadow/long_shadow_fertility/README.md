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

## Phase 2 status: complete (2026-05-22)

**Deliverables (29 commits ahead of Phase 1 baseline):**

**Multi-country data outputs:**
- `france_fertility_annual.parquet` (HMD FRATNP 1806-2020)
- `italy_fertility_annual.parquet` (HMD ITA 1862-2019)
- `sweden_fertility_annual.parquet` (HMD SWE 1749-2022 — Tabellverket era)
- `france_dept_fertility_annual.parquet` (Cassini 1851-1897 + ModE-RA spine 1421-2008)
- `country_climate_annual.parquet` (4-country ModE-RA cropland-weighted)
- `maddison_multicountry_annual.parquet` (GBR/FRA/ITA/SWE GDPpc)
- `panel_multi_country_year.parquet` (unified Phase 2 panel)

**Controls panels:**
- `war_panel_country_year.parquet` (Brecke conflict catalogs)
- `pandemic_panel_country_year.parquet` (pre-1500 plague + Black Death + modern epidemics)
- `emdat_panel_country_year.parquet` (EMDAT 1962+ — FRA absent from source)
- `climate_extremes_country_year.parquet` (heat/drought via pre-1900 quantiles)
- `controls_panel_country_year.parquet` (unified)

**Princeton EFP:** BLOCKED (OPR 403); stub builder + figure ready for manual data fetch.

**Phase 2 figures:**
- `fig1_rolling_multi_country.{pdf,png}` — 4-panel rolling-window
- `fig2_pooled_smooth_transition.{pdf,png}` — HEADLINE pooled STR with controls
- `fig3_stacked_volcanic.{pdf,png}` — 1815/1883/1991 stacked
- `fig4_country_decade_heatmap.{pdf,png}` — β heatmap
- `fig6_france_subnational.{pdf,png}` — France dept rolling robustness

**Phase 1 retrospective figures (with controls):**
- `fig1r_rolling_with_controls_england.{pdf,png}`
- `fig2r_smooth_transition_with_controls_england.{pdf,png}`
- `fig3r_volcanic_with_controls_england.{pdf,png}`

**Phase 2 memo:** sibling Fertility repo at `docs/long_shadow_fertility_phase2_memo.md`

**Key findings (honest):**
- Pooled STR identification failure persists across 4 countries with controls: θ pegs at 5.0, β_M = -0.459 (SE 0.342, n.s.), β_T = -0.026; both regimes negative.
- Phase 1 Malthusian-positive rolling result shrinks 7× when controls are partialled out (1700 β: +0.035 → +0.005).
- Volcanic event-study positive signs survive controls in all three regimes (Pinatubo h=0 triples to +0.094).
- Rolling-window sign-flip pattern visible across all 4 countries (Italy earliest demographic transition).

**Test suite:** ~66 passing (Phase 1 36 + Phase 2 ~30).

**Phase 3 directions:** alternative non-parametric STR (kernel/spline); fill 1838-1937 England gap (Mitchell historical); aggregate FRA dept-level EMDAT for FRA control; unblock EFP via manual download.

## Phase 3 status: complete (2026-05-22)

**Four methodological additions:**
- Distributed-lag climate (k=0..3) replacing contemporaneous T
- Climate volatility promoted from control to primary regressor
- Joint fertility + mortality via SUR (with cross-equation Wald)
- Allen real-wage Z + T → wage → fertility mediation

**New builders:**
- `build_country_mortality_annual.py` — HMD Deaths_1x1 → log_cdr per (iso3, year)
- `build_real_wage_panel.py` — Allen (1421-1913) + Maddison GDPpc proxy (1914+)

**New estimators:**
- `distributed_lag.py` — per-horizon β + cumulative B + HAC SE
- `bivariate_sur.py` — two-step FGLS + cross-equation Wald
- `mediation.py` — direct/indirect/total + bootstrap SEs

**New figures (Fig 7-12):**
- Fig 7 — distributed-lag IRF per country
- Fig 8 — volatility as treatment
- Fig 9 — joint fertility+mortality SUR
- Fig 10 — pooled STR with real-wage Z
- Fig 11 — mediation diagram
- Fig 12 — distributed-lag by STR regime

**Phase 3 memo:** sibling Fertility repo at `docs/long_shadow_fertility_phase3_memo.md`

**Key findings (honest):**
- Distributed lags REVEAL clean negative climate-fertility response across all 4 countries (cum β -0.24 to -0.59, all significant) — Phase 2's mixed-sign contemporaneous-only results were misleading
- Joint fertility+mortality SUR: both negative in T (cold-mortality channel dominates in temperate Europe)
- Mediation: 83.7% of climate-fertility effect runs through real wages (caveat: Allen/Maddison splice creates unit discontinuity)
- Fig 10 STR with wage-Z shows apparent regime flip but is largely an artifact of the pre/post-1914 splice
- Phase 2 STR threshold c=7.17 turned out to be at the edge of Maddison's England GDPpc support (Malthus regime collapses to 1 obs)

**Test suite:** Phase 1 (36) + Phase 2 (60) + Phase 3 (~15) ≈ 111 passing, 2 skipped (EFP).

## Phase 4 status: complete (2026-05-22)

**Two-tier methodological refinement** (Tier 1 = pooled rigor, Tier 2 = IV identification):

**New builders:**
- `build_sigl_volcanic_panel.py` — Sigl-Toohey 2024 eVolv2k VSSI (500 BCE - 1900 CE)
- `build_teleconnection_panel.py` — NAO/AMO/ENSO from NOAA (BLOCKED on fetch; stub committed)

**New estimators:**
- `pooled_distributed_lag.py` — country FE + year FE + cluster SE
- `pooled_volatility_dl.py` — joint level + volatility DL
- `cluster_bootstrap.py` — block-bootstrap utility
- `iv_2sls.py` — 2SLS with first-stage F + Anderson-Rubin diagnostics

**New figures:**
- `fig7v2_pooled_dl_irf` — supersedes Phase 3 Fig 7
- `fig8v2_pooled_vol_dl` — supersedes Phase 3 Fig 8
- `fig11v2_mediation_cluster_bootstrap` — supersedes Phase 3 Fig 11
- `fig13_sigl_volcanic_iv` — new 2SLS figure
- `fig14_teleconnection_iv` — stub (teleconnection BLOCKED)

**Phase 4 memo:** sibling Fertility repo at `docs/long_shadow_fertility_phase4_memo.md`

**Key findings:**
- Phase 3 distributed-lag cum β -0.41 (significant) shrinks to **-0.016 (NOT significant)** in pooled spec with FE + 9 controls + cluster SE. Phase 3 finding was largely OVB + iid SE understatement.
- Phase 3 volatility-as-treatment sign-flip (FRA/ITA β^V positive) DISAPPEARS with controls.
- Mediation through wages SURVIVES cluster bootstrap (indirect SE inflates 2.5× but still 4.4σ).
- Sigl volcanic-IV: 2SLS β = -1.10 (vs OLS -0.016 → 70× LATE-vs-ATE wedge), weak F=7.4. AR p-value 1e-60.
- Teleconnection IV BLOCKED (NOAA 404); manual download needed.

**Test suite:** Phase 1 (36) + Phase 2 (60) + Phase 3 (~21) + Phase 4 (~13) ≈ ~130 passing, 3 skipped.

## Phase 5 status: complete (2026-05-22)

Three pillars:
- **Pillar 1 (gap fix):** HFD → HMD swap for England. Gap shrinks 100yr → 3yr (1838-1840 only). GBR log_cbr observations 376 → 473.
- **Pillar 2 (statistical refinements):** harmonized Allen↔Maddison wages (eliminates 1914 unit discontinuity); 3-equation SUR (fertility+mortality+wages jointly); wild cluster bootstrap (Cameron-Gelbach-Miller).
- **Pillar 3 (expansion):** 7-country panel — added BEL/NLD/ESP. Skipped DEU (reunification).

**New builders:** `build_england_fertility_annual.py` (v2 HMD), `build_bel_fertility_annual.py`, `build_nld_fertility_annual.py`, `build_esp_fertility_annual.py`, `build_real_wage_panel_v2.py`. Mortality/climate/Maddison panels extended in-place.

**New estimators:** `triple_sur.py`, `wild_cluster_bootstrap.py`.

**New figures:**
- Fig 1v3 — 7-panel rolling-window
- Fig 7v3 — pooled DL on 7-country gap-filled panel
- Fig 10v2 — STR with harmonized wage Z
- Fig 11v3 — mediation with harmonized wages + wild cluster bootstrap
- Fig 15 — 3-equation SUR
- Fig 16 — cross-phase progression chart

**Phase 5 memo:** sibling Fertility repo at `docs/long_shadow_fertility_phase5_memo.md`

**Key findings:** Phase 4 reversal CONFIRMED on better panel (cum β = +0.009, n.s.). **Mediated share rises to 100% with harmonized wages** — direct climate-fertility effect ≈ 0; entire transmission via wage channel. Triple SUR: T raises wages (+0.50), lowers fertility (-0.20), lowers mortality (-0.16) — all distinct.

**Test suite:** ~155 passing (Phase 1-4: 135 + Phase 5: ~20).

## Phase 6 status: complete (2026-05-22, autonomous)

User-delegated execution while away. Two non-code deliverables plus one test fix:

1. **Stale Phase 2 test fix** — `test_four_countries` → `test_seven_countries` (BIGDATA branch `long-shadow-fertility-phase6`).
2. **Paper draft v0.1** — `Fertility/docs/paper_long_shadow_fertility.{tex,pdf}`. 14 pages, 7 headline figures, structured for workshop submission.
3. **Beamer slides v0.1** — `Fertility/docs/slides_long_shadow_fertility.{tex,pdf}`. 18 slides for 30-minute seminar (2 backup slides).

**Phase 6 memo:** `Fertility/docs/long_shadow_fertility_phase6_memo.md`

**Test suite:** 160 passing, 5 skipped, 0 failed.
