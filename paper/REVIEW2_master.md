# REVIEW2 — Pre-Submission Audit Consolidation

**Papers:** `long_shadow.tex` (Volcanic Forcing and the Pre-Industrial Malthusian Trap), `horserace.tex` (deep-roots horserace)
**Date:** 2026-06-18
**Companion machine-applyable patch list:** `REVIEW2_findings.json`

---

## Executive summary

This audit consolidates a multi-agent, post-verification pass over both manuscripts. Refuted items have been removed; what remains has been checked against the committed parquets/scripts. Counts below are of *distinct* findings after light deduplication.

| Paper | Blocker | Major | Minor | Presentation |
|---|---|---|---|---|
| long_shadow | 4 | 6 | 19 | ~16 |
| horserace | 1 | 8 | 13 | ~10 |
| both (cross-paper) | 0 | 2 | 2 | ~12 |

`REVIEW2_findings.json` contains 65 records with exact `old_text` → `new_text` replacements; the remaining vague/structural/code-only notes are described here in prose only.

### The headline blocker: the long_shadow post-1700 pastoral over-claim

The single most consequential error is repeated in **four places** (abstract l.29, intro l.48, body l.247, conclusion l.422): the manuscript states that **both** the crop-dominant late and the pastoral/mixed late demographic coefficients survive restriction to the post-1700 decadal portion of HYDE. They do not. Re-running `joint_var_post1700_results.parquet` (2026-06-18):

- **Crop-dominant late** survives every post-1700 window — full $p=9.7\times10^{-7}$; 1700–1900 $p=0.0125$; 1750–1900 $p=0.0058$; 1800–1900 $p=0.018$.
- **Pastoral/mixed late** *collapses* — full $p=1.7\times10^{-6}$ but post-1700 $p=0.95$ (1700), $0.74$ (1750), $0.49$ (1800). Its full-panel significance comes **entirely from the pre-1700 century-resolution cells.**

This is correctly stated in the paper's own Appendix L.3 (`p = 0.95` post-1700) and Table `tab:joint-var-post1700`, so the abstract/intro/body/conclusion must be brought into line with the appendix — not the other way round. The Bonferroni, Webb wild-cluster bootstrap, and orthogonal-re-clustering robustness claims **do** hold for both pathways; only the post-1700 clause fails, and only for the pastoral/mixed pathway. The recommended edits keep both coefficients' full-panel significance and surgically restrict the post-1700 survival claim to crop-dominant late.

### The recurring horserace blocker: "five" vs "eight" FWER survivors

`horserace.tex` line 302 (and the abstract/intro) correctly state that **eight** cells survive the Westfall–Young family-wise correction (the df-fair minP correction the paper adopts as primary). But the robustness section (lines 343, 534, 543, 548) and the discussion (line 557) still say "five" (the df-biased maxt special case) and in one place wrongly assign ancestral-crop-yield a surviving density-2025 cell (it is the lone near-miss, $p_{\\text{WY}}=0.054$). `robustness_battery.parquet` confirms eight survivors: functional alleles ×{gdppc 2015, dt-timing, pop-growth, density 2025}; ancestral yield ×{density 1500, pop-growth}; Neolithic fraction ×{density 2025, dt-timing}.

### The other long_shadow blockers

1. **Modern prevpos table (l.799–801):** the "1900–2022 full" row is stale (paper: +0.0001, p=0.840, N=2,117; parquet: +0.0074, p=0.049, N=2,023), the "1950–2008" label should be "1950–2022" (the script window), and the caption's "collapses monotonically" claim is false — the pooled full-period coefficient is *larger and more significant* than the modern sub-period coefficient.

---

## long_shadow.tex

### Blockers

**[blocker] l.247 — post-1700 over-claim (body).** Paper: "both ... surviving restriction to the post-1700 decadal portion of HYDE." Correct: only crop-dominant late survives ($p\\le0.018$); pastoral/mixed late collapses ($p=0.95$). Source: `joint_var_post1700_results.parquet`. See JSON for the surgical rewrite.

**[blocker] l.48 — post-1700 over-claim (intro First finding).** Same fix; restrict the post-1700 clause to crop-dominant late.

**[blocker] l.29 — post-1700 over-claim (abstract).** "Bonferroni-significant and robust across every stress test we run" must split: crop-dominant late robust across every stress test; pastoral/mixed late identified off pre-1700 cells, does not survive the post-1700 restriction. (Also attaches the missing exponent to the first coefficient.)

**[blocker] l.422 — post-1700 over-claim (conclusion).** Same fix.

**[blocker] l.801 — `1900--2022 full` table row stale.** Paper: +0.0001, (0.0007), 0.840, 2,117. Parquet (`prevpos_modern.parquet`, m0/t_anom): +0.0074, (0.0038), 0.049, 2,023.

**[blocker] l.800 — `1950--2008` label stale.** Script window is 1950–2022; numeric cell already matches. Relabel to `1950--2022`.

**[blocker] l.791 — caption "collapses monotonically" false.** Full-period coefficient (+0.0074, p=0.049) exceeds the 1950–2022 sub-period (+0.0057, p=0.32). Reword to "the sub-period channel attenuates … though the pooled full-period coefficient stays positive and marginally significant."

### Major

**[major] l.786 — pathway country composition stale.** Paper lists High-density intensive={France, Netherlands}; Early extensifiers={Britain, Sweden, Denmark, Norway, Switzerland}; Pastoral/mixed late={Italy, Iceland, Finland}. Replicating `prevpos_pathway_stratified.py`: {FRA, ITA, NLD}; {CHE, DNK, GBR}; {NOR, SWE}. Finland sits alone in crop-dominant late; Iceland is absent from the merged panel.

**[major] l.910 — CF2 welfare figure stale.** Paper attributes a $-0.32$ log-unit welfare gain to $\\hat\\beta=-0.285$. Current parquet: CF2 crop-dominant late median $\\Delta\\ln P=+0.21$ (+23%), so welfare $=-0.325\\times0.21=-0.07$ log-units. Refresh the figure (and `tab:struct-cf` rows), do not re-attribute the betas.

**[major] l.291 — stale welfare cross-reference.** Change "$-0.32$ log-units" to "$-0.07$ log-units"; leave the +0.95% real-wage rise and $\\hat\\beta_{\\ln P}=-0.325$ untouched (internally consistent).

**[major] l.910 — pathway proportional gains misstated.** Paper: "pastoral/mixed late countries lose a similar amount in proportional terms." Parquet: all pathways *gain* under no-volcanism — pastoral/mixed +0.66 (+93%), high-density +0.24, extensifiers +0.17 — none lose; +93% is roughly half of crop-dominant's +212%, not "similar."

**[major] l.370 — figure caption p-value mis-paired.** Caption: $\\hat\\beta=-2.48$ ($p<10^{-24}$, $R^2=0.45$). The $p<10^{-24}$ is the N=199 deep-determinants baseline ($R^2=0.19$); the N=154 pathway-FE spec (which carries $R^2=0.45$) has $p<10^{-15}$. Change the caption exponent to $10^{-15}$ to match the body (l.374) and `tab:shadow`.

**[major] l.48 (intro) panel-VAR shorthand** *(also tagged presentation in JSON).* `joint_landuse_var.py` estimates three separate within-country FE-OLS equations — no SUR/companion/IRF structure. Soften the "reduced-form panel VAR" gloss to "a set of contemporaneously-identified FE-OLS equations rather than a structural VAR."

### Minor (number/text fixes — all in JSON)

- **l.38** Tambora "60 teragrams of sulfur" → "roughly 30 teragrams of sulfur (about 60 Tg of SO₂)" (eVolv2k VSSI = 28.08 Tg S; the 60 is the SO₂ mass).
- **l.44** "1500–1900 contains roughly sixty" → "the panel window 1421–1900 contains roughly sixty" (1500–1900 has only 52; 1421–1900 has 61).
- **l.48** "price-on-population coefficient" → "real-wage-on-population coefficient."
- **l.101 / l.106 / l.849** ANOVA $F=6.31$ → $6.30$ (productive_months = 6.3048; 6.31 is dry_months).
- **l.247** crop-dominant coefficient $-9.7\\cdot10^{-5}$ → $-9.6\\cdot10^{-5}$ (parquet = −9.6465e-5). *Note: abstract/intro round to −9.7; harmonise the rounding convention.*
- **l.129** "3.5–5 times" → "3.5–6 times" (range is 3.56–5.85).
- **l.296** "country-linear trends" → "a common linear time trend" (single shared trend, no iso3 interaction).
- **l.1059** Appendix L bootstrap "agree to within rounding" — two urban-equation cells diverge (0.086→0.280; 0.126→0.045); reword, both still far from Bonferroni.
- **l.765** volatility coefficient $p=0.036$ → $0.035$.
- **l.642** "$\\Delta$ urban share ... no relationship ... ($p>0.30$)" — +|lat| cell is $p=0.298<0.30$; reword to "no significant relationship; largest $|\\hat\\beta|$ carries $p=0.30$."
- **l.799** `1900--1950` row: +0.0078/(0.0057)/0.175/659 → +0.0079/(0.0058)/0.169/659.
- **l.733** two-way-clustered pooled $\\hat\\gamma_T$ $p=0.008$ → $0.020$ (stale; significance at 5% holds).
- **l.816** precipitation uncertainty start 31.6 → 31.5 mm/month.
- **l.894** CF2 crop-dominant $-0.12$ ($-12\\%$) → ($-11\\%$).
- **l.900** high-density baseline $+0.36$ ($+43\\%$) → ($+44\\%$).
- **l.483** conflict within-variance "58–98%" → "81–98%" (four named regressors span 80.6–97.6%; 58% is unnamed log_fatalities).
- **l.483** "Kuwae 1453" → "Kuwae 1458 (... fall of Constantinople (1453))" (eVolv2k dates Kuwae at 1458).
- **l.552** $\\beta_{L=2}$ first two cells carry spurious significance stars; drop them.
- **l.442** Allen "4,739 ... across 18 cities, 568 pre-1500" → "across these 17 cities (568 pre-1500), augmented with 560 Tuscan observations ..., for 18 cities in all."

### Presentation / style (selected — most are JSON-applyable)

- **l.29 abstract** "collapsing to zero by 1950" → "collapsing by an order of magnitude by 1950" (modern FEVD point 0.016, CI excludes zero; collapse is 11×, not literally zero).
- **Abstract/intro title reframe** — foreground the pathway-heterogeneous demographic margin as the central result; demote the long-shadow and Boserup findings to supporting roles. *(Structural; .md only.)*
- **N reconciliation** (196 vs 197 vs 199 vs 154 across `tab:lat-robust`, `tab:shadow`, `tab:placebo-window`, `tab:deep`) — add a one-line footnote on listwise variation. *(.md only.)*
- **e-notation / artefact spelling / "honest null" framing / UK spelling** — standardise. *(.md only.)*
- **§4 reordering, §4.5 Boserup length, figure float placement** — structural editorial suggestions. *(.md only.)*

---

## horserace.tex

### Blockers

**[blocker] l.343 / l.534 / l.543 / l.548 — "five" vs "eight" FWER survivors.** Line 302 and the abstract/intro say eight; these say five. `robustness_battery.parquet` (check=wy_correction, p_adj_wy_minp<0.05) = eight survivors. Replace "five FWER-surviving cells/findings" with "eight FWER-surviving cells" at each. (l.543 also fixes the stale "162-country panel" → R1b coverage is 195, and strikes "horserace v3".)

### Major

**[major] l.557 — "Three of the five" ancestral-yield survivors.** Correct: two of the eight (pop-growth $F=26.97$, density 1500 $F=15.55$); density-2025 is the lone near-miss ($F=10.40$, $p_{\\text{WY}}=0.054$). Density-2025 survivors are Neolithic fraction and functional alleles.

**[major] l.559 — GS dominance range mis-attributed.** "1.8 to 17 across the within-bundle decomposition" is the long_shadow within-country panel range. The horserace within-bundle range is ~2.8× to 23× across demographically informative cells, with $\\ln D_{2025}$ the exception (GS marginally leads).

**[major] l.228 — Table 6 caption false identity.** "By construction the column sum equals the climate bundle's Shapley R² from Table tab:shapley" is false (0.0534 vs 0.0875; conditioning set is 3 scalar substrates, not 4). Reword to a marginal-R²-conditional-on-everything-else statement.

**[major] l.343 — climate-window placebo numbers.** pre-1500 climate on D1500: 0.024→**0.040** (baseline 0.065→**0.088**); modern-window on pop growth: 0.163→**0.136** (baseline 0.082→**0.077**); modern on D1500: 0.024→**0.028**. The 0.163 was a mixed-up row.

**[major] l.515 — climate–A partialling numbers.** D1500 climate 0.065→0.069 should be **0.087→0.092**; ancestral 0.049→0.045 should be **0.012→0.007**. Pop-growth climate 0.082→0.096 should be **0.077→0.090**; ancestral 0.122→0.106 should be **0.096→0.077**. (Also "five FWER" → "eight".)

**[major] l.427 — Table 7 caption false identity.** Same defect as Table 6; reword to "approximate ... but differ because the within-bundle decomposition uses a different residual baseline (e.g. 0.166 vs 0.175 on log GDPpc)."

**[major] l.436 — malaria-locus exclusion subtraction.** $0.175-0.037-0.017=0.121$ mixes scales. Use the within-bundle start point: $0.166-0.037-0.017\\approx0.11$, with an explicit baseline caveat.

**[major] l.194 — Shapley formula indexed to four players.** Change summation to five players: $\\sum_{S\\subseteq\\{1,\\dots,5\\}\\setminus\\{s\\}}\\frac{|S|!\\,(5-|S|-1)!}{5!}[\\cdot]$.

**[major] l.474 — typo "principal specificationification's".** → "principal specification's".

### Cross-paper (both)

**[major] Table 7 caption (l.427)** — see above; bundled under horserace majors but flagged `both` in JSON as it co-occurs with the Table 6 caption defect.

### Minor (all in JSON)

- **l.55** "robust on the deep-past density margins" → "... density, convergence-growth, and transition-timing margins."
- **l.88** "$|r|\\in[0.24,0.67]$" → "$[0.23,0.67]$" (min |r|=0.235).
- **l.167** Maddison-1500 parenthetical lists 12 names for a count of 11 — drop "Italy via inferred series" (Italy's 1500 gdppc is NaN).
- **l.224** climate Shapley D1500 0.088 → 0.087; pandemic Shapley range 0.001–0.015 → 0.002–0.015.
- **l.249** scalar-substrate list "predicted heterozygosity" → "the agricultural-Neolithic ancestry fraction" (SCALAR_SUBSTRATES = neolithic_frac, ancestral_yield_log, pandemic_intensity_norm).
- **l.302** functional GDPpc reference $p=2.4\\times10^{-9}$ → $2.7\\times10^{-9}$.
- **l.322** sub-sample mediation fraction "58%→62%, 54–62%" → "63%→70%, 63–67%".
- **l.313** Americas count "(24 ISO3)" → "(26 ISO3)".
- **l.356** climate D2025 "$\\phi=0.034$ ... smallest of the six" → "among the smallest of the six" (it is third-smallest).
- **l.532** `cum_vssi` "71 to 123 ... NH mid-latitudes cluster near 123" → "30 to 123 ... equatorial countries cluster near 123; high-latitude NH receive least" (largest eruptions were tropical).

### Presentation / style (selected — JSON-applyable where concrete)

- **l.29 abstract** "dominant determinant of the modern outcomes" → "the broadest of the five substrates ... family-wise robust on four of the six outcomes" (argmax on 3 of 4 modern outcomes, lead distinguishable only on gdppc and dt-timing).
- **l.447 Table caption** "in all four columns ... DARC and HBB attenuation" → "in all three estimated columns ... per-allele attenuation from column (1) to (3)."
- **l.210 / l.178** "four pre-industrial channels" → "five pre-industrial substrates."
- **Version labels** ("horserace v3"/"v2"), "LS" abbreviation definition, abstract paragraph splits, greyscale colormap caption, "metaphor"/"multi-witness" prose trims, DARC reframe in §4.6/§4.7 — described in JSON where a clean swap exists; the rest are .md-only structural notes.

---

## Cross-paper (both)

- **[editorial] long_shadow l.50** — soften the institutional-development alignment to a hedged "consistent with, not evidence for" reading and point to the companion horserace's DARC sample-composition finding. *(Structural sentence-rewrite; .md only.)*
- **[editorial] long_shadow l.424 / l.29** — add a within-battery / pre-multiplicity qualifier to the "single strongest predictor" superlative, cross-referencing the horserace family-wise verdict. *(.md only.)*
- **[editorial] horserace l.172 / l.121** — narrow the orthogonality claim (the pathway clustering is *not* orthogonal to the climate bundle, since inter-annual temperature volatility is in both bases) and cross-flag the GAEZ mean-T/mean-P overlap. *(.md only.)*
- **[editorial] Table cross-references** (`tab:malthus` SEs/cluster counts, `tab:joint-var` per-column N, `tab:lat-robust` per-column N, `tab:mediation` bolding rule, side-by-side joint vs single-equation pathway coefficients, figure-directory pruning) — presentation completeness; .md only.

---

## Reproducibility gaps (no prose change; pipeline fixes)

These are correct in the manuscript but not regenerable from committed output:

- **horserace** — the 91% DARC sample-composition share (l.53) is not emitted by any committed script; have `exercise_colonial_partial_extended.py` emit the varying→fixed→fixed+state_hist swing and add a `verify_against_paper.py` assertion.
- **horserace** — Table `tab:colonial-partial-extended` AJR-only and EUR-only rows (l.495–496) are not in the shipped parquet; add the two single-control specs.
- **long_shadow** — `tab:boserup-robust` Panel C (l.940–942) verified by hand but emitted by no committed script; add a pop-weighted WLS variant to `boserup_robustness.py`.
