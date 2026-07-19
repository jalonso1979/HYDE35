# Pre-1751 and medieval mortality extensions

**Date:** 2026-05-15
**Paper:** *The Long Shadow of Seasonality* (long_shadow.tex)
**Author:** Jorge Alonso Ortiz (with coauthor José-María Da-Rocha)

## 1. Motivation

The current Section 4.3 preventive/positive-check decomposition (`prevpos`)
recovers the historically documented summer-mortality pattern of pre-modern
Europe: warm years killed infants and children disproportionately, while
fertility was largely unresponsive. The exercise rests on the 547-observation
annual European panel 1751--1900 anchored by Sweden's 149-year continuous
vital statistics series.

Two natural extensions are available:

1. **Push the prevpos exercise back into the deep pre-industrial Malthusian
   regime** by adding the Wrigley-Schofield annual aggregate England series
   for 1541--1750.
2. **Add a medieval English mortality event-study** by linking the Hatcher
   manorial-level mortality reconstructions (1300--1500) to the eVolv2k
   volcanic record, the OWDA / PAGES2k climate reconstructions, and the
   hand-coded Black Death wave catalogue we already maintain.

Two further mechanism checks become free given the data we already have:

3. **Pathway-stratified prevpos** on the existing 547-obs 1751--1900 panel.
4. **Modern-era prevpos** on the 1900--2022 segment of `FertilityData.xlsx`.

Together these four sub-exercises form a unified "deep-time, cross-pathway,
modern-era" decomposition of the Malthusian climate response and a sharper
test of the summer-mortality mechanism.

## 2. Scope and out-of-scope guards

In scope (this design):

- W-S annual England CBR / CDR / population 1541--1871 → §4.3.2.
- Hatcher-Bailey medieval manorial mortality 1300--1500 → new Appendix C.3.
- Pathway-stratified prevpos on the existing 1751--1900 panel → §4.3.3.
- Modern-era 1900--2022 prevpos on the existing FertilityData panel → §4.3.4.

Out of scope (deferred):

- A full structural model of preventive vs positive checks with deep
  parameters. We stay reduced-form.
- Joint-VAR-style integration of the prevpos exercise with the §4.4
  population/cropland/urban VAR. Each exercise stays single-equation per
  outcome.
- An attempt to recover the age-specific decomposition ($m_0$, $m_5$, $m_{10}$)
  pre-1751. W-S provides only crude vital rates (CBR, CDR); the age-specific
  decomposition must remain on the 1751-1900 sub-sample.
- A national aggregation of Hatcher manorial mortality. The medieval
  exercise stays manor-level event-study; no national synthesis.

## 3. Data retrieval cascade

### 3.1 Wrigley-Schofield

Target output: a CSV
`analysis/data/wrigley_schofield/ws_england_annual.csv` with columns
`year, CBR, CDR, population, source` covering 1541--1871. CBR and CDR in
per-1000 per year; population in persons.

Cascade execution, with stop conditions:

1. **Crafts-Mills (2020 *EJ*) AEA replication.** Search `openicpsr.org` and
   `aeaweb.org/journals/code` for "Crafts Mills Malthus Solow" replication.
   Expected target: a Stata/R dataset with annual England demographic
   series. Stop condition: CBR, CDR, and population (or birth-count,
   death-count, and population from which CBR, CDR can be constructed)
   all available at annual resolution 1541--1871. Partial coverage (e.g.,
   population only) does not satisfy the stop condition; proceed to step 2.
2. **Gregory Clark UC Davis page.** `faculty.econ.ucdavis.edu/faculty/gclark/data.html`
   has *Farewell to Alms* spreadsheets. Stop condition: same as above.
3. **Bob Allen Nuffield archive scrape.** We have 17 city xls files already
   downloaded. Check whether any have an England-population or
   births/deaths column we did not previously parse.
4. **OCR Schofield (2022) *Parish Register Aggregate Analyses* 2nd ed PDF.**
   Freely available at `localpopulationstudies.org.uk`. Use vision-API table
   extraction on the Appendix A3 tables. Validate against the 1801 census
   population anchor (~8.7 M).
5. **UK Data Service registration.** Last resort only. Requires user
   academic credentials; not pursued unless steps 1--4 all fail.

If only decadal (not annual) W-S data is retrievable, run the §4.3.2
regression at decadal frequency for the pre-1751 segment as a sub-period
robustness rather than abandoning the exercise.

### 3.2 Hatcher-Bailey

Target output: a CSV
`analysis/data/hatcher_bailey/medieval_mortality.csv` with columns
`manor, year, mortality_rate, sample_size, source` covering 1300--1500 for
as many manors as we can recover (Halesowen, Westminster, Winchester,
others if available).

Cascade execution:

1. **Voigtländer-Voth (2013 *RES*) "Three Horsemen of Riches" replication.**
   Supplementary materials on `restud.oxfordjournals.org`. Stop condition:
   at least one continuous manorial series with ≥ 30 years of coverage
   1300--1500.
2. **Campbell (2016) *The Great Transition* Cambridge UP appendix.**
   Online supplementary data. Stop condition: same as above.
3. **OCR Hatcher (1986 *EHR*) "Mortality in the fifteenth century"
   appendix tables** and Hatcher-Bailey (2001) *Modelling the Middle Ages*
   tables. Vision-API extraction.

If only point estimates (1348, 1361 specific years) are recoverable rather
than continuous series, downgrade to a narrative documentation block in
Appendix C.3 rather than a regression. Honesty in the framing.

## 4. Regression specifications

### 4.1 §4.3.2 W-S pre-1751 deepening

For each (country $i$, year $t$) in the augmented 1541--1900 panel, with
England 1541--1900 added from W-S and the other 9 countries entering as
they start (Sweden 1751, France 1820, ...):

$$
\Delta \log \mathrm{CBR}_{i,t} = \alpha^B_i + \gamma^B_T\, T_{it} + \gamma^B_P\, P_{it} + \delta^B_T\, \sigma^T_{i,[t-2,t+2]} + \theta^B\, \mathrm{trend}_{i,t} + \varepsilon^B_{it}
$$
$$
\Delta \log \mathrm{CDR}_{i,t} = \alpha^D_i + \gamma^D_T\, T_{it} + \gamma^D_P\, P_{it} + \delta^D_T\, \sigma^T_{i,[t-2,t+2]} + \theta^D\, \mathrm{trend}_{i,t} + \varepsilon^D_{it}
$$

Country FE; country-linear trend; SEs clustered by country.
$T_{it}$ and $P_{it}$ are ModE-RA + CRU annual anomalies from the country
mean. $\sigma^T_{i,[t-2,t+2]}$ is the 5-year centred rolling SD of $T$.

Sample: England 1541--1900 (W-S, then Registrar General post-1871) + the
9 other countries 1751--1900 from FertilityData. Expected $N \approx 760$.

Substantive prediction: warm-year-positive coefficient on $\Delta \log
\mathrm{CDR}$ is statistically significant on the deep pre-industrial
1541--1750 sub-sample, consistent with the 1751--1900 baseline. Fertility
coefficient remains small and insignificant.

### 4.2 §4.3.3 Pathway-stratified prevpos

For each pathway $p$, run the four 1751--1900 prevpos equations on the
within-pathway subsample. Expected sample sizes by pathway:
- High-density intensive: ~80--120 (FRA, NLD)
- Early extensifiers: ~250--350 (GBR, NOR, DNK, CHE, SWE)
- Crop-dominant late: ~150--200 (ITA, ESP, others)
- Pastoral/mixed late: ~0--50 (panel countries are mostly not pastoral)

Substantive prediction: positive-check dominance is strongest in
rain-fed-crop pathways (crop-dominant late) and weakest in capital-intensive
systems with better-developed storage and trade (high-density intensive).

### 4.3 §4.3.4 Modern-era 1900--2022 prevpos

Same four equations as §4.3.1, on the 1900--2022 sub-sample of the
FertilityData MAIN PANEL. Climate from ERA5.

Substantive prediction: the warm-year-positive-mortality coefficient flips
sign as sanitation, refrigeration, and antibiotics develop through the
twentieth century. If the prediction holds, it strengthens the
summer-mortality interpretation by showing the channel is sanitation-mediated.
If it does not hold, we acknowledge the mechanism is not pinned down.

### 4.4 Appendix C.3 medieval mortality event study

For manor $m$, year $t \in [1300, 1500]$:

$$
\log \mu_{m,t} = \alpha_m + \delta_d + \sum_{L=0}^{5} \beta_L\, V_{t-L} + \gamma_T\, T^{\mathrm{summer}}_{t,\mathrm{OWDA}} + \gamma_\sigma\, \sigma^T_{[t-5,t-1]} + \beta_{\mathrm{BD}}\, \mathbf{1}\{\text{plague year}\} + \varepsilon_{m,t}
$$

Manor FE $\alpha_m$, decade FE $\delta_d$, manor-clustered SEs. Climate
inputs:
- $T^{\mathrm{summer}}_{t,\mathrm{OWDA}}$: Old World Drought Atlas summer
  drought index for the relevant manor's grid cell, 1300--1500.
- $\sigma^T_{[t-5,t-1]}$: 5-year SD of OWDA summer T.
- $V_{t-L}$: continuous distributed-lag eVolv2k VSSI as in §4.1.3.
- Plague-window indicator: hand-coded from our existing pandemic catalogue
  (Black Death 1347--53, Pestis Secunda 1361--63, 1374--75, 1399--1402, etc.).

## 5. Paper integration

### 5.1 Section edits

§4.3 (prevpos) grows to a 4-piece structure:
- **4.3.1 Baseline 1751--1900** (existing as written).
- **4.3.2 Pre-1751 deepening (Wrigley-Schofield)** — one paragraph plus
  one short table with the CBR/CDR coefficients on the 1541--1900 panel.
- **4.3.3 Pathway-stratified prevpos** — one paragraph plus one short
  figure with per-pathway $T$-anomaly coefficients on $\Delta \log m_0$.
- **4.3.4 Modern-era 1900--2022 prevpos** — one paragraph plus one short
  table with the 1900--2022 coefficients alongside 1751--1900 for contrast.

New **Appendix C.3 "Medieval English mortality, 1300--1500"** — one paragraph
of data description, one figure (event-study coefficients), one table
(continuous distributed-lag VSSI coefficients). Honest scope disclaimer
about small N and manor-level (not national) identification.

Abstract: add one short sentence noting the deeper pre-industrial coverage
and the modern-era reversal test.

Introduction (findings paragraph): condense the prevpos description to
mention the four-piece structure rather than just the headline.

Conclusion: add one sentence noting the modern-era reversal confirms the
mechanism interpretation.

### 5.2 Bibliography additions

- `hatcher1986` (Hatcher 1986 *EHR* "Mortality in the 15th century")
- `hatcherbailey2001` (Hatcher and Bailey 2001 *Modelling the Middle Ages*)
- `campbell2016` (Campbell 2016 *The Great Transition*) -- if used
- `crafts_mills2020` -- if W-S retrieved from their replication
- `clark2007` -- if W-S retrieved from his page (the citation already exists)

## 6. Files to add or modify

### 6.1 New files

- `analysis/data/wrigley_schofield/ws_england_annual.csv`
- `analysis/data/wrigley_schofield/provenance.txt`
- `analysis/data/hatcher_bailey/medieval_mortality.csv`
- `analysis/data/hatcher_bailey/provenance.txt`
- `analysis/paper4_shadow/retrieve_ws_hatcher.py` (cascade execution script)
- `analysis/paper4_shadow/prevpos_extended_ws.py` (extended prevpos regression)
- `analysis/paper4_shadow/prevpos_pathway_stratified.py` (pathway-stratified)
- `analysis/paper4_shadow/prevpos_modern_era.py` (1900--2022 regressions)
- `analysis/paper4_shadow/medieval_mortality_event_study.py` (Hatcher event study)

### 6.2 Modified files

- `paper/long_shadow.tex` — add §4.3.2--4.3.4 subsubsections, add App. C.3,
  update abstract/intro/conclusion, add bibliography entries.
- `analysis/paper4_shadow/run_pre1500_volcanic.py` — add the new medieval
  mortality script to the orchestrator if it logically belongs there.

## 7. Validation and honesty constraints

- **Provenance.** Every retrieved series must carry a `provenance.txt`
  recording the exact source URL/citation and the cascade step at which it
  was retrieved. No silently-synthesised numbers.
- **Anchor checks.** The W-S series must match known anchor values: 1801
  census population (~8.7 M); Black Death mortality 1348--49 (CDR ≈ 200 per
  1000 or equivalent). The Hatcher series must show a documented mortality
  spike in 1348.
- **Negative-result reporting.** If a prediction fails (e.g., the
  modern-era prevpos does NOT flip sign, or the pathway-stratified
  coefficients are not heterogeneous as predicted), the paper text reports
  the negative result honestly rather than dropping the exercise.
- **Sample disclosures.** Every regression's $N$, country count, and time
  range explicitly reported in every table caption.

## 8. Risks and mitigations

- **Retrieval failure for W-S.** Mitigated by the cascade structure: even
  if the AEA dataverse search fails, Clark's UC Davis page or OCR of the
  Schofield 2022 PDF should yield the annual series. If only decadal is
  retrievable, downgrade to a decadal-frequency sub-period regression.
- **Retrieval failure for Hatcher.** If continuous series are not
  recoverable, the appendix becomes a narrative documentation block rather
  than a regression, with at most a small table of documented mortality
  spike years (1348, 1361, 1375) cross-referenced to climate and volcanic
  records. This is suggestive triangulation, not primary identification.
- **Modern-era data quality.** ERA5 is well-calibrated; FertilityData
  MAIN PANEL is HMD-derived. No anticipated quality issue.
- **Statistical power for pathway-stratified prevpos.** Pastoral/mixed
  late has very few HMD countries; we may not get a coefficient. If $N_p
  < 50$ for a pathway, we drop it from Table caption with a footnote
  rather than report unreliable estimates.
- **Scope creep.** Four sub-exercises plus a new appendix subsection
  could expand the paper by 6--8 pages. Mitigation: each sub-piece is one
  short paragraph plus one short table; we do not pad. Section 4.3 should
  remain about 4 pages total even after the addition.

## 9. Timeline estimate

- Cascade retrieval: 1--3 hours depending on which tier succeeds.
- W-S extended regression: 30 minutes.
- Pathway-stratified prevpos: 20 minutes.
- Modern-era prevpos: 20 minutes.
- Hatcher event study: 1--2 hours (depending on data shape).
- Paper text and tables: 1--2 hours.
- Verification compile and cross-reference audit: 30 minutes.

Total estimate: 4--8 hours of focused work.
