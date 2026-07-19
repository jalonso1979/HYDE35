---
title: "Horserace v3: modern population-genetics substrates replace AG heterozygosity"
date: 2026-05-19
status: draft
supersedes: 2026-05-18-horserace-climate-bundle-density-outcomes-design.md (partial)
---

# Horserace v3 — replacing predicted heterozygosity with functional alleles + Neolithic fraction

## Motivation

The horserace v2 (commit `873c606`) retained the Ashraf-Galor (2013) ancestry-adjusted predicted-heterozygosity measure $H_i^{\text{pwadj}}$ as one of four substrates. That measure has drawn three lines of well-developed critique:

1. **Ethical/interpretive.** Guedes et al. (2013, PNAS commentary) and Coop et al. (PNAS letter) explicitly criticised the AG framework for reifying abstract "genetic diversity" as a deep determinant of development, with the inverted-U framing reading uncomfortably close to eugenicist logic about an "optimal diversity" level for civilisation. The critique has been sustained in the population-genetics literature and is the single most common objection raised in seminar Q&A.

2. **Substantive.** $H^{\text{pred}}$ is a scalar function of migratory distance from Addis Ababa, which makes it nearly inseparable from "distance from Ethiopia" (or equivalently, an Africa-vs-non-Africa indicator). The post-1500 ancestry adjustment partially fixes the modern-composition issue but doesn't address the underlying measurement.

3. **Empirical (within Eurasia).** Our own §6.7 Lazaridis exercise (horserace v2, on the 79-country Eurasian sub-sample) shows the $H^{\text{pred}} \to$ log GDPpc coefficient moves from $-10.68$ (se 10.61) to $-1.70$ (se 13.11) when Anatolian Neolithic farmer (ANF) and Yamnaya steppe-pastoralist ancestry are added as controls. The "scalar diversity" interpretation has already failed inside the Eurasian core; the question is whether the same absorption holds globally.

The horserace v3 redesign replaces $H^{\text{pred}}$ with two functionally-interpretable substrates that are individually more defensible and jointly span what $H^{\text{pred}}$ was measuring.

## Substrate changes

### New substrate 1: Functional-allele bundle (coalition player)

Eight SNPs with established functional/economic-history channels, treated as a Shapley coalition player parallel to the climate bundle:

| Locus | SNP / variant | Selection pressure | Economic-history channel |
|---|---|---|---|
| LCT/MCM6 | rs4988235 | lactase persistence | dairy-pastoral package (post-Neolithic NW Europe, East African pastoral) |
| ADH1B | rs1229984 | fast alcohol metabolism (Arg47His) | East Asian rice/cereal fermentation cultures |
| AMY1 | rs4244372 (tag for copy-number) | salivary amylase activity | grain-based agriculture (starch digestion) |
| EDAR | rs3827760 (V370A) | hair/sweat-gland morphology | East Asian Neolithic |
| DARC/FY | rs2814778 | Duffy-negative (malaria resistance, *P. vivax*) | tropical malarial-zone selection |
| SLC24A5 | rs1426654 | skin pigmentation (Ala111Thr) | latitude/UV adaptation |
| HBB | rs334 (HbS) | sickle-cell trait (malaria resistance, *P. falciparum*) | malarial-zone selection (Sub-Saharan Africa) |
| FADS1/2 | rs174570 | long-chain PUFA conversion | post-agricultural diet shift (vegetable-oil reliance) |

**Data sources**: 1000 Genomes Project (1KG Phase 3) per-population derived-allele frequencies via NCBI ALFA, supplemented by SGDP (Simons Genome Diversity Project, 142 populations) for non-1KG populations. For each SNP, build a population-level frequency table covering ~70–142 source populations; map to country-level frequency via the Putterman-Weil 1500 ancestry matrix (same operation as $H^{\text{pwadj}}$). This produces an 8-element vector per country.

**Treated as a Shapley coalition player.** When the bundle is "in" subset $S$, all 8 SNPs enter the regression jointly; mediation share is the partial-R² form (eq. 6 of horserace v2). Within-bundle sub-Shapley (new Table 7) decomposes the bundle's contribution across the 8 SNPs per outcome.

**Coverage estimate**: ~185 countries (matching the existing full-substrate sample), since the Putterman-Weil ancestry matrix already spans the world. The 8-SNP frequency table at the population level is small and easy to assemble from published genome data and ALFA.

### New substrate 2: Neolithic-ancestry fraction (scalar)

Country-level share of modern ancestry derived from populations that underwent an *agricultural-Neolithic transition* of any kind, versus pre-Neolithic forager ancestry:

**Post-Neolithic (counted toward the fraction):**
- Eurasia: Anatolian Neolithic farmer (ANF), Iranian Neolithic, Caucasus Hunter-Gatherer-derived Neolithic (CHG), Yamnaya/steppe pastoralist (post-Neolithic agropastoral package) (Lazaridis 2014, 2022)
- East Asia: Yangtze rice-farmer, Han-Neolithic, Tibetan agricultural (Wang 2019, Yang 2020, Liu 2024)
- Sub-Saharan Africa: Bantu agriculturalist (Patin 2017, Lipson 2020)
- Americas: Mesoamerican maize-farmer, Andean potato-farmer — counted post-Neolithic because they underwent independent agricultural transitions ~5000–4000 BP (Reich 2012, Posth 2018)
- Pacific: Austronesian (Skoglund 2016, Bergström 2017)

**Pre-Neolithic (excluded from fraction):**
- Western Hunter-Gatherer (WHG), Eastern Hunter-Gatherer (EHG), Ancient North Eurasian (ANE)
- Beringian / pre-agricultural Native American
- Khoisan, ancient Pygmy-like, ancient West African forager
- Papuan, Aboriginal Australian, Onge-like

**Construction**: For each country, sum the ancestry shares of all post-Neolithic components from the relevant regional ancient-DNA decomposition; the result is a scalar in $[0, 1]$.

**Coverage and imputation strategy**: Lazaridis (Eurasia) covers ~80 countries with fine-grained decompositions. For other regions, ~6–8 published papers need to be aggregated. Where component-level percentages are unavailable (some Pacific microstates, parts of Central Asia, isolated highland regions), we impute from the regional modal classification (e.g., a Bantu-speaking African country without a published qpAdm decomposition gets the Bantu-region mean). The imputation rule is documented per country.

**Conceptual note**: Steppe (Yamnaya) ancestry is counted as post-Neolithic because the package it brought (horse-pastoralism, wheel, possibly Indo-European languages) was a post-Holocene adaptation to extensive land use, and is grouped with the agricultural transitions for our purposes. We test sensitivity to this choice by reporting an alternative "strict-agriculturalist" classification that excludes Yamnaya.

### Substrate 3 (unchanged): Climate bundle

Same as horserace v2: $(\bar T, \bar P, \sigma_v^T, \sigma_v^P)$ over 1421--1750.

### Substrate 4 (unchanged): Ancestral crop yield

Same as horserace v2: $\ln(A_i + 1)$ from Galor-Özak 2016 Caloric Suitability Index.

### Substrate 5 (unchanged): Pandemic intensity

Same as horserace v2: $\Pi_i$ hand-coded.

### Demoted: H_pred_pwadj

$H_i^{\text{pwadj}}$ remains in the panel for the coefficient-evolution robustness section but is **not** a substrate in the headline Shapley matrix. The headline matrix becomes 5 substrates × 6 outcomes = 30 cells.

## Empirical exercises

### Exercise 1 (rerun): headline Shapley on 5×6 matrix

Same as horserace v2 but with 5 substrates. Climate bundle and functional-allele bundle are coalition players; Neolithic fraction, ancestral crop yield, and pandemic intensity are scalars. Reports Shapley R² per (substrate, outcome) cell, plus the within-functional-bundle sub-Shapley (new Table 7).

### Exercise 2 (rerun): pathway mediation on 30-cell matrix

Same as horserace v2, extended to 30 cells. Climate bundle and functional-allele bundle use partial-R² mediation share; Neolithic fraction, ancestral yield, pandemic use β-attenuation.

### Exercise 3 (NEW): H_pred coefficient-evolution test

On the full 185-country panel, run for each FWER-surviving outcome (the v2 cells where H_pred was a survivor: log GDPpc, log pop growth):
1. Baseline: $y \sim H^{\text{pwadj}} + \text{controls}$
2. + Neolithic fraction: $y \sim H^{\text{pwadj}} + \text{Neolithic} + \text{controls}$
3. + functional bundle: $y \sim H^{\text{pwadj}} + \text{Neolithic} + \text{functional bundle} + \text{controls}$

Document how $\hat\beta_{H^{\text{pwadj}}}$ moves across the three specifications. The strong prior (from §6.7 Lazaridis Eurasian result) is that the global coefficient attenuates substantially when the two modern substrates enter. This is the empirical defense of the substrate replacement choice.

### Exercise 4 (NEW): within-functional-bundle sub-Shapley

Decompose the functional-allele bundle's Shapley R² contribution across the 8 SNPs per outcome. The baseline for this sub-decomposition is (geography controls + the 4 non-functional substrates). Expected substantive patterns:
- LCT dominates the bundle on outcomes for Northern Europe / East African pastoral cultures
- ADH1B and EDAR dominate on outcomes where East Asian rice-cereal cultures are over-represented
- DARC and HBB dominate where malarial-Africa is over-represented
- AMY1 dominates broadly across grain-cultivating regions

The sub-Shapley clarifies *which functional adaptation* drives the bundle's contribution. We will report a "malaria-loci-removed" variant (drop DARC + HBB) to isolate the agricultural-adaptation signal from the tropical-disease signal.

### Exercise 5 (existing): robustness battery on 30-cell matrix

Continent FE, LOO, climate-window placebos, climate-A partialling, conflict/volcanic controls. Westfall-Young FWER over 30 cells (was 24 in v2). Same machinery, expanded grid.

## Files modified

### Data builders (`analysis/paper5_horserace/`)

- `build_functional_alleles.py` (new) — assemble 8-SNP × 70-population frequency table from 1KG + SGDP; map to country via Putterman-Weil 1500 ancestry matrix; output `analysis/data/deep_determinants/functional_alleles_pwadj.parquet` with 8 columns (`fa_lct`, `fa_adh1b`, ...).
- `build_neolithic_fraction.py` (new) — aggregate ancestry decompositions from ~8 papers; classify each component as pre/post-Neolithic; compute country-level post-Neolithic share. Output `analysis/data/deep_determinants/neolithic_fraction.parquet` (single `neolithic_frac` column).
- `build_horserace_panel.py` — merge the two new datasets into the master horserace panel.

### Analysis kernel (`analysis/paper5_horserace/`)

- `exercise1_shapley.py` — change `SUBSTRATES` from 4 to 5; add functional-allele bundle as a second coalition player. New `FUNCTIONAL_BUNDLE` tuple.
- `exercise1_functional_subshapley.py` (new) — mirror of `exercise1_climate_subshapley.py` for the functional-allele bundle.
- `exercise2_mediation.py`, `exercise2_subsample_stability.py` — extend to 5 substrates; functional bundle uses partial-R² mediation.
- `exercise3_h_pred_evolution.py` (new) — coefficient-evolution test for H_pred under sequential addition of Neolithic and functional substrates.
- `robustness_battery.py` — extend to 30 cells; WY FWER over the new grid.

### Tables and figures (`analysis/figures/paper5_horserace/`)

- `tab01_descriptives.tex` — add 8 functional-allele rows + Neolithic fraction row; demote H_pred to a footnote sub-section.
- `tab02_substrate_correlations.tex` — 5×5 cross-substrate correlation (functional bundle represented by AMY1 as bundle representative, or by PC1 of the 8 SNPs).
- `tab02c_functional_subcorrelations.tex` (new) — within-functional-bundle 8×8 SNP correlation matrix.
- `tab03_full_ols.tex` — 6 columns (one per outcome), substrate block expanded to ~14 rows (4 climate + 8 functional + Neolithic + A + Π).
- `tab04_shapley_table.tex` — 5×6 Shapley matrix.
- `tab07_functional_subshapley.tex` (new) — within-bundle 8×6 sub-decomposition.
- `tab_h_pred_evolution.tex` (new) — three-column table showing H_pred coefficient moving across the three sequential specifications, for log GDPpc and log pop growth.
- `fig01_substrate_covariance.py` — 5-substrate scatter matrix.
- `fig03_shapley_heatmap.py` — 5×6 heatmap.
- `fig04_mediation.py` — 5×6 mediation matrix.
- `fig06_subsample_stability.py` — 5×6 stability ribbons.
- `figA_robustness_battery.py` — 30-cell WY panel; F-stat heatmap.

### Paper text (`paper/horserace/horserace.tex`)

- **Abstract**: insert one sentence on the modern-genetics framing immediately after the climate-bundle sentence.
- **§1 Introduction**: cite Guedes et al. 2013 and Coop et al. critiques; reframe the substrate description; rewrite the findings paragraphs after empirics.
- **§2 Theory**: substitute the substrate enumeration; theoretical channel for each functional locus stated explicitly.
- **§3 Data**:
  - new §3.1.5 "Functional-allele bundle" (8 SNPs, sources, Putterman-Weil mapping)
  - new §3.1.6 "Neolithic-ancestry fraction" (sources, classification, imputation)
  - rewrite §3.1.2 (was the H_pred description) to note H_pred is demoted to the coefficient-evolution robustness section
- **§4 Shapley**: extend headline narrative to 5 substrates; add a paragraph on the within-functional-bundle sub-Shapley.
- **§5 Mediation**: extend to 5 substrates; functional bundle's mediation share interpreted.
- **§6 Robustness**:
  - new §6.x "H_pred coefficient-evolution: modern substrates absorb predicted heterozygosity globally"
  - new §6.y "Malaria-loci sensitivity" (functional-bundle Shapley with DARC + HBB removed)
- **§7 Discussion**: rewrite around 5 substrates; discuss the AG-critique resolution explicitly
- **§8 Conclusion**: update findings enumeration

### Bibliography additions

- Guedes et al. (2013), "Is poverty in our genes?" PNAS critique of AG
- Coop et al. (PNAS letter) — critique of AG 2013
- 1000 Genomes Project (Auton et al. 2015 Nature) — reference panel
- SGDP (Mallick et al. 2016 Nature) — reference panel
- Mathieson et al. (2015) Nature — selection in 230 ancient Eurasians
- Bersaglieri et al. (2004) AJHG — LCT selection
- Tishkoff et al. (2007) Nature Genetics — LCT in East African pastoralists
- Patin et al. (2017) Science — Bantu expansion
- Lipson et al. (2020) Nature — Ancient West African
- Wang et al. (2019) Nature Communications — Caucasus/East Asia
- Yang et al. (2020) Science — Yangtze rice farmer ancestry
- Liu et al. (2024) — Han Chinese genetic history
- Posth et al. (2018) Cell — ancient South Americans
- Skoglund et al. (2016) Nature — ancient Vanuatu
- Bergström et al. (2017) Nature — Aboriginal Australian
- Reich (2012) Nature — Native American three-ancestry model

## Acceptance criteria

1. `deep_determinants_horserace.parquet` includes 8 functional-allele columns + `neolithic_frac` column with ≥160 non-null countries each (matches realistic coverage given Putterman-Weil + ancestry-decomposition gaps).
2. Headline 5×6 Shapley runs without error; sum of Shapley contributions per outcome equals R²(full) − R²(baseline) within 1e-4.
3. The H_pred coefficient-evolution exercise shows the $\hat\beta_{H^{\text{pwadj}}}$ on log GDPpc attenuates by at least 50% when Neolithic + functional bundle are added (the strong empirical defense of the substrate replacement).
4. The 5 (or more) FWER-surviving findings under the new substrate set are reported transparently. If H_pred → GDPpc and H_pred → pop growth disappear under the new design, that's a substantive finding and the paper says so explicitly.
5. Paper compiles clean (no undefined refs, no duplicate labels). Page count ≤ 35.
6. All existing 72 tests pass; new tests for the new builders pass.

## Risks and trade-offs

- **Functional-allele bundle may absorb part of climate or ancestral-yield Shapley.** DARC and HBB (malaria-resistance) covary with tropical climate; AMY1 covaries with grain-suitable agro-ecologies. The malaria-loci-removed sensitivity column directly addresses this.
- **Neolithic-ancestry classification has debatable cases.** The "strict-agriculturalist" alternative (excluding Yamnaya) is reported as sensitivity.
- **Coverage outside Eurasia for the Neolithic fraction is sparse.** Several countries will end up with imputed values from regional means rather than direct decompositions. We report this honestly and flag the imputed countries in the data documentation. Robustness column: re-run on the sub-sample with direct (non-imputed) decompositions.
- **The post-1500 ancestry adjustment is shared with H_pred_pwadj.** The functional-allele bundle and Neolithic fraction use the same Putterman-Weil ancestry matrix, so they inherit the same geographic-vs-pwadj asymmetry on the density-1500 outcome. We acknowledge this in §3 and report unadjusted variants where possible.
- **Several functional alleles have well-known interactions** (LCT × climate; HBB × DARC × malaria). The Shapley is computed jointly so these interactions are accounted for in the variance attribution, but the within-bundle sub-Shapley may show high collinearity among related SNPs.
- **The Coop and Guedes critiques are sustained academic positions.** Some referees who hold these positions may push back even on the substrate-replacement framing. We will be transparent about the critique and the empirical resolution rather than defensive.

## Sequence

Dependency graph:

```
build_functional_alleles.py        build_neolithic_fraction.py
              \                     /
        build_horserace_panel.py
                    │
                    ├── exercise1_shapley.py
                    │       ├── exercise1_functional_subshapley.py
                    │       └── exercise3_h_pred_evolution.py
                    ├── exercise2_mediation.py + subsample_stability.py
                    └── robustness_battery.py
                            │
                    figures and tables
                            │
                    paper text revisions
                            │
                    final compile
```

Implementation will proceed in this order. Spec approval first; implementation plan optional (the spec is detailed enough that a separate plan file may not be necessary).
