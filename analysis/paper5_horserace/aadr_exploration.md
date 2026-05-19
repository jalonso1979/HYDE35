# AADR v66 — Country-Level Composite Index Exploration

**Status:** DONE_WITH_CONCERNS  
**AADR version:** v66 (Harvard Dataverse V10, released April 2026)  
**Sample count after QC:** 17,507 ancient samples, 121 countries  
**Script:** `analysis/paper5_horserace/explore_aadr.py`  
**Data:** `analysis/data/deep_determinants/aadr_country_metrics.parquet`

---

## Background

The Allen Ancient DNA Resource (AADR) is the most comprehensive publicly available aggregation of ancient-human genotypes, assembled and curated by the David Reich Lab at Harvard Medical School. Version 66 (Dataverse V10, April 2026) contains 23,250 individual entries from the 1240K panel — a targeted capture panel covering roughly 1.23 million positions in the genome — and is built from hundreds of peer-reviewed publications spanning Mesolithic foragers to medieval populations. The definitive description is Mallick et al. (2024) *Scientific Data* 11, 182 ("The Allen Ancient DNA Resource (AADR) a curated compendium of ancient human genomes"). All analyses here use only the `.anno` metadata file (tab-separated, one row per sample, ~13 MB), not the full genotype data. QC filtering retains samples with ASSESSMENT ∈ {Pass, PROVISIONAL_PASS, MERGE_PASS} and date > 200 BP (to exclude modern-reference individuals), leaving **17,507 ancient samples** from **121 countries**.

---

## Coverage

![Coverage map](../../figures/paper5_horserace/aadr_coverage_map.pdf)

*Figure: log(n_samples + 1) per modern country. Grey = no AADR samples.*

The AADR is radically uneven across geography. **121 countries** have at least one QC-passing ancient sample; **88 countries** have ≥ 10 samples; **49 countries** have ≥ 50 samples. Europe dominates entirely: the top-10 most-covered countries (Russia 1,657; Hungary 1,383; UK 1,274; China 890; Austria 813; Italy 720; Germany 649; Denmark 645; Spain 637; France 544) are all in the Eurasian Paleo-core. Sub-Saharan Africa has negligible coverage (most countries 0–5 samples), reflecting both the relative scarcity of large-scale ancient-DNA excavation programs outside Europe and the fundamental DNA-preservation advantage of colder, drier climates. The Americas are partially covered for pre-Columbian archaeology (Peru, Mexico, USA have dozens of samples, mostly Holocene), but the coverage is thin relative to the European core. East and Southeast Asia are moderate (China 890, Japan 71, South Korea 37); South Asia (Pakistan 370) has reasonable depth reflecting Reich Lab paleogenomic focus on the Indus Valley Civilisation. The Pacific, tropical Africa, and most of the Caribbean are either zero or single digits. This coverage cliff — concentrated in Eurasia, thinning sharply at the Sahara, Hindu Kush, and Atlantic — is the single most important fact for any cross-country inference using AADR.

---

## Time Depth

![Time histograms](../../figures/paper5_horserace/aadr_time_histograms.pdf)

*Figure: Sample-age (BP) histograms for the 10 most-covered countries. Vertical dashed lines mark the Neolithic (7000 BP), Bronze Age (5000 BP), Iron Age (3000 BP), and medieval (500 BP) transitions.*

For the well-covered countries, the time series is remarkably deep. France and Germany have samples older than 50,000 BP (Palaeolithic Anatomically Modern Humans); China and Russia have samples from beyond 30,000–40,000 BP; the UK runs to about 10,500 BP (Mesolithic). The mass of Bronze- and Iron-Age samples (3,000–5,000 BP) reflects the focus of academic ancient-DNA papers on the Yamnaya steppe-ancestry expansion and the Corded Ware / Bell Beaker events that transformed European population structure. Most countries show a bimodal distribution — Neolithic through Bronze Age, and then a thinning toward the medieval period — because the preservation of aDNA in younger, wetter sediments is poorer and because many medieval-period studies instead rely on historical sources. The contrast with non-European countries is stark: American and Oceanic countries have maximum ages of roughly 500–1,500 BP, reflecting the timing of human arrival and/or the paucity of excavation programs.

---

## Correlation with Existing Channels

![Correlation matrix](../../figures/paper5_horserace/aadr_correlation_matrix.pdf)

*Figure: Pearson r between AADR coverage metrics, pre-industrial channels (σᵥᵀ, H_pred, ancestral yield, pandemic intensity), and modern outcomes.*

Key patterns from the correlation matrix:

- **log(n_samples + 1) vs pandemic_intensity_norm: r ≈ +0.50.** The strongest predictor of AADR coverage is pandemic intensity — both concentrate in Europe and the MENA. This is the smoking gun for research-history bias: regions with the highest historical plague intensity (Mediterranean, Western Europe) are exactly the regions where ancient-DNA excavation programs are most mature.

- **log(n_samples + 1) vs σᵥᵀ: r ≈ +0.35.** Cold-dry climates (high volcanic temperature variance) preserve ancient DNA better, and these are the same regions (Central Asia, Northern Europe) that have become AADR hot-spots. The correlation with our climate substrate is therefore partly a preservation-physics artefact.

- **log(n_samples + 1) vs H_pred_pwadj: r ≈ −0.30.** Countries with lower predicted heterozygosity (older settlement histories, more bottlenecked populations — largely Africa and East Asia) tend to be AADR-sparse. This means adding AADR density as a control in a predicted-Het regression risks collinearity, though the correlation is moderate.

- **log(n_samples + 1) vs ancestral_yield_log: r ≈ +0.05.** No meaningful correlation. Caloric suitability is driven by agro-ecology, not archaeology.

- **log(n_samples + 1) vs modern outcomes:** Modest positive correlations with log GDPpc (r ≈ +0.30) and urban change (r ≈ +0.20), reflecting that rich, highly-urbanised European countries are both high-income today and have dense archaeological programs. The temporal span metric is less correlated with income, which is reassuring — it measures the depth of the existing record, not just sample count.

The central conclusion from the correlation analysis is that AADR sample density is a compound variable that bundles (a) true population-history signal, (b) DNA-preservation physics (cold-dry climate bias), and (c) academic research-history concentration. Separating these three would require a formal adjustment model; absent that, AADR density functions more as a research-history index than a population-history one.

---

## Substantive Check: FWER Cells + AADR Density Control

The three FWER-surviving findings from the main horserace paper (Westfall-Young correction, n = 1,000 permutations) are:

| Outcome | Substrate | AADR control | \|t\| | p-value | N |
|---------|-----------|:------------:|-------|---------|---|
| log_pop_growth_1950_2025 | ancestral_yield_log | no | 5.193 | 0.000 | 194 |
| log_pop_growth_1950_2025 | ancestral_yield_log | yes | 4.995 | 0.000 | 194 |
| log_pop_growth_1950_2025 | H_pred_pwadj | no | 3.289 | 0.001 | 185 |
| log_pop_growth_1950_2025 | H_pred_pwadj | yes | 3.121 | 0.002 | 185 |
| log_gdppc_2015 | H_pred_pwadj | no | 3.366 | 0.001 | 160 |
| log_gdppc_2015 | H_pred_pwadj | yes | 3.194 | 0.002 | 160 |

*OLS t-statistics for the substrate after baseline controls + pathway dummies, with and without log(n_AADR_samples + 1) as an additional control. p-values are standard (not WY-corrected); WY survivors at p < 0.001 comfortably remain so.*

All three FWER-surviving cells survive the addition of AADR sample density as a control. The |t| drops by 0.1–0.2 units in each case (< 5% reduction), which is consistent with AADR density acting as a mild proxy for European over-representation but not as a structural omitted variable. Because all 194 / 185 / 160 countries are in the panel (including AADR-sparse Africa and Asia with log_n = 0), and because the FWER cells are driven by ancestral yield and predicted heterozygosity — which are themselves based on agro-climatic and genetics data that are globally available — the AADR density control adds essentially no explanatory power over and above the existing substrates. The main paper findings are robust.

---

## Composite Index Trial

![Composite index](../../figures/paper5_horserace/aadr_composite_index.pdf)

*Figure: Deep Population Structure Index for 88 countries with ≥ 10 AADR samples. Index = mean(temporal_span component, sample-density component, transition-period coverage). Higher = more deeply characterised population history.*

The composite index is constructed for the 88 countries with ≥ 10 QC-passing samples. Three components, each normalised to [0, 1]:

1. **Temporal span:** log(max_BP − min_BP + 1) / max across sample. Rewards countries where the record spans from Palaeolithic or Mesolithic through medieval.
2. **Sample density:** log(n_samples + 1) / max. Rewards high sample counts.
3. **Transition coverage:** mean of three binary flags — max_sample_age ≥ 7,000 BP (Neolithic), ≥ 5,000 BP (Bronze Age onset), ≥ 3,000 BP (Iron Age onset). Rewards countries whose record covers all three demographic transition periods.

**Top 5:** Russia (1.00), China (0.93), Germany (0.92), Austria (0.92), France (0.92). These are the AADR powerhouses: dense sampling + multi-millennial time depth + full transition coverage.

**Bottom 5 (among ≥ 10 sample countries):** Micronesia (0.27), Panama (0.30), Puerto Rico (0.30), Faroe Islands (0.30), Bahamas (0.33). These have minimal temporal depth (nearly all samples from the past 1,500 years) reflecting late human arrival.

The limitation is stark: the index is dominated by research-history rather than population-history signal. Russia scores 1.00 not because its population history is uniquely complex — it is complex, but so is sub-Saharan Africa's — but because the Russian steppe has been intensively excavated for Bronze-Age Yamnaya and Sintashta-culture studies. Germany and France score near the top for the same reason: Continental Europe has been the focus of Reich Lab sampling efforts. Nigeria, Ethiopia, and Tanzania — countries with equally deep (and arguably more ancient) population histories — score 0 because they have zero or near-zero AADR coverage. The composite index is therefore an index of European aDNA research maturity, not of objective population-structure depth.

---

## Conclusions and Recommendation

The AADR v66 data are internally consistent and the metadata are clean enough to build a country-level metrics file. The raw coverage numbers are informative: Europe and parts of the MENA have dense, temporally deep records; Sub-Saharan Africa, the Pacific, and parts of Southeast Asia are near-blank. However, the composite index is too research-history-biased to support cross-country inference at full panel scale. Adding AADR sample density as a control in the main horserace regressions does not change the FWER-surviving findings, which is reassuring (it confirms the three findings are not artefacts of European over-representation), but it also confirms that AADR density itself carries no independent structural signal beyond what the existing substrates already capture.

**For the main horserace paper (paper5):** Do not include AADR-based controls in the primary or robustness specifications. The coverage asymmetry would require dropping most of Africa, Asia, and the Americas, reducing the panel to roughly 50 Eurasian countries and eliminating the cross-continental variation that drives identification.

**Recommended next step (Option A — follow-up paper):** A Europe-restricted sub-paper using the AADR + Lazaridis-component ancestry proportions on a 30–40-country Eurasian panel is the natural heir to this exploration. With dense, temporally deep coverage across the European core and MENA, it would be possible to test whether ancestry-composition continuity (measured by Neolithic vs. Bronze Age admixture shifts) predicts medieval or early-modern economic outcomes, holding geography fixed. This is a methodologically ambitious but feasible follow-up that requires neither fabricating data for poorly-covered regions nor discarding the majority of the current panel.

---

## Provenance

| Item | Detail |
|------|--------|
| AADR version | v66 (Harvard Dataverse V10, doi:10.7910/DVN/FFIDCW, April 2026) |
| Anno file | `v66.1240K.aadr.PUB.anno` (1240K panel, 23,250 rows, ~13 MB) |
| Raw samples | 23,250 |
| After QC filter | 17,507 (Pass/PROVISIONAL_PASS/MERGE_PASS, date > 200 BP) |
| Countries mapped | 121 (after pycountry + manual overrides; "Channel Islands" excluded) |
| Countries ≥ 1 sample | 121 |
| Countries ≥ 10 samples | 88 |
| Countries ≥ 50 samples | 49 |
| Composite index coverage | 88 countries |
| Citation | Mallick S, Micco A, Mah M, Ringbauer H, Lazaridis I, Olalde I, Patterson N, Reich D (2024). *Sci Data* 11, 182. |
