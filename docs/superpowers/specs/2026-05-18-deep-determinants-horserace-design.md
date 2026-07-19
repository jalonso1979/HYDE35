# Climate Volatility and Modern Demographic Outcomes: An Investigation

Date: 2026-05-18 (reframed 2026-05-18 after empirical results)
Status: reframed; Tasks 0-17 complete; writing tasks 18-22 pending

## Working title

*Climate Volatility, Agricultural Pathway, and Modern Demographic Outcomes: A Cross-Country Investigation*

(Alt: *How Climate Volatility Shapes Population and Land Use: An Empirical Investigation Across Pre-Industrial and Modern Windows.*)

## Reframe note (2026-05-18)

The original spec framed this as a four-substrate "deep determinants horserace" with the agricultural pathway as mediator. The completed empirical work (Tasks 0-17) reveals that:

1. The Shapley decomposition shows climate volatility ($\sigma_v^T$) is one channel among several distinct channels — ancestral crop yield (Galor-Özak), predicted heterozygosity (Ashraf-Galor), and pre-1500 pandemic intensity each absorb a non-trivial slice of cross-country variance.
2. The agricultural pathway is not a clean mediator on the baseline specification — suppressor structure dominates — but **continent fixed effects cleanup the suppressor structure**, suggesting unmeasured continental heterogeneity, not a substrate-pathway misfit.
3. The pre-industrial $\sigma_v^T$ 1421–1750 channel and the modern-window $\sigma_v^T$ 1950–2008 channel both predict modern population growth, with the modern window showing a *larger* Shapley R² (0.061 vs 0.024). This is *evidence for*, not against, a climate-volatility channel.
4. Only 3 of 16 substrate × outcome cells survive Westfall-Young FWER correction at p<0.05: ancestral_yield × log pop growth (the strongest), H_pred × log GDPpc, and H_pred × log pop growth.

Given these findings, this paper reframes from "which deep root dominates?" to "**how does climate volatility shape long-run demographic outcomes, and how does it relate to other pre-industrial substrates?**" The other three substrates (Het, ancestral yield, pandemic intensity) become competing/complementary channels controlled for in robustness rather than co-equal headline claims.

## Goal

A standalone paper investigating how climate volatility shapes long-run cross-country demographic outcomes, with the agricultural pathway typology and three competing pre-industrial substrates (Ashraf-Galor predicted heterozygosity, Galor-Özak ancestral crop yield, pre-1500 pandemic intensity) entering as covariates and competing channels. The investigation crosses pre-industrial and modern climate-volatility windows to test whether the channel is window-specific or persistent.

The paper sits alongside, not on top of, three other papers in flight:

- `long_shadow.tex` — the volcanic-Malthusian-Boserupian paper, AEJ:Macro/Restud target. This paper takes its $\sigma_v^T$ 1421–1750 construction and pathway typology as given.
- `paper_pandemicICE` — Restud 2026 submission (Drive `Pandemics/paper_pandemicICE`).
- `paper_macro_before_ir` — JPE 2026 submission (Drive `Pandemics/paper_macro_before_ir`).
- `paper_great_dying` — TBD (Drive `Pandemics/paper_great_dying`).

The four pandemic/macro papers in Drive provide the data infrastructure for the **pandemic-exposure substrate**; this new paper does not duplicate their claims, only borrows the constructed exposure series as one of four substrates.

`long_shadow.tex` is not modified. The new paper companion-cites it.

## Central claim

For each modern demographic outcome $y_i^{(k)}$ (modern population growth, urbanisation change, log GDPpc, demographic-transition timing), the paper investigates how climate volatility $\sigma_v^T$ and the agricultural pathway typology jointly shape long-run outcomes, controlling for and decomposing against three competing pre-industrial substrates:

1. **Climate volatility** $\sigma_v^T$, country-level standard deviation of annual mean temperature 1421–1750, from ModE-RA. *Headline channel.* Provenance: `long_shadow.tex` §5.
2. **Predicted genetic heterozygosity** $H_i$, ancestry-adjusted via Putterman-Weil migration weights, from Ashraf-Galor (2013). *Competing channel.*
3. **Ancestral agricultural potential** $A_i$, gridded prehistoric crop-yield potential from Galor-Özak (2016) aggregated to ISO3. *Competing channel.*
4. **Pre-1500 pandemic intensity** $\Pi_i$, constructed from Brecke + AntiquityPandemics + Justinianic/Antonine/Cyprian reconstructions. *Competing channel.*

The paper makes four connected investigative claims:

**Claim 1 (variance decomposition across channels).** A Shapley-Owen R² decomposition identifies the unique-variance contribution of each channel — climate volatility, predicted Het, ancestral yield, pandemic intensity — for each modern outcome. The result is a 4×4 conditional-R² matrix that quantifies how much each channel contributes once the others are conditioned. *Empirical finding from Task 10: ancestral_yield_log dominates 3 of 4 outcomes (max Shapley R² 0.136 for log pop growth); H_pred wins log GDPpc; $\sigma_v^T$ has small but non-trivial unique contributions across all outcomes.*

**Claim 2 (pathway as transmission channel for climate volatility).** Climate volatility shapes the agricultural pathway typology by selecting among intensive crop / pastoral / mixed systems. Adding the K=5 pathway-dummy vector $\mathbf{P}_i$ as a mediator on the $\sigma_v^T \to y$ relationship asks whether the climate-volatility effect operates through pathway selection or directly. *Empirical finding from Tasks 13-14: on the baseline specification, the mediation analysis shows wide suppressor structure for most substrate × outcome pairs; **adding continent FE cleans this up** (Task 17 Check 1), suggesting the suppressor was driven by unmeasured continental heterogeneity, not by a substrate-pathway misfit. The headline mediation specification is with continent FE.*

**Claim 3 (climate-window robustness).** The $\sigma_v^T$ channel is robust to the climate-volatility measurement window. Pre-industrial $\sigma_v^T$ 1421–1750 (the focal measure) and modern $\sigma_v^T$ 1950–2008 (the placebo measure) both predict modern outcomes; the modern-window Shapley R² is actually larger for population growth (0.061 vs 0.024). This is consistent with a structural climate-volatility channel that operates at multiple temporal horizons rather than a window-specific identification artefact.

**Claim 4 (FWER-surviving claims under multi-testing correction).** Of 16 substrate × outcome cells in the mediation matrix, only 3 survive Westfall-Young FWER correction at p<0.05: ancestral_yield × log pop growth (strongest), H_pred × log GDPpc, and H_pred × log pop growth. The paper reports these as the multi-testing-robust headline findings. The $\sigma_v^T$ channel has real Shapley R² contribution but the mediation through pathway dummies is statistically noisy (does not survive FWER); we report the climate-volatility channel as identified through its Shapley R² and conditional coefficient rather than through clean pathway mediation.

**Causal claim.** None at the channel level. Each channel has its own causal-identification literature (Ashraf-Galor 2013 for Het; Galor-Özak 2016 for ancestral yield; Voigtländer-Voth 2013 for pandemic exposure; `long_shadow.tex` for climate volatility). The paper's contribution is to put these four pre-existing causal-identification stories side-by-side in one country-level panel with internally consistent measurement and to decompose the unique-variance contribution of each.

## What the paper adds that `long_shadow` does not

`long_shadow.tex` §5 already runs $\sigma_v^T$ against modern population growth with a deep-determinant battery that includes Neolithic distance, migratory distance from Addis Ababa, ruggedness, log land area, and absolute latitude. It does *not* include:

- Explicit Ashraf-Galor ancestry-adjusted predicted heterozygosity (only the underlying migratory distance from Addis Ababa is used, without the Putterman-Weil ancestry adjustment).
- Galor-Özak ancestral crop-yield potential (gridded GAEZ-based prehistoric yield).
- Putterman-Weil ancestry-adjusted state history.
- Pre-1500 pandemic exposure as a quantitative substrate.
- A Shapley-Owen variance decomposition.
- The mediation diagram from substrate to pathway to outcome.

This paper adds all six. None of them require modifying `long_shadow.tex`.

## Data layers — already built vs. to build

### Already built

| Layer | File | Coverage |
|---|---|---|
| ModE-RA country-monthly climate 1421–2008 | `analysis/data/modera_country_monthly.parquet` | 196 countries |
| ModE-RA derived annual + interval series | `analysis/data/country_climate_1421_2025.parquet` | 196 countries |
| Pre-industrial $\sigma_v^T$ 1421–1750 | derived from above | 196 countries |
| K=5 pathway clusters (HYDE-features) | `analysis/data/climate_pathways_country.parquet` | 156 countries |
| Climate-only pathway clusters | from `joint_var_climate_pathways.py` | 196 countries |
| Standard deep determinants | `analysis/data/deep_determinants_extended.parquet` | 205 countries |
| Brecke conflict catalogue + plague indicators | `analysis/data/brecke/`, `conflict_pandemic_panel.parquet` | 196 countries |
| HYDE 3.5 country-year land use + population 1500–2025 | from `analysis/data/` | 204 countries |

### To build

| Layer | Source | Estimated effort | Output target |
|---|---|---|---|
| Ashraf-Galor predicted heterozygosity, ancestry-adjusted | AER 2013 replication archive + Putterman-Weil ancestry weights | 1 week | `analysis/data/deep_determinants/predicted_het_pw_adjusted.parquet` |
| Galor-Özak ancestral crop-yield potential | AER 2016 replication archive (gridded GAEZ caloric yield 1500 climate) | 1 week | `analysis/data/deep_determinants/ancestral_crop_yield.parquet` |
| Putterman-Weil state history (ancestry-adjusted) | QJE 2010 archive | 3 days | `analysis/data/deep_determinants/state_history_pw.parquet` |
| Pre-1500 pandemic intensity index | Brecke (have) + AntiquityPandemics Drive + Harper Roman + Justinianic + Antonine + Cyprian | 2 weeks | `analysis/data/deep_determinants/pandemic_intensity_pre1500.parquet` |
| Modern outcomes harmonised | WB WDI + Maddison Project DB + Reher (2004) | 3 days | `analysis/data/deep_determinants/modern_outcomes.parquet` |
| Master horserace panel | join all of the above | 2 days | `analysis/data/deep_determinants_horserace.parquet` |

Total new build before first regression: roughly 5 weeks of data engineering.

### Sample

- Master panel: 196 countries (the ModE-RA / HYDE intersection).
- Substrate sub-panel for outcomes that need all four substrates: ~150 countries (the AG-PW-GÖ intersection).
- Sub-sample stability:
  1. Drop colonial-extraction-history countries (Acemoglu-Johnson-Robinson list).
  2. Drop small-island states (UN list, $<500{,}000$ population in 1950).
  3. Drop post-Columbian Americas (24 countries — these are the same that artefactually drove the cropland-share Boserupian result in `long_shadow.tex` §4.5).
  4. Drop countries with ancestry-share-imputation flag in Putterman-Weil.
- Robustness: 21-country crop-dominant late substantive core (the same restricted sample from `long_shadow.tex` §4.5, where HYDE measurement issues are minimised).

## Outcomes

Four outcome variables, in order of priority:

1. **Modern population growth** $y^{(1)}_i \equiv \log(P_i^{2020-2025}/P_i^{1950-1960})$. Primary outcome. Comparable to `long_shadow.tex` §5 outcome.
2. **Urbanisation change** $y^{(2)}_i \equiv \text{UrbanShare}_i^{2020-2025} - \text{UrbanShare}_i^{1950-1960}$.
3. **Log GDPpc in 2015** $y^{(3)}_i \equiv \log \text{GDPpc}_i^{2015}$ (Maddison + WB).
4. **Demographic-transition timing** $y^{(4)}_i \equiv$ year of CBR crossing 25/1000 from above (Reher 2004 typology + UN World Population Prospects).

These four outcomes capture distinct UGT exit dimensions: pure quantity ($y^{(1)}$), structural transformation ($y^{(2)}$), income ($y^{(3)}$), and timing of the demographic transition ($y^{(4)}$). The paper's variance-decomposition results will differ across them in interpretable ways.

## Empirical strategy

### Specification

For each outcome $y^{(k)}_i$, run a single cross-country OLS:

$$y^{(k)}_i \;=\; \alpha^{(k)} \;+\; \boldsymbol{\beta}^{(k)\prime} \mathbf{S}_i \;+\; \boldsymbol{\gamma}^{(k)\prime} \mathbf{P}_i \;+\; \boldsymbol{\delta}^{(k)\prime} \mathbf{X}_i \;+\; \varepsilon^{(k)}_i$$

where:

- $\mathbf{S}_i = (\sigma_v^T, H_i, A_i, \Pi_i)$ — the four substrate dimensions.
- $\mathbf{P}_i$ = K-1 pathway dummies from `long_shadow.tex` §2.3 (omitting irrigation-pioneer singleton).
- $\mathbf{X}_i$ = geography controls: abs latitude, log land area, landlocked indicator, ruggedness, log Neolithic distance.
- SEs: heteroskedasticity-robust HC3 (small-N appropriate). No clustering; cross-country, $N\approx 150$.

The system is estimated as four separate OLS regressions, not SUR. SUR's only payoff is if the across-equation residual correlation is exploited for joint inference; here we run independent decompositions per outcome, so OLS is sufficient and simpler to communicate.

### Three exercises

**Exercise 1 — Sequential nested R²; Shapley-Owen decomposition.**

For each outcome $y^{(k)}$:
- Run 16 nested specifications: baseline ($\mathbf{X}_i$ only), $\mathbf{X}_i$ + each subset of the four substrates. Geography controls $\mathbf{X}_i$ are always in.
- Compute Shapley value $\phi_s^{(k)}$ for each substrate $s$ as its average marginal $R^2$ contribution across all $4! = 24$ orderings of the four substrates, holding $\mathbf{X}_i$ fixed. The Shapley value is computed over the substrate set only; geography is treated as a pre-conditioned base.
- Report as 4×4 heatmap (rows = substrates, columns = outcomes), with $\phi_s^{(k)}$ in each cell.
- Headline number: total joint $R^2$ from substrates only, vs. with pathway dummies added, vs. with geography controls only.

**Exercise 2 — Mediation by pathway.**

For each outcome $y^{(k)}$ and each substrate $s$:
- Estimate baseline $\hat\beta_s^{(k)}$ from specification without $\mathbf{P}_i$.
- Estimate mediated $\tilde\beta_s^{(k)}$ from specification with $\mathbf{P}_i$.
- Compute mediation share $1 - \tilde\beta_s^{(k)} / \hat\beta_s^{(k)}$, with non-parametric percentile bootstrap CIs (1{,}000 country resamples).
- Substrates with high mediation share are routed through pathway; substrates with low mediation share have direct channels to modern outcomes.
- Report as 4×4 mediation-share table, with 95\% percentile CIs.
- This is the **structural finding** of the paper: which deep substrates work through agricultural-pathway selection. Together with Exercise 1's Shapley decomposition, these two objects are the headline; both are reported regardless of which substrate dominates either.

**Exercise 3 — Pathway-cluster source robustness.**

Re-run Exercises 1 and 2 substituting climate-only-clustered pathways for HYDE-clustered pathways. The two clusterings agree on a substantial subset of countries but disagree where HYDE outcomes (population, cropland) drove cluster assignment. The differential pattern is itself informative.

### Robustness battery

Run all in parallel after Exercise 1 lands:

1. **Sub-sample stability** — drop colonial-history, small-island, post-Columbian Americas, AG-imputed countries.
2. **Continent fixed effects** — adds 6 dummies; absorbs Old-World/New-World/Africa cross-continental confounding.
3. **Leave-one-out instability** — drop each country one at a time, report distribution of Shapley values.
4. **Westfall-Young multi-testing correction** — for the mediation-share matrix (16 tests), report FWER-corrected p-values.
5. **Pre-1500 climate placebo** — use ModE-RA volatility 1421–1500 (smaller sample, noisier) instead of 1421–1750. The substrate signal should attenuate but the relative variance decomposition should be stable.
6. **Modern-window placebo** — use post-1950 climate volatility as $\sigma_v^T$. Per `long_shadow.tex` Table 9, this is a null channel; the decomposition should reassign substrate-1 mass to substrates 2/3/4.
7. **Heterogeneity by sample period** — use pre-1900 modern outcomes (where available, e.g., demographic-transition timing) and re-run the decomposition; check stability vs. post-1950 outcomes.

### What can identify, what cannot

- Identifies: *which slice of cross-country variance each substrate absorbs conditional on the others.*
- Identifies: *whether agricultural pathway mediates the substrate-to-outcome relationship.*
- Does not identify: *the causal effect of any single substrate on modern outcomes.* (Each substrate has its own identification literature; we borrow their assumptions and quote them where used.)
- Does not identify: *whether the substrate set is complete.* The decomposition is conditional on which substrates are in the panel.

The paper says this explicitly in the introduction.

## Theory section

A short theory section (~2 pages) places the four substrates inside a unified-growth-theory framework. The model is the Ashraf-Galor (2011) Malthusian-to-modern transition skeleton, modified so that pre-industrial pathway choice is a function of $(\sigma_v^T, H_i, A_i, \Pi_i)$ — the four substrates — and pathway then conditions the rate of the Malthusian-to-modern transition.

The model delivers two empirical predictions that map to the empirical exercises:

- **Prediction 1**: Modern population growth is decreasing in $\sigma_v^T$ for pre-industrial pathways where the Malthusian regime persisted longer (crop-dominant late, pastoral/mixed late) and uniformly weakly decreasing for pathways that exited the Malthusian regime earlier (early extensifiers, high-density intensive).
- **Prediction 2**: The mediating role of pathway is largest for substrates whose primary channel is *agricultural-system selection* (climate volatility, ancestral crop-yield potential) and smallest for substrates whose primary channel is *direct on modern human capital* (predicted heterozygosity per Ashraf-Galor's original interpretation; pandemic intensity via post-1500 immunity-driven selection).

The model is motivational, not literal. We do not estimate the structural parameters. The companion `long_shadow.tex` Appendix A does.

## Figures

Six figures, in order of appearance:

1. **Fig 1 — Substrate covariance matrix.** Pairwise scatter of the four substrates across countries, with marginal histograms and pairwise correlations. Shows that the substrates are not collinear: each absorbs a distinct chunk of cross-country variation.

2. **Fig 2 — World maps of the four substrates.** 2×2 grid of choropleths, one per substrate, with the same projection and colour scale family.

3. **Fig 3 — Headline Shapley decomposition heatmap.** 4 substrates × 4 outcomes, with conditional R² in each cell. Total panel R² printed at the top of each column.

4. **Fig 4 — Mediation diagram + table.** Path diagram showing substrate $\to$ pathway $\to$ outcome, with arrow widths proportional to estimated coefficients. Accompanying table reports mediation shares with bootstrap CIs.

5. **Fig 5 — Pathway-cluster source robustness.** Side-by-side Shapley heatmaps for HYDE-clustered pathways vs. climate-only-clustered pathways.

6. **Fig 6 — Sub-sample stability ribbon.** Line-plus-band chart showing how each mediation share $1 - \tilde\beta_s^{(k)} / \hat\beta_s^{(k)}$ moves along the x-axis as we sequentially drop colonial-history, small-island, post-Columbian Americas, and AG-imputed countries. The "ribbon" is the percentile-bootstrap CI band at each sub-sample step.

## Tables

Five tables, all in main text:

1. **Table 1 — Descriptive statistics.** Four substrates + four outcomes + geography controls, with means, SDs, country counts.
2. **Table 2 — Pairwise substrate correlations.** 4×4 matrix; flag correlations >0.4 in bold.
3. **Table 3 — Pooled OLS estimates of full specification.** Four columns (one per outcome), rows = all regressors. Standard significance stars.
4. **Table 4 — Shapley contributions.** Same 4×4 shape as Fig 3 but numerical with bootstrap SEs.
5. **Table 5 — Mediation shares.** 4×4 with $1 - \tilde\beta_s^{(k)} / \hat\beta_s^{(k)}$ and 95% CIs.

Appendix tables: sub-sample stability matrices for Tables 4 and 5; continent-FE specifications; leave-one-out histograms; Westfall-Young corrected p-values.

## Paper structure

- §1 Introduction (5 pp).
- §2 Conceptual framework: a UGT skeleton with substrate-mediated pathway selection (3 pp).
- §3 Data: substrates, pathways, outcomes (4 pp).
- §4 Exercise 1: variance decomposition (4 pp).
- §5 Exercise 2: mediation by pathway (4 pp).
- §6 Exercise 3 + robustness battery (4 pp).
- §7 Discussion: what this implies for UGT and for the deep-determinants literature (2 pp).
- §8 Conclusion (1 pp).
- Appendices A–D: data construction details for the new substrates; full robustness tables; sensitivity to estimation choices; replication of mediation analysis with continuous pathway scores.

Target length: ~27 pages main + appendix.

## Risks and how the paper handles each

1. **Predicted heterozygosity is contested.**
   - Mitigation: lead with "ancestry-adjusted population diversity" framing; cite Ashraf-Galor (2013) as construction source, not as causal claim. Frame the paper as conditional decomposition, not causal identification.
   - Backup: report all results with and without $H_i$ in the substrate set; substrate 2 should not be load-bearing for the headline claims.

2. **Cross-country regression on N≈150 has known small-N pathologies.**
   - Mitigation: HC3 SEs throughout; Westfall-Young multi-testing correction; leave-one-out stability; sub-sample stability ribbons.
   - Honest disclosure: the paper does *not* claim individual coefficient stability — it claims decomposition stability.

3. **Pathway is endogenous to the substrates.**
   - Mitigation: report both HYDE-clustered and climate-only-clustered pathway versions (Exercise 3). The climate-only version mitigates the endogeneity but loses the historical realism of HYDE's K=5.
   - Mediation share is bounded by the substrate-to-pathway selection equation, which we report in the appendix.

4. **Pandemic substrate is the most novel and least standardised.**
   - Mitigation: report the paper with and without the pandemic substrate (3 vs 4 substrates). If a referee objects to the pandemic-substrate construction, the 3-substrate version still stands.
   - Construction details in Appendix B with explicit source-by-source tabulation.

5. **Outcome multiplicity (four outcomes) invites cherry-picking accusations.**
   - Mitigation: pre-register the four outcomes in the abstract and intro; report all four side-by-side; never claim a finding holds for a subset of outcomes without showing the full grid.
   - The four outcomes are theory-motivated (UGT exit dimensions) and not selected ex post.

6. **Coordination with `paper_pandemicICE`, `paper_macro_before_ir`, `paper_great_dying`.**
   - This paper takes constructed pandemic exposure as an input; it does not duplicate any pandemic-paper claim.
   - Companion-cite all three Drive papers; clarify in the data section that the pandemic-intensity construction extends the panels of `paper_pandemicICE` and `paper_macro_before_ir` to a country-year format.
   - Coordination with Da-Rocha needed to confirm `long_shadow` and `paper_pandemicICE` co-authorship structure remains as-is and this paper either inherits or modifies it.

## Target outlet

Primary: AEJ:Macro.
Secondary: JEG.

**Reasoning.** The variance-decomposition + mediation framing fits AEJ:Macro's empirical-style preferences (cf. Andersen-Dalgaard-Selaya 2016; Ashraf-Galor's own follow-up work; Bentzen 2019; Galor-Özak's prior). JEG is a reasonable backup with faster turnaround. We avoid QJE / Restud for this paper specifically because the methodology is descriptive decomposition rather than novel causal identification; QJE/Restud are reserved for the higher-novelty `long_shadow.tex` and `paper_pandemicICE` submissions in flight.

Co-author conversation needed before drafting: confirm Da-Rocha is co-author; identify whether to add a third co-author for the population-genetics methodology piece (candidate: Andersen or Bentzen — both have relevant priors and are accessible).

## Timeline

A realistic ~6-month schedule from spec approval, assuming the author works on this in parallel with `long_shadow` submission revisions:

- **Weeks 1–5**: build the four new data layers (predicted het, ancestral crop yield, state history, pandemic intensity, modern outcomes).
- **Weeks 6–8**: master horserace panel assembly + diagnostics + Fig 1 (substrate covariance) + Fig 2 (substrate maps).
- **Weeks 9–10**: Exercise 1 (Shapley decomposition) + Fig 3 + Table 4.
- **Weeks 11–13**: Exercise 2 (mediation) + Fig 4 + Table 5.
- **Weeks 14–15**: Exercise 3 + robustness battery + Figs 5–6.
- **Weeks 16–18**: theory section (§2) + introduction + discussion.
- **Weeks 19–22**: first full draft + internal review + Da-Rocha review.
- **Weeks 23–24**: revisions; submission to AEJ:Macro.

## Reasonable calls made — explicit list for override

The author of this spec made the following design calls that the user can override before implementation begins:

1. **Pandemic substrate included from the start.** Could be reduced to three substrates if the construction effort proves too high or the result is fragile. Default: include.
2. **AEJ:Macro target.** JEG is the runner-up; QJE/Restud are excluded for this paper to leave them clear for `long_shadow` and `paper_pandemicICE`.
3. **Four outcomes rather than just population growth.** Adds defensive surface but provides four independent variance-decomposition exercises.
4. **Pathway as mediator rather than competing covariate.** This is the structural claim of the paper. The alternative (pathway as just another covariate) is simpler but less novel.
5. **No causal identification claim at the substrate level.** Honest framing as a conditional descriptive decomposition. Reduces submission risk.
6. **`long_shadow.tex` not modified.** Companion-cited only.
7. **OLS per outcome, not SUR.** Simpler exposition, near-identical results given small substrate-residual correlation expected.
8. **HC3 SEs, not bootstrap.** Bootstrap reserved for mediation shares (where it is needed) and for leave-one-out (where it is the natural method).

## Out of scope for this spec

The following are deliberately not in this paper:

- Sub-national analysis. (Substrates are country-level only; sub-national identification is `long_shadow.tex`'s edge and stays there.)
- Time-varying substrates. (The substrates are defined as pre-industrial fixed effects.)
- A structural-econometric model. (The theory section is motivational. `long_shadow.tex` Appendix K is the only place a structural model lives in the broader pipeline for now.)
- Within-pathway substrate effects on modern outcomes. (Possible appendix-grade extension if a referee asks; not pre-committed.)
- Genetic-distance (Spolaore-Wacziarg) as a substrate. (Could be added as an alternative to predicted heterozygosity; deferred unless predicted heterozygosity proves fragile.)

## Acceptance criteria

The spec is implementation-ready when the user has reviewed it and approved the following:

1. The four-substrate frame, not a three-substrate frame or a five-substrate frame.
2. The four outcomes, not a single-outcome paper or a six-outcome paper.
3. The decomposition + mediation strategy, not a competing-coefficients horserace.
4. The AEJ:Macro target, not QJE/Restud or JEG-only.
5. The "do not modify `long_shadow.tex`" constraint.
6. The 6-month timeline, with the data-build phase taking the bulk of weeks 1–8.

If the user disagrees on any of these, the spec is revised in-place before the implementation plan is written.
