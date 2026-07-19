# Long Shadow paper: tighten, extend, retitle — design spec

Date: 2026-05-16
Status: approved, in execution

## Goal

After the Appendix L robustness audit (commits 5f51b1e, f9b331b, 6017922) reframed the paper's central claim from "three-margin cross-pathway asymmetry" to "demographic margin + Boserupian intensification + long shadow", four follow-up tasks remain to make the paper submission-ready:

1. Add four new figures that sell the database quality and the central findings visually.
2. Extend the calibrated structural model in Appendix K from one equation (demographic) to three (demographic + cropland share + urban share), with a new "Boserup-shutdown" counterfactual that puts a number on the welfare value of the intensification response.
3. Shorten the abstract from ~280 to ~155 words.
4. Trim the full paper from 52 to ≤40 pages by compressing appendices and moving the discrete-event volcanic study out of the main text.

## Plot designs

### Fig 1: source-coverage timeline (new)

Horizontal-bar chart, one bar per data source, x-axis is calendar year. Sources to show, in chronological start-year order: Harper Roman-Egypt wheat prices (45–650 CE), Allen-Nuffield wage-and-price archive (1259–1914), eVolv2k volcanic record (500 BCE–1900, but truncate display at 500 CE for legibility), Brecke pre-1400 European conflicts (900–1402), HYDE 3.5 land use & population (we show the 1500–2025 portion, with a note that HYDE extends to 10,000 BCE), Brecke conflict catalogue v18 (1400–1999), ModE-RA paleo-reanalysis (1421–2008), CamPop family reconstitution (1538–1851), CRU climatology (1901–1950), HMD/HFD vital statistics (Sweden 1751–, others later), ERA5 reanalysis (1950–2025). Place as new figure right after the §2 opening paragraph as Fig 1 of the data section.

### Fig 2: famine cross-validation strip (new)

5 small-multiples in one row (or 5 panels in a 1×5 grid). One panel per documented famine, each showing the affected country's annual T anomaly (and optionally P anomaly as a second line) for a 30-year window centered on the famine: Great Frost 1709 (France/UK), Tambora year-without-a-summer 1816 (Switzerland/Germany), Irish Famine 1845–49 (Ireland/UK), North China Famine 1876–79 (China), Bengal Famine 1943 (Bangladesh/India). Highlight famine years with a vertical band. Place after the calibration figure in §2.2 Validation.

### Fig 3: Tambora propagation heatmap (new)

4×4 grid of world maps showing monthly T anomalies from June 1815 to September 1816 (16 months). Use the ModE-RA monthly data. Common diverging colorbar (cool blue, warm red), maybe ±3°C scale. Place in §2.1 or §2.2 as the "wow figure" of the data section, demonstrating that ModE-RA's monthly resolution lets us watch an event propagate spatially.

### Fig 4: long-shadow choropleth (new)

World choropleth colored by pre-industrial σ_v^T 1421–1750 (the long-shadow predictor). Overlay modern population growth 1955–2020 as either a second color layer (e.g., diverging hatch) or as numeric labels on selected countries. Place at the start of §5 as the visual hook for the long-shadow finding.

## Structural model extension (Appendix K)

### Current form (1 equation)

```
Δ_ann ln N_{it} = α_τ + β_τ(t) · d̃_{it} + γ_τ · T̃_{it} + δ_τ · σ̃^T_{it} + ε_{it}
```

with `β_τ(t) = β_{0,τ} + η_τ · (t - 1421) / 100`.

### Extended form (3 equations)

For each pathway τ and country i in pathway τ, jointly:

```
Δ_ann ln N_{it} = α^N_τ + β^N_τ(t) · d̃ + γ^N_τ · T̃ + δ^N_τ · σ̃^T + φ^N_τ · V_{it} + ε^N
Δ_ann ln s_{it} = α^s_τ + β^s_τ(t) · d̃ + γ^s_τ · T̃ + δ^s_τ · σ̃^T + φ^s_τ · V_{it} + ε^s
Δ_ann ln u_{it} = α^u_τ + β^u_τ(t) · d̃ + γ^u_τ · T̃ + δ^u_τ · σ̃^T + φ^u_τ · V_{it} + ε^u
```

Pathway-specific coefficients sourced directly from the joint VAR pathway-stratified estimates (already computed in `joint_landuse_var_results.parquet`).

### New parameter table

Replace Table K's 5-column (α, β_0, η, γ, δ) per-pathway table with a 3-block table: one block of 5 parameters per equation, four pathways × three equations = 60 parameters. Compact form: pathway-by-pathway 3-row, 5-column blocks.

### Counterfactuals (updated)

- **CF1 pathway reassignment**: simulate pastoral/mixed-late countries under high-density intensive parameters across all 3 equations. Report population, cropland-share, and urban-share trajectories.
- **CF2 no volcanism**: set V_{it} = 0 in eruption decades, re-simulate all 3 equations. Report population, cropland, urban outcomes per pathway.
- **CF3 Boserup-shutdown** (NEW, replaces CF3 volatility-decoupling): force φ^s_{crop-dominant late} = 0 (no cropland response to volcanic forcing), keep all other coefficients intact, re-simulate. Report: how much extra cumulative population loss do crop-dominant late countries suffer in the absence of the Boserupian cropland-expansion response? Expected sign: positive (more pop loss without intensification).

Welfare counterfactuals via Allen wages: extend to a 2-channel welfare measure that combines the population path and the cropland path. The cropland path is a proxy for agricultural-output gain; an honest version says per-capita welfare moves by `β_pop · Δlog N + β_crop · Δlog s` if we assume cropland share is a sufficient statistic for agricultural-output-per-capita changes.

### Narrative honest framing

Keep the "this is calibration, not deep structural estimation" framing. Add a note that the 3-equation model is silent on within-decade dynamics (HYDE decadal resolution) and on the urbanization-as-exit channel beyond a reduced-form coefficient.

## 40-page trim plan

Current: 52 pages. Target: ≤40 pages. Cuts to apply:

| Cut | Pages saved |
|---|---|
| App. A two-stage framework 4→2 pp | -2 |
| App. C.3 medieval mortality 2→0.5 pp | -1.5 |
| App. F dynamic IRF fold into App. C | -1 |
| App. B appendix figures inline with relevant appendix | -1 |
| App. G prevpos pathway+modern compact-table | -1 |
| App. J deep determinants compact-table only | -1 |
| §3.2 prose tighten (post-coauthor-share orphans) | -1 |
| §4.2 discrete event study move to App. F | -2 |
| §5 placebo table compact inline | -1 |
| Total | ~12 |

New figures add ~2 pages, so net target: 40–42.

## Abstract (replacement, ~155 words)

```
We assemble a new monthly paleo-economic panel covering 196 countries from
1421 CE — ModE-RA × HYDE × eVolv2k × Brecke × Allen — and use it to identify
three findings about the pre-industrial Malthusian trap. First, continuous
volcanic forcing 1500–1900 produces a pathway-heterogeneous demographic
response, with crop-dominant late and pastoral/mixed late countries losing
population at −9.7 and −5.5×10⁻⁵ per Tg of stratospheric sulfur injection
respectively, Bonferroni-significant across multiple-testing and robustness
slices. Second, the same crop-dominant late countries expanded cropland share
after cold shocks in the modern instrumental era (+7.8×10⁻⁴ per Tg, p<0.01),
the first cross-country pre-industrial test of the Boserupian
intensification hypothesis using exogenous climate forcing. Third,
pre-industrial inter-annual climate volatility 1421–1750 is the single
strongest cross-country predictor of modern population growth at R²=0.45,
surviving latitude and deep-determinant controls. Seasonality, properly
measured, casts a long shadow.
```

## Execution order

1. Build the four figures in parallel (independent).
2. Extend structural model and rewrite App. K narrative.
3. Apply 40-page trim cuts.
4. Replace abstract.
5. Compile and verify.
6. Commit in a single coherent commit.
