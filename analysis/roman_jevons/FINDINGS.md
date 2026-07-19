# Roman antiquity experiments — findings note

**For potential integration into the *Jevons Visits Rome* / *Pandemics Written on Ice* project.**

Data: `roman_v2_panel.csv` from the Pandemic project (889 years, 86 BCE — 802 CE).
Activity proxy: Greenland ice-core lead (`lead_z`), a proxy for Roman-era mining and smelting (Hong et al. 1994; McConnell et al. 2018).
Treatments tested: volcanic forcing, pandemic intensity (pooled and by family), war intensity.

All scripts and outputs are in `/Volumes/BIGDATA/HYDE35/analysis/roman_jevons/`.

---

## Summary

The cleanest single statement: **pandemics and wars contracted Roman economic activity, but volcanic forcing—after controlling for them—does not separately predict lead-Z**. Family-specific dating exercises reveal that the Antonine and Cyprian events have severe pre-trend problems (Roman activity was already declining before them), while Justinianic is the cleanest candidate event.

---

## (1) Pooled dynamic event-studies

### Volcanic forcing → lead-Z, leads/lags h ∈ {−5, …, +10}

| Diagnostic | Value |
|---|---|
| N | 829 |
| Parallel-trends F (h < 0) | F = 1.61, p = 0.156 → **PASS** |
| Notable coefficients | Mostly null. Scattered significance at h = +5 (β = −0.047, p = 0.009) and h = +6 (β = +0.017, p = 0.037). |

Volcanic forcing alone shows no systematic effect on Roman lead-Z. Parallel trends pass.

### Pandemic intensity (pooled) → lead-Z

| Diagnostic | Value |
|---|---|
| N | 829 |
| Parallel-trends F (h < 0) | **FAILS** — h = −5 (β = −0.045, p = 0.032) and h = −1 (β = −0.042, p = 0.050) significant |
| Post-event peak | Pattern of mild post-event negative coefficients (e.g. h = +1: β = −0.021), but pre-trends dominate |

The pooled pandemic IRF fails parallel trends. Roman activity was already declining before pandemics on average.

## (2) Family-specific IRFs

| Pandemic | Parallel-trends F | p | Verdict |
|---|---:|---:|---|
| Antonine (~165–180 CE) | 11.36 | < 10⁻¹⁰ | **FAIL** — strong pre-trends |
| Cyprian (~249–262 CE) | 5.37 | < 10⁻⁴ | **FAIL** — strong pre-trends |
| Justinianic (~541 onward) | 1.61 | 0.155 | **PASS** in simple spec |

### Interpretation of the family-by-family pattern

The Antonine and Cyprian leads suggest one of:
1. **Pre-existing economic decline** that may have set the stage for pandemic outbreaks (reverse causality at long horizons).
2. **Dating uncertainty** in the lead-Z proxy and/or the pandemic-onset dummies.
3. **Entanglement with other Roman crises** (e.g., Marcomannic Wars overlap with Antonine; Crisis of the Third Century overlaps with Cyprian).

Justinianic stands out as the cleanest event for causal interpretation. This matters substantively for the Pandemics paper: arguments about "pandemic effects" on Roman activity should weight Justinianic more heavily than Antonine or Cyprian.

## (3) Joint static specification

Lead-Z regressed on pandemic intensity + volcanic + war intensity + temperature, HAC standard errors.

| Variable | β | SE | p |
|---|---:|---:|---:|
| Pandemic intensity | **−0.120** | 0.040 | **0.003** |
| Volcanic forcing | +0.003 | 0.014 | 0.816 |
| War intensity | **−0.066** | 0.033 | **0.044** |
| Temperature | −0.037 | 0.057 | 0.512 |

R² = 0.020, N = 837. Pandemics and wars matter; volcanic forcing does not separately.

## (4) Justinianic IRF with controls (volcanic + war)

Extending the lead/lag window to h ∈ {−10, …, +20} and adding controls. Parallel-trends F = 2.23, p = 0.015 — **fails** at the 10% threshold once the additional controls absorb part of the apparent pre-trends-passing in the simpler spec.

Notable post-event coefficients:
- h = +12: β = −0.033, p = 0.005
- h = +15: β = −0.027, p = 0.031

A scattered pattern of negative coefficients 12-15 years post-541 CE, but the parallel-trends violation is concerning.

## (5) Decade-level volcanic IRF

Aggregating annual data to decade bins, replicating the Long-Shadow paper's Sigl-style approach. N = 84 decades.

Parallel-trends F = 0.85, p = 0.47 → **PASS**.
All coefficients individually insignificant. Volcanic forcing has no systematic decade-level effect on Roman lead-Z.

---

## Implications for the Jevons-Visits-Rome project

1. **Headline finding for the Pandemics paper**: Pandemic intensity reduces Roman lead-Z by 0.12 SD per unit intensity (p = 0.003) in a joint specification that holds volcanic and war constant. This is the clean welfare-on-activity number.

2. **Robustness concern to address**: Antonine and Cyprian families fail parallel-trends pretests. The paper should either (a) report family-specific IRFs and note the pre-trends explicitly, (b) restrict the headline to Justinianic, or (c) instrument the pandemic dummies (with ice-core dust-veil signals, for instance, since plague spread and volcanic cooling were sometimes linked).

3. **Volcanic-channel null is itself informative**: After controlling for pandemic and war, volcanic forcing does not separately predict Roman lead-Z. This contrasts with our Long-Shadow paper finding that volcanic VSSI 1500–1900 strongly predicts decade-level pop growth (p < 10⁻⁹). Two possible explanations: (i) the Roman lead-Z measures a narrower margin (mining/smelting activity) than population dynamics; (ii) the classical-era ice-core volcanic record is too sparse to identify decade-level effects in this proxy.

4. **War effects deserve their own treatment**: War intensity comes in at p = 0.04 in the joint spec. The Pandemics project could add a war-vs-pandemic horse-race section building on this.

5. **Possible extension**: cross-site pollen/charcoal panel using `roman_multiproxy_site_meta.csv` (1,757 sites, lat/lon, pandemic-family tags). Would allow geographic heterogeneity in the pandemic response, parallel to the within-country sub-national identification in the Long Shadow paper.

---

## (6) Robustness — CHRE coin-hoards as a parallel-trends control

`01b_dynamic_event_studies_with_hoards.py` adds the Coin Hoards of the Roman Empire (CHRE; 18,310 dated hoards, 25-year bins forward-filled to annual) as an additional time-varying covariate. Two variants tested: empire-wide pooled log1p hoards (`hoards_empire_z`) and Italy-region z-score (`hoards_italy_z`). Hypothesis: pre-existing decline before Antonine/Cyprian could partly reflect monetary-crisis hoarding cycles that the lead-Z proxy also picks up.

| Treatment | Baseline F (p) | + empire hoards F (p) | + Italy hoards F (p) | + both F (p) | Verdict |
|---|---:|---:|---:|---:|---|
| Volcanic | 1.61 (0.16) | 1.45 (0.20) | 1.34 (0.24) | 1.91 (0.09) | unchanged (PASS) |
| Pandemic intensity (pooled) | 1.64 (0.15) | 2.19 (0.05) | 2.63 (0.02) | 2.04 (0.07) | **degrades** |
| Antonine | 11.36 (10⁻¹⁰) | 10.70 (10⁻¹⁰) | 10.90 (10⁻¹⁰) | 9.68 (10⁻⁹) | **still FAIL** |
| Cyprian | 5.38 (10⁻⁴) | 4.38 (6×10⁻⁴) | 3.83 (2×10⁻³) | 3.53 (4×10⁻³) | **still FAIL** |
| Justinianic | 1.61 (0.16) | 1.88 (0.09) | 2.25 (0.05) | 1.84 (0.10) | **degrades** |

**Headline**: coin-hoards do NOT rescue the Antonine/Cyprian parallel-trends failures — the pre-trends are not just a co-moving monetary-crisis artifact. Antonine F drops only from 11.4 to 10.7 (still p < 10⁻¹⁰); Cyprian from 5.4 to 4.4 (still p < 10⁻³). Adding hoards actually **degrades** the Justinianic spec from a clean PASS to borderline (p = 0.095), suggesting hoards co-vary with Justinianic-era activity in ways that absorb event-time variation. Empire-wide hoards spike at year-bin 250 (z = +1.25) coincident with the Crisis of the Third Century, so the proxy is doing real work — it just doesn't explain the lead-Z pre-trends.

**Substantive implication**: argument (a) in §5.2 ("report family-specific IRFs and note pre-trends explicitly") becomes the cleanest path. The pre-existing decline before Antonine/Cyprian appears to be a real economic phenomenon, not a coinage-crisis artifact. For the Pandemics paper, restricting headline causal claims to Justinianic (the parallel-trends survivor in the baseline spec) remains the conservative move.

**Output**: `irf_parallel_trends_with_hoards.csv` (20 rows: 5 treatments × 4 specs).

---

## (7) Robustness — volcanic dust-veil IV for pandemic timing

`03_volcanic_iv_pandemics.py` (annual) and `03b_volcanic_iv_decade.py` (decade-level) implement FINDINGS.md §5.2 option (c): instrument pandemic intensity with lagged volcanic forcing, motivated by the historical literature linking 536 CE dust-veil eruptions to the Justinianic plague onset (Büntgen et al. 2016; Harper 2017).

| Spec | First-stage F | OLS β (SE) | 2SLS β (SE) | Verdict |
|---|---:|---:|---:|---|
| Annual (volc lags 5/7/10 as IV) | 0.11 | −0.122 (0.042) | −2.50 (5.25) | weak IV; uninformative |
| Decade (volc lags 1/2/3 dec as IV) | 1.37 | −0.033 (0.014) | −0.027 (0.071) | weak IV; sign matches |

**Headline**: the volcanic-pandemic link historians describe is **not** a systematic statistical first-stage relationship across 86 BCE–802 CE. At annual scale, none of the volcanic lags 5/7/10 individually predicts pandemic intensity (all p>0.6). At decade scale, the 30-year lag is marginally significant (p=0.045) but the joint F is 1.37, far below Stock-Yogo's 10. The 2SLS point estimates have the right sign but unidentifiably wide confidence intervals.

**Substantive implication**: the IV approach is not a viable rescue for the parallel-trends concern. Option (a) — report family-specific IRFs and note pre-trends explicitly — remains the cleanest path. The Buentgen 2016 / 536 CE story may reflect a single anecdotal coincidence rather than a systematic mechanism.

**Outputs**: `iv_pandemic_volcanic.csv`, `iv_pandemic_volcanic_decade.csv`.

---

## Files in this directory

```
01_dynamic_event_studies.py             ← scripts 1, 2, 3 above
01b_dynamic_event_studies_with_hoards.py ← script 6 (CHRE coin-hoards covariate)
02_joint_and_justinianic.py             ← scripts A, B, C above
03_volcanic_iv_pandemics.py             ← script 7 (annual volcanic IV)
03b_volcanic_iv_decade.py               ← script 7 (decade volcanic IV)
irf_volcanic_lead.csv                   ← (1) IRF table
irf_pandemic_lead.csv                   ← (2) IRF table
irf_pandemic_by_family.csv              ← (3) family-specific IRFs
irf_parallel_trends_with_hoards.csv     ← (6) hoards-augmented F-stats
iv_pandemic_volcanic.csv                ← (7) annual IV summary
iv_pandemic_volcanic_decade.csv         ← (7) decade IV summary
joint_static.csv                        ← (A) joint specification
irf_justinianic_long.csv                ← (B) Justinianic with controls
irf_volcanic_decade.csv                 ← (C) decade-level
figures/
  fig_irf_volcanic.pdf
  fig_irf_pandemic.pdf
  fig_irf_pandemic_by_family.pdf
  fig_justinianic_long_irf.pdf
```
