# Long Shadow paper — coauthor brief, 2026-05-17

**For:** José-María Da-Rocha (RGEA, Vigo)
**From:** Jorge Alonso Ortiz (ITAM)
**Re:** Substantive restructure since the Boserup-led version you reviewed on 2026-05-16

## TL;DR

The Boserup-led restructure I sent you yesterday turned out not to survive sample stress tests. I demoted Boserup back to a supporting finding, restored the Malthus-led framing, but in the process discovered something more interesting: HYDE's land-use back-projection has a previously-unsurfaced two-axis identification problem, and the KK10 reconstruction (which is population-independent) surfaces a small positive Boserupian signal that HYDE obscures. Net: paper is now at 49 pages, three substantive findings + one methodological contribution, and is in better shape for the AEJ:Macro / Restud push than the Boserup-led version was. **I would like your sign-off before sharing further or submitting.**

## What changed and why

**Starting position (2026-05-16):** Boserup-led, +7.8e-4 cropland-share coefficient as the headline, 1800-1900 sub-sample driving identification.

**Round of measurement-robustness diagnostics I ran today:**

1. **Drop 24 small-island / tropical-Africa countries from the crop-dominant late cluster.** These countries have HYDE colonial-transition encoding in their 1840, 1850, and 1890 cropland-share trajectories (sample mean Δlog s of -0.027 in 1840, with -0.26 outliers). The headline 1800-1900 coefficient drops from +7.8e-4 (p=0.003) to +1.1e-5 (p=0.24) — vanishes on the substantive 21-country Mediterranean / Eurasian-steppe / SE-Asia core.

2. **Substitute cropland share with absolute cropland area (km²).** The share-coefficient is decomposable into cropland-area change minus grazing-area change. On the full sample, log cropland area actually *contracts* under volcanic forcing (-9.6e-5, p=0.077), while log grazing area contracts 70× faster. The cropland-share rises only because the denominator falls faster than the numerator — this is *grazing abandonment*, not Boserupian intensification.

3. **Pop-weighted estimation.** The headline coefficient shrinks from +7.8e-4 to +6.1e-5 (p=0.10) under population weighting — a 13× attenuation. Small countries were driving the headline.

The Boserup-led framing didn't survive any of these stress tests, so I demoted it.

## What I found instead

**A. HYDE has a two-axis identification problem.** The share specification fails for denominator-instability reasons (1). The area specification fails for a different reason — it's *fully mediated by the demographic margin*. Adding Δlog P as a control collapses the area coefficient to zero in every pathway, every window, on both full and core samples. The reason is mechanical: HYDE's pre-1850 reconstruction uses population density as an input variable, so HYDE-land follows HYDE-people by construction.

**B. KK10 cross-validation surfaces the actual Boserupian signal.** KK10 (Kaplan et al. 2011) is methodologically population-independent. On the 21-country crop-dominant late core post-1700, the KK10 anthropogenic-area coefficient is **+1.07e-5 per Tg, p=0.005, and survives the Δlog P control essentially unchanged** (+1.08e-5). On the 16-country high-density intensive pathway it's **+4.3e-5 per Tg, p=0.047**, also surviving pop-control — 4× larger.

The Boserupian channel exists at pre-industrial cross-country scale, but its magnitude is roughly two orders of magnitude smaller than the HYDE-cropland-share artefact had originally implied. A Tambora-class decade adds ~0.32% to cumulative anthropogenic land in the crop-dominant late core, ~1.3% in high-density intensive.

**C. Channel substitution: pathway-level pattern, country-level null.** The pathway-stratified KK10 results trace a visible substitution pattern: high-density intensive (small demographic, large Boserupian) sits opposite crop-dominant late core (large demographic, small Boserupian). I escalated this to country-level (116 countries with both coefficients estimable) and ran an errors-in-variables Deming regression of φ^L on φ^P. The slope is +0.007 (not negative), bootstrap 95% CI [-0.003, +0.19], 91% positive. The cross-country substitution slope is NOT identified. I report the pathway-level pattern as *descriptive* rather than as identification, with the country-level null surfaced honestly.

## Current paper structure (49 pages)

- §1 Intro: three findings — pathway-heterogeneous demographic margin, long shadow, HYDE Boserupian null + KK10 small-positive cross-validation
- §2 Data: the new paleo-economic panel (unchanged)
- §3 Identification: storage-demand and Malthusian regression (unchanged)
- §4 Volcanic forcing and the Malthusian channel:
  - §4.1 Pathway-heterogeneous demographic margin (joint VAR, the headline)
  - §4.2 Single-equation cross-checks + pre-XV-century extension
  - §4.3 Welfare evidence (Allen wages)
  - §4.4 Preventive and positive Malthusian checks
  - §4.5 Cropland response: an honest null + KK10 cross-validation + channel-substitution exercise
- §5 The long shadow on modern outcomes (compressed)
- §6 Robustness
- §7 Conclusion
- Appendices A-L unchanged in structure; App. I (calibrated model) reframed to note the CF3 cropland counterfactual is partial-equilibrium arithmetic of a measurement-driven coefficient, not a structural identification of a Boserupian channel.

## What I need from you

1. **Sign-off on the demoted-Boserup framing.** The 2026-05-16 restructure was Boserup-led. I've now reverted to Malthus-led with the KK10 cross-validation as the substantive Boserupian payoff. This is closer to the option-1 framing you saw before that restructure. Are you comfortable with this direction for the AEJ:Macro / Restud push?

2. **Read §4.5 (honest null + KK10) and the new Table 8 + Table 9 + Table 10 + Table 11 in particular.** The two-axis identification critique of HYDE-cropland regressions is the methodological contribution that I think gives the paper venue value at top-5. I want to make sure I'm not overclaiming.

3. **Cover letters drafts: Restud and AEJ:Macro.** Drafts attached as `paper/cover_letter_restud.tex` and `paper/cover_letter_aejmacro.tex`. Please review tone and substance.

4. **Reference suggestions.** Suggested referee lists in each cover letter — replace any names you think are inappropriate or add others.

## Files to look at

- `paper/long_shadow.tex` (49 pages, current draft)
- `paper/long_shadow.pdf` (rendered)
- `paper/cover_letter_restud.tex` + `.pdf`
- `paper/cover_letter_aejmacro.tex` + `.pdf`
- `analysis/paper4_shadow/boserup_cropland_area.py` (the alternative-outcome decomposition)
- `analysis/paper4_shadow/kk10_orthogonality.py` (KK10 cross-validation)
- `analysis/paper4_shadow/kk10_pathway_heterogeneity.py` (pathway-by-pathway KK10)
- `analysis/paper4_shadow/country_substitution.py` (country-level substitution test)
- `analysis/data/kk10_country_panel.parquet` (new KK10 country aggregation)

## Open items (not blocking submission, but worth flagging)

- The pathway clustering still uses HYDE-trajectory features. The climate-only re-clustering robustness exercise (App. L.1) confirms the demographic margin is robust to typology choice, but the cropland exercise on KK10 has not been re-run on climate-only clusters. Probably ~1 hour of work if you want it.
- The CamPop England pre-1751 extension (App. I) and the medieval English mortality narrative (App. C.3) are still in. No changes to those.
- The structural model (App. K) is reframed but the CF3 cropland counterfactual numbers still reproduce the measurement-driven coefficient. I think the honest disclosure in the appendix prose suffices, but if you'd prefer to drop CF3 entirely, easy to do.

## Decision points

| Question | My recommendation | Alternatives |
|---|---|---|
| Primary venue | Restud first (better fit for methodological + identification contributions) | AEJ:Macro (more applied, faster turnaround) |
| Title | Current: "Volcanic Forcing and the Pre-Industrial Malthusian Trap" | Open to alternatives |
| Abstract length | ~225 words currently, may need to trim for AEJ (max 100 words) | Trim for AEJ submission |
| Drop §4.5.x (the Boserup null) entirely? | No — the methodological contribution is real and useful | Could be moved to App. M with one summary paragraph in §4 |
| Submit timing | After your review, no rush | --- |

Happy to talk through any of this on a call.

— J
