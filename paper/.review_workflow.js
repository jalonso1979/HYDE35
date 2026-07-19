export const meta = {
  name: 'two-paper-critical-review',
  description: 'Critical referee + code review of the long_shadow and horserace papers: 7 lenses each, adversarial verification, synthesized referee reports',
  phases: [
    { title: 'Review', detail: '7 critical lenses x 2 projects, each reads paper+code directly' },
    { title: 'Verify', detail: 'independent skeptic re-opens cited file to confirm/refute each finding' },
    { title: 'Synthesize', detail: 'per-paper referee report + cross-cutting themes' },
  ],
}

// ------------------------------------------------------------------ schemas
const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    summary: { type: 'string', description: 'One-paragraph overall assessment for this lens.' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          id: { type: 'string', description: 'short slug, e.g. ls-id-var-ordering' },
          title: { type: 'string' },
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          category: { type: 'string' },
          location: { type: 'string', description: 'file:line or paper section/line number' },
          claim_or_code: { type: 'string', description: 'what the paper claims or what the code does' },
          problem: { type: 'string', description: 'the specific defect' },
          evidence: { type: 'string', description: 'exact quote / line numbers you actually read' },
          why_it_matters: { type: 'string' },
          suggested_fix: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['id', 'title', 'severity', 'location', 'problem', 'evidence', 'why_it_matters', 'confidence'],
      },
    },
  },
  required: ['summary', 'findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['confirmed', 'partially_confirmed', 'refuted', 'uncertain'] },
    reasoning: { type: 'string', description: 'what you found when you re-opened the cited source' },
    corrected_severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'none'] },
    correction: { type: 'string', description: 'if the finding was wrong or overstated, the corrected statement; else empty' },
  },
  required: ['verdict', 'reasoning', 'corrected_severity'],
}

// ------------------------------------------------------------------ context
const OUTPUT_RULES = `
RULES OF EVIDENCE:
- Report ONLY problems you verified by opening the actual file or paper section. Quote the exact line(s) and give file:line or LaTeX section + line number.
- Never invent numbers or cite a result you did not read. If you suspect but could not confirm, set confidence "low" and say what you'd need to confirm.
- Severity: "blocker" = invalidates a headline claim or produces a wrong result; "major" = materially weakens a claim, or a real code bug affecting reported numbers; "minor" = imprecision, clarity, robustness gap, or correctness-relevant code smell.
- Prefer a FEW high-certainty, material findings over many speculative ones. Returning 0 findings is fine if the work is sound for your lens — say so in the summary.
- For code, trace the actual computation line by line. For paper claims, check against what the code/outputs actually produce where you can (Read the JSON in /analysis/output, or run python via Bash to recompute a conversion).
- You have Read, Grep, Glob, and Bash. Parquet data files can be inspected with python (pandas) via Bash if needed, but that's optional.
`

const CTX_LS = `
PROJECT: "long_shadow"
Paper (LaTeX, ~1390 lines): /Volumes/BIGDATA/HYDE35/paper/long_shadow.tex
  Title: "Volcanic Forcing and the Pre-Industrial Malthusian Trap". Authors: Alonso Ortiz & Da-Rocha. Target: macro/econ-history journal.
Code dir: /Volumes/BIGDATA/HYDE35/analysis/paper4_shadow/  (the *.py at the top level are the core modules feeding THIS paper).
  IMPORTANT: the subdir long_shadow_fertility/ belongs to a DIFFERENT paper (its .tex is not in this folder) — IGNORE it.
  Orchestrator: paper4_shadow/run_all.py
Reproduced outputs: JSON in /Volumes/BIGDATA/HYDE35/analysis/output/  ; figures in /Volumes/BIGDATA/HYDE35/analysis/figures/paper4_v2/
Note: /Volumes/BIGDATA/HYDE35/paper/main.tex is an OLDER superseded draft of this project — you may consult it but the live paper is long_shadow.tex.

HEADLINE NUMBERS THE PAPER CLAIMS (verify against code / outputs / tables):
- Joint 3-equation panel VAR pop-growth VSSI coefficients: crop-dominant-late = -9.7e-5 /Tg, pastoral/mixed-late = -5.5e-5 /Tg; Bonferroni-significant across 12 pathway x equation tests; robust to Webb wild-cluster bootstrap and to re-clustering pathways on climate primitives; survives post-1700 restriction.
- 30-Tg Tambora-class decade => ~3% cumulative pop loss (crop-dominant-late); 10-Tg ordinary => ~1%.
- Allen real-wage panel, 6 European countries 1421-1850: canonical Malthusian price-on-population coefficient; implied real-wage rise after a Tambora-class contraction ~ +0.95% for survivors.
- Mortality channel: HMD/HFD 10 countries 1751-1900 trace positive check on infant/child mortality; collapses to ~0 by 1950-2022.
- Long shadow: pre-industrial inter-annual climate volatility 1421-1750 is the single strongest cross-country predictor of modern pop growth, R^2=0.45; survives latitude + deep-determinant battery; placebo over modern 1950-2008 window p=0.16; specific to demographic outcomes (urban share, GDPpc absorbed by latitude).
- GS decomposition: predictive content concentrated in NON-growing-season variance; |beta_nonGS| exceeds |beta_GS| by 2x to 17x across nine specification cells.
- Boserup HYDE cropland-share: +7.8e-4 /Tg, p=0.003 (1800-1900 sub-sample); artefactual — driven by HYDE colonial-transition encoding in 24 small-island/tropical-Africa countries; vanishes on 21-country substantive core; attenuates 10x under pop-weighting. Cropland-AREA fallback: robust on core but fully mediated by demographic margin (adding annual dlnP collapses it). KK10 (population-independent): +1.07e-5 /Tg p=0.005 (core), +4.3e-5 /Tg p=0.047 (high-density intensive) — survives demographic-margin control; ~2 orders below HYDE.
- Panel: 196 countries, 3,187 sub-national units, from 1421 CE; ModE-RA monthly climate aggregated to HYDE 3.5 cells, calibrated to ERA5, linked to Sigl-Toohey eVolv2k VSSI.
`

const CTX_HR = `
PROJECT: "horserace"
Paper (LaTeX, ~818 lines): /Volumes/BIGDATA/HYDE35/paper/horserace/horserace.tex
  Title: "Good genes, good weather and a prosperous land: A re-examination of the deep root factors of economic development". Authors: Alonso Ortiz & Da-Rocha.
Code dir: /Volumes/BIGDATA/HYDE35/analysis/paper5_horserace/  (build_*.py construct the panel; exercise*.py / shapley.py / mediation.py / robustness_battery.py run the analyses; fig*.py make figures; tests/ has the test suite).

HEADLINE STRUCTURE / CLAIMS (verify against code / outputs / tables):
- A 196-country panel joins FIVE deep-determinant "substrates": (1) ancient genetic diversity / predicted heterozygosity REPLACED by two alternatives: an 8-locus functional-allele bundle (LCT/MCM6, AMY1, FADS1/2, DARC, HBB, SLC24A5, ADH1B, EDAR) and a global agricultural-Neolithic ancestry fraction; (2) climate regime selecting agricultural pathway; (3) ancestral caloric crop-yield potential; (4) pre-colonial state history (Bockstette-Putterman); (5) pre-1500 pandemic exposure.
- Method: grouped Shapley-Owen variance decomposition averaging each substrate's marginal R^2 over all 5!=120 orderings, conditional on a geography control battery + pathway dummies; Westfall-Young FWER correction across 30 substrate x outcome cells (5 substrates x 6 outcomes).
- "Temporal partition" headline: climate (via pathway selection) dominates the Malthusian density margin at 1500; ancestral crop-yield potential dominant across the whole demographic-density trajectory (Malthusian density, convergence-window growth 1950-2025, cumulative density 2025); agricultural-Neolithic ancestry fraction dominates cumulative density at 2025 + demographic-transition timing; functional-allele bundle dominates per-capita income 2015; pandemic intensity contributes the smallest, least-robust signal.
- DARC graded-partialling exercise: DARC's negative coefficient on log GDPpc survives colonial-binary and raw African-ancestry controls but is ~65% attenuated (to insignificance) by the Bockstette-Putterman state-history index — reinterpreted as a proxy for low pre-1500 state capacity, not biological causation or colonial extraction. Same partialling sharpens FADS1/2.

CRITICAL INTERNAL-CONSISTENCY FLAG (from a prior internal analysis of this project):
- An earlier internal write-up of this same project recorded: "Shapley dominance — ancestral crop yield wins 3 of 4 outcomes" and "FAILED mediation — 14 of 16 cells show a suppressor structure, robust across clusterings and subsamples." The current paper instead presents 6 outcomes / 30 cells and a clean "temporal partition" narrative. Check whether (a) the paper's outcome/cell counts and dominance claims match the CURRENT code outputs, and (b) the mediation results are honestly represented or whether a "failed/suppressor" mediation result has been quietly reframed. This is a high-priority cross-check.
`

// ------------------------------------------------------------------ lenses
const LENS_DEFS = [
  {
    key: 'identification',
    focusLS: `Scrutinize CAUSAL IDENTIFICATION. (a) The joint 3-equation panel VAR (paper ~line 209; code joint_landuse_var.py, joint_var_bootstrap.py, joint_var_post1700.py, joint_var_climate_pathways.py): is decade-summed VSSI a credible exogenous forcing? Does the headline depend on the VAR ordering / contemporaneous-vs-lagged climate timing? (b) The long-shadow regression (paper ~line 371; long_shadow.py, deep_determinants*.py, latitude_controls.py): is "strongest predictor, R^2=0.45" sold as causal when it is cross-sectional/predictive? Examine the placebo (1950-2008, p=0.16) logic and whether it actually isolates the claimed mechanism. (c) HYDE uses population as an input — the paper acknowledges this for Boserup; check whether the SAME endogeneity contaminates the pop/cropland VAR or the long-shadow regressor. (d) Pathways are ESTIMATED groupings — does conditioning/heterogeneity on an estimated partition bias inference?`,
    focusHR: `Scrutinize IDENTIFICATION and causal language over a variance decomposition. (a) Shapley-Owen attributes VARIANCE, not causation — flag every place the paper slides into causal verbs ("dominates", "drives", "selected"). (b) The DARC graded-partialling (exercise_colonial_partial*.py, build_functional_alleles.py): is the 65%-attenuation conclusion an artifact of control ORDERING, or is state-history a "bad control"/collider/post-treatment for DARC? Is the causal reinterpretation ("proxy for state capacity") over-told from a partialling exercise? (c) Ancestral crop yield and Neolithic-ancestry fraction as regressors — are they plausibly exogenous or jointly determined with outcomes? (d) Pathway mediation (mediation.py, exercise2_mediation.py) — are the causal-step / no-unmeasured-confounding assumptions stated and defensible?`,
  },
  {
    key: 'inference',
    focusLS: `Audit STATISTICAL INFERENCE and multiple testing. (a) Bonferroni across "12 pathway x equation tests" — is the count and family correct, and is Bonferroni the right correction given correlated tests? (b) Webb wild-cluster bootstrap + cluster bootstrap (joint_var_bootstrap.py): correct sign weights (Webb 6-point?), null imposed correctly, ENOUGH clusters? How many countries/pathways cluster — is this a few-clusters problem that invalidates cluster-robust SEs? (c) SE clustering level and whether it matches the variation being exploited. (d) AR degrees of freedom (a recent commit fixed "AR dof n-2" — verify it's right now). (e) Manski bounds (manski_bounds.py) and the bootstrap difference-CI (phase outputs) — logic correct? (f) spatial_se_robustness.py — Conley SEs implemented correctly (cutoff, kernel)?`,
    focusHR: `Audit STATISTICAL INFERENCE and multiple testing. (a) Westfall-Young FWER across 30 cells (robustness_battery.py): is the resampling done under the COMPLETE null with the maxT/minP step-down done correctly, and is the family really 30? (b) Grouped Shapley-Owen over 5!=120 orderings (shapley.py, exercise1_shapley*.py): is the marginal-R^2 averaging and the coalition grouping (climate + functional bundle treated as group players) implemented correctly, with no double counting? (c) Do the Shapley-share confidence intervals account for the fact that R^2 is itself estimated (bootstrap over observations)? (d) n=196 cross-section with many regressors + pathway dummies + geography battery — degrees-of-freedom / overfitting; is adjusted R^2 or out-of-sample used anywhere? (e) Are the "dominance" rankings within sampling noise of each other (do CIs overlap)?`,
  },
  {
    key: 'consistency',
    focusLS: `Cross-check EVERY headline number against (a) the body tables and (b) the code outputs/JSON in /analysis/output and figures. Locate or recompute: -9.7e-5, -5.5e-5; R^2=0.45; placebo p=0.16; +7.8e-4 p=0.003; +1.07e-5 p=0.005; +4.3e-5 p=0.047; "2x to 17x" GS ratio; ~3% and ~1% cumulative loss; +0.95% real wage; 196 countries / 3,187 units. Flag any number that appears in the abstract/intro but is NOT derivable from a table or output, or that disagrees between abstract, body, and code. Check internal arithmetic explicitly with python via Bash (e.g., does -9.7e-5/Tg compounded over a 30-Tg decade actually yield ~3%? does the real-wage figure follow from the price-on-pop coefficient?).`,
    focusHR: `Cross-check EVERY headline claim against the body tables and the code outputs. Verify each "dominance" assignment in the temporal partition against the ACTUAL Shapley table/output (not just the abstract's prose). Locate the 30-cell Westfall-Young table and confirm which cells are FWER-significant. Verify the DARC 65% attenuation against exercise_colonial_partial output. THEN execute the critical-consistency flag from the context: compare the paper's "6 outcomes / 30 cells / temporal partition" against the prior internal record of "3 of 4 outcomes" and "failed mediation / 14-of-16 suppressor cells" — determine whether the current CODE outputs support the paper's framing or the earlier finding, and flag any quiet reframe. Use Read/Grep on horserace.tex + the output JSON/CSV; recompute with python where possible.`,
  },
  {
    key: 'data',
    focusLS: `Audit DATA CONSTRUCTION and measurement. (a) ModE-RA monthly -> HYDE-cell aggregation + cropland-weighting (build_gs_climate.py, long_shadow_gs.py, long_shadow_monthly_decomp.py, long_shadow_harvest_mask.py): correct area/crop weights, no temporal leakage in growing-season masks. (b) eVolv2k VSSI linkage and decade-summing alignment with the decadal HYDE grid vs annual climate — off-by-one / aggregation correctness. (c) KK10 cross-validation (build_kk10_country_panel.py, kk10_orthogonality.py, kk10_pathway_heterogeneity.py): is KK10 truly population-INDEPENDENT as claimed, and is the orthogonality check sound? (d) Allen wage splices (allen_wage_malthus*.py, build_allen_pre1500_panel.py). (e) Provenance of the 3,187 sub-national units. (f) Lag construction look-ahead errors. Verify suspicious merges by inspecting the parquet with python if needed.`,
    focusHR: `Audit DATA CONSTRUCTION. (a) 8-locus functional-allele bundle (build_functional_alleles.py): allele-frequency sourcing, country mapping, ancestry weighting; are the loci/signs handled consistently? (b) Ancestry measures (build_lazaridis_ancestry.py, build_neolithic_fraction.py, build_r1b_m269.py, build_predicted_het.py): Putterman-Weil migration-matrix weighting applied correctly; coverage/missingness. (c) Ancestral crop yield (build_ancestral_crop_yield.py): construction and units. (d) State history (build_state_history.py) and pandemic intensity (build_pandemic_intensity.py). (e) Master join (build_horserace_panel.py): does it drop or DUPLICATE rows, silently coerce NA, or change n across substrates so Shapley R^2 is computed on inconsistent samples? Inspect row counts / keys with python.`,
  },
  {
    key: 'claims',
    focusLS: `Assess CLAIMS vs EVIDENCE and the honesty of exposition. (a) The Boserup "honest null" framing (paper ~line 305) — is it genuinely honest or does it bury/soften a result? (b) "iron laws" language and "climate-deep rather than agronomically specific" — does the GS decomposition actually license that conclusion? (c) Are pathway-heterogeneity conclusions robust to the (estimated) pathway definition? (d) External validity: conclusions drawn from a 6-country wage panel and a 10-country mortality panel generalized to "pre-industrial economies". (e) Does the conclusion (paper ~line 424) claim more than Sections 3-5 showed? Read intro, Section 5 (long shadow ~371), robustness (~417), conclusion and compare.`,
    focusHR: `Assess CLAIMS vs EVIDENCE. (a) The "temporal partition / each witness speaks for its own epoch" narrative — is it a robust structure or a post-hoc story imposed on multiply-tested cells whose CIs may overlap? (b) Is the mediation result honestly reported (prior internal finding: mediation FAILED with a suppressor structure)? Read the mediation section + fig04_mediation and check the spin. (c) Does replacing Ashraf-Galor heterozygosity with the functional bundle + Neolithic fraction genuinely improve interpretability, or relabel the same variance? (d) Is the DARC institutional-proxy reinterpretation over-claimed from one partialling exercise? Read sec:discussion + sec:conclusion and compare to what Sections shapley/mediation/robustness actually establish.`,
  },
  {
    key: 'code-estimators',
    focusLS: `Read the CORE ESTIMATOR implementations for correctness bugs. Targets: joint_landuse_var.py (panel VAR: lag indexing, FE demeaning, IRF/FEVD math), volcanic_event_study.py / sigl_event_study_dynamic.py / volcanic_price_event_study.py (event-study leads/lags, binning, reference period), structural_3eq.py / structural_calibration.py / structural_simulate.py / structural_counterfactuals.py / welfare_counterfactuals.py (calibration targets, simulation loop, welfare math), preindustrial_malthus*.py (regression spec), preventive_positive_check.py. Look for: wrong lag/sign indexing, incorrect within-transformation, matrix-algebra/inversion errors, biased variance, uncontrolled NaN drops that silently change the sample, nondeterminism (unseeded RNG), unit mismatches, and IRF accumulation errors. Quote the buggy lines.`,
    focusHR: `Read the CORE ESTIMATOR implementations for correctness bugs. Targets: shapley.py + exercise1_shapley.py + exercise1_shapley_chre_robust.py + exercise1_functional_subshapley.py (marginal-R^2 averaging over orderings, coalition handling, no double counting, correct base/full sets), mediation.py + exercise2_mediation.py (direct/indirect decomposition, suppressor detection, bootstrap), robustness_battery.py (Westfall-Young resampling), subsamples.py + exercise2_subsample_stability.py. Look for: data leakage between predictors, incorrect standardization, R^2 computed on the wrong/non-common sample, off-by-one over the 120 orderings, unseeded RNG / nondeterminism, mishandled NaNs changing n. Quote the buggy lines.`,
  },
  {
    key: 'pipeline-repro',
    focusLS: `Audit DATA-PIPELINE correctness + REPRODUCIBILITY. (a) run_all.py: does it actually regenerate the paper's numbers end to end, and in the right dependency order? (b) Is there any harness tying long_shadow.tex numbers to outputs? (verify_against_paper.py targets a DIFFERENT paper — confirm.) (c) Hardcoded absolute paths, missing/!inconsistent random seeds. (d) git status shows many modified committed parquet/PDF outputs — are the committed outputs in sync with the current code, or stale? (e) Merges/joins across modules: key collisions, many-to-one row multiplication, silent missing-data coercion that changes n. (f) Environment/requirements pinning. Trace whether a committed JSON output in /analysis/output is consistent with what the current code path would produce.`,
    focusHR: `Audit DATA-PIPELINE correctness + REPRODUCIBILITY. (a) build_horserace_panel.py master join: key integrity, row counts, NA coercion, consistent n across substrates. (b) Hardcoded paths, missing seeds. (c) Test suite quality (tests/test_*.py): do the tests actually exercise the estimators (Shapley/mediation/WY) and assert correct numbers, or only smoke-test shapes? Run pytest if quick. (d) Are the committed figures/tables in sync with the code, or stale? (e) Trace whether the Shapley/mediation outputs the figures consume match what the CURRENT code would produce. (f) Reproduction entry point — is there one, and does dependency order hold?`,
  },
]

// build the flat lens list (project x lens)
const LENSES = []
for (const project of ['long_shadow', 'horserace']) {
  const ctx = project === 'long_shadow' ? CTX_LS : CTX_HR
  for (const d of LENS_DEFS) {
    const focus = project === 'long_shadow' ? d.focusLS : d.focusHR
    LENSES.push({
      key: `${project}:${d.key}`,
      project,
      label: `review:${project.split('_')[0]}:${d.key}`,
      prompt: `You are a rigorous, skeptical referee + senior code reviewer for a quantitative economic-history paper. Your single lens for this task: ${d.key.toUpperCase()}.
${ctx}
YOUR LENS — ${d.key}:
${focus}
${OUTPUT_RULES}`,
    })
  }
}

// ------------------------------------------------------------------ run
log(`Launching ${LENSES.length} review lenses (7 x 2 projects); each finding will be adversarially verified.`)

const verifyPrompt = (f, lens) => `You are an independent verifier. A reviewer (lens "${lens.key}") filed the finding below about the ${lens.project} project. Your job is to REFUTE it: open the cited source yourself and check whether it actually holds. Default to skepticism — if you cannot reproduce the evidence, mark it refuted or uncertain.

Project paths:
- long_shadow: paper /Volumes/BIGDATA/HYDE35/paper/long_shadow.tex ; code /Volumes/BIGDATA/HYDE35/analysis/paper4_shadow/ ; outputs /Volumes/BIGDATA/HYDE35/analysis/output/
- horserace: paper /Volumes/BIGDATA/HYDE35/paper/horserace/horserace.tex ; code /Volumes/BIGDATA/HYDE35/analysis/paper5_horserace/

FINDING TO VERIFY:
- title: ${f.title}
- severity (claimed): ${f.severity}
- location: ${f.location}
- problem: ${f.problem}
- evidence cited: ${f.evidence}
- why it matters: ${f.why_it_matters || ''}

Open the cited file/section (Read/Grep/Bash). Confirm the quoted lines exist and say what the reviewer claims. Judge whether the problem is real, overstated, or wrong. If the reviewer misread the code/paper, set verdict "refuted" and explain. If real but milder/sharper than stated, "partially_confirmed" with corrected_severity. Be concrete; cite what you saw.`

const reviewed = await pipeline(
  LENSES,
  lens => agent(lens.prompt, { label: lens.label, phase: 'Review', schema: FINDINGS_SCHEMA })
    .then(r => ({ lens, review: r })),
  ({ lens, review }) => {
    const fs = (review && review.findings ? review.findings : []).slice(0, 8)
    if (!fs.length) return { lens, summary: review ? review.summary : '', verified: [] }
    return parallel(fs.map(f => () =>
      agent(verifyPrompt(f, lens), { label: `verify:${lens.key}:${f.id}`, phase: 'Verify', schema: VERDICT_SCHEMA })
        .then(v => ({ ...f, lens: lens.key, project: lens.project, verdict: v }))
        .catch(() => null)
    )).then(vs => ({ lens, summary: review ? review.summary : '', verified: vs.filter(Boolean) }))
  }
)

// collect verified findings by project, keep only confirmed / partially_confirmed / uncertain-but-material
const byProject = { long_shadow: [], horserace: [] }
const lensSummaries = { long_shadow: [], horserace: [] }
for (const r of reviewed.filter(Boolean)) {
  lensSummaries[r.lens.project].push(`[${r.lens.key}] ${r.summary}`)
  for (const f of r.verified) {
    const vd = f.verdict ? f.verdict.verdict : 'uncertain'
    if (vd === 'refuted') continue
    byProject[f.project].push(f)
  }
}

const countSev = (arr, sev) => arr.filter(f => {
  const s = (f.verdict && f.verdict.corrected_severity && f.verdict.corrected_severity !== 'none')
    ? f.verdict.corrected_severity : f.severity
  return s === sev
}).length

const stats = {}
for (const p of ['long_shadow', 'horserace']) {
  stats[p] = {
    surviving: byProject[p].length,
    blocker: countSev(byProject[p], 'blocker'),
    major: countSev(byProject[p], 'major'),
    minor: countSev(byProject[p], 'minor'),
  }
}
log(`Verification complete. long_shadow: ${JSON.stringify(stats.long_shadow)} | horserace: ${JSON.stringify(stats.horserace)}`)

// ------------------------------------------------------------------ synthesize
phase('Synthesize')

const synthPrompt = (project, ctx, findings, summaries) => `You are the LEAD REFEREE writing the critical review for the "${project}" paper. Below are findings from 7 independent review lenses, each already adversarially verified (each carries a verdict + corrected_severity from a skeptic who re-opened the source). REFUTED findings have already been removed.
${ctx}
PER-LENS SUMMARIES:
${summaries.join('\n')}

VERIFIED FINDINGS (JSON):
${JSON.stringify(findings, null, 1)}

Write a rigorous, referee-quality critical review in GitHub Markdown. Use the verdict/corrected_severity to weight each item (treat "partially_confirmed" as real but calibrate the claim; treat "uncertain" as a flag-for-author, not an accusation). De-duplicate findings that multiple lenses raised. Structure:

# Critical Review — <paper title>
## Verdict
2-3 paragraphs: is the core contribution sound and publishable? What is the single biggest threat to the headline result? Be direct.
## Blockers
Numbered. Each: **what**, why it matters, evidence/location (file:line or §), concrete fix. Only genuine result-invalidating issues. If none, say "None identified."
## Major issues
Numbered, same format.
## Minor issues & robustness gaps
Tight bullets.
## Code & reproducibility
What works, what's broken, what's stale, what a replicator would hit.
## Prioritized fix list
Ordered checklist, highest-leverage first.

Be specific and cite locations. Do not pad or flatter. Distinguish genuine errors from defensible judgment calls. If the paper is fundamentally sound, say so clearly rather than manufacturing problems.`

const [reviewLS, reviewHR] = await parallel([
  () => agent(synthPrompt('long_shadow', CTX_LS, byProject.long_shadow, lensSummaries.long_shadow), { label: 'synth:long_shadow', phase: 'Synthesize' }),
  () => agent(synthPrompt('horserace', CTX_HR, byProject.horserace, lensSummaries.horserace), { label: 'synth:horserace', phase: 'Synthesize' }),
])

const crossCutting = await agent(
  `You are the editor. Two referee reports follow for two papers by the SAME authors that share data infrastructure (HYDE 3.5, ModE-RA, a 196-country panel) and methods (Shapley/bootstrap/pathway typology). Write a SHORT cross-cutting note (Markdown, no preamble): (1) issues that recur in BOTH papers (shared-infrastructure or shared-method risks worth fixing once), (2) any tension/contradiction BETWEEN the two papers (e.g., the same variable described differently, or a result one paper relies on that the other undercuts), (3) the 3-5 highest-leverage fixes across both. Be concise and concrete.

=== REVIEW: long_shadow ===
${reviewLS}

=== REVIEW: horserace ===
${reviewHR}`,
  { label: 'synth:cross-cutting', phase: 'Synthesize' }
)

return {
  stats,
  reviewLS,
  reviewHR,
  crossCutting,
  allFindings: byProject,
}
