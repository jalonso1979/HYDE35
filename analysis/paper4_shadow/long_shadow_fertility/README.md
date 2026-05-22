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
