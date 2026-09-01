import re

with open("analysis/paper4_shadow/run_all.py", "r") as f:
    content = f.read()

helpers = """
def _prepare_seasonality_endowments(data: dict, endowments: pd.DataFrame | None) -> pd.DataFrame:
    if endowments is None:
        from analysis.paper4_shadow.seasonality import build_seasonality_panel
        endowments = build_seasonality_panel(
            data["climate_panel"],
            entity_col="region",
            historical_window=50,
            historical_period=(1, 1850),
        )

    # Rename region -> era5_region for the cross-section builder
    seas_endow = endowments.rename(
        columns={
            "region": "era5_region",
            "hist_seasonality_proxy": "seasonality_historical",
        },
    )
    return seas_endow

def _build_and_run_cross_section(data: dict, seas_endow: pd.DataFrame) -> pd.DataFrame:
    from analysis.paper4_shadow.long_shadow import (
        build_long_shadow_cross_section,
        run_long_shadow_regressions,
    )
    ep = data["extended_panel"]
    pw = data["pathways"]

    _sub("Building cross-section")
    xs = build_long_shadow_cross_section(
        ep,
        seas_endow,
        pw,
        entity_col="country_id",
    )
    print(f"  Cross-section: {len(xs)} countries")

    _sub("OLS: modern outcomes ~ historical seasonality")
    results = run_long_shadow_regressions(
        xs,
        seasonality_col="seasonality_historical",
        pathway_col="cluster",
        include_pathway_fe=True,
    )
    if not results.empty:
        print(results.to_string(index=False))
    else:
        print("  [warn] No results from long-shadow regressions")

    return xs

def _plot_long_shadow_figures(xs: pd.DataFrame) -> None:
    figs = _import_figures()
    if figs and len(xs) > 0:
        seas_col = "seasonality_historical" if "seasonality_historical" in xs.columns else "hist_seasonality_proxy"
        if seas_col in xs.columns:
            _try_figure(figs.fig9_long_shadow, xs, seas_col)
"""

new_exercise_6 = """def exercise_6(
    data: dict,
    endowments: pd.DataFrame | None = None,
    *,
    figures_only: bool = False,
) -> None:
    \"\"\"Exercise 6: Long shadow — do pre-industrial endowments predict modern outcomes?\"\"\"
    _banner("Exercise 6: The Long Shadow Cross-Section")

    seas_endow = _prepare_seasonality_endowments(data, endowments)

    xs = pd.DataFrame()
    if not figures_only:
        xs = _build_and_run_cross_section(data, seas_endow)

    if not xs.empty:
        _plot_long_shadow_figures(xs)
"""

pattern = re.compile(
    r"def exercise_6\([\s\S]*?_try_figure\(figs\.fig9_long_shadow, xs, seas_col\)\n",
    re.MULTILINE
)

# find the replacement position
match = pattern.search(content)
if match:
    new_content = content[:match.start()] + helpers + "\n" + new_exercise_6 + content[match.end():]
    with open("analysis/paper4_shadow/run_all.py", "w") as f:
        f.write(new_content)
    print("Replaced!")
else:
    print("Match not found.")
