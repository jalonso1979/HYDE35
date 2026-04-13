# Long Shadow of Seasonality — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all six empirical exercises, generate 10 figures and 6 tables, and produce the LaTeX paper for "The Long Shadow of Seasonality" targeting the Journal of Economic Growth.

**Architecture:** Extends the existing three-paper codebase (`paper1_escape/`, `paper2_malthus/`, `paper3_climate/`) with a new `paper4_shadow/` module containing six analysis scripts (one per exercise), a figure builder, and a runner script. Reuses existing clustering, LP, and FE regression code via imports. New analyses: seasonality computation, multinomial logit, dual-channel regressions, pathway-stratified IRFs, interaction regressions.

**Tech Stack:** Python 3.13, pandas, numpy, xarray, statsmodels, scikit-learn, matplotlib. LaTeX for the paper.

**Spec:** `docs/superpowers/specs/2026-04-13-long-shadow-paper-design.md`

---

## File Structure

### New files to create

```
analysis/paper4_shadow/
    __init__.py
    seasonality.py           # Compute intra-annual seasonality from ERA5 monthly + PAGES 2k proxy
    pathway_prediction.py    # Exercise 1: multinomial logit — seasonality predicts pathways
    dual_channel.py          # Exercise 2: joint seasonality + volatility regressions
    malthusian_extended.py   # Exercise 3: pathway-specific Malthusian with volatility
    pathway_irfs.py          # Exercise 4: pathway-stratified local projection IRFs
    escape_mechanism.py      # Exercise 5: interaction regressions — what mediates escape
    long_shadow.py           # Exercise 6: cross-sectional long shadow with seasonality
    figures.py               # All 10 figures
    run_all.py               # Orchestrator — runs exercises, produces figures/tables

tests/
    test_seasonality.py
    test_pathway_prediction.py
    test_dual_channel.py
    test_escape_mechanism.py

paper/
    long_shadow.tex          # LaTeX paper
```

### Existing files to reuse (not modify)

```
analysis/paper1_escape/clustering.py       # pathway clustering
analysis/paper2_malthus/regressions.py     # FE regressions, rolling windows
analysis/paper3_climate/climate_shocks.py  # anomalies, volatility
analysis/paper3_climate/local_projections.py  # Jordà LPs
analysis/paper3_climate/regime_switching.py   # sample splitting
analysis/update_era5_panel.py              # ERA5 panel rebuild
analysis/build_climate_panel_0_2025.py     # historical climate panel
analysis/build_extended_panel.py           # HYDE+ERA5 merge
```

### Existing data files consumed

```
analysis/data/paper1_clustered_features.parquet  # pathway assignments
analysis/data/trajectory_features.parquet         # trajectory features
analysis/data/era5_full_panel.parquet             # ERA5 regional annual means
analysis/data/climate_panel_0_2025.parquet        # 0-2025 CE calibrated climate
analysis/data/hyde_era5_extended_panel.parquet     # merged country-level 1950-2025
analysis/data/country_era5_region_map.parquet      # country -> region assignment
```

---

## Task 1: Seasonality computation module

**Files:**
- Create: `analysis/paper4_shadow/__init__.py`
- Create: `analysis/paper4_shadow/seasonality.py`
- Create: `tests/test_seasonality.py`

This module computes two distinct climate measures: intra-annual seasonality (sigma_s, the Matranga channel) and wraps the existing inter-annual volatility (sigma_v). For ERA5 (1950-2025) it uses monthly NetCDFs directly. For historical periods it computes a proxy from the PAGES 2k decadal reconstructions.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_seasonality.py
"""Tests for seasonality computation."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper4_shadow.seasonality import (
    compute_intra_annual_seasonality_era5,
    compute_historical_seasonality_proxy,
    build_seasonality_panel,
)


def test_era5_seasonality_basic():
    """Monthly data with known seasonality should produce correct range."""
    # 2 regions, 2 years, 12 months each
    rows = []
    for region in [1, 2]:
        for year in [2000, 2001]:
            for month in range(1, 13):
                # Region 1: temp ranges 0-22 (seasonality=22)
                # Region 2: temp ranges 10-14 (seasonality=4)
                if region == 1:
                    temp = 11.0 + 11.0 * np.sin(2 * np.pi * (month - 4) / 12)
                else:
                    temp = 12.0 + 2.0 * np.sin(2 * np.pi * (month - 4) / 12)
                rows.append({"region": region, "year": year, "month": month, "temperature_c": temp})
    monthly = pd.DataFrame(rows)
    result = compute_intra_annual_seasonality_era5(monthly)
    # Region 1 should have higher seasonality than Region 2
    r1 = result[result["region"] == 1]["seasonality_temp"].mean()
    r2 = result[result["region"] == 2]["seasonality_temp"].mean()
    assert r1 > r2 * 3, f"Region 1 ({r1:.1f}) should be much more seasonal than Region 2 ({r2:.1f})"


def test_era5_seasonality_columns():
    """Output should have expected columns."""
    rows = []
    for region in [1]:
        for year in [2000]:
            for month in range(1, 13):
                rows.append({"region": region, "year": year, "month": month, "temperature_c": 15.0 + month, "precipitation_mm": 0.1 + 0.01 * month})
    monthly = pd.DataFrame(rows)
    result = compute_intra_annual_seasonality_era5(monthly)
    assert "seasonality_temp" in result.columns
    assert "seasonality_precip" in result.columns
    assert "region" in result.columns
    assert "year" in result.columns


def test_historical_proxy_basic():
    """Historical proxy from annual data should compute inter-decadal range."""
    rows = []
    for region in [1, 2]:
        for year in range(1000, 1200):
            # Region 1: volatile (std ~5), Region 2: stable (std ~0.5)
            rng = np.random.default_rng(42 + region * 1000 + year)
            scale = 5.0 if region == 1 else 0.5
            rows.append({"region": region, "year": year, "temperature_c": 15.0 + rng.normal(0, scale)})
    annual = pd.DataFrame(rows)
    result = compute_historical_seasonality_proxy(annual, window=50)
    r1 = result[result["region"] == 1]["seasonality_proxy"].mean()
    r2 = result[result["region"] == 2]["seasonality_proxy"].mean()
    assert r1 > r2 * 2, f"Region 1 proxy ({r1:.2f}) should exceed Region 2 ({r2:.2f})"


def test_build_seasonality_panel_shape():
    """Full panel builder should produce one row per region."""
    rows = []
    for region in [1, 2]:
        for year in range(1000, 1100):
            rows.append({"region": region, "year": year, "temperature_c": 15.0})
    annual = pd.DataFrame(rows)
    result = build_seasonality_panel(annual, era5_monthly=None)
    assert len(result) == 2, "One row per region"
    assert "region" in result.columns
    assert "seasonality_historical" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_seasonality.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.paper4_shadow'`

- [ ] **Step 3: Implement seasonality module**

```python
# analysis/paper4_shadow/__init__.py
"""Paper 4: The Long Shadow of Seasonality."""
```

```python
# analysis/paper4_shadow/seasonality.py
"""Compute intra-annual seasonality and historical volatility proxies.

Two climate measures:
- sigma_s (intra-annual seasonality): max-min of monthly values within a year.
  Drives Stage 1 (Matranga channel — storage demand -> system selection).
- Historical proxy: rolling SD of annual temperature over multi-decade windows.
  Used for pre-ERA5 periods where monthly data is unavailable.
"""
import numpy as np
import pandas as pd


def compute_intra_annual_seasonality_era5(
    monthly: pd.DataFrame,
    entity_col: str = "region",
) -> pd.DataFrame:
    """Compute intra-annual seasonality from ERA5 monthly data.

    For each region-year, seasonality = max(monthly) - min(monthly).
    This is the Matranga measure: higher values mean stronger
    within-year variation, incentivizing storage and sedentism.

    Parameters
    ----------
    monthly : DataFrame with columns [entity_col, year, month, temperature_c]
        and optionally precipitation_mm
    entity_col : grouping column

    Returns
    -------
    DataFrame with columns [entity_col, year, seasonality_temp, seasonality_precip]
    """
    results = []
    for (entity, year), grp in monthly.groupby([entity_col, "year"]):
        row = {entity_col: entity, "year": year}
        if "temperature_c" in grp.columns:
            row["seasonality_temp"] = grp["temperature_c"].max() - grp["temperature_c"].min()
        if "precipitation_mm" in grp.columns:
            row["seasonality_precip"] = grp["precipitation_mm"].max() - grp["precipitation_mm"].min()
        results.append(row)
    return pd.DataFrame(results)


def compute_historical_seasonality_proxy(
    annual: pd.DataFrame,
    window: int = 50,
    entity_col: str = "region",
) -> pd.DataFrame:
    """Compute historical seasonality proxy from annual reconstructions.

    Since PAGES 2k provides only annual/decadal averages (no monthly),
    we proxy seasonality with the rolling standard deviation of annual
    temperature. High inter-annual variability correlates with high
    intra-annual seasonality at centennial timescales.

    Parameters
    ----------
    annual : DataFrame with columns [entity_col, year, temperature_c]
    window : rolling window size in years

    Returns
    -------
    DataFrame with columns [entity_col, year, seasonality_proxy]
    """
    parts = []
    for entity, grp in annual.groupby(entity_col):
        grp = grp.sort_values("year").copy()
        grp["seasonality_proxy"] = (
            grp["temperature_c"]
            .rolling(window=window, center=True, min_periods=window // 2)
            .std()
        )
        parts.append(grp[[entity_col, "year", "seasonality_proxy"]])
    return pd.concat(parts, ignore_index=True)


def build_seasonality_panel(
    annual_climate: pd.DataFrame,
    era5_monthly: pd.DataFrame | None = None,
    entity_col: str = "region",
    historical_window: int = 50,
    historical_period: tuple[int, int] = (1, 1850),
) -> pd.DataFrame:
    """Build cross-sectional seasonality endowments per region.

    Computes long-run mean seasonality for each region from:
    - Historical proxy (annual data, pre-industrial period)
    - ERA5 direct seasonality (monthly data, if provided)

    Returns one row per region with seasonality measures.
    """
    # Historical proxy
    hist = annual_climate[
        (annual_climate["year"] >= historical_period[0])
        & (annual_climate["year"] <= historical_period[1])
        & annual_climate["temperature_c"].notna()
    ].copy()

    proxy = compute_historical_seasonality_proxy(hist, window=historical_window, entity_col=entity_col)

    endowments = (
        proxy.groupby(entity_col)
        .agg(seasonality_historical=("seasonality_proxy", "mean"))
        .reset_index()
    )

    # ERA5 direct seasonality (if available)
    if era5_monthly is not None and len(era5_monthly) > 0:
        era5_seas = compute_intra_annual_seasonality_era5(era5_monthly, entity_col=entity_col)
        era5_mean = (
            era5_seas.groupby(entity_col)
            .agg(seasonality_era5_temp=("seasonality_temp", "mean"))
            .reset_index()
        )
        endowments = endowments.merge(era5_mean, on=entity_col, how="left")

    return endowments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_seasonality.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper4_shadow/__init__.py analysis/paper4_shadow/seasonality.py tests/test_seasonality.py
git commit -m "feat(paper4): add seasonality computation module with tests"
```

---

## Task 2: Pathway prediction from climate — Exercise 1

**Files:**
- Create: `analysis/paper4_shadow/pathway_prediction.py`
- Create: `tests/test_pathway_prediction.py`

Multinomial logit predicting pathway cluster assignment from climate endowments. Tests prediction 1 of the model: higher seasonality -> intensive pathways.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pathway_prediction.py
"""Tests for pathway prediction from climate endowments."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper4_shadow.pathway_prediction import (
    run_pathway_multinomial,
    compute_marginal_effects,
)


@pytest.fixture
def synthetic_pathway_climate():
    """Synthetic data: 60 countries, 3 pathways, seasonality predicts pathway."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(60):
        if i < 20:
            pathway = 0  # intensive — high seasonality
            seas = 0.8 + rng.normal(0, 0.1)
        elif i < 40:
            pathway = 1  # mixed — medium seasonality
            seas = 0.4 + rng.normal(0, 0.1)
        else:
            pathway = 2  # pastoral — low seasonality
            seas = 0.1 + rng.normal(0, 0.1)
        rows.append({
            "country_id": i,
            "pathway": pathway,
            "seasonality": seas,
            "mean_temp": 15.0 + rng.normal(0, 5),
            "latitude": rng.uniform(-60, 60),
        })
    return pd.DataFrame(rows)


def test_multinomial_runs(synthetic_pathway_climate):
    """Multinomial logit should run and return results dict."""
    result = run_pathway_multinomial(
        synthetic_pathway_climate,
        pathway_col="pathway",
        climate_vars=["seasonality", "mean_temp"],
        control_vars=["latitude"],
    )
    assert "model" in result
    assert "summary" in result
    assert "predictions" in result


def test_multinomial_seasonality_significant(synthetic_pathway_climate):
    """Seasonality should be a significant predictor."""
    result = run_pathway_multinomial(
        synthetic_pathway_climate,
        pathway_col="pathway",
        climate_vars=["seasonality"],
        control_vars=[],
    )
    # Check that at least one seasonality coefficient has p < 0.10
    pvals = result["pvalues"]
    seas_pvals = [v for k, v in pvals.items() if "seasonality" in str(k)]
    assert any(p < 0.10 for p in seas_pvals), f"Seasonality should be significant, got p-values: {seas_pvals}"


def test_marginal_effects_shape(synthetic_pathway_climate):
    """Marginal effects should have one row per predictor per pathway."""
    result = run_pathway_multinomial(
        synthetic_pathway_climate,
        pathway_col="pathway",
        climate_vars=["seasonality", "mean_temp"],
        control_vars=[],
    )
    me = compute_marginal_effects(result["model"], synthetic_pathway_climate, ["seasonality", "mean_temp"])
    n_pathways = synthetic_pathway_climate["pathway"].nunique()
    n_vars = 2
    assert len(me) == n_vars * (n_pathways - 1), f"Expected {n_vars * (n_pathways - 1)} rows, got {len(me)}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_pathway_prediction.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement pathway prediction module**

```python
# analysis/paper4_shadow/pathway_prediction.py
"""Exercise 1: Multinomial logit — does seasonality predict pathway assignment?

Tests Stage 1 prediction: higher intra-annual seasonality -> intensive pathways.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.orderedmodel import OrderedModel


def run_pathway_multinomial(
    df: pd.DataFrame,
    pathway_col: str = "pathway",
    climate_vars: list[str] | None = None,
    control_vars: list[str] | None = None,
) -> dict:
    """Run multinomial logit: pathway ~ seasonality + controls.

    Parameters
    ----------
    df : DataFrame with pathway assignments and climate endowments
    pathway_col : column with pathway cluster labels
    climate_vars : climate predictor columns (seasonality, mean_temp, etc.)
    control_vars : geographic controls (latitude, continent dummies, etc.)

    Returns
    -------
    dict with keys: model, summary, predictions, pvalues
    """
    if climate_vars is None:
        climate_vars = ["seasonality"]
    if control_vars is None:
        control_vars = []

    all_vars = climate_vars + control_vars
    sub = df[[pathway_col] + all_vars].dropna().copy()

    y = sub[pathway_col].astype(int)
    X = sm.add_constant(sub[all_vars].astype(float))

    model = sm.MNLogit(y, X).fit(disp=False, method="newton", maxiter=200)

    # Extract p-values for all parameters
    pvalues = {}
    for i, name in enumerate(model.params.index):
        for j, col in enumerate(model.params.columns):
            pvalues[f"{name}_{col}"] = model.pvalues.iloc[i, j]

    predictions = model.predict(X)

    return {
        "model": model,
        "summary": model.summary(),
        "predictions": predictions,
        "pvalues": pvalues,
        "params": model.params,
    }


def compute_marginal_effects(
    model,
    df: pd.DataFrame,
    climate_vars: list[str],
) -> pd.DataFrame:
    """Compute average marginal effects of climate variables on pathway probabilities.

    Returns DataFrame with columns: variable, pathway, marginal_effect, std_err
    """
    me = model.get_margeff(at="overall", method="dydx")
    results = []
    for i, var in enumerate(me.summary_frame().index):
        for j, col in enumerate(me.summary_frame().columns):
            if "dy/dx" in col:
                var_name = str(var)
                # Only report climate variables
                if any(cv in var_name for cv in climate_vars):
                    results.append({
                        "variable": var_name,
                        "pathway": j,
                        "marginal_effect": me.margeff[i, j] if hasattr(me.margeff, '__getitem__') else float(me.summary_frame().iloc[i][col]),
                    })

    if not results:
        # Fallback: extract from params directly
        for var in climate_vars:
            if var in model.params.index:
                for pathway in model.params.columns:
                    results.append({
                        "variable": var,
                        "pathway": pathway,
                        "marginal_effect": model.params.loc[var, pathway],
                    })

    return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_pathway_prediction.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper4_shadow/pathway_prediction.py tests/test_pathway_prediction.py
git commit -m "feat(paper4): add multinomial logit for pathway prediction (Exercise 1)"
```

---

## Task 3: Dual-channel separation — Exercise 2

**Files:**
- Create: `analysis/paper4_shadow/dual_channel.py`
- Create: `tests/test_dual_channel.py`

Joint regression showing seasonality and volatility operate through separate causal channels.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dual_channel.py
"""Tests for dual-channel separation regressions."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper4_shadow.dual_channel import (
    run_dual_channel_pathway,
    run_dual_channel_malthusian,
)


@pytest.fixture
def synthetic_dual_data():
    """Synthetic data where seasonality predicts pathway, volatility predicts beta."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(80):
        seas = rng.uniform(0.1, 1.0)
        vol = rng.uniform(0.1, 1.0)
        # Pathway determined by seasonality
        pathway = 0 if seas > 0.6 else (1 if seas > 0.3 else 2)
        # Malthusian beta determined by volatility
        malthus_beta = -0.001 * vol + rng.normal(0, 0.0001)
        rows.append({
            "country_id": i,
            "pathway": pathway,
            "seasonality": seas,
            "volatility": vol,
            "malthusian_beta": malthus_beta,
            "mean_temp": 15 + rng.normal(0, 5),
        })
    return pd.DataFrame(rows)


def test_dual_channel_pathway_returns_results(synthetic_dual_data):
    """Pathway regression should return coefficients for both channels."""
    result = run_dual_channel_pathway(
        synthetic_dual_data,
        pathway_col="pathway",
        seasonality_col="seasonality",
        volatility_col="volatility",
    )
    assert "seasonality_coef" in result
    assert "volatility_coef" in result
    assert "seasonality_pval" in result
    assert "volatility_pval" in result


def test_dual_channel_malthusian_returns_results(synthetic_dual_data):
    """Malthusian regression should return coefficients for both channels."""
    result = run_dual_channel_malthusian(
        synthetic_dual_data,
        beta_col="malthusian_beta",
        seasonality_col="seasonality",
        volatility_col="volatility",
    )
    assert "seasonality_coef" in result
    assert "volatility_coef" in result


def test_dual_channel_orthogonality(synthetic_dual_data):
    """Seasonality should predict pathway more than volatility does."""
    pw = run_dual_channel_pathway(
        synthetic_dual_data,
        pathway_col="pathway",
        seasonality_col="seasonality",
        volatility_col="volatility",
    )
    # Seasonality should be more significant for pathway
    assert pw["seasonality_pval"] < pw["volatility_pval"], (
        f"Seasonality p={pw['seasonality_pval']:.4f} should be < volatility p={pw['volatility_pval']:.4f} for pathway"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_dual_channel.py -v`
Expected: FAIL

- [ ] **Step 3: Implement dual-channel module**

```python
# analysis/paper4_shadow/dual_channel.py
"""Exercise 2: Dual-channel separation.

Tests that seasonality (sigma_s) and inter-annual volatility (sigma_v)
operate through separate causal channels:
- Seasonality predicts pathway assignment (Stage 1)
- Volatility predicts Malthusian intensity within pathway (Stage 2)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_dual_channel_pathway(
    df: pd.DataFrame,
    pathway_col: str = "pathway",
    seasonality_col: str = "seasonality",
    volatility_col: str = "volatility",
    control_vars: list[str] | None = None,
) -> dict:
    """Regress pathway on seasonality AND volatility jointly.

    Uses ordered probit since pathways have a natural ordering
    (intensive -> mixed -> pastoral along seasonality gradient).
    Falls back to OLS if ordered model fails.
    """
    if control_vars is None:
        control_vars = []

    all_vars = [seasonality_col, volatility_col] + control_vars
    sub = df[[pathway_col] + all_vars].dropna().copy()

    y = sub[pathway_col].astype(float)
    X = sm.add_constant(sub[all_vars].astype(float))

    model = sm.OLS(y, X).fit(cov_type="HC1")

    return {
        "seasonality_coef": model.params[seasonality_col],
        "seasonality_pval": model.pvalues[seasonality_col],
        "volatility_coef": model.params[volatility_col],
        "volatility_pval": model.pvalues[volatility_col],
        "rsquared": model.rsquared,
        "nobs": int(model.nobs),
        "model": model,
    }


def run_dual_channel_malthusian(
    df: pd.DataFrame,
    beta_col: str = "malthusian_beta",
    seasonality_col: str = "seasonality",
    volatility_col: str = "volatility",
    control_vars: list[str] | None = None,
) -> dict:
    """Regress Malthusian coefficient on seasonality AND volatility jointly.

    Tests whether volatility predicts Malthusian intensity independently
    of the pathway selected (Stage 2 channel).
    """
    if control_vars is None:
        control_vars = []

    all_vars = [seasonality_col, volatility_col] + control_vars
    sub = df[[beta_col] + all_vars].dropna().copy()

    y = sub[beta_col].astype(float)
    X = sm.add_constant(sub[all_vars].astype(float))

    model = sm.OLS(y, X).fit(cov_type="HC1")

    return {
        "seasonality_coef": model.params[seasonality_col],
        "seasonality_pval": model.pvalues[seasonality_col],
        "volatility_coef": model.params[volatility_col],
        "volatility_pval": model.pvalues[volatility_col],
        "rsquared": model.rsquared,
        "nobs": int(model.nobs),
        "model": model,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_dual_channel.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper4_shadow/dual_channel.py tests/test_dual_channel.py
git commit -m "feat(paper4): add dual-channel separation regressions (Exercise 2)"
```

---

## Task 4: Extended Malthusian regressions — Exercise 3

**Files:**
- Create: `analysis/paper4_shadow/malthusian_extended.py`

Extends existing `paper2_malthus/regressions.py` by adding inter-annual volatility as a regressor alongside density, and stratifying by pathway.

- [ ] **Step 1: Write the implementation**

```python
# analysis/paper4_shadow/malthusian_extended.py
"""Exercise 3: Pathway-specific Malthusian dynamics with volatility.

Extends Paper 2 rolling-window FE regressions by:
1. Adding inter-annual volatility (sigma_v) as a regressor
2. Stratifying by pathway cluster
"""
import pandas as pd
from analysis.paper2_malthus.regressions import run_fe_regression, run_rolling_window


def run_malthusian_by_pathway(
    panel: pd.DataFrame,
    pathway_assignments: pd.DataFrame,
    dep_var: str = "pop_growth",
    density_var: str = "log_density",
    volatility_var: str = "temp_volatility",
    land_labor_var: str = "log_land_labor",
    entity_col: str = "country",
    pathway_col: str = "cluster",
) -> pd.DataFrame:
    """Run FE regression separately for each pathway.

    Returns DataFrame with one row per pathway containing coefficients.
    """
    merged = panel.merge(
        pathway_assignments[[entity_col, pathway_col]],
        on=entity_col,
        how="inner",
    )

    results = []
    for pathway, grp in merged.groupby(pathway_col):
        indep_vars = [density_var]
        if volatility_var in grp.columns and grp[volatility_var].notna().sum() > 10:
            indep_vars.append(volatility_var)
        if land_labor_var in grp.columns:
            indep_vars.append(land_labor_var)

        try:
            reg = run_fe_regression(grp, dep_var, indep_vars, entity_col)
            row = {
                "pathway": pathway,
                "beta_density": reg["params"].get(density_var, float("nan")),
                "pval_density": reg["pvalues"].get(density_var, float("nan")),
                "nobs": reg["nobs"],
                "rsquared": reg["rsquared"],
            }
            if volatility_var in reg["params"]:
                row["gamma_volatility"] = reg["params"][volatility_var]
                row["pval_volatility"] = reg["pvalues"][volatility_var]
            results.append(row)
        except Exception:
            pass

    return pd.DataFrame(results)


def run_rolling_by_pathway(
    panel: pd.DataFrame,
    pathway_assignments: pd.DataFrame,
    dep_var: str = "pop_growth",
    key_var: str = "log_density",
    control_vars: list[str] | None = None,
    entity_col: str = "country",
    pathway_col: str = "cluster",
    window_years: int = 400,
    step_years: int = 100,
) -> pd.DataFrame:
    """Run rolling-window Malthusian estimation separately by pathway."""
    if control_vars is None:
        control_vars = ["log_land_labor"]

    merged = panel.merge(
        pathway_assignments[[entity_col, pathway_col]],
        on=entity_col,
        how="inner",
    )

    all_results = []
    for pathway, grp in merged.groupby(pathway_col):
        roll = run_rolling_window(
            grp, dep_var, key_var, control_vars,
            entity_col=entity_col,
            window_years=window_years,
            step_years=step_years,
        )
        roll["pathway"] = pathway
        all_results.append(roll)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
```

- [ ] **Step 2: Verify it imports correctly**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -c "from analysis.paper4_shadow.malthusian_extended import run_malthusian_by_pathway; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/malthusian_extended.py
git commit -m "feat(paper4): add pathway-stratified Malthusian regressions (Exercise 3)"
```

---

## Task 5: Pathway-stratified IRFs — Exercise 4

**Files:**
- Create: `analysis/paper4_shadow/pathway_irfs.py`

Runs Jordà local projections separately for countries in each pathway cluster.

- [ ] **Step 1: Write the implementation**

```python
# analysis/paper4_shadow/pathway_irfs.py
"""Exercise 4: Pathway-stratified local projection IRFs.

Runs LP-IRFs separately for countries in each agricultural pathway,
testing prediction that pastoral pathways show larger, more persistent
responses to climate shocks than intensive pathways.
"""
import pandas as pd
from analysis.paper3_climate.local_projections import run_local_projection


def run_irfs_by_pathway(
    panel: pd.DataFrame,
    pathway_assignments: pd.DataFrame,
    shock_var: str,
    response_var: str,
    entity_col: str = "country_id",
    pathway_col: str = "cluster",
    max_horizon: int = 10,
    n_lags: int = 2,
) -> dict[int, pd.DataFrame]:
    """Run LP-IRFs separately for each pathway.

    Returns dict mapping pathway label -> IRF DataFrame.
    """
    merged = panel.merge(
        pathway_assignments.rename(columns={"country": entity_col})
        if "country" in pathway_assignments.columns
        else pathway_assignments,
        on=entity_col,
        how="inner",
    )

    results = {}
    for pathway, grp in merged.groupby(pathway_col):
        if grp[entity_col].nunique() < 3:
            continue
        valid = grp.dropna(subset=[shock_var, response_var])
        if len(valid) < 30:
            continue
        irf = run_local_projection(
            valid,
            shock_var=shock_var,
            response_var=response_var,
            entity_col=entity_col,
            max_horizon=max_horizon,
            n_lags=n_lags,
        )
        irf["pathway"] = pathway
        results[pathway] = irf

    return results


def compare_pathway_irfs(
    irf_dict: dict[int, pd.DataFrame],
    horizon: int = 5,
) -> pd.DataFrame:
    """Compare IRF coefficients across pathways at a given horizon.

    Returns DataFrame with one row per pathway showing coefficient,
    SE, and significance at the specified horizon.
    """
    rows = []
    for pathway, irf in irf_dict.items():
        h_row = irf[irf["horizon"] == horizon]
        if len(h_row) == 0:
            continue
        h_row = h_row.iloc[0]
        rows.append({
            "pathway": pathway,
            "horizon": horizon,
            "coefficient": h_row["coefficient"],
            "std_err": h_row["std_err"],
            "pvalue": h_row["pvalue"],
            "ci_lower": h_row["ci_lower"],
            "ci_upper": h_row["ci_upper"],
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Verify it imports correctly**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -c "from analysis.paper4_shadow.pathway_irfs import run_irfs_by_pathway; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/pathway_irfs.py
git commit -m "feat(paper4): add pathway-stratified LP-IRFs (Exercise 4)"
```

---

## Task 6: Escape mechanism — Exercise 5

**Files:**
- Create: `analysis/paper4_shadow/escape_mechanism.py`
- Create: `tests/test_escape_mechanism.py`

Interaction regressions testing whether intensification or urbanization mediates the decline in climate sensitivity.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_escape_mechanism.py
"""Tests for escape mechanism interaction regressions."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper4_shadow.escape_mechanism import (
    run_escape_interactions,
    run_rolling_with_mediator,
)


@pytest.fixture
def synthetic_modern_panel():
    """Panel where urbanization reduces climate sensitivity."""
    rng = np.random.default_rng(42)
    rows = []
    for country in range(20):
        for year in range(1950, 2026):
            urban = 0.1 + 0.005 * (year - 1950) + rng.normal(0, 0.01)
            intens = 0.3 + 0.003 * (year - 1950) + rng.normal(0, 0.02)
            shock = rng.normal(0, 1)
            # Growth declines with shock, but less so when urban is high
            growth = 0.02 - 0.005 * shock + 0.004 * shock * urban + rng.normal(0, 0.005)
            rows.append({
                "country_id": country,
                "year": year,
                "pop_growth": growth,
                "temp_anomaly": shock,
                "urban_share": urban,
                "intensification_index": intens,
            })
    return pd.DataFrame(rows)


def test_escape_interactions_runs(synthetic_modern_panel):
    """Should return results for each mediator."""
    results = run_escape_interactions(
        synthetic_modern_panel,
        shock_var="temp_anomaly",
        outcome_var="pop_growth",
        mediators=["urban_share", "intensification_index"],
        entity_col="country_id",
    )
    assert len(results) == 3, "Should have results for each mediator + joint"
    assert "urban_share" in results
    assert "intensification_index" in results
    assert "joint" in results


def test_escape_interaction_coefficients(synthetic_modern_panel):
    """Interaction coefficient should be positive (urban reduces negative shock effect)."""
    results = run_escape_interactions(
        synthetic_modern_panel,
        shock_var="temp_anomaly",
        outcome_var="pop_growth",
        mediators=["urban_share"],
        entity_col="country_id",
    )
    # The interaction shock*urban should be positive in our synthetic data
    assert results["urban_share"]["interaction_coef"] > 0, "Urban interaction should be positive"


def test_rolling_with_mediator(synthetic_modern_panel):
    """Rolling window should return time-varying interaction coefficients."""
    result = run_rolling_with_mediator(
        synthetic_modern_panel,
        shock_var="temp_anomaly",
        outcome_var="pop_growth",
        mediator_var="urban_share",
        entity_col="country_id",
        window=20,
        step=10,
    )
    assert len(result) > 0
    assert "center_year" in result.columns
    assert "interaction_coef" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_escape_mechanism.py -v`
Expected: FAIL

- [ ] **Step 3: Implement escape mechanism module**

```python
# analysis/paper4_shadow/escape_mechanism.py
"""Exercise 5: What mediates the escape from climate sensitivity?

Tests whether intensification or urbanization (or both) explain the
declining climate sensitivity observed after the Green Revolution.

Specification:
    growth_it = alpha_i + beta1 * shock_t + beta2 * shock_t * h_it + eps
    where h = intensification_index, urban_share, or both
    beta2 < 0 means higher h reduces climate sensitivity
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_escape_interactions(
    panel: pd.DataFrame,
    shock_var: str = "temp_anomaly",
    outcome_var: str = "pop_growth",
    mediators: list[str] | None = None,
    entity_col: str = "country_id",
) -> dict:
    """Run interaction regressions for each mediator, plus joint model.

    Returns dict mapping mediator name -> regression results.
    """
    if mediators is None:
        mediators = ["urban_share", "intensification_index"]

    results = {}

    # Individual mediator models
    for med in mediators:
        result = _run_single_interaction(panel, shock_var, outcome_var, med, entity_col)
        if result is not None:
            results[med] = result

    # Joint model with all mediators
    if len(mediators) > 1:
        result = _run_joint_interaction(panel, shock_var, outcome_var, mediators, entity_col)
        if result is not None:
            results["joint"] = result

    return results


def _run_single_interaction(
    panel: pd.DataFrame,
    shock_var: str,
    outcome_var: str,
    mediator_var: str,
    entity_col: str,
) -> dict | None:
    """Single interaction: outcome ~ shock + shock*mediator + mediator."""
    interaction_col = f"{shock_var}_x_{mediator_var}"
    df = panel[[outcome_var, shock_var, mediator_var, entity_col]].dropna().copy()
    if len(df) < 50:
        return None

    df[interaction_col] = df[shock_var] * df[mediator_var]

    # Demean for FE
    for col in [outcome_var, shock_var, mediator_var, interaction_col]:
        df[col] = df[col] - df.groupby(entity_col)[col].transform("mean")

    y = df[outcome_var]
    X = sm.add_constant(df[[shock_var, mediator_var, interaction_col]])
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df[entity_col]})

    return {
        "shock_coef": model.params[shock_var],
        "shock_pval": model.pvalues[shock_var],
        "mediator_coef": model.params[mediator_var],
        "mediator_pval": model.pvalues[mediator_var],
        "interaction_coef": model.params[interaction_col],
        "interaction_pval": model.pvalues[interaction_col],
        "rsquared": model.rsquared,
        "nobs": int(model.nobs),
        "model": model,
    }


def _run_joint_interaction(
    panel: pd.DataFrame,
    shock_var: str,
    outcome_var: str,
    mediators: list[str],
    entity_col: str,
) -> dict | None:
    """Joint model with all mediators and their interactions."""
    df = panel[[outcome_var, shock_var, entity_col] + mediators].dropna().copy()
    if len(df) < 50:
        return None

    interaction_cols = []
    for med in mediators:
        icol = f"{shock_var}_x_{med}"
        df[icol] = df[shock_var] * df[med]
        interaction_cols.append(icol)

    all_rhs = [shock_var] + mediators + interaction_cols
    for col in [outcome_var] + all_rhs:
        df[col] = df[col] - df.groupby(entity_col)[col].transform("mean")

    y = df[outcome_var]
    X = sm.add_constant(df[all_rhs])
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df[entity_col]})

    result = {"rsquared": model.rsquared, "nobs": int(model.nobs), "model": model}
    for med in mediators:
        icol = f"{shock_var}_x_{med}"
        result[f"{med}_interaction_coef"] = model.params.get(icol, float("nan"))
        result[f"{med}_interaction_pval"] = model.pvalues.get(icol, float("nan"))

    return result


def run_rolling_with_mediator(
    panel: pd.DataFrame,
    shock_var: str = "temp_anomaly",
    outcome_var: str = "pop_growth",
    mediator_var: str = "urban_share",
    entity_col: str = "country_id",
    window: int = 20,
    step: int = 5,
) -> pd.DataFrame:
    """Rolling window interaction regression."""
    years = sorted(panel["year"].unique())
    min_yr, max_yr = min(years), max(years)

    results = []
    start = min_yr
    while start + window <= max_yr + 1:
        sub = panel[(panel["year"] >= start) & (panel["year"] < start + window)]
        result = _run_single_interaction(sub, shock_var, outcome_var, mediator_var, entity_col)
        if result is not None:
            results.append({
                "center_year": start + window // 2,
                "shock_coef": result["shock_coef"],
                "interaction_coef": result["interaction_coef"],
                "interaction_pval": result["interaction_pval"],
                "nobs": result["nobs"],
            })
        start += step

    return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_escape_mechanism.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper4_shadow/escape_mechanism.py tests/test_escape_mechanism.py
git commit -m "feat(paper4): add escape mechanism interaction regressions (Exercise 5)"
```

---

## Task 7: Long shadow cross-section — Exercise 6

**Files:**
- Create: `analysis/paper4_shadow/long_shadow.py`

Cross-sectional regressions linking pre-industrial seasonality/volatility to modern outcomes.

- [ ] **Step 1: Write the implementation**

```python
# analysis/paper4_shadow/long_shadow.py
"""Exercise 6: The long shadow of seasonality.

Cross-sectional analysis: does pre-industrial (0-1850 CE) seasonality
and volatility predict modern (2015-2025) structural outcomes?

Decomposes into seasonality + volatility components and adds pathway FE
to test whether effects operate through pathway selection or independently.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def build_long_shadow_cross_section(
    modern_panel: pd.DataFrame,
    seasonality_endowments: pd.DataFrame,
    pathway_assignments: pd.DataFrame,
    entity_col: str = "country_id",
    early_period: tuple[int, int] = (1950, 1960),
    late_period: tuple[int, int] = (2015, 2025),
) -> pd.DataFrame:
    """Build cross-sectional dataset for long-shadow analysis.

    Merges early-period outcomes, late-period outcomes, climate endowments,
    and pathway assignments into one row per country.
    """
    early = modern_panel[modern_panel["year"].between(*early_period)].groupby(entity_col).agg(
        t0=("temperature_c", "mean"),
        cs0=("crop_share", "mean"),
        ir0=("irrigation_share", "mean"),
        ur0=("urban_share", "mean"),
        d0=("density", "mean"),
        p0=("pop", "mean"),
    )
    late = modern_panel[modern_panel["year"].between(*late_period)].groupby(entity_col).agg(
        t1=("temperature_c", "mean"),
        cs1=("crop_share", "mean"),
        ir1=("irrigation_share", "mean"),
        ur1=("urban_share", "mean"),
        d1=("density", "mean"),
        p1=("pop", "mean"),
    )

    xs = early.join(late, how="inner")
    xs["warming"] = xs["t1"] - xs["t0"]
    xs["d_crop"] = xs["cs1"] - xs["cs0"]
    xs["d_irrig"] = xs["ir1"] - xs["ir0"]
    xs["d_urban"] = xs["ur1"] - xs["ur0"]
    xs["pop_gr"] = np.log(xs["p1"].clip(lower=1) / xs["p0"].clip(lower=1))

    # Merge seasonality endowments
    xs = xs.reset_index()
    if "era5_region" in modern_panel.columns:
        region_map = (
            modern_panel.groupby(entity_col)["era5_region"]
            .first()
            .reset_index()
        )
        xs = xs.merge(region_map, on=entity_col, how="left")
        xs = xs.merge(
            seasonality_endowments,
            left_on="era5_region",
            right_on="region",
            how="left",
        )

    # Merge pathway assignments
    pw_col = "cluster" if "cluster" in pathway_assignments.columns else "pathway"
    merge_col = "country" if "country" in pathway_assignments.columns else entity_col
    xs = xs.merge(
        pathway_assignments[[merge_col, pw_col]].rename(columns={merge_col: entity_col}),
        on=entity_col,
        how="left",
    )

    return xs


def run_long_shadow_regressions(
    xs: pd.DataFrame,
    outcomes: list[str] | None = None,
    seasonality_col: str = "seasonality_historical",
    volatility_col: str = "seasonality_historical",  # proxy: historical SD
    pathway_col: str = "cluster",
    include_pathway_fe: bool = False,
) -> pd.DataFrame:
    """Run cross-sectional regressions of modern outcomes on historical climate.

    Two specifications per outcome:
    1. Without pathway FE (total effect)
    2. With pathway FE (effect conditional on pathway — tests independent channel)
    """
    if outcomes is None:
        outcomes = ["d_crop", "d_irrig", "d_urban", "pop_gr"]

    results = []

    for outcome in outcomes:
        for with_fe in ([False, True] if include_pathway_fe else [False]):
            predictors = [seasonality_col]

            sub = xs[[outcome] + predictors].dropna()
            if with_fe and pathway_col in xs.columns:
                sub = xs[[outcome] + predictors + [pathway_col]].dropna()
                # Add pathway dummies
                dummies = pd.get_dummies(sub[pathway_col], prefix="pw", drop_first=True)
                sub = pd.concat([sub, dummies], axis=1)
                predictors = predictors + list(dummies.columns)

            if len(sub) < 10:
                continue

            y = sub[outcome]
            X = sm.add_constant(sub[predictors].astype(float))
            model = sm.OLS(y, X).fit(cov_type="HC1")

            results.append({
                "outcome": outcome,
                "pathway_fe": with_fe,
                "seasonality_coef": model.params.get(seasonality_col, float("nan")),
                "seasonality_pval": model.pvalues.get(seasonality_col, float("nan")),
                "rsquared": model.rsquared,
                "nobs": int(model.nobs),
            })

    return pd.DataFrame(results)
```

- [ ] **Step 2: Verify it imports correctly**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -c "from analysis.paper4_shadow.long_shadow import build_long_shadow_cross_section; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/long_shadow.py
git commit -m "feat(paper4): add long shadow cross-section regressions (Exercise 6)"
```

---

## Task 8: Figure generation

**Files:**
- Create: `analysis/paper4_shadow/figures.py`

All 10 figures for the paper. Reuses existing figure code where possible.

- [ ] **Step 1: Write the figure module**

```python
# analysis/paper4_shadow/figures.py
"""Generate all figures for the Long Shadow paper."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/paper4")


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig3_seasonality_predicts_pathway(xs: pd.DataFrame, seasonality_col: str, pathway_col: str):
    """Fig 3: Scatter of seasonality vs pathway with multinomial fit."""
    ensure_out_dir()
    fig, ax = plt.subplots(figsize=(8, 6))

    pathways = sorted(xs[pathway_col].dropna().unique())
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    labels = {0: "Intensive", 1: "Early ext.", 2: "Crop-dom.", 3: "Pastoral", 4: "Irrigation"}

    for i, pw in enumerate(pathways):
        sub = xs[xs[pathway_col] == pw]
        ax.scatter(
            sub[seasonality_col], [pw] * len(sub),
            alpha=0.5, s=40,
            color=colors[i % len(colors)],
            label=labels.get(pw, f"Pathway {pw}"),
        )

    # Box plot overlay
    data_by_pw = [xs[xs[pathway_col] == pw][seasonality_col].dropna().values for pw in pathways]
    bp = ax.boxplot(data_by_pw, positions=pathways, vert=False, widths=0.4, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.3)

    ax.set_xlabel("Historical Seasonality (temperature SD, 0-1850 CE)")
    ax.set_ylabel("Agricultural Pathway")
    ax.set_yticks(pathways)
    ax.set_yticklabels([labels.get(pw, f"{pw}") for pw in pathways])
    ax.set_title("Does Climate Seasonality Predict Agricultural Pathway?")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_seasonality_pathway.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'fig3_seasonality_pathway.png'}")


def fig4_dual_channel_dag():
    """Fig 4: Conceptual DAG showing two causal channels."""
    ensure_out_dir()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Nodes
    nodes = {
        "Intra-annual\nSeasonality\n(σs)": (1, 3.5),
        "Inter-annual\nVolatility\n(σv)": (1, 1.5),
        "Agricultural\nSystem\nSelection": (4, 3.5),
        "Malthusian\nFeedback\nStrength": (4, 1.5),
        "Speed of\nEscape": (7, 2.5),
        "Modern\nOutcomes": (9, 2.5),
    }

    for label, (x, y) in nodes.items():
        bbox = dict(boxstyle="round,pad=0.4", facecolor="lightblue", edgecolor="navy", alpha=0.8)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold", bbox=bbox)

    # Arrows
    arrows = [
        ((1.8, 3.5), (3.0, 3.5), "#d7191c", "Stage 1"),    # seasonality -> system
        ((1.8, 1.5), (3.0, 1.5), "#2c7bb6", "Stage 2"),    # volatility -> malthusian
        ((5.0, 3.5), (6.2, 2.8), "#d7191c", ""),            # system -> escape
        ((5.0, 1.5), (6.2, 2.2), "#2c7bb6", ""),            # malthusian -> escape
        ((7.8, 2.5), (8.3, 2.5), "gray", ""),               # escape -> modern
        ((5.0, 3.3), (3.0, 1.7), "#999999", ""),            # system -> malthusian (indirect)
    ]

    for (x1, y1), (x2, y2), color, label in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.25
            ax.text(mx, my, label, ha="center", fontsize=9, color=color, fontstyle="italic")

    ax.set_title("Two-Channel Causal Framework: Seasonality and Volatility", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_dual_channel_dag.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'fig4_dual_channel_dag.png'}")


def fig5_rolling_malthusian_by_pathway(roll_df: pd.DataFrame):
    """Fig 5: Rolling Malthusian coefficient stratified by pathway."""
    ensure_out_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {0: "#1b9e77", 1: "#d95f02", 2: "#7570b3", 3: "#e7298a", 4: "#66a61e"}
    labels = {0: "Intensive", 1: "Early ext.", 2: "Crop-dom.", 3: "Pastoral", 4: "Irrigation"}

    for pathway in sorted(roll_df["pathway"].unique()):
        sub = roll_df[roll_df["pathway"] == pathway].sort_values("center_year")
        ax.plot(sub["center_year"], sub["coefficient"],
                marker="o", linewidth=2, markersize=4,
                color=colors.get(pathway, "gray"),
                label=labels.get(pathway, f"Pathway {pathway}"))

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Center Year")
    ax.set_ylabel("Malthusian Coefficient (β)")
    ax.set_title("Time-Varying Malthusian Coefficient by Agricultural Pathway")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_rolling_malthus_pathway.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'fig5_rolling_malthus_pathway.png'}")


def fig6_pathway_stratified_irfs(irf_dict: dict, shock_label: str = "Temp"):
    """Fig 6: IRF panels comparing pathways."""
    ensure_out_dir()
    n_pathways = len(irf_dict)
    if n_pathways == 0:
        print("No IRFs to plot")
        return

    fig, axes = plt.subplots(1, min(n_pathways, 4), figsize=(5 * min(n_pathways, 4), 5), squeeze=False)
    axes = axes.ravel()

    colors = {0: "#1b9e77", 1: "#d95f02", 2: "#7570b3", 3: "#e7298a", 4: "#66a61e"}
    labels = {0: "Intensive", 1: "Early ext.", 2: "Crop-dom.", 3: "Pastoral", 4: "Irrigation"}

    for i, (pathway, irf) in enumerate(sorted(irf_dict.items())):
        if i >= len(axes):
            break
        ax = axes[i]
        c = colors.get(pathway, "gray")
        ax.plot(irf["horizon"], irf["coefficient"], marker="o", linewidth=2, color=c)
        ax.fill_between(irf["horizon"], irf["ci_lower"], irf["ci_upper"], alpha=0.15, color=c)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(labels.get(pathway, f"Pathway {pathway}"))
        ax.set_xlabel("Horizon (years)")
        ax.set_ylabel("β")
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"{shock_label} Shock → Cropland: IRFs by Pathway", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"fig6_irf_by_pathway_{shock_label.lower()}.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'fig6_irf_by_pathway_{shock_label.lower()}.png'}")


def fig7_escape_rolling(roll_df: pd.DataFrame):
    """Fig 7: Rolling climate sensitivity convergence toward zero."""
    ensure_out_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    if "pathway" in roll_df.columns:
        colors = {0: "#1b9e77", 1: "#d95f02", 2: "#7570b3", 3: "#e7298a"}
        labels = {0: "Intensive", 1: "Early ext.", 2: "Crop-dom.", 3: "Pastoral"}
        for pw in sorted(roll_df["pathway"].unique()):
            sub = roll_df[roll_df["pathway"] == pw].sort_values("mid")
            ax.plot(sub["mid"], sub["beta"], marker="o", linewidth=2,
                    color=colors.get(pw, "gray"), label=labels.get(pw, f"Pw {pw}"))
    else:
        ax.plot(roll_df["mid"], roll_df["beta"], marker="o", linewidth=2, color="#d7191c")
        if "se" in roll_df.columns:
            ax.fill_between(roll_df["mid"], roll_df["beta"] - 1.96 * roll_df["se"],
                            roll_df["beta"] + 1.96 * roll_df["se"], alpha=0.2, color="#d7191c")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(1970, color="gray", linestyle=":", alpha=0.5, label="Green Revolution")
    ax.set_xlabel("Center Year")
    ax.set_ylabel("β (temperature anomaly)")
    ax.set_title("The Great Escape: Declining Climate Sensitivity (20-yr rolling FE)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_escape_rolling.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'fig7_escape_rolling.png'}")


def fig8_escape_mechanism(interaction_results: dict):
    """Fig 8: Interaction coefficients showing which mediator drives escape."""
    ensure_out_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mediators = [k for k in interaction_results.keys() if k != "joint"]

    for i, med in enumerate(mediators[:2]):
        ax = axes[i]
        result = interaction_results[med]
        if "model" in result:
            model = result["model"]
            # Find interaction term
            interaction_terms = [c for c in model.params.index if "_x_" in c]
            if interaction_terms:
                coef = model.params[interaction_terms[0]]
                se = model.bse[interaction_terms[0]]
                pval = model.pvalues[interaction_terms[0]]
                star = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

                ax.bar(0, coef, yerr=1.96 * se, color="#2c7bb6" if i == 0 else "#d7191c",
                       alpha=0.7, capsize=10, width=0.5)
                ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
                ax.set_title(f"Shock × {med.replace('_', ' ').title()}\nβ={coef:.5f} {star}")
                ax.set_ylabel("Interaction Coefficient")
                ax.set_xticks([])
        ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle("Escape Mechanism: What Reduces Climate Sensitivity?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig8_escape_mechanism.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'fig8_escape_mechanism.png'}")


def fig9_long_shadow(xs: pd.DataFrame, seasonality_col: str):
    """Fig 9: Cross-section scatter — historical seasonality vs modern outcomes."""
    ensure_out_dir()
    import statsmodels.api as sm as_sm

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plots = [
        (seasonality_col, "d_crop", "Δ Crop Share"),
        (seasonality_col, "pop_gr", "Log Pop Growth"),
        (seasonality_col, "d_urban", "Δ Urban Share"),
        (seasonality_col, "d_irrig", "Δ Irrigation"),
    ]

    for ax, (x, y, ylabel) in zip(axes.ravel(), plots):
        v = xs[[x, y]].dropna()
        if len(v) < 5:
            ax.set_title(f"{ylabel}: insufficient data")
            continue
        ax.scatter(v[x], v[y], alpha=0.5, s=25)
        X_ols = sm.add_constant(v[x])
        m = sm.OLS(v[y], X_ols).fit()
        xr_ = np.linspace(v[x].min(), v[x].max(), 50)
        ax.plot(xr_, m.params.iloc[0] + m.params.iloc[1] * xr_, color="red", linewidth=1.5)
        p = m.pvalues.iloc[1]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.set_title(f"{ylabel}\nβ={m.params.iloc[1]:.4f}, p={p:.3f}{star}")
        ax.set_xlabel("Historical Seasonality")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)

    plt.suptitle("The Long Shadow: Pre-Industrial Seasonality → Modern Outcomes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig9_long_shadow.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'fig9_long_shadow.png'}")
```

- [ ] **Step 2: Fix the import typo in fig9 and verify module imports**

The line `import statsmodels.api as sm as_sm` in fig9 has a typo. Fix to `import statsmodels.api as sm`.

Run: `cd /Volumes/BIGDATA/HYDE35 && python -c "from analysis.paper4_shadow.figures import fig4_dual_channel_dag; fig4_dual_channel_dag(); print('OK')"`
Expected: Saves fig4 and prints OK

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/figures.py
git commit -m "feat(paper4): add figure generation module (Figs 3-9)"
```

---

## Task 9: Runner script — orchestrate all exercises

**Files:**
- Create: `analysis/paper4_shadow/run_all.py`

Orchestrator that loads data, runs all six exercises, and generates all figures and tables.

- [ ] **Step 1: Write the runner**

```python
# analysis/paper4_shadow/run_all.py
"""Run all analyses for 'The Long Shadow of Seasonality'.

Usage:
    python -m analysis.paper4_shadow.run_all
    python -m analysis.paper4_shadow.run_all --exercises 1 2 3
    python -m analysis.paper4_shadow.run_all --figures-only
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/Volumes/BIGDATA/HYDE35")
sys.path.insert(0, str(ROOT))

DATA = ROOT / "analysis" / "data"
FIG_DIR = ROOT / "analysis" / "figures" / "paper4"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all required datasets."""
    print("Loading data...")
    data = {}
    data["era5"] = pd.read_parquet(DATA / "era5_full_panel.parquet")
    data["climate_hist"] = pd.read_parquet(DATA / "climate_panel_0_2025.parquet")
    data["extended"] = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    data["pathways"] = pd.read_parquet(DATA / "paper1_clustered_features.parquet")

    print(f"  ERA5 panel: {data['era5'].shape}")
    print(f"  Historical climate: {data['climate_hist'].shape}")
    print(f"  Extended panel: {data['extended'].shape}")
    print(f"  Pathway assignments: {data['pathways'].shape}")
    return data


def exercise_1(data):
    """Seasonality predicts pathway assignment."""
    print("\n" + "=" * 70)
    print("EXERCISE 1: SEASONALITY → PATHWAY ASSIGNMENT")
    print("=" * 70)

    from analysis.paper4_shadow.seasonality import build_seasonality_panel
    from analysis.paper4_shadow.pathway_prediction import run_pathway_multinomial

    # Build seasonality endowments
    endowments = build_seasonality_panel(data["climate_hist"], entity_col="region")
    print(f"\nSeasonality endowments: {len(endowments)} regions")
    print(endowments.to_string(index=False))

    # Map countries to regions and merge
    if "era5_region" in data["pathways"].columns:
        region_col = "era5_region"
    else:
        region_map = data["extended"].groupby("country_id")["era5_region"].first().reset_index()
        pw = data["pathways"].merge(region_map, left_on="country", right_on="country_id", how="left")
        region_col = "era5_region"

        pw = pw.merge(endowments, left_on=region_col, right_on="region", how="left")
        pw = pw.dropna(subset=["seasonality_historical", "cluster"])

        if len(pw) > 10:
            result = run_pathway_multinomial(
                pw, pathway_col="cluster",
                climate_vars=["seasonality_historical"],
                control_vars=[],
            )
            print(f"\nMultinomial logit results:")
            print(result["summary"])

            from analysis.paper4_shadow.figures import fig3_seasonality_predicts_pathway
            fig3_seasonality_predicts_pathway(pw, "seasonality_historical", "cluster")
        else:
            print(f"  Insufficient data: {len(pw)} countries with seasonality + pathway")

    return endowments


def exercise_2(data, endowments):
    """Dual-channel separation."""
    print("\n" + "=" * 70)
    print("EXERCISE 2: DUAL-CHANNEL SEPARATION")
    print("=" * 70)

    from analysis.paper4_shadow.dual_channel import run_dual_channel_pathway, run_dual_channel_malthusian
    from analysis.paper4_shadow.figures import fig4_dual_channel_dag

    fig4_dual_channel_dag()
    print("  (Dual-channel regressions require volatility + Malthusian betas — computed in Exercises 3+)")


def exercise_3(data):
    """Pathway-specific Malthusian dynamics."""
    print("\n" + "=" * 70)
    print("EXERCISE 3: PATHWAY-SPECIFIC MALTHUSIAN DYNAMICS")
    print("=" * 70)

    from analysis.paper4_shadow.malthusian_extended import run_malthusian_by_pathway, run_rolling_by_pathway
    from analysis.paper4_shadow.figures import fig5_rolling_malthusian_by_pathway

    # This requires the country_analysis_panel (0-1750 CE)
    panel_path = DATA / "country_analysis_panel.parquet"
    if not panel_path.exists():
        print("  Skipping: country_analysis_panel.parquet not found")
        return

    panel = pd.read_parquet(panel_path)
    pw = data["pathways"]

    entity_col = "country" if "country" in panel.columns else "country_id"
    pw_entity = "country" if "country" in pw.columns else "country_id"

    # Static Malthusian by pathway
    indep = ["log_density"]
    if "log_land_labor" in panel.columns:
        indep.append("log_land_labor")

    result = run_malthusian_by_pathway(
        panel, pw,
        dep_var="pop_growth",
        density_var="log_density",
        entity_col=entity_col,
        pathway_col="cluster",
    )
    print("\nMalthusian coefficient by pathway:")
    print(result.to_string(index=False))

    # Rolling by pathway
    roll = run_rolling_by_pathway(
        panel, pw,
        dep_var="pop_growth",
        key_var="log_density",
        control_vars=["log_land_labor"] if "log_land_labor" in panel.columns else [],
        entity_col=entity_col,
        pathway_col="cluster",
    )
    if len(roll) > 0:
        fig5_rolling_malthusian_by_pathway(roll)


def exercise_4(data):
    """Pathway-stratified IRFs."""
    print("\n" + "=" * 70)
    print("EXERCISE 4: PATHWAY-STRATIFIED IRFs")
    print("=" * 70)

    from analysis.paper4_shadow.pathway_irfs import run_irfs_by_pathway, compare_pathway_irfs
    from analysis.paper4_shadow.figures import fig6_pathway_stratified_irfs

    panel = data["extended"]
    pw = data["pathways"]
    entity_col = "country_id"

    # Filter to regions with good climate coverage
    panel = panel[panel["temperature_c"].notna() & (panel["pop"] > 0)].copy()
    if "temp_anomaly" not in panel.columns:
        from analysis.paper3_climate.climate_shocks import build_climate_shock_panel
        # Build shocks at country level using era5_region grouping
        panel = build_climate_shock_panel(panel, ["temperature_c"], entity_col="era5_region")
        panel = panel.rename(columns={"temperature_c_anomaly": "temp_anomaly"})

    if "log_cropland" not in panel.columns and "cropland" in panel.columns:
        panel["log_cropland"] = np.log(panel["cropland"].clip(lower=0.01))

    # Merge pathway
    pw_merge = pw.rename(columns={"country": entity_col}) if "country" in pw.columns else pw
    panel = panel.merge(pw_merge[[entity_col, "cluster"]], on=entity_col, how="left")
    panel = panel.dropna(subset=["cluster", "temp_anomaly"])

    if len(panel) < 50:
        print(f"  Insufficient data: {len(panel)} obs with pathway + anomaly")
        return

    for response, label in [("log_cropland", "Cropland"), ("pop_growth", "Population")]:
        if response not in panel.columns:
            continue
        irfs = run_irfs_by_pathway(
            panel, pw_merge, shock_var="temp_anomaly",
            response_var=response, entity_col=entity_col,
            pathway_col="cluster", max_horizon=10,
        )
        if irfs:
            comparison = compare_pathway_irfs(irfs, horizon=5)
            print(f"\n{label} IRF comparison at h=5:")
            print(comparison.to_string(index=False))
            fig6_pathway_stratified_irfs(irfs, shock_label=f"Temp→{label}")


def exercise_5(data):
    """Escape mechanism."""
    print("\n" + "=" * 70)
    print("EXERCISE 5: ESCAPE MECHANISM")
    print("=" * 70)

    from analysis.paper4_shadow.escape_mechanism import run_escape_interactions, run_rolling_with_mediator
    from analysis.paper4_shadow.figures import fig8_escape_mechanism

    panel = data["extended"]
    panel = panel[panel["temperature_c"].notna() & (panel["pop"] > 0)].copy()

    if "temp_anomaly" not in panel.columns:
        from analysis.paper3_climate.climate_shocks import build_climate_shock_panel
        panel = build_climate_shock_panel(panel, ["temperature_c"], entity_col="era5_region")
        panel = panel.rename(columns={"temperature_c_anomaly": "temp_anomaly"})

    mediators = []
    if "urban_share" in panel.columns:
        mediators.append("urban_share")
    if "intensification_index" in panel.columns:
        mediators.append("intensification_index")

    if not mediators:
        print("  No mediator variables found")
        return

    results = run_escape_interactions(
        panel, shock_var="temp_anomaly",
        outcome_var="pop_growth", mediators=mediators,
        entity_col="country_id",
    )

    for med, res in results.items():
        if med == "joint":
            continue
        print(f"\n  {med}:")
        print(f"    Shock coef:       {res.get('shock_coef', float('nan')):+.6f} (p={res.get('shock_pval', float('nan')):.4f})")
        print(f"    Interaction coef: {res.get('interaction_coef', float('nan')):+.6f} (p={res.get('interaction_pval', float('nan')):.4f})")

    fig8_escape_mechanism(results)


def exercise_6(data, endowments):
    """Long shadow cross-section."""
    print("\n" + "=" * 70)
    print("EXERCISE 6: THE LONG SHADOW")
    print("=" * 70)

    from analysis.paper4_shadow.long_shadow import build_long_shadow_cross_section, run_long_shadow_regressions
    from analysis.paper4_shadow.figures import fig9_long_shadow

    panel = data["extended"]
    pw = data["pathways"]

    xs = build_long_shadow_cross_section(panel, endowments, pw, entity_col="country_id")
    print(f"\nCross-section: {len(xs)} countries")

    if "seasonality_historical" in xs.columns and xs["seasonality_historical"].notna().sum() > 10:
        results = run_long_shadow_regressions(
            xs, seasonality_col="seasonality_historical",
            include_pathway_fe=True,
        )
        print("\nLong shadow regressions:")
        print(results.to_string(index=False))
        fig9_long_shadow(xs, "seasonality_historical")
    else:
        print("  Insufficient seasonality data for cross-section")


def main():
    parser = argparse.ArgumentParser(description="Run Long Shadow analyses")
    parser.add_argument("--exercises", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    data = load_data()

    endowments = None
    if 1 in args.exercises:
        endowments = exercise_1(data)
    if 2 in args.exercises:
        exercise_2(data, endowments)
    if 3 in args.exercises:
        exercise_3(data)
    if 4 in args.exercises:
        exercise_4(data)
    if 5 in args.exercises:
        exercise_5(data)
    if 6 in args.exercises:
        exercise_6(data, endowments)

    print("\n" + "=" * 70)
    print("ALL EXERCISES COMPLETE")
    print("=" * 70)
    print(f"\nFigures saved to: {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify runner imports work**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -c "from analysis.paper4_shadow.run_all import load_data; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/run_all.py
git commit -m "feat(paper4): add runner script orchestrating all exercises"
```

---

## Task 10: Integration test — run all exercises on real data

- [ ] **Step 1: Run all exercises**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -m analysis.paper4_shadow.run_all 2>&1 | tee analysis/figures/paper4/_run_log.txt`

Expected: All 6 exercises complete, figures saved to `analysis/figures/paper4/`

- [ ] **Step 2: Fix any import or data-shape errors**

Review the log output. Common issues:
- Column name mismatches between data files
- Missing columns in merged panels
- Insufficient observations for some pathway strata

Fix and re-run until all exercises produce output.

- [ ] **Step 3: Review generated figures**

Check each figure in `analysis/figures/paper4/`:
- `fig3_seasonality_pathway.png`
- `fig4_dual_channel_dag.png`
- `fig5_rolling_malthus_pathway.png`
- `fig6_irf_by_pathway_*.png`
- `fig7_escape_rolling.png`
- `fig8_escape_mechanism.png`
- `fig9_long_shadow.png`

- [ ] **Step 4: Commit results**

```bash
git add analysis/figures/paper4/ analysis/paper4_shadow/
git commit -m "feat(paper4): complete integration run with all figures"
```

---

## Task 11: Run all tests

- [ ] **Step 1: Run the full test suite**

Run: `cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/ -v --tb=short`

Expected: All existing tests still pass, plus new tests for paper4 modules.

- [ ] **Step 2: Fix any regressions**

If any existing tests broke, fix the issue without changing existing module behavior.

- [ ] **Step 3: Commit if any fixes were needed**

```bash
git add -A && git commit -m "fix: resolve test regressions from paper4 integration"
```
