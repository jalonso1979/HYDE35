"""Smoke tests for Fig 25 (regime FEVD) — stacking helper + share invariants."""
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.paper4_shadow.long_shadow_fertility.figures import fig25_regime_fevd as f25

JSON_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase11_regime_fevd.json"
)


@pytest.mark.skipif(not JSON_PATH.exists(), reason="phase11 JSON not yet generated")
def test_stack_matrix_shape_and_order():
    data = json.loads(JSON_PATH.read_text())
    entry = data["core_by_regime"]["malthusian"]
    mat = f25._stack_matrix(entry)
    # 4 shocks x 16 horizons.
    assert mat.shape == (len(f25.STACK_VARS), len(data["core_by_regime"]["horizons"]))


@pytest.mark.skipif(not JSON_PATH.exists(), reason="phase11 JSON not yet generated")
def test_stacked_shares_sum_to_one():
    data = json.loads(JSON_PATH.read_text())
    for regime in ("malthusian", "modern"):
        mat = f25._stack_matrix(data["core_by_regime"][regime])
        col_sums = mat.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones_like(col_sums), atol=1e-6)


@pytest.mark.skipif(not JSON_PATH.exists(), reason="phase11 JSON not yet generated")
def test_figure_writes_files(tmp_path, monkeypatch):
    # Redirect outputs to a temp dir so the test does not clobber real figures.
    monkeypatch.setattr(f25, "FIG_DIR", tmp_path)
    monkeypatch.setattr(f25, "PAPER_FIG_DIR", tmp_path / "paper")
    out = f25.make_fig25()
    assert out["pdf"].exists()
    assert out["png"].exists()
    # Headline: weather share present and finite for both regimes.
    ws = out["weather_share_h15"]
    assert np.isfinite(ws["malthusian"]) and np.isfinite(ws["modern"])
