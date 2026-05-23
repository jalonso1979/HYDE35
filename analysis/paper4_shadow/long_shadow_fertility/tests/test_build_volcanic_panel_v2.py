"""Phase 7 Pillar C2: volcanic panel splice (Sigl VSSI + Sato AOD rescaled)."""
import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_volcanic_panel_v2 import (
    splice_sigl_sato,
    BLOCKED,
)


def _make_sigl(years, vssi):
    return pd.DataFrame({"year": years, "vssi": vssi})


def _make_sato(years, aod):
    return pd.DataFrame({"year": years, "aod_max": aod})


def test_splice_continuous_years():
    sigl = _make_sigl(list(range(1421, 1901)), [0.0] * (1901 - 1421))
    sato = _make_sato(list(range(1850, 2013)), [0.01] * (2013 - 1850))
    out = splice_sigl_sato(sigl, sato)
    assert out is not BLOCKED
    assert out["year"].min() == 1421
    assert out["year"].max() == 2012
    years = sorted(out["year"].tolist())
    assert years == list(range(1421, 2013))


def test_splice_source_tagged():
    sigl = _make_sigl(list(range(1421, 1901)), [0.0] * 480)
    sato = _make_sato(list(range(1850, 2013)), [0.01] * 163)
    out = splice_sigl_sato(sigl, sato)
    pre_1900 = out[out["year"] < 1901]["source"].unique().tolist()
    post_1900 = out[out["year"] > 1900]["source"].unique().tolist()
    assert pre_1900 == ["Sigl_VSSI"]
    assert any("Sato" in s for s in post_1900)


def test_splice_rescaling_aligns_variance():
    """OLS-rescale on 1850-1900 overlap; post-rescale Sato should sit near
    the same magnitude as Sigl in the overlap."""
    rng = np.random.default_rng(0)
    sigl_v = rng.exponential(2.0, size=480)  # heavy-tailed VSSI values 1421-1900
    sigl = _make_sigl(list(range(1421, 1901)), sigl_v.tolist())
    # Sato AOD on overlap 1850-1900 has known linear relation to Sigl
    overlap_sigl = sigl.loc[sigl["year"].between(1850, 1900), "vssi"].to_numpy()
    overlap_aod = 0.5 + 0.3 * overlap_sigl + rng.normal(0, 0.1, size=len(overlap_sigl))
    post_aod = rng.exponential(0.5, size=2013 - 1901)
    sato_years = list(range(1850, 1901)) + list(range(1901, 2013))
    sato_aod = list(overlap_aod) + list(post_aod)
    sato = _make_sato(sato_years, sato_aod)
    out = splice_sigl_sato(sigl, sato)
    pre = out.loc[out["year"].between(1850, 1900), "vssi"].mean()
    post = out.loc[out["year"].between(1901, 1950), "vssi"].mean()
    # Means should be within an order of magnitude (loose check)
    assert 0.1 * pre < post < 10 * pre


def test_blocked_sato_returns_blocked():
    sigl = _make_sigl(list(range(1421, 1901)), [0.0] * 480)
    out = splice_sigl_sato(sigl, BLOCKED)
    assert out is BLOCKED
