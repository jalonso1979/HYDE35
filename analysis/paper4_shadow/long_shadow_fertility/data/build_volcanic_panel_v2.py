"""Splice Sigl VSSI (1700-1900) with NASA GISS Sato AOD (1850-2012) rescaled
to VSSI units via OLS on the 1850-1900 overlap.

Returns BLOCKED sentinel propagating from fetch_sato_aod if Sato is unavailable.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.fetch_sato_aod import (
    fetch_sato_aod,
    BLOCKED,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/volcanic_panel_v2.parquet")
OVERLAP = (1850, 1900)
MIN_OVERLAP_OBS = 10


def _load_sigl() -> pd.DataFrame:
    """Read existing Sigl VSSI panel (built in Phase 3)."""
    from analysis.paper4_shadow.long_shadow_fertility.data.build_sigl_volcanic_panel import (
        build_sigl_volcanic_panel,
    )
    df = build_sigl_volcanic_panel()
    if "vssi" not in df.columns:
        # adapt to whatever column name the Phase 3 builder uses
        for c in df.columns:
            if c.lower() in ("vssi", "vssi_tg", "vssi_tg_s", "sulfur_load_tg"):
                df = df.rename(columns={c: "vssi"})
                break
    # Phase 3 panel is a (iso3, year) long format; collapse to year-only since
    # VSSI is a global forcing series (identical across countries within a year).
    keep = df[["year", "vssi"]].dropna().drop_duplicates(subset=["year"]).copy()
    return keep.loc[keep["year"] <= 1900].reset_index(drop=True)


def splice_sigl_sato(sigl: pd.DataFrame,
                       sato: pd.DataFrame | object) -> pd.DataFrame | object:
    """Splice Sigl pre-1900 with rescaled Sato post-1900. Returns BLOCKED
    if Sato is the BLOCKED sentinel.

    Parameters
    ----------
    sigl : DataFrame with columns (year, vssi), coverage 1700-1900
    sato : DataFrame with columns (year, aod_max), or BLOCKED sentinel
    """
    if sato is BLOCKED:
        return BLOCKED

    sigl = sigl.copy()
    sato = sato.copy()
    sigl["source"] = "Sigl_VSSI"

    overlap = sigl.merge(sato, on="year").query(f"year >= {OVERLAP[0]} and year <= {OVERLAP[1]}")
    if len(overlap) >= MIN_OVERLAP_OBS:
        x = overlap["aod_max"].to_numpy()
        y = overlap["vssi"].to_numpy()
        # OLS through the origin: VSSI = b * AOD (physically, AOD=0 implies
        # no aerosol load, hence VSSI=0). Free-intercept polyfit was attempted
        # but extrapolates badly when post-1900 AOD distribution sits below
        # the overlap intercept; through-origin keeps the rescale on the same
        # ray and yields physically interpretable magnitudes post-splice.
        b = float((x * y).sum() / (x * x).sum())
        a = 0.0
        method = "Sato_AOD_rescaled"
    else:
        s_med = sato.loc[sato["year"].between(*OVERLAP), "aod_max"].median()
        v_med = sigl.loc[sigl["year"].between(*OVERLAP), "vssi"].median()
        if pd.isna(s_med) or pd.isna(v_med):
            a, b = 0.0, 1.0
        else:
            a, b = float(v_med - s_med), 1.0
        method = "Sato_AOD_median_shifted"

    post = sato.loc[sato["year"] > 1900, ["year", "aod_max"]].copy()
    post["vssi"] = a + b * post["aod_max"]
    post["source"] = method
    post = post[["year", "vssi", "source"]]

    spliced = pd.concat([sigl[["year", "vssi", "source"]], post], ignore_index=True)
    spliced = spliced.sort_values("year").drop_duplicates("year", keep="first")
    return spliced.reset_index(drop=True)


def build_volcanic_panel_v2(write: bool = False) -> pd.DataFrame | object:
    sato = fetch_sato_aod(cache_dir=OUT.parent, raise_on_failure=False)
    if sato is BLOCKED:
        # Phase 8: try NetCDF fallback (NASA migrated .txt -> NetCDF)
        from analysis.paper4_shadow.long_shadow_fertility.data.fetch_sato_aod import (
            fetch_sato_aod_netcdf,
        )
        sato = fetch_sato_aod_netcdf(cache_dir=OUT.parent, raise_on_failure=False)
        if sato is BLOCKED:
            return BLOCKED
    sigl = _load_sigl()
    out = splice_sigl_sato(sigl, sato)
    if out is BLOCKED:
        return BLOCKED
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)
    return out


if __name__ == "__main__":
    out = build_volcanic_panel_v2(write=True)
    if out is BLOCKED:
        print("BLOCKED: Sato fetch unavailable; volcanic panel v2 not built")
    else:
        print(f"wrote {OUT}: {len(out)} years; "
              f"{out['year'].min()}-{out['year'].max()}; "
              f"sources: {out['source'].value_counts().to_dict()}")
