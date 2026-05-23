"""NASA GISS Sato Tau AOD fetcher (monthly stratospheric aerosol optical depth).

URL: https://data.giss.nasa.gov/modelforce/strataer/tau.line_2012.12.txt
Coverage: 1850-2012 monthly. Includes Krakatau 1883, Santa Maria 1902,
Katmai 1912, Agung 1963, El Chichon 1982, Pinatubo 1991.

Returns annual max AOD per year as a 2-column DataFrame (year, aod_max).
BLOCKED sentinel returned if fetch fails (no network, schema unexpected).
"""
from __future__ import annotations
from io import StringIO
from pathlib import Path
import urllib.request
import urllib.error
import pandas as pd

SATO_URL = "https://data.giss.nasa.gov/modelforce/strataer/tau.line_2012.12.txt"
BLOCKED = object()  # sentinel returned when fetch fails


def _parse_sato_text(text: str) -> pd.DataFrame:
    """Parse the Sato Tau line text. Header rows start with '#' or are non-numeric.

    Expected data layout (after header skip):
        decimal_year  global_AOD  NH_AOD  SH_AOD
    We use the global column (col index 1) and aggregate to annual max.
    """
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            float(parts[0])
        except ValueError:
            continue
        lines.append(s)
    if not lines:
        raise ValueError("No numeric data lines parsed from Sato file")

    df = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+", header=None,
                       names=["decimal_year", "global_aod", "nh_aod", "sh_aod"])
    df["year"] = df["decimal_year"].astype(int)
    annual = df.groupby("year", as_index=False)["global_aod"].max()
    annual = annual.rename(columns={"global_aod": "aod_max"})
    return annual


def fetch_sato_aod(cache_dir: Path | None = None,
                    raise_on_failure: bool = False) -> pd.DataFrame | object:
    """Fetch Sato Tau AOD annual maxima. Returns BLOCKED on failure.

    Parameters
    ----------
    cache_dir : optional directory to cache raw text + parsed parquet
    raise_on_failure : if True, propagate exceptions; else return BLOCKED
    """
    try:
        with urllib.request.urlopen(SATO_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        if raise_on_failure:
            raise
        return BLOCKED

    try:
        annual = _parse_sato_text(text)
    except (ValueError, KeyError, pd.errors.EmptyDataError):
        if raise_on_failure:
            raise
        return BLOCKED

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "sato_tau_raw.txt").write_text(text)
        annual.to_parquet(cache_dir / "sato_aod_annual.parquet", index=False)
    return annual


SATO_NETCDF_URL = "https://data.giss.nasa.gov/modelforce/strataer/tau_reff_Sato-Lacis.nc"


def fetch_sato_aod_netcdf(cache_dir: Path | None = None,
                            raise_on_failure: bool = False) -> pd.DataFrame | object:
    """Fetch Sato AOD via NetCDF (post-2020 NASA distribution).

    Returns DataFrame (year, aod_max) — global-mean stratospheric AOD aggregated
    to annual maximum across months. Returns BLOCKED on network/schema failure.
    """
    import tempfile
    try:
        import xarray as xr
    except ImportError:
        if raise_on_failure:
            raise
        return BLOCKED

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        nc_path = cache_dir / "tau_reff_Sato-Lacis.nc"
    else:
        nc_path = Path(tempfile.mkstemp(suffix=".nc")[1])

    if not nc_path.exists():
        try:
            with urllib.request.urlopen(SATO_NETCDF_URL, timeout=60) as resp:
                nc_path.write_bytes(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            if raise_on_failure:
                raise
            return BLOCKED

    try:
        ds = xr.open_dataset(nc_path)
        # Find the AOD variable — likely named 'tau' or 'aod' or 'optical_depth'
        candidates = [v for v in ds.data_vars if "tau" in v.lower() or "aod" in v.lower()]
        if not candidates:
            raise ValueError(f"No AOD variable in {list(ds.data_vars)}")
        aod_var = candidates[0]
        aod = ds[aod_var]

        # If 3D (time, lat, lon), take global mean over space
        spatial_dims = [d for d in aod.dims if d not in ("time", "month")]
        if spatial_dims:
            aod = aod.mean(dim=spatial_dims)

        df = aod.to_dataframe().reset_index()
        time_col = "time" if "time" in df.columns else "month"
        if df[time_col].dtype.kind == "M":
            df["year"] = pd.to_datetime(df[time_col]).dt.year
        else:
            # numeric month-since-epoch; assume month-since-1850 convention
            df["year"] = (df[time_col] // 12 + 1850).astype(int)
        annual = df.groupby("year", as_index=False)[aod_var].max().rename(columns={aod_var: "aod_max"})
    except Exception:  # noqa: BLE001
        if raise_on_failure:
            raise
        return BLOCKED

    if cache_dir is not None:
        annual.to_parquet(cache_dir / "sato_aod_annual_from_netcdf.parquet", index=False)
    return annual


if __name__ == "__main__":
    out_dir = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility")
    res = fetch_sato_aod(cache_dir=out_dir, raise_on_failure=False)
    if res is BLOCKED:
        print("BLOCKED: NASA GISS Sato fetch failed; commit fetcher stub only")
    else:
        print(f"fetched {len(res)} years; range {res['year'].min()}-{res['year'].max()}")
        print(res.head())
