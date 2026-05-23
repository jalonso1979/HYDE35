"""NAO + AMO + ENSO teleconnection reconstructions from NOAA NCEI paleo archive.

BLOCKABLE: NOAA endpoints may 403 or redirect. If unreachable, log and exit;
downstream IV figure (Fig 14) will skip gracefully.
"""
from __future__ import annotations
from pathlib import Path
import urllib.error
import urllib.request
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility")
OUT = ROOT / "teleconnection_panel.parquet"

URLS = {
    "nao": "https://www.ncei.noaa.gov/pub/data/paleo/historical/europe/luterbacher2002/nao_djf-trim.dat",
    "amo": "https://www.ncei.noaa.gov/pub/data/paleo/contributions_by_author/mann2009/mann2009amorecon.txt",
    "enso": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/enso/cook2008nino34.txt",
}


def _try_fetch(url: str, timeout: int = 15) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return None


def _parse_two_col(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "%", "*", "Citation", "References")):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            year = int(float(parts[0]))
            val = float(parts[1])
            rows.append({"year": year, "value": val})
        except ValueError:
            continue
    return pd.DataFrame(rows)


def build_teleconnection_panel(write: bool = False) -> pd.DataFrame:
    series = {}
    for name, url in URLS.items():
        text = _try_fetch(url)
        if text is None:
            raise RuntimeError(
                f"Could not fetch {name} from {url}. BLOCKED: manually download "
                f"and place at {ROOT / (name + '.txt')} in plain 'year value' format."
            )
        parsed = _parse_two_col(text)
        if parsed.empty:
            raise RuntimeError(f"Parsed 0 rows from {name}; check format")
        series[name] = parsed.rename(columns={"value": name})

    df = series["nao"]
    for name in ("amo", "enso"):
        df = df.merge(series[name], on="year", how="outer")
    df = df.sort_values("year").reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    try:
        df = build_teleconnection_panel(write=True)
        print(f"wrote {OUT}: {len(df)} rows")
    except RuntimeError as exc:
        print(f"BLOCKED: {exc}")
