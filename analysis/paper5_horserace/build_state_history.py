"""Build Putterman-Weil ancestry-adjusted state-history index.

Source: Putterman & Weil (2010, QJE) "Post-1500 Population Flows and
the Long-Run Determinants of Economic Growth and Inequality",
replication archive. Statehist v3 (normalized, 0.5 discount rate).

The raw file is `statehist.xls` (sheet `sratiov3`), downloaded from
Louis Putterman's Brown faculty page (Google Drive):
https://sites.google.com/brown.edu/louis-putterman

Ancestry adjustment uses the World Migration Matrix (WMM) v1.1:
  `pw_migration_matrix_v1p1.xlsx`
WMM layout: rows = destination countries, columns = source countries.
WMM[j, s] = fraction of destination j's year-2000 population with
ancestry from source country s. Row sums = 1.

For each destination j:
  state_hist_pwadj[j] = sum_s( WMM[j,s] * state_hist[s] )
                        / sum_s( WMM[j,s] )   [normalise for partial coverage]

Output: analysis/data/deep_determinants/state_history_pw.parquet
Columns: iso3, state_hist, state_hist_pwadj, source
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
RAW = ROOT / "analysis/data/deep_determinants/_raw/putterman_weil_2010"
OUT = ROOT / "analysis/data/deep_determinants/state_history_pw.parquet"

# The state-history file uses column `statehistn05v3`:
#   "v3" = third revision; "n" = normalised to [0,1];
#   "05" = 0.5 discount rate (each 50-year period weighted by 0.5 relative
#          to the most recent 50-year window). This is the standard PW measure.
SH_FILE = RAW / "statehist.xls"
SH_SHEET = "sratiov3"
SH_NORM_COL = "statehistn05v3"

WMM_FILE = RAW / "pw_migration_matrix_v1p1.xlsx"


def load_state_history() -> pd.DataFrame:
    """Load and parse the raw state-history Excel file.

    Returns a DataFrame with columns: iso3, state_hist
    """
    if not SH_FILE.exists():
        raise FileNotFoundError(
            f"State-history file missing: {SH_FILE}\n"
            "Download from https://sites.google.com/brown.edu/louis-putterman "
            "(State Antiquity Index > Statehist.xls)"
        )

    df = pd.read_excel(SH_FILE, sheet_name=SH_SHEET, header=1)
    # After header=1: row 1 of original file becomes column names.
    # Columns present: wbcode, wbname, aosnew1..39, statehist*v3 variants.

    if "wbcode" not in df.columns:
        raise KeyError(f"Expected 'wbcode' column in {SH_FILE}:{SH_SHEET}")
    if SH_NORM_COL not in df.columns:
        available = [c for c in df.columns if "statehist" in c.lower()]
        raise KeyError(
            f"Expected '{SH_NORM_COL}' column; available state-history cols: {available}"
        )

    df = df.rename(columns={"wbcode": "iso3", SH_NORM_COL: "state_hist"})
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()

    # Keep only valid ISO3 codes and non-null state_hist
    df = df[df["iso3"].str.match(r"^[A-Z]{3}$")].copy()
    df = df[["iso3", "state_hist"]].dropna(subset=["state_hist"])
    df["state_hist"] = pd.to_numeric(df["state_hist"], errors="coerce")
    df = df.dropna(subset=["state_hist"])

    return df.reset_index(drop=True)


def load_wmm() -> pd.DataFrame:
    """Load the World Migration Matrix.

    Returns a DataFrame with wbcode (uppercase) as index and lowercase
    source-country codes as columns, values = ancestry fractions.
    """
    if not WMM_FILE.exists():
        raise FileNotFoundError(f"World Migration Matrix missing: {WMM_FILE}")

    wmm = pd.read_excel(WMM_FILE, sheet_name=0)
    wmm["wbcode"] = wmm["wbcode"].astype(str).str.upper().str.strip()

    # Drop metadata columns; keep only source-fraction columns
    src_cols = [c for c in wmm.columns if c not in ("wbcode", "wbname", "update")]

    # Convert all source columns to numeric
    wmm[src_cols] = wmm[src_cols].apply(pd.to_numeric, errors="coerce")

    return wmm.set_index("wbcode")[src_cols]


def compute_pwadj(sh: pd.DataFrame, wmm: pd.DataFrame) -> pd.Series:
    """Compute ancestry-adjusted state-history for each destination country.

    Parameters
    ----------
    sh : DataFrame with columns iso3, state_hist (N_sh rows)
    wmm : DataFrame indexed by destination iso3, columns = source iso3 (lowercase)

    Returns
    -------
    Series indexed by destination iso3, values = ancestry-adjusted state_hist
    """
    # Build source lookup: uppercase iso3 -> state_hist
    sh_dict = sh.set_index("iso3")["state_hist"].to_dict()

    results = {}
    for dest_iso, src_row in wmm.iterrows():
        weighted_sh = 0.0
        total_weight = 0.0
        for src_lower, w in src_row.items():
            if pd.isna(w) or w <= 0:
                continue
            src_iso = src_lower.upper()
            if src_iso in sh_dict:
                weighted_sh += w * sh_dict[src_iso]
                total_weight += w
        if total_weight > 0:
            results[dest_iso] = weighted_sh / total_weight
        else:
            results[dest_iso] = np.nan

    return pd.Series(results, name="state_hist_pwadj")


def main() -> None:
    # 1. Load state history
    sh = load_state_history()
    print(f"Loaded state history: {len(sh)} countries")

    # 2. Load WMM and compute ancestry-adjusted values
    wmm = load_wmm()
    print(f"Loaded WMM: {len(wmm)} destination countries x {wmm.shape[1]} source countries")

    pwadj = compute_pwadj(sh, wmm)

    # 3. Merge: start from state_hist, left-join pwadj
    df = sh.copy()
    df = df.merge(
        pwadj.reset_index().rename(columns={"index": "iso3"}),
        on="iso3",
        how="left",
    )

    # Also add WMM destinations that may not be in state_hist (unlikely, but safe)
    # -- skip: we only want countries with a valid raw state_hist

    df["source"] = (
        "Putterman-Weil 2010 QJE state-history index v3, "
        "normalised (0.5 discount rate); ancestry adjustment via WMM v1.1"
    )

    # Validate ranges before writing
    assert df["state_hist"].between(0, 1).all(), "state_hist out of [0,1]"
    pwadj_valid = df["state_hist_pwadj"].dropna()
    if len(pwadj_valid) > 0:
        assert pwadj_valid.between(0, 1).all(), "state_hist_pwadj out of [0,1]"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    n_adj = df["state_hist_pwadj"].notna().sum()
    print(f"Wrote {OUT}")
    print(f"  {len(df)} countries with raw state_hist")
    print(f"  {n_adj} countries with ancestry-adjusted state_hist_pwadj")


if __name__ == "__main__":
    main()
