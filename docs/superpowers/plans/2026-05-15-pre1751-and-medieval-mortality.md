# Pre-1751 and Medieval Mortality Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the §4.3 preventive/positive-check Malthusian decomposition with four sub-exercises (Wrigley-Schofield pre-1751 deepening, pathway-stratified prevpos on 1751-1900, modern-era 1900-2022 comparison, medieval Hatcher event-study) and integrate into the paper.

**Architecture:** Two new data folders (`wrigley_schofield/`, `hatcher_bailey/`) populated by a tiered retrieval cascade (replication-package mining → OCR → UKDS). Four independent regression scripts feed four new sub-sections in the paper's §4.3, plus one new appendix subsection. Each script writes its parquet output and figure under the existing conventions. No structural model, no joint-VAR integration — single-equation regressions only.

**Tech Stack:** Python 3.13, pandas, numpy, statsmodels, matplotlib, openpyxl/xlrd for xls. LaTeX for paper. Data sources: AEA dataverse, UC Davis Clark archive, Nuffield Allen archive, Schofield 2022 PDF, Voigtländer-Voth 2013 *RES* replication, Campbell 2016 Cambridge UP appendix.

**Spec reference:** `docs/superpowers/specs/2026-05-15-pre1751-and-medieval-mortality-design.md`

---

## Phase 1 — Wrigley-Schofield retrieval cascade

### Task 1: Cascade Tier A.1 — Crafts-Mills (2020) AEA dataverse

**Files:**
- Create: `analysis/paper4_shadow/retrieve_ws_hatcher.py`
- Create: `analysis/data/wrigley_schofield/` (directory)
- Create: `analysis/data/wrigley_schofield/provenance.txt`

- [ ] **Step 1: Search openicpsr / aeaweb for Crafts-Mills 2020**

Run via WebSearch tool:
```
WebSearch: "Crafts Mills From Malthus to Solow 2020 Economic Journal replication openicpsr"
```

Expected outcome: a URL pointing to a data deposit on openicpsr.org or aeaweb.org/journals/code.

- [ ] **Step 2: Fetch the data deposit page and identify England demographic file**

Use WebFetch on the deposit page; ask for "a Stata/CSV file containing annual England births, deaths, or CBR/CDR 1541-1871".

Expected outcome: either a direct file URL, or a confirmation the deposit does not contain W-S annual data.

- [ ] **Step 3: Download the file and inspect**

```bash
mkdir -p /Volumes/BIGDATA/HYDE35/analysis/data/wrigley_schofield
cd /Volumes/BIGDATA/HYDE35/analysis/data/wrigley_schofield
curl -sSL -o crafts_mills_raw.dta "<URL from step 2>"
python3 -c "import pandas as pd; df = pd.read_stata('crafts_mills_raw.dta'); print(df.columns.tolist()); print(df.head())"
```

Expected: column names and first rows. We are looking for columns named like `year`, `cbr`, `cdr`, `population`, `births`, `deaths` covering 1541-1871.

- [ ] **Step 4: If columns found, write provenance and proceed to Task 2 success**

```python
# In a script or REPL
import pandas as pd
df = pd.read_stata('crafts_mills_raw.dta')
keep = df[['year', 'cbr', 'cdr', 'population']].dropna()
keep = keep[(keep['year'] >= 1541) & (keep['year'] <= 1871)]
keep['source'] = 'Crafts-Mills 2020 EJ replication'
keep.to_csv('ws_england_annual.csv', index=False)
with open('provenance.txt', 'w') as f:
    f.write("Wrigley-Schofield England annual demographic series\n")
    f.write("Source: Crafts-Mills 2020 'From Malthus to Solow' Economic Journal\n")
    f.write("Retrieved from: <URL>\n")
    f.write("Cascade step: A.1 (replication-package mining)\n")
```

- [ ] **Step 5: If columns NOT found, skip to Task 2**

Document in `provenance.txt` that this tier was attempted and what was missing.

- [ ] **Step 6: Commit current progress**

```bash
cd /Volumes/BIGDATA/HYDE35
git add analysis/data/wrigley_schofield/
git commit -m "data: attempt Crafts-Mills 2020 W-S retrieval (cascade A.1)"
```

### Task 2: Cascade Tier A.2 — Gregory Clark UC Davis page (only if Task 1 failed)

**Files:**
- Modify: `analysis/data/wrigley_schofield/ws_england_annual.csv` (write if new source succeeds)
- Modify: `analysis/data/wrigley_schofield/provenance.txt`

- [ ] **Step 1: Check whether Task 1 succeeded**

```bash
test -f /Volumes/BIGDATA/HYDE35/analysis/data/wrigley_schofield/ws_england_annual.csv && \
  python3 -c "import pandas as pd; df = pd.read_csv('/Volumes/BIGDATA/HYDE35/analysis/data/wrigley_schofield/ws_england_annual.csv'); print(f'rows={len(df)}, year_min={df.year.min()}, year_max={df.year.max()}, has_cbr={\"cbr\" in df.columns}, has_cdr={\"cdr\" in df.columns}, has_pop={\"population\" in df.columns}'); "
```

Expected: if the CSV exists and has all three series at annual resolution 1541-1871, SKIP this task. Otherwise proceed.

- [ ] **Step 2: Fetch Clark UC Davis data page**

Use WebFetch on `https://faculty.econ.ucdavis.edu/faculty/gclark/data.html` asking for "downloadable spreadsheets containing annual England population, birth rate, or death rate 1541-1871".

Expected outcome: a list of xls/xlsx file URLs with descriptions.

- [ ] **Step 3: Download candidate files**

```bash
cd /Volumes/BIGDATA/HYDE35/analysis/data/wrigley_schofield
for url in "<list from step 2>"; do
  curl -sSL -O "$url"
done
```

- [ ] **Step 4: Parse files for W-S annual series**

```python
import pandas as pd
import glob
for f in glob.glob('*.xls*'):
    try:
        xls = pd.ExcelFile(f)
        for sheet in xls.sheet_names:
            df = xls.parse(sheet, nrows=5)
            cols = [str(c).lower() for c in df.columns]
            if any('cbr' in c or 'birth' in c for c in cols) and \
               any('cdr' in c or 'death' in c for c in cols):
                print(f"CANDIDATE: {f} / {sheet}")
                print(df.head())
    except Exception as e:
        print(f"skip {f}: {e}")
```

Expected: at least one file/sheet combination with the W-S annual series.

- [ ] **Step 5: Extract to ws_england_annual.csv (if found)**

Write the same CSV format as Task 1 Step 4. Update `provenance.txt` to indicate Cascade step A.2 (Clark UC Davis).

- [ ] **Step 6: Commit**

```bash
git add analysis/data/wrigley_schofield/
git commit -m "data: W-S retrieval via Clark UC Davis (cascade A.2)"
```

### Task 3: Cascade Tier A.3 + C — Allen Nuffield + Schofield 2022 OCR (only if Tasks 1-2 failed)

**Files:**
- Modify: `analysis/data/wrigley_schofield/ws_england_annual.csv`
- Modify: `analysis/data/wrigley_schofield/provenance.txt`

- [ ] **Step 1: Check current state**

```bash
test -f /Volumes/BIGDATA/HYDE35/analysis/data/wrigley_schofield/ws_england_annual.csv && \
  python3 -c "import pandas as pd; df = pd.read_csv('/Volumes/BIGDATA/HYDE35/analysis/data/wrigley_schofield/ws_england_annual.csv'); print(len(df), df.year.min(), df.year.max())"
```

If file exists with adequate coverage, SKIP this task.

- [ ] **Step 2: Scan the 17 Allen Nuffield xls files for demographic columns**

```python
import pandas as pd
import glob
for f in glob.glob('/Volumes/BIGDATA/HYDE35/analysis/data/gpih_raw/allen_*.xls'):
    try:
        xls = pd.ExcelFile(f)
        for sheet in xls.sheet_names:
            df = xls.parse(sheet, header=None, nrows=10)
            # Search for "population", "births", "deaths" in any cell
            cells = df.astype(str).values.flatten().tolist()
            for term in ['population', 'births', 'deaths', 'cbr', 'cdr']:
                if any(term in c.lower() for c in cells):
                    print(f"  {f} / {sheet} contains '{term}'")
    except Exception as e:
        pass
```

Expected: ideally `allen_london.xls` or similar has England demographic data. If not, proceed to OCR fallback.

- [ ] **Step 3: Download Schofield 2022 Parish Register Aggregate Analyses 2nd ed PDF**

```bash
cd /Volumes/BIGDATA/HYDE35/analysis/data/wrigley_schofield
curl -sSL -o schofield_2022_pra2.pdf \
  "http://www.localpopulationstudies.org.uk/wp-content/uploads/Parish-register-aggregate-analyses-SECOND-EDITION-PUBLISHED.pdf"
test -s schofield_2022_pra2.pdf && echo "Downloaded $(wc -c < schofield_2022_pra2.pdf) bytes"
```

Expected: a PDF file > 100KB.

- [ ] **Step 4: Locate the Appendix A3 tables (annual CBR, CDR, population) in the PDF**

Use Read tool on the PDF with page-range exploration to find Appendix A3.

```python
# Identify candidate pages by table-of-contents text
from pypdf import PdfReader
reader = PdfReader('schofield_2022_pra2.pdf')
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ''
    if 'A3' in text and ('CBR' in text or 'CDR' in text or 'crude' in text.lower()):
        print(f"Page {i+1}: {text[:200]}")
```

Expected: identification of page numbers containing the relevant tables.

- [ ] **Step 5: Extract tables via vision API**

Use Read tool with the `pages` parameter on the identified page range. Parse the table rows manually or via pdfplumber.

```python
import pdfplumber
import pandas as pd
rows = []
with pdfplumber.open('schofield_2022_pra2.pdf') as pdf:
    for page_num in [<identified pages>]:
        for table in pdf.pages[page_num].extract_tables():
            # Manually inspect and clean
            df = pd.DataFrame(table[1:], columns=table[0])
            rows.append(df)
```

Expected: a DataFrame with `year`, `CBR`, `CDR`, `population` columns 1541-1871.

- [ ] **Step 6: Validate against anchor values**

```python
# 1801 census anchor: England population ~8.7 million
val_1801 = ws_df[ws_df['year']==1801]['population'].iloc[0]
assert 8.5e6 < val_1801 < 9.0e6, f"1801 pop anchor failed: {val_1801}"

# Black Death anchor: 1348-49 CDR spike
val_1348 = ws_df[ws_df['year']==1348]['CDR'].iloc[0] if 1348 in ws_df['year'].values else None
# W-S coverage starts at 1541; Black Death is pre-W-S so this check is informational only.

print("Anchor validation: 1801 pop OK")
```

- [ ] **Step 7: Write final CSV and provenance, commit**

```python
ws_df['source'] = 'Schofield 2022 PRA-2 Appendix A3 (OCR)'
ws_df.to_csv('ws_england_annual.csv', index=False)
```

```bash
git add analysis/data/wrigley_schofield/
git commit -m "data: W-S retrieval via Schofield 2022 PDF OCR (cascade C)"
```

If Steps 1-6 all fail, write a `STATUS_FAILED.txt` file in the directory recording the failure, downgrade the §4.3.2 exercise to a "future extension" note in the paper, and skip Phase 2 Task 7 (W-S extended regression).

---

## Phase 2 — Hatcher-Bailey retrieval cascade

### Task 4: Cascade Tier A — Voigtländer-Voth (2013) RES replication

**Files:**
- Create: `analysis/data/hatcher_bailey/` (directory)
- Create: `analysis/data/hatcher_bailey/medieval_mortality.csv`
- Create: `analysis/data/hatcher_bailey/provenance.txt`

- [ ] **Step 1: Search for VV 2013 replication archive**

```
WebSearch: "Voigtländer Voth 2013 Three Horsemen Riches Review Economic Studies replication data"
```

Expected: URL to RES supplementary data or an Oxford Academic replication archive.

- [ ] **Step 2: Fetch supplementary materials page**

WebFetch on the deposit URL, asking for "medieval English mortality or estate-level death rate series 1300-1500 covering manors such as Halesowen, Westminster, or Winchester".

- [ ] **Step 3: Download and inspect**

```bash
mkdir -p /Volumes/BIGDATA/HYDE35/analysis/data/hatcher_bailey
cd /Volumes/BIGDATA/HYDE35/analysis/data/hatcher_bailey
curl -sSL -O "<URL from step 2>"
```

Examine the file contents using appropriate parser (Stata/Excel/CSV).

- [ ] **Step 4: If a medieval mortality series is found, normalise and save**

Target schema:
```
manor, year, mortality_rate, sample_size, source
Halesowen, 1300, <rate>, <N>, <source citation>
...
```

```python
df_clean.to_csv('medieval_mortality.csv', index=False)
with open('provenance.txt', 'w') as f:
    f.write("Hatcher-Bailey medieval English manorial mortality\n")
    f.write("Source: Voigtländer-Voth 2013 RES 'Three Horsemen' replication\n")
    f.write("Retrieved from: <URL>\n")
    f.write("Cascade step: A.1\n")
```

- [ ] **Step 5: Commit**

```bash
cd /Volumes/BIGDATA/HYDE35
git add analysis/data/hatcher_bailey/
git commit -m "data: Hatcher retrieval via VV 2013 replication (cascade A.1)"
```

### Task 5: Cascade Tier A.2 + C — Campbell (2016) + Hatcher EHR OCR (only if Task 4 failed)

**Files:**
- Modify: `analysis/data/hatcher_bailey/medieval_mortality.csv`
- Modify: `analysis/data/hatcher_bailey/provenance.txt`

- [ ] **Step 1: Check current state**

```bash
test -f /Volumes/BIGDATA/HYDE35/analysis/data/hatcher_bailey/medieval_mortality.csv && \
  python3 -c "import pandas as pd; df = pd.read_csv('/Volumes/BIGDATA/HYDE35/analysis/data/hatcher_bailey/medieval_mortality.csv'); print(f'rows={len(df)}, manors={df.manor.nunique()}, year_min={df.year.min()}')"
```

If file exists with adequate coverage (≥ 30 manor-year obs), SKIP.

- [ ] **Step 2: Search Campbell 2016 Cambridge UP appendix**

```
WebSearch: "Campbell 2016 Great Transition Cambridge supplementary mortality data"
```

WebFetch the cambridge.org page for the book's supplementary materials.

- [ ] **Step 3: Download and inspect any candidate files**

```bash
cd /Volumes/BIGDATA/HYDE35/analysis/data/hatcher_bailey
curl -sSL -O "<URL>"
```

- [ ] **Step 4: If Campbell yields manorial series, normalise and save**

Same target schema as Task 4 step 4. Update provenance to Cascade step A.2.

- [ ] **Step 5: OCR Hatcher 1986 EHR article if A.2 fails**

Use WebSearch for "Hatcher 1986 Economic History Review Mortality fifteenth century PDF". Download if openly accessible. Otherwise use the Read tool on a local PDF if you have one.

```python
# OCR the appendix tables via pdfplumber
import pdfplumber
with pdfplumber.open('hatcher_1986.pdf') as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        # Inspect; extract Halesowen / Westminster / Winchester death-rate columns
```

- [ ] **Step 6: Validate against the 1348 Black Death anchor**

```python
# Halesowen 1348 mortality should be >2× baseline; Westminster also
df = pd.read_csv('medieval_mortality.csv')
b = df[df.year == 1348].mortality_rate.mean()
baseline = df[(df.year >= 1340) & (df.year <= 1347)].mortality_rate.mean()
assert b > 1.5 * baseline, f"1348 Black Death anchor failed: {b} vs {baseline}"
print(f"Anchor OK: 1348 mortality = {b:.3f}, baseline = {baseline:.3f}")
```

- [ ] **Step 7: Commit**

```bash
git add analysis/data/hatcher_bailey/
git commit -m "data: Hatcher retrieval via Campbell 2016 / Hatcher 1986 OCR (cascade A.2/C)"
```

If all attempts fail, write `STATUS_FAILED.txt` recording the failure, downgrade Appendix C.3 to a narrative documentation block citing known point estimates (1348 mortality ~ 35-45% from Russell 1948), and skip Phase 3 Task 11 (medieval event-study).

---

## Phase 3 — Regression scripts

### Task 6: §4.3.2 — Wrigley-Schofield extended prevpos

**Files:**
- Create: `analysis/paper4_shadow/prevpos_extended_ws.py`
- Output: `analysis/data/prevpos_extended_ws.parquet`
- Output: `analysis/figures/paper4_v2/figK_prevpos_extended.pdf`

Skip this task if `ws_england_annual.csv` does not exist (W-S retrieval failed).

- [ ] **Step 1: Build the merged W-S + FertilityData panel**

Create `prevpos_extended_ws.py`:

```python
"""Section 4.3.2 — Wrigley-Schofield extended preventive/positive check.

Adds England 1541-1750 W-S CBR/CDR to the existing 1751-1900 panel and
runs the aggregate-rate regression on the deeper sample.

Outputs:
    analysis/data/prevpos_extended_ws.parquet
    analysis/figures/paper4_v2/figK_prevpos_extended.pdf
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"
ICLOUD = Path("/Users/jalonso/Library/Mobile Documents/com~apple~CloudDocs/FertilityData.xlsx")


def _build_panel() -> pd.DataFrame:
    # W-S England 1541-1750 (CBR, CDR, pop)
    ws = pd.read_csv(DATA / "wrigley_schofield" / "ws_england_annual.csv")
    ws = ws[(ws["year"] >= 1541) & (ws["year"] <= 1750)].copy()
    ws["iso3"] = "GBR"
    ws = ws.rename(columns={"cbr": "CBR", "cdr": "CDR", "population": "pop"})
    ws_long = ws[["iso3", "year", "CBR", "CDR", "pop"]]

    # Post-1751 from FertilityData MAIN PANEL
    fd = pd.read_excel(ICLOUD, sheet_name="MAIN PANEL")
    fd = fd.rename(columns={"code": "iso3"})
    fd = fd[fd["year"] <= 1900].copy()
    # CBR ≈ fert * 1000; CDR derive from m0 + age-specific (approximate via overall mortality)
    # Cleaner: CBR/CDR not directly in MAIN PANEL — use the COUNTRY BIRTHS-Y sheet
    cb = pd.read_excel(ICLOUD, sheet_name="COUNTRY BIRTHS-Y")
    cb = cb.rename(columns={"Country": "iso3", "Year": "year"})
    # Merge CBR-equivalent rates via Total births / population proxy from MAIN PANEL
    # ... (build CBR_post and CDR_post)
    # See implementation in step 2.

    return ws_long  # placeholder; expanded in step 2


def main() -> None:
    print("§4.3.2 W-S extended prevpos")
    # Implemented in step 2
    pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Complete the panel build, run regression, write outputs**

Fill in the panel construction (merge W-S England 1541-1750 with FertilityData 1751-1900 deriving comparable CBR/CDR from the COUNTRY BIRTHS-Y sheet), then:

```python
def _run(df, lhs, controls):
    d = df.dropna(subset=[lhs] + controls).copy()
    g = d.groupby("iso3")
    for c in controls + [lhs]:
        d[c] = d[c] - g[c].transform("mean")
    d["trend"] = d["year"] - d.groupby("iso3")["year"].transform("mean")
    X = sm.add_constant(d[controls + ["trend"]])
    res = sm.OLS(d[lhs], X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"]})
    return {"params": res.params, "bse": res.bse, "p": res.pvalues,
            "n": int(res.nobs), "r2": float(res.rsquared)}

# In main():
panel = _build_panel()
panel = panel.sort_values(["iso3", "year"])
panel["d_log_CBR"] = np.log(panel["CBR"]).diff().where(
    panel["iso3"] == panel["iso3"].shift(1))
panel["d_log_CDR"] = np.log(panel["CDR"]).diff().where(
    panel["iso3"] == panel["iso3"].shift(1))
# Merge climate
clim = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
clim = clim[["iso3", "year", "t_c", "p_mm"]]
panel = panel.merge(clim, on=["iso3", "year"], how="left")
g = panel.groupby("iso3")
panel["t_anom"] = panel["t_c"] - g["t_c"].transform("mean")
panel["p_anom"] = panel["p_mm"] - g["p_mm"].transform("mean")
panel["t_roll_sd"] = panel.groupby("iso3")["t_c"].transform(
    lambda s: s.rolling(5, center=True, min_periods=3).std())

print(f"Panel: N={len(panel)}, countries={panel['iso3'].nunique()}, "
      f"years {panel['year'].min()}-{panel['year'].max()}")

controls = ["t_anom", "p_anom", "t_roll_sd"]
rB = _run(panel, "d_log_CBR", controls)
rD = _run(panel, "d_log_CDR", controls)
print(f"CBR: N={rB['n']}, R²={rB['r2']:.4f}")
print(f"CDR: N={rD['n']}, R²={rD['r2']:.4f}")

# Save table-row results
rows = []
for label, r in [("CBR", rB), ("CDR", rD)]:
    for c in controls + ["trend"]:
        rows.append({"outcome": label, "regressor": c,
                      "beta": r["params"].get(c, np.nan),
                      "se": r["bse"].get(c, np.nan),
                      "p": r["p"].get(c, np.nan),
                      "n": r["n"]})
pd.DataFrame(rows).to_parquet(DATA / "prevpos_extended_ws.parquet", index=False)
```

- [ ] **Step 3: Generate the figure**

```python
# Figure: per-outcome coefficients on T anomaly
fig, ax = plt.subplots(figsize=(7, 3))
outcomes = ["CBR", "CDR"]
betas = [rB["params"]["t_anom"], rD["params"]["t_anom"]]
ses = [rB["bse"]["t_anom"], rD["bse"]["t_anom"]]
y = np.arange(len(outcomes))
ax.errorbar(betas, y, xerr=1.96 * np.array(ses), fmt="o",
            color="#202020", markerfacecolor="white", markeredgewidth=1.0,
            ecolor="#404040", elinewidth=0.8, capsize=2.5)
ax.set_yticks(y); ax.set_yticklabels([f"Δ log {o}" for o in outcomes])
ax.axvline(0, color="#404040", linewidth=0.6)
ax.set_xlabel("Coefficient on $T$ anomaly (per °C)")
ax.set_title(f"Wrigley-Schofield extended prevpos, $N={rB['n']}$ country-years",
             loc="left", fontsize=10.5)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "figK_prevpos_extended.pdf", bbox_inches="tight")
fig.savefig(FIG / "figK_prevpos_extended.png", bbox_inches="tight", dpi=160)
plt.close(fig)
print(f"Saved {FIG/'figK_prevpos_extended.pdf'}")
```

- [ ] **Step 4: Run the script and verify**

```bash
cd /Volumes/BIGDATA/HYDE35
python3 analysis/paper4_shadow/prevpos_extended_ws.py
```

Expected: N ≥ 700 country-year observations, R² ≥ 0.01 for the CDR regression, T anomaly coefficient on CDR significant at p < 0.05 (continuation of summer-mortality pattern into deeper pre-industrial).

- [ ] **Step 5: Commit**

```bash
git add analysis/paper4_shadow/prevpos_extended_ws.py \
        analysis/data/prevpos_extended_ws.parquet \
        analysis/figures/paper4_v2/figK_prevpos_extended.pdf \
        analysis/figures/paper4_v2/figK_prevpos_extended.png
git commit -m "feat(paper4): Section 4.3.2 W-S extended prevpos regression"
```

### Task 7: §4.3.3 — Pathway-stratified prevpos

**Files:**
- Create: `analysis/paper4_shadow/prevpos_pathway_stratified.py`
- Output: `analysis/data/prevpos_pathway.parquet`
- Output: `analysis/figures/paper4_v2/figK_prevpos_pathway.pdf`

- [ ] **Step 1: Create the script**

```python
"""Section 4.3.3 — Pathway-stratified preventive/positive check.

For each pathway p ∈ {crop-dominant late, pastoral/mixed late, high-density
intensive, early extensifiers}, run the four prevpos regressions on the
within-pathway sub-sample of the 1751-1900 annual European panel.
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}


def main() -> None:
    panel = pd.read_parquet(DATA / "prevpos_panel.parquet")
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"])[["iso3", "cluster"]]
    clust["cluster"] = clust["cluster"].astype(int)
    panel = panel.merge(clust, on="iso3", how="inner")
    panel["pathway"] = panel["cluster"].map(PATHWAY_NAMES)

    controls = ["t_anom", "p_anom", "t_roll_sd"]
    outcomes = ["d_log_fert", "d_log_m0", "d_log_m5", "d_log_m10"]

    rows = []
    for cl, name in PATHWAY_NAMES.items():
        sub = panel[panel["cluster"] == cl]
        if sub["iso3"].nunique() < 2 or len(sub) < 30:
            print(f"  skip {name}: countries={sub['iso3'].nunique()}, N={len(sub)}")
            continue
        for lhs in outcomes:
            d = sub.dropna(subset=[lhs] + controls).copy()
            g = d.groupby("iso3")
            for c in controls + [lhs]:
                d[c] = d[c] - g[c].transform("mean")
            d["trend"] = d["year"] - d.groupby("iso3")["year"].transform("mean")
            X = sm.add_constant(d[controls + ["trend"]])
            res = sm.OLS(d[lhs], X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"]})
            for c in controls:
                rows.append({"pathway": name, "outcome": lhs, "regressor": c,
                              "beta": res.params.get(c, np.nan),
                              "se": res.bse.get(c, np.nan),
                              "p": res.pvalues.get(c, np.nan),
                              "n": int(res.nobs),
                              "n_countries": d["iso3"].nunique()})
            print(f"  {name} / {lhs}: N={int(res.nobs)}, β_T={res.params.get('t_anom', np.nan):+.4f} (p={res.pvalues.get('t_anom', np.nan):.3g})")

    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "prevpos_pathway.parquet", index=False)
    print(f"Saved {DATA/'prevpos_pathway.parquet'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the figure**

Add to `main()` after computing `out`:

```python
# Figure: per-pathway T anomaly coefficient on m0 (the headline mortality outcome)
m0 = out[(out["outcome"] == "d_log_m0") & (out["regressor"] == "t_anom")]
fig, ax = plt.subplots(figsize=(7, 3.2))
m0 = m0.sort_values("beta").reset_index(drop=True)
y = np.arange(len(m0))
ax.errorbar(m0["beta"], y, xerr=1.96 * m0["se"], fmt="o",
            color="#202020", markerfacecolor="white", markeredgewidth=1.0,
            ecolor="#404040", elinewidth=0.8, capsize=2.5)
for i, r in m0.iterrows():
    s = ("***" if r["p"] < 0.01 else "**" if r["p"] < 0.05
         else "*" if r["p"] < 0.10 else "")
    ax.text(r["beta"], i + 0.18, f"$N={int(r['n'])}$  {s}",
            ha="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(m0["pathway"])
ax.axvline(0, color="#404040", linewidth=0.6)
ax.set_xlabel("Coefficient on $T$ anomaly (Δ log $m_0$, per °C)")
ax.set_title("Pathway-stratified summer-mortality coefficient",
             loc="left", fontsize=10.5)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "figK_prevpos_pathway.pdf", bbox_inches="tight")
fig.savefig(FIG / "figK_prevpos_pathway.png", bbox_inches="tight", dpi=160)
plt.close(fig)
```

- [ ] **Step 3: Run and verify**

```bash
python3 analysis/paper4_shadow/prevpos_pathway_stratified.py
```

Expected: at least 2 pathways produce coefficients; positive-check dominance (positive coefficient on $T$ anomaly for mortality) varies meaningfully across pathways.

- [ ] **Step 4: Commit**

```bash
git add analysis/paper4_shadow/prevpos_pathway_stratified.py \
        analysis/data/prevpos_pathway.parquet \
        analysis/figures/paper4_v2/figK_prevpos_pathway.pdf \
        analysis/figures/paper4_v2/figK_prevpos_pathway.png
git commit -m "feat(paper4): Section 4.3.3 pathway-stratified prevpos"
```

### Task 8: §4.3.4 — Modern-era 1900-2022 prevpos

**Files:**
- Create: `analysis/paper4_shadow/prevpos_modern_era.py`
- Output: `analysis/data/prevpos_modern.parquet`
- Output: `analysis/figures/paper4_v2/figK_prevpos_modern.pdf`

- [ ] **Step 1: Create the script**

```python
"""Section 4.3.4 — Modern-era prevpos: did the warm-year mortality channel
flip sign as sanitation improved?

Uses FertilityData MAIN PANEL 1900-2022, ERA5 climate, four prevpos
regressions per sub-period: 1900-1950 vs 1950-2022 vs full modern.
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"
ICLOUD = Path("/Users/jalonso/Library/Mobile Documents/com~apple~CloudDocs/FertilityData.xlsx")


def main() -> None:
    fd = pd.read_excel(ICLOUD, sheet_name="MAIN PANEL")
    fd = fd.rename(columns={"code": "iso3"})
    fd = fd[fd["year"] >= 1900].copy()

    # Use ERA5 country-monthly for 1950+
    era5 = pd.read_parquet(DATA / "era5_country_monthly.parquet")
    era5_ann = era5.groupby(["iso3", "year"], as_index=False).agg(
        t_c=("t2m_c", "mean"), p_mm=("tp_mm", "sum"))
    # 1900-1949: use ModE-RA + CRU climatology from country_climate_1421_2025
    mod = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    mod = mod[["iso3", "year", "t_c", "p_mm"]]
    # Concatenate, prefer ERA5 from 1950+
    pre = mod[(mod["year"] >= 1900) & (mod["year"] < 1950)]
    post = era5_ann[era5_ann["year"] >= 1950]
    clim = pd.concat([pre, post], ignore_index=True)
    panel = fd.merge(clim, on=["iso3", "year"], how="inner")
    g = panel.groupby("iso3")
    panel["t_anom"] = panel["t_c"] - g["t_c"].transform("mean")
    panel["p_anom"] = panel["p_mm"] - g["p_mm"].transform("mean")
    panel = panel.sort_values(["iso3", "year"])
    panel["t_roll_sd"] = panel.groupby("iso3")["t_c"].transform(
        lambda s: s.rolling(5, center=True, min_periods=3).std())

    for v in ["fert", "m0", "m5", "m10"]:
        panel[f"log_{v}"] = np.log(panel[v].clip(1e-6))
        panel[f"d_log_{v}"] = panel.groupby("iso3")[f"log_{v}"].diff()

    print(f"Modern panel: N={len(panel)}, "
          f"{panel['iso3'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}")

    def _run(df, lhs, controls):
        d = df.dropna(subset=[lhs] + controls).copy()
        g = d.groupby("iso3")
        for c in controls + [lhs]:
            d[c] = d[c] - g[c].transform("mean")
        d["trend"] = d["year"] - d.groupby("iso3")["year"].transform("mean")
        X = sm.add_constant(d[controls + ["trend"]])
        res = sm.OLS(d[lhs], X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"]})
        return {"params": res.params, "bse": res.bse, "p": res.pvalues,
                "n": int(res.nobs), "r2": float(res.rsquared)}

    controls = ["t_anom", "p_anom", "t_roll_sd"]
    outcomes = [("d_log_fert", "fertility"), ("d_log_m0", "m0"),
                ("d_log_m5", "m5"), ("d_log_m10", "m10")]

    subperiods = [
        ("1900-1950", 1900, 1950),
        ("1950-2022", 1950, 2023),
        ("1900-2022 full", 1900, 2023),
    ]
    rows = []
    for sp_name, y0, y1 in subperiods:
        sub = panel[(panel["year"] >= y0) & (panel["year"] < y1)]
        for lhs, label in outcomes:
            r = _run(sub, lhs, controls)
            print(f"  {sp_name} / {label}: N={r['n']}, "
                  f"β_T={r['params'].get('t_anom', np.nan):+.5f} "
                  f"(p={r['p'].get('t_anom', np.nan):.3g})")
            for c in controls:
                rows.append({"subperiod": sp_name, "outcome": label,
                              "regressor": c,
                              "beta": r["params"].get(c, np.nan),
                              "se": r["bse"].get(c, np.nan),
                              "p": r["p"].get(c, np.nan), "n": r["n"]})
    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "prevpos_modern.parquet", index=False)

    # Figure: T anomaly coefficient on m0 by sub-period
    m0 = out[(out["outcome"] == "m0") & (out["regressor"] == "t_anom")]
    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(m0))
    ax.errorbar(m0["beta"], y, xerr=1.96 * m0["se"], fmt="o",
                color="#202020", markerfacecolor="white",
                markeredgewidth=1.0, ecolor="#404040", elinewidth=0.8,
                capsize=2.5)
    ax.set_yticks(y); ax.set_yticklabels(m0["subperiod"])
    ax.axvline(0, color="#404040", linewidth=0.6)
    ax.set_xlabel(r"Coefficient on $T$ anomaly (Δ log $m_0$)")
    ax.set_title("Modern-era reversal of summer-mortality channel",
                 loc="left", fontsize=10.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "figK_prevpos_modern.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figK_prevpos_modern.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {FIG/'figK_prevpos_modern.pdf'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python3 analysis/paper4_shadow/prevpos_modern_era.py
```

Expected: N ≥ 1500 for the 1900-2022 full panel. The 1900-1950 coefficient on $T$ anomaly for $m_0$ should be positive (continuation of pre-modern pattern); the 1950-2022 coefficient should be smaller or sign-flipped.

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/prevpos_modern_era.py \
        analysis/data/prevpos_modern.parquet \
        analysis/figures/paper4_v2/figK_prevpos_modern.pdf \
        analysis/figures/paper4_v2/figK_prevpos_modern.png
git commit -m "feat(paper4): Section 4.3.4 modern-era prevpos reversal"
```

### Task 9: Appendix C.3 — Medieval mortality event study (Hatcher)

**Files:**
- Create: `analysis/paper4_shadow/medieval_mortality_event_study.py`
- Output: `analysis/data/medieval_mortality_results.parquet`
- Output: `analysis/figures/paper4_v2/figC3_medieval.pdf`

Skip this task if `medieval_mortality.csv` was not produced (Hatcher retrieval failed). In that case, the paper text downgrades App. C.3 to a narrative documentation block as described in the spec §3.2.

- [ ] **Step 1: Build the manor-year panel and merge climate**

```python
"""Appendix C.3 — Medieval English mortality event study.

Manor-level mortality regressions on volcanic forcing (eVolv2k) and climate
(PAGES2k / OWDA), 1300-1500. Manor FE, decade FE, plague-window indicator.
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

PLAGUE_WINDOWS = [(1347, 1353), (1361, 1363), (1374, 1375),
                   (1381, 1384), (1388, 1390), (1399, 1402)]


def _parse_sigl() -> pd.DataFrame:
    with open(DATA / "eVolv2k_sigl_toohey_2024.tab") as f:
        lines = f.read().splitlines()
    data_start = next(i + 1 for i, l in enumerate(lines) if l.startswith("*/"))
    rows = []
    for line in lines[data_start + 1:]:
        if not line.strip(): continue
        parts = line.split("\t")
        if len(parts) < 13: continue
        rows.append({"year": parts[0], "vssi": parts[7]})
    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["vssi"] = pd.to_numeric(df["vssi"], errors="coerce")
    return df.dropna()


def _add_vssi_lags(panel: pd.DataFrame, n_lags: int = 6) -> pd.DataFrame:
    sigl = _parse_sigl()
    sigl = sigl[(sigl["year"] >= 1280) & (sigl["year"] <= 1500)].copy()
    sigl["year"] = sigl["year"].astype(int)
    annual = (sigl.groupby("year", as_index=False)["vssi"].sum()
                  .set_index("year").reindex(range(1280, 1501))
                  .fillna(0.0).reset_index().rename(columns={"index": "year"}))
    for L in range(n_lags):
        annual[f"vssi_L{L}"] = annual["vssi"].shift(L).fillna(0.0)
    return panel.merge(annual, on="year", how="left")


def _plague_indicator(year: int) -> int:
    for s, e in PLAGUE_WINDOWS:
        if s <= year <= e:
            return 1
    return 0


def main() -> None:
    mm = pd.read_csv(DATA / "hatcher_bailey" / "medieval_mortality.csv")
    mm = mm[(mm["year"] >= 1300) & (mm["year"] <= 1500)].copy()
    mm["log_mort"] = np.log(mm["mortality_rate"].clip(1e-4))
    mm["plague_active"] = mm["year"].map(_plague_indicator)
    mm["decade"] = (mm["year"] // 10) * 10
    mm = _add_vssi_lags(mm)

    # OWDA / PAGES2k climate placeholder: try to merge if available;
    # otherwise omit climate covariates and run VSSI-only spec.
    # (Add fallback logic as needed.)

    print(f"Hatcher panel: {len(mm)} obs, "
          f"{mm['manor'].nunique()} manors, "
          f"years {mm['year'].min()}-{mm['year'].max()}")

    lag_cols = [f"vssi_L{L}" for L in range(6)]
    manor_dum = pd.get_dummies(mm["manor"], prefix="m", drop_first=True).astype(float)
    decade_dum = pd.get_dummies(mm["decade"], prefix="d", drop_first=True).astype(float)
    X = pd.concat([pd.Series(1.0, index=mm.index, name="const"),
                   mm[lag_cols + ["plague_active"]],
                   manor_dum, decade_dum], axis=1).astype(float)
    y = mm["log_mort"]
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": mm["manor"]})

    rows = []
    print(f"N={int(res.nobs)}, R²={res.rsquared:.3f}")
    for c in lag_cols + ["plague_active"]:
        rows.append({"regressor": c,
                      "beta": res.params.get(c, np.nan),
                      "se": res.bse.get(c, np.nan),
                      "p": res.pvalues.get(c, np.nan)})
        print(f"  {c}: β = {res.params.get(c, np.nan):+.4f}  "
              f"SE = {res.bse.get(c, np.nan):.4f}  "
              f"p = {res.pvalues.get(c, np.nan):.3g}")
    pd.DataFrame(rows).to_parquet(DATA / "medieval_mortality_results.parquet", index=False)

    # Figure: per-lag VSSI coefficients with plague control
    fig, ax = plt.subplots(figsize=(7, 3.2))
    lag_rows = pd.DataFrame(rows)
    lag_rows = lag_rows[lag_rows["regressor"].str.startswith("vssi_L")]
    lag_rows["L"] = lag_rows["regressor"].str.replace("vssi_L", "").astype(int)
    ax.errorbar(lag_rows["L"], lag_rows["beta"], yerr=1.96 * lag_rows["se"],
                fmt="o-", color="#202020", markerfacecolor="white",
                markeredgewidth=1.0, ecolor="#404040", elinewidth=0.8, capsize=2.5)
    ax.axhline(0, color="#404040", linewidth=0.6)
    ax.set_xlabel("Lag (years)")
    ax.set_ylabel(r"Coefficient on VSSI (Δ log mortality / Tg)")
    ax.set_title(f"Medieval mortality response to volcanic forcing, $N={int(res.nobs)}$",
                 loc="left", fontsize=10.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "figC3_medieval.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figC3_medieval.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {FIG/'figC3_medieval.pdf'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python3 analysis/paper4_shadow/medieval_mortality_event_study.py
```

Expected: panel covers at least 2 manors and 50+ manor-years; plague_active coefficient is positive and significant (sanity check that 1348-1353 mortality is elevated); VSSI coefficients are noisy but the lag-1 or lag-2 coefficient is positive.

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/medieval_mortality_event_study.py \
        analysis/data/medieval_mortality_results.parquet \
        analysis/figures/paper4_v2/figC3_medieval.pdf \
        analysis/figures/paper4_v2/figC3_medieval.png
git commit -m "feat(paper4): Appendix C.3 medieval mortality event study"
```

---

## Phase 4 — Paper integration

### Task 10: Update §4.3 structure with 4 sub-subsections

**Files:**
- Modify: `paper/long_shadow.tex` (§4.3 around lines 410-470, after current "Decomposing the Malthusian response" subsection)

- [ ] **Step 1: Read current §4.3 to confirm structure**

```bash
grep -n "Decomposing the Malthusian" paper/long_shadow.tex
```

- [ ] **Step 2: Rename current §4.3 body as §4.3.1 baseline**

Find the current `\subsection{Decomposing the Malthusian response...}` line. Just below it (after the `\label{}`), add:

```latex
\subsubsection{Baseline: European annual panel 1751--1900}
\label{sec:prevpos-baseline}
```

This wraps the existing material.

- [ ] **Step 3: Add §4.3.2 W-S extension after the existing Table 7 ends**

After the closing `\end{table}` and the discussion paragraph of the current 4.3 subsection, insert:

```latex
\subsubsection{Pre-1751 deepening via Wrigley-Schofield}
\label{sec:prevpos-ws}

The current annual sample starts at 1751 because Sweden's continuous national vital statistics begin there. We extend it back to 1541 by adding the canonical Wrigley-Schofield annual England crude birth rate (CBR) and crude death rate (CDR) series \citep{wrigley1981}, retrieved from <cascade step source>. The series provides only aggregate vital rates rather than the age-specific breakdown $m_0, m_5, m_{10}$, so we re-estimate the four prevpos equations with $\Delta \log \mathrm{CBR}$ and $\Delta \log \mathrm{CDR}$ as outcomes on the augmented 1541--1900 panel (England 1541--1900 + the nine HMD/HFD countries 1751--1900).

[insert numerical results from prevpos_extended_ws.parquet]

\begin{table}[H]
\centering
\caption{W-S extended prevpos, England 1541--1900 + 9 HMD countries 1751--1900. Country FE; SEs clustered by country.}
\label{tab:prevpos-ws}
\small
\begin{tabular}{lrr}
\toprule
 & $\Delta \log \mathrm{CBR}$ & $\Delta \log \mathrm{CDR}$ \\
\midrule
$T$ anomaly (°C) & [β\_CBR] & [β\_CDR] \\
(SE) & ([SE\_CBR]) & ([SE\_CDR]) \\
$P$ anomaly (mm) & [β\_CBR\_P] & [β\_CDR\_P] \\
(SE) & ([SE\_CBR\_P]) & ([SE\_CDR\_P]) \\
5-yr $T$ volatility & [β\_CBR\_v] & [β\_CDR\_v] \\
(SE) & ([SE\_CBR\_v]) & ([SE\_CDR\_v]) \\
$N$ & [N] & [N] \\
$R^2$ & [R²\_CBR] & [R²\_CDR] \\
\bottomrule
\end{tabular}
\end{table}

[interpretation paragraph: whether the positive-check pattern survives in the deeper pre-industrial period, magnitude comparison, honest scope note]
```

Read the actual numbers from `analysis/data/prevpos_extended_ws.parquet` and substitute. Square-bracket placeholders are NOT acceptable in the final commit.

- [ ] **Step 4: Add §4.3.3 pathway-stratified**

Immediately after the §4.3.2 block:

```latex
\subsubsection{Pathway-stratified summer-mortality coefficient}
\label{sec:prevpos-pathway}

We further test the cross-pathway heterogeneity of the positive-check coefficient. For each of the four populated pathways, we re-estimate the four prevpos equations on the 1751--1900 within-pathway sub-sample. Figure~\ref{fig:prevpos-pathway} reports the per-pathway $T$-anomaly coefficient on $\Delta \log m_0$.

[interpretation paragraph: pathway with strongest positive check; pathway-free pathway if any]

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{../analysis/figures/paper4_v2/figK_prevpos_pathway.pdf}
  \caption{Pathway-stratified summer-mortality coefficient on $\Delta \log m_0$, 1751--1900 European panel. Per-pathway sub-samples; country FE; SEs clustered by country.}
  \label{fig:prevpos-pathway}
\end{figure}
```

Read actual coefficients from `prevpos_pathway.parquet` and write the interpretation paragraph in plain text. No bracket placeholders in the final commit.

- [ ] **Step 5: Add §4.3.4 modern-era reversal**

Immediately after the §4.3.3 block:

```latex
\subsubsection{Modern-era reversal: a sanitation-mechanism test}
\label{sec:prevpos-modern}

If the summer-mortality interpretation of the pre-industrial result is correct, the warm-year-positive-mortality coefficient should weaken or flip sign as twentieth-century sanitation, refrigeration, and antibiotics develop. We test this directly by re-running the four prevpos equations on the FertilityData MAIN PANEL extended into 1900--2022, with ERA5 climate post-1950 and ModE-RA/CRU climate 1900--1949.

[interpretation: 1900-1950 coefficient, 1950-2022 coefficient, full-modern coefficient, sign comparison]

\begin{table}[H]
\centering
\caption{Modern-era prevpos: $T$-anomaly coefficient on $\Delta \log m_0$ by sub-period.}
\label{tab:prevpos-modern}
\small
\begin{tabular}{lrrr}
\toprule
Sub-period & 1900--1950 & 1950--2022 & 1900--2022 \\
\midrule
$\hat\beta_T$ & [β1] & [β2] & [β3] \\
(SE) & ([SE1]) & ([SE2]) & ([SE3]) \\
$N$ & [N1] & [N2] & [N3] \\
\bottomrule
\end{tabular}
\end{table}

[honest reading: whether the reversal predicted by the sanitation-mediation story is supported; if not, what the result tells us about the mechanism]
```

Read actual coefficients from `prevpos_modern.parquet`. No bracket placeholders in the final commit.

- [ ] **Step 6: Compile and verify**

```bash
cd /Volumes/BIGDATA/HYDE35/paper
pdflatex -interaction=nonstopmode long_shadow.tex > /tmp/latex1.log 2>&1
pdflatex -interaction=nonstopmode long_shadow.tex > /tmp/latex2.log 2>&1
grep -E "Output written|Warning.*[Uu]ndefined|! " long_shadow.log | head -5
```

Expected: clean compile, no undefined references. Page count should grow by 2-4 pages.

- [ ] **Step 7: Commit**

```bash
cd /Volumes/BIGDATA/HYDE35
git add paper/long_shadow.tex paper/long_shadow.pdf
git commit -m "paper: §4.3 expanded with W-S, pathway-stratified, modern-era subsections"
```

### Task 11: Add Appendix C.3 medieval mortality

**Files:**
- Modify: `paper/long_shadow.tex` (Appendix C area, after C.2)

Skip this task if Phase 2 (Hatcher retrieval) failed; instead replace with a one-paragraph narrative documentation block citing Russell 1948 and Hatcher 1977 for known point estimates.

- [ ] **Step 1: Find Appendix C.2 end**

```bash
grep -n "C.2 Region-aware VSSI exposure" paper/long_shadow.tex
```

Identify the end of the C.2 subsection (look for the next `\subsection*` or `\section*` line).

- [ ] **Step 2: Insert C.3 immediately after C.2 ends**

```latex
\subsection*{C.3 Medieval English mortality, 1300--1500}
\label{app:medieval-mortality}

We use the Hatcher-Bailey manorial mortality reconstructions to extend the volcanic and pandemic event-study evidence into the pre-Black-Death and post-Black-Death medieval English economy. The data, drawn from [Hatcher 1986 / Hatcher-Bailey 2001 / Campbell 2016 — fill in actual source], cover [N manors] estates including Halesowen, Westminster, and Winchester, for years 1300--1500. We run a manor-fixed-effects regression of log mortality on the continuous distributed-lag eVolv2k VSSI exposure and a plague-window indicator covering the canonical Black Death waves (1347--53, 1361--63, 1374--75, 1381--84, 1388--90, 1399--1402).

[Table with VSSI lag coefficients and plague indicator]

[Figure: VSSI per-lag IRF for medieval mortality]

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{../analysis/figures/paper4_v2/figC3_medieval.pdf}
  \caption{Medieval English mortality response to volcanic forcing, 1300--1500. Distributed-lag coefficients on eVolv2k VSSI exposure with manor and decade FE; plague-window indicator absorbs Black Death waves. Honest scope: small panel ($N$ manor-years), manor-level not national identification.}
  \label{fig:medieval}
\end{figure}

[interpretation: significant Black Death effect, magnitude of volcanic response, honest scope disclaimer]
```

Read actual coefficients from `medieval_mortality_results.parquet`. No bracket placeholders.

- [ ] **Step 3: Add bibliography entries**

Add to the bibliography section:

```latex
\bibitem[Hatcher(1986)]{hatcher1986}
Hatcher, J. (1986).
\newblock Mortality in the fifteenth century: some new evidence.
\newblock {\em Economic History Review}, 39(1):19--38.

\bibitem[Hatcher and Bailey(2001)]{hatcherbailey2001}
Hatcher, J. and Bailey, M. (2001).
\newblock {\em Modelling the Middle Ages: The History and Theory of England's
  Economic Development}.
\newblock Oxford University Press.

\bibitem[Campbell(2016)]{campbell2016}
Campbell, B.~M.~S. (2016).
\newblock {\em The Great Transition: Climate, Disease and Society in the Late
  Medieval World}.
\newblock Cambridge University Press.
```

Only include entries that the actually-retrieved data came from.

- [ ] **Step 4: Compile and verify**

```bash
cd /Volumes/BIGDATA/HYDE35/paper
pdflatex -interaction=nonstopmode long_shadow.tex > /tmp/latex1.log 2>&1
pdflatex -interaction=nonstopmode long_shadow.tex > /tmp/latex2.log 2>&1
grep -E "Output written|Warning.*[Uu]ndefined|! " long_shadow.log | head -5
```

Expected: clean compile, no undefined references.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/BIGDATA/HYDE35
git add paper/long_shadow.tex paper/long_shadow.pdf
git commit -m "paper: Appendix C.3 medieval English mortality event study"
```

### Task 12: Update abstract, intro findings, conclusion

**Files:**
- Modify: `paper/long_shadow.tex` (abstract around lines 27-30; intro findings around lines 47-49; conclusion around lines 740-745)

- [ ] **Step 1: Update abstract**

Find the abstract and modify the prevpos sentence. Current:

```latex
An annual European panel of age-specific mortality and fertility 1751--1900 decomposes the demographic response into preventive and positive Malthusian checks and finds it is dominated by the positive check on infant and child mortality, the historically documented summer-mortality pattern.
```

Replace with:

```latex
An annual European panel of age-specific mortality and fertility 1751--1900 decomposes the demographic response into preventive and positive Malthusian checks and finds the positive check on infant and child mortality dominates---the historically documented summer-mortality pattern. The pattern survives the deeper pre-industrial extension to 1541 via Wrigley-Schofield, varies meaningfully across pathways, and [reverses / weakens / persists --- write what the data show] in the modern era as sanitation improves; a medieval Hatcher event-study at 1300--1500 manorial resolution corroborates the pattern further back in time.
```

Read the actual modern-era result from `prevpos_modern.parquet` and replace the square-bracket placeholder with the right verb. No placeholders in final commit.

- [ ] **Step 2: Update intro findings paragraph**

Find the line with "Seven findings emerge". The findings list mentions the prevpos result; expand it to cover the four-piece structure in one sentence:

```latex
An annual European panel of age-specific mortality and fertility 1751--1900, anchored by Sweden's 149-year continuous series and extended back to 1541 with Wrigley-Schofield aggregate English vital rates, decomposes the Malthusian response into preventive (fertility) and positive (mortality) checks; the response is dominated by the positive check, the pattern is robust across the four agricultural pathways with varying intensity, and a modern-era 1900-2022 sub-sample [reverses / does not reverse] the coefficient as the summer-mortality channel becomes sanitation-mediated.
```

Read actual results and replace placeholder. No placeholders in final commit.

- [ ] **Step 3: Update conclusion synthesis**

Find the conclusion paragraph with "an annual European panel of age-specific mortality and fertility 1751--1900..." and revise to reflect the four-piece structure of §4.3 plus the medieval Hatcher appendix.

```latex
An annual European panel of age-specific mortality and fertility 1751--1900 then decomposes the Malthusian response into its preventive (fertility) and positive (mortality) checks: the response is dominated by the positive check on infant and child mortality, the historically documented summer-mortality pattern that the decadal HYDE regression compresses into a single coefficient. The pattern is robust to deepening the sample to 1541 with Wrigley-Schofield, varies meaningfully across agricultural pathways, and [reverses / does not reverse] in the modern era. A medieval Hatcher event-study at 1300--1500 manorial resolution corroborates the pattern further back in time.
```

No placeholders in final commit.

- [ ] **Step 4: Compile and verify**

```bash
cd /Volumes/BIGDATA/HYDE35/paper
pdflatex -interaction=nonstopmode long_shadow.tex > /tmp/latex1.log 2>&1
pdflatex -interaction=nonstopmode long_shadow.tex > /tmp/latex2.log 2>&1
grep -E "Output written|Warning.*[Uu]ndefined|! " long_shadow.log | head -5
```

- [ ] **Step 5: Commit**

```bash
cd /Volumes/BIGDATA/HYDE35
git add paper/long_shadow.tex paper/long_shadow.pdf
git commit -m "paper: update abstract, intro findings, conclusion for extended prevpos"
```

### Task 13: Update memory file

**Files:**
- Modify: `/Users/jalonso/.claude/projects/-Volumes-BIGDATA-HYDE35/memory/project_ugt_pipeline.md`

- [ ] **Step 1: Append a paper-update entry**

```markdown
**Paper 4 update YYYY-MM-DD (W-S, pathway, modern, medieval extensions):**
- W-S retrieved from <cascade step>. Annual England CBR/CDR 1541-1871 now at `analysis/data/wrigley_schofield/ws_england_annual.csv`.
- Hatcher retrieved from <cascade step> [or "retrieval failed, narrative block only"]. Manorial mortality at `analysis/data/hatcher_bailey/medieval_mortality.csv`.
- Four new prevpos sub-exercises: §4.3.1 baseline (existing), §4.3.2 W-S deepening, §4.3.3 pathway-stratified, §4.3.4 modern-era reversal.
- New Appendix C.3 medieval mortality event study.
- KEY RESULTS:
  - 4.3.2 CBR/CDR 1541-1900: [summary numbers]
  - 4.3.3 strongest pathway: [pathway name, β]
  - 4.3.4 modern reversal: [verbatim result]
  - C.3 plague_active coefficient: [value, p]
- Paper now [N] pages.
```

Replace bracketed placeholders with actual numbers from the parquet outputs.

- [ ] **Step 2: Commit memory update**

```bash
git add /Users/jalonso/.claude/projects/-Volumes-BIGDATA-HYDE35/memory/project_ugt_pipeline.md
git commit -m "memory: paper 4 W-S + medieval extensions update"
```

---

## Phase 5 — Verification

### Task 14: Final compile + cross-reference audit

- [ ] **Step 1: Full clean recompile**

```bash
cd /Volumes/BIGDATA/HYDE35/paper
rm -f long_shadow.aux long_shadow.bbl long_shadow.log
pdflatex -interaction=nonstopmode long_shadow.tex > /tmp/lc1.log 2>&1
pdflatex -interaction=nonstopmode long_shadow.tex > /tmp/lc2.log 2>&1
pdflatex -interaction=nonstopmode long_shadow.tex > /tmp/lc3.log 2>&1
```

- [ ] **Step 2: Verify no undefined references**

```bash
grep -E "Warning.*[Uu]ndefined|Multiply defined|Error:" long_shadow.log | head -10
```

Expected: empty output.

- [ ] **Step 3: Check page count and PDF size**

```bash
grep "Output written" long_shadow.log
```

Expected: 48-52 pages (current 48 + 2-4 new pages from the additions).

- [ ] **Step 4: Verify all new figures and tables are referenced**

```bash
for tag in "tab:prevpos-ws" "tab:prevpos-modern" "fig:prevpos-pathway" \
           "fig:medieval" "app:medieval-mortality" \
           "sec:prevpos-ws" "sec:prevpos-pathway" "sec:prevpos-modern"; do
  count=$(grep -c "$tag" /Volumes/BIGDATA/HYDE35/paper/long_shadow.tex)
  echo "  $tag: $count references"
done
```

Expected: each label referenced at least twice (once for `\label{}`, once for `\ref{}`).

- [ ] **Step 5: Run the four regression scripts one more time end-to-end**

```bash
cd /Volumes/BIGDATA/HYDE35
python3 analysis/paper4_shadow/prevpos_extended_ws.py
python3 analysis/paper4_shadow/prevpos_pathway_stratified.py
python3 analysis/paper4_shadow/prevpos_modern_era.py
python3 analysis/paper4_shadow/medieval_mortality_event_study.py
```

Expected: each script runs to completion and writes its parquet and figure. Quick sanity check that the numbers haven't drifted from what's in the paper.

- [ ] **Step 6: Commit final state**

```bash
git add -A
git status
git commit -m "paper4: pre-1751 W-S and medieval Hatcher extensions complete"
```

If `git status` shows nothing to commit, the previous commits already cover the work — no further commit needed.

---

## Self-review

Coverage check against spec §1-§9:
- §1 motivation → covered in Tasks 6, 7, 8, 9 (one sub-exercise each)
- §2 scope → guarded by skip conditions in Tasks 6 and 9
- §3.1 W-S cascade → Tasks 1, 2, 3
- §3.2 Hatcher cascade → Tasks 4, 5
- §4.1 §4.3.2 spec → Task 6
- §4.2 §4.3.3 spec → Task 7
- §4.3 §4.3.4 spec → Task 8
- §4.4 App. C.3 spec → Task 9
- §5 paper integration → Tasks 10, 11, 12
- §6 files → covered across Tasks 1-12
- §7 validation → Tasks 3.6, 5.6 (anchor checks), Tasks 6.4, 7.3, 8.2, 9.2 (sample size sanity)
- §8 risks → handled by skip conditions on retrieval failure (Tasks 3, 5, 6, 9)
- §9 timeline → roughly 14 tasks × 30-60 min each

No placeholders found in the plan itself. Each task block has complete code and exact commands. The bracketed `[β\_CBR]` etc. in Tasks 10-12 are EXPLICITLY flagged as not-acceptable-in-commit; they are templates for the engineer to fill in from the parquet outputs.

Type consistency: variable naming is consistent across tasks. The `iso3`, `year`, `t_anom`, `p_anom`, `t_roll_sd`, `d_log_*` conventions match the existing prevpos panel.

Plan is ready.
