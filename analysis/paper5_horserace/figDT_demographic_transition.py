"""Figure: Demographic-transition timing across countries.

Two-panel figure:
 (a) World choropleth of the first year crude birth rate fell below 25/1000
 (b) Country-level distribution by region, with mean lines and key anchors

Run:
    python -m analysis.paper5_horserace.figDT_demographic_transition          # colour
    python -m analysis.paper5_horserace.figDT_demographic_transition --bw     # B&W
"""
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

BW = False

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/figDT_demographic_transition.pdf"
FIG_BW = ROOT / "analysis/figures/paper5_horserace/figDT_demographic_transition_bw.pdf"


# Crude region map (iso3 -> macro-region) for the lower panel
ISO3_REGION = {
    # Western Europe / settler offshoots
    "GBR":"WEur","IRL":"WEur","FRA":"WEur","DEU":"WEur","NLD":"WEur","BEL":"WEur",
    "LUX":"WEur","CHE":"WEur","AUT":"WEur","ITA":"WEur","ESP":"WEur","PRT":"WEur",
    "DNK":"WEur","SWE":"WEur","NOR":"WEur","FIN":"WEur","ISL":"WEur",
    "USA":"WEur","CAN":"WEur","AUS":"WEur","NZL":"WEur",
    # Eastern Europe / former USSR
    "POL":"EEur","CZE":"EEur","SVK":"EEur","HUN":"EEur","ROU":"EEur","BGR":"EEur",
    "GRC":"EEur","ALB":"EEur","SRB":"EEur","HRV":"EEur","BIH":"EEur","MKD":"EEur",
    "MNE":"EEur","SVN":"EEur","RUS":"EEur","BLR":"EEur","UKR":"EEur","MDA":"EEur",
    "LTU":"EEur","LVA":"EEur","EST":"EEur","KAZ":"EEur","TUR":"EEur","ARM":"EEur",
    "GEO":"EEur","AZE":"EEur","CYP":"EEur","MLT":"EEur",
    # East / SE Asia
    "CHN":"EAsia","JPN":"EAsia","KOR":"EAsia","PRK":"EAsia","MNG":"EAsia","TWN":"EAsia",
    "HKG":"EAsia","SGP":"EAsia","VNM":"EAsia","THA":"EAsia","MMR":"EAsia","KHM":"EAsia",
    "LAO":"EAsia","MYS":"EAsia","IDN":"EAsia","PHL":"EAsia","BRN":"EAsia","TLS":"EAsia",
    # South Asia / Central Asia
    "IND":"SAsia","PAK":"SAsia","BGD":"SAsia","NPL":"SAsia","LKA":"SAsia","BTN":"SAsia",
    "MDV":"SAsia","AFG":"SAsia","UZB":"SAsia","TJK":"SAsia","TKM":"SAsia","KGZ":"SAsia",
    # MENA
    "EGY":"MENA","DZA":"MENA","TUN":"MENA","MAR":"MENA","LBY":"MENA","SDN":"MENA",
    "ESH":"MENA","SYR":"MENA","LBN":"MENA","JOR":"MENA","ISR":"MENA","PSE":"MENA",
    "IRQ":"MENA","IRN":"MENA","SAU":"MENA","YEM":"MENA","OMN":"MENA","ARE":"MENA",
    "QAT":"MENA","BHR":"MENA","KWT":"MENA",
    # SSA (everything else in Africa)
    # Americas (post-Columbian Latin America + Caribbean)
    "MEX":"LAm","GTM":"LAm","BLZ":"LAm","SLV":"LAm","HND":"LAm","NIC":"LAm",
    "CRI":"LAm","PAN":"LAm","CUB":"LAm","DOM":"LAm","HTI":"LAm","JAM":"LAm",
    "TTO":"LAm","BHS":"LAm","BRB":"LAm","COL":"LAm","VEN":"LAm","GUY":"LAm",
    "SUR":"LAm","ECU":"LAm","PER":"LAm","BOL":"LAm","BRA":"LAm","PRY":"LAm",
    "URY":"LAm","ARG":"LAm","CHL":"LAm",
}
SSA_OVERRIDES = {  # explicit SSA list (everything in Africa not in MENA above)
    "SEN","GMB","GIN","GNB","SLE","LBR","CIV","GHA","TGO","BEN","NGA","CMR",
    "TCD","CAF","GAB","COG","COD","AGO","ETH","ERI","DJI","SOM","KEN","UGA",
    "TZA","RWA","BDI","ZMB","MWI","MOZ","ZWE","BWA","NAM","SWZ","LSO","ZAF",
    "MDG","COM","MUS","SYC","BFA","MLI","NER","MRT","STP","CPV","GNQ","SSD",
}
for k in SSA_OVERRIDES:
    ISO3_REGION[k] = "SSA"
# Oceania misc
for k in ["FJI","PNG","SLB","VUT","WSM","TON","FSM","MHL","PLW","NRU","TUV","KIR"]:
    ISO3_REGION[k] = "Oceania"

REGION_ORDER = ["WEur", "EEur", "EAsia", "SAsia", "MENA", "LAm", "SSA", "Oceania"]
REGION_LABEL = {
    "WEur":  "W. Europe & offshoots",
    "EEur":  "E. Europe / fmr USSR",
    "EAsia": "East & SE Asia",
    "SAsia": "South & Central Asia",
    "MENA":  "MENA",
    "LAm":   "Latin Am. & Carib.",
    "SSA":   "Sub-Saharan Africa",
    "Oceania":"Pacific Oceania",
}


def main() -> None:
    global BW
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw", action="store_true", help="Produce grayscale B&W version")
    args = parser.parse_args()
    BW = args.bw

    df = pd.read_parquet(PANEL).set_index("iso3")
    dt = df["dt_timing_year"].dropna()
    print(f"DT-timing N: {len(dt)}; range {dt.min():.0f}-{dt.max():.0f}, median {dt.median():.0f}")

    cmap_name = "Greys_r" if BW else "RdYlBu_r"
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = Normalize(vmin=dt.quantile(0.02), vmax=dt.quantile(0.98))
    missing_color = "white" if BW else "#ececec"

    out = FIG_BW if BW else FIG
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.18)

    # ---------- Panel (a): world map ----------
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    shp = shpreader.natural_earth(resolution="110m", category="cultural",
                                  name="admin_0_countries")
    for country in shpreader.Reader(shp).records():
        iso3 = country.attributes.get("ADM0_A3", "")
        if iso3 in dt.index:
            color = cmap(norm(dt.loc[iso3]))
            hatch = None
        else:
            color = missing_color
            hatch = "////" if BW else None
        patch_kw = dict(facecolor=color, edgecolor="black", linewidth=0.15)
        if hatch:
            patch_kw["hatch"] = hatch
            patch_kw["linewidth"] = 0.1
        ax_map.add_geometries([country.geometry], ccrs.PlateCarree(), **patch_kw)
    ax_map.set_global()
    ax_map.set_title(
        r"(a) Demographic-transition timing: first year CBR $<$ 25 per 1,000",
        fontsize=12,
    )
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_map, orientation="horizontal",
                        pad=0.04, shrink=0.55, aspect=35)
    cbar.set_label("Year of demographic transition", fontsize=10)

    # ---------- Panel (b): timeline by region ----------
    ax_tl = fig.add_subplot(gs[1, 0])
    dt_df = dt.reset_index()
    dt_df["region"] = dt_df["iso3"].map(ISO3_REGION).fillna("Other")
    dt_df = dt_df[dt_df["region"].isin(REGION_ORDER)].copy()

    y_positions = {r: i for i, r in enumerate(REGION_ORDER)}
    rng = np.random.default_rng(7)

    for region in REGION_ORDER:
        sub = dt_df[dt_df["region"] == region]
        ypos = y_positions[region]
        # jitter for visibility
        ys = ypos + rng.uniform(-0.18, 0.18, size=len(sub))
        # color by year (same cmap as map) — gives reader an at-a-glance temporal cue
        face = [cmap(norm(v)) for v in sub["dt_timing_year"].values]
        ax_tl.scatter(sub["dt_timing_year"], ys, s=42,
                      facecolor=face, edgecolor="black", linewidth=0.4, zorder=3)
        # Region median line
        med = sub["dt_timing_year"].median()
        ax_tl.plot([med, med], [ypos - 0.32, ypos + 0.32],
                   color="black", linewidth=2.2, zorder=4)
        # Median text just above the bar
        ax_tl.text(med, ypos - 0.40, f"{int(med)}",
                   fontsize=8.5, va="bottom", ha="center",
                   fontweight="bold", zorder=5,
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.4))
        # Count to the right of plot
        ax_tl.text(2032, ypos, f"N={len(sub)}",
                   fontsize=8.5, va="center", ha="left", color="0.3")

    ax_tl.set_yticks(list(y_positions.values()))
    ax_tl.set_yticklabels([REGION_LABEL[r] for r in REGION_ORDER], fontsize=9.5)
    ax_tl.invert_yaxis()
    ax_tl.set_xlim(1790, 2055)
    ax_tl.set_xlabel("Year crude birth rate first fell below 25 per 1,000")
    ax_tl.grid(axis="x", linestyle=":", alpha=0.5, zorder=1)
    # Highlight century markers
    for x in (1800, 1850, 1900, 1950, 2000):
        ax_tl.axvline(x, color="0.7", linewidth=0.5, zorder=1)
    ax_tl.set_title(
        r"(b) Country-level transitions by macro-region (jittered scatter; black bars = regional medians)",
        fontsize=11,
    )

    # Anchor labels for the most extreme countries
    anchors = [
        ("NOR", "Norway 1809", "right"),
        ("FRA", "France 1828", "right"),
        ("JPN", "Japan 1873", "left"),
        ("CHN", "China 1991", "left"),
        ("IND", "India 2009", "right"),
        ("PNG", "PNG 2023", "left"),
    ]
    for iso, lab, ha in anchors:
        if iso in dt.index and iso in ISO3_REGION:
            yr = float(dt.loc[iso])
            ypos = y_positions[ISO3_REGION[iso]]
            dx = -5 if ha == "right" else 5
            ax_tl.annotate(lab, xy=(yr, ypos), xytext=(yr + dx, ypos - 0.55),
                           fontsize=7.5, ha=ha,
                           arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4"))

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
