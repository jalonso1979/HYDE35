"""Map ERA5 download regions (1-25) to ISO3 country codes via centroid lookup.

For the 7-country Long Shadow panel we only need 7 ISO3 codes. The mapping
is derived by checking each country capital's lat/lon against each region's
bounding box.
"""
from __future__ import annotations
from pathlib import Path
import xarray as xr

ERA5_ROOT = Path("/Volumes/BIGDATA/HYDE35/ERA5")

# Approximate capital coordinates for the 7 Long Shadow countries
COUNTRY_CAPITAL_COORDS = {
    "GBR": (51.5074, -0.1278),   # London
    "FRA": (48.8566, 2.3522),    # Paris
    "ITA": (41.9028, 12.4964),   # Rome
    "SWE": (59.3293, 18.0686),   # Stockholm
    "BEL": (50.8503, 4.3517),    # Brussels
    "NLD": (52.3676, 4.9041),    # Amsterdam
    "ESP": (40.4168, -3.7038),   # Madrid
}

# Approximate national bounding boxes (lat_min, lat_max, lon_min, lon_max)
# generous to capture rural climate alongside urban centres.
COUNTRY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "GBR": (49.9, 60.9, -8.6, 1.8),     # UK
    "FRA": (42.3, 51.1, -5.1, 9.6),     # France (metropolitan)
    "ITA": (36.6, 47.1, 6.6, 18.5),     # Italy
    "SWE": (55.3, 69.1, 11.1, 24.2),    # Sweden
    "BEL": (49.5, 51.5, 2.5, 6.4),      # Belgium
    "NLD": (50.7, 53.6, 3.4, 7.2),      # Netherlands
    "ESP": (36.0, 43.8, -9.3, 3.3),     # Spain (peninsula)
}


def _region_bbox(region_id: int) -> tuple[float, float, float, float] | None:
    """Return (lat_min, lat_max, lon_min, lon_max) for an ERA5 region, or None."""
    reg_dir = ERA5_ROOT / f"region={region_id}"
    if not reg_dir.exists():
        return None
    for yr_dir in sorted(reg_dir.glob("year=*"), reverse=True):
        ext_dir = yr_dir / "_extracted"
        if not ext_dir.exists():
            continue
        nc_files = sorted(ext_dir.glob("*.nc"))
        if not nc_files:
            continue
        try:
            ds = xr.open_dataset(nc_files[0])
        except Exception:
            continue
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        if lat_name not in ds.coords or lon_name not in ds.coords:
            ds.close()
            continue
        bbox = (
            float(ds[lat_name].min()),
            float(ds[lat_name].max()),
            float(ds[lon_name].min()),
            float(ds[lon_name].max()),
        )
        ds.close()
        return bbox
    return None


def build_region_country_map(countries: dict[str, tuple[float, float]] | None = None,
                                n_regions: int = 25) -> dict[str, int]:
    """Map ISO3 → region_id by checking which region's bbox contains the capital."""
    if countries is None:
        countries = COUNTRY_CAPITAL_COORDS
    region_bboxes = {r: _region_bbox(r) for r in range(1, n_regions + 1)}
    region_bboxes = {r: b for r, b in region_bboxes.items() if b is not None}

    result: dict[str, int] = {}
    for iso, (lat, lon) in countries.items():
        # ERA5 longitudes can be 0..360 or -180..180; normalize both
        lon_alt = lon + 360 if lon < 0 else lon - 360
        for region_id, (lat_min, lat_max, lon_min, lon_max) in region_bboxes.items():
            in_lat = lat_min <= lat <= lat_max
            in_lon = (lon_min <= lon <= lon_max) or (lon_min <= lon_alt <= lon_max)
            if in_lat and in_lon:
                result[iso] = region_id
                break
        else:
            result[iso] = -1  # not found
    return result


def build_region_country_map_with_bboxes(
    countries: dict[str, tuple[float, float]] | None = None,
    bboxes: dict[str, tuple[float, float, float, float]] | None = None,
    n_regions: int = 25,
) -> dict[str, dict]:
    """Map ISO3 → {region: int, bbox: (lat_min, lat_max, lon_min, lon_max)}.

    The region tells you which ERA5 tile to read; the bbox tells you which
    grid points within that tile to average for country-specific values.
    """
    if countries is None:
        countries = COUNTRY_CAPITAL_COORDS
    if bboxes is None:
        bboxes = COUNTRY_BBOXES
    region_map = build_region_country_map(countries=countries, n_regions=n_regions)
    return {
        iso: {"region": region_map.get(iso, -1), "bbox": bboxes.get(iso)}
        for iso in countries
    }


if __name__ == "__main__":
    full = build_region_country_map_with_bboxes()
    print("ISO3 → ERA5 region + country bbox:")
    for iso, entry in sorted(full.items()):
        bbox = entry["bbox"]
        if bbox:
            print(f"  {iso}: region={entry['region']}, "
                  f"lat [{bbox[0]:.1f}, {bbox[1]:.1f}], "
                  f"lon [{bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            print(f"  {iso}: region={entry['region']}, bbox=None")
