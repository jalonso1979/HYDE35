"""Region and country mask construction for HYDE 3.5 grids."""
import numpy as np
import xarray as xr
from typing import Optional


def build_bbox_mask(
    template: xr.DataArray,
    bbox: dict,
    land_mask: xr.DataArray,
    subtract_boxes: Optional[list[dict]] = None,
) -> xr.DataArray:
    """Build a binary mask from a lat/lon bounding box.

    Parameters
    ----------
    template:
        DataArray with ``lat`` and ``lon`` coordinates that define the grid.
    bbox:
        Dict with keys ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.
    land_mask:
        Binary DataArray (1 = land, 0 = ocean/excluded) on the same grid.
    subtract_boxes:
        Optional list of bbox dicts to subtract from the primary mask.

    Returns
    -------
    xr.DataArray
        Float mask (1.0 = included land cell, 0.0 = excluded).
    """
    lat = template["lat"]
    lon = template["lon"]
    mask = (
        (lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"])
        & (lon >= bbox["lon_min"]) & (lon <= bbox["lon_max"])
    )
    mask = xr.where(mask, 1.0, 0.0) * land_mask
    if subtract_boxes:
        for sub in subtract_boxes:
            sub_mask = (
                (lat >= sub["lat_min"]) & (lat <= sub["lat_max"])
                & (lon >= sub["lon_min"]) & (lon <= sub["lon_max"])
            )
            mask = mask * xr.where(sub_mask, 0.0, 1.0)
    return mask


def build_country_mask(
    country_grid: xr.DataArray,
    country_id: int | float,
    land_mask: xr.DataArray,
) -> xr.DataArray:
    """Build a binary mask for a single country from the ISO numeric grid.

    Parameters
    ----------
    country_grid:
        DataArray where each cell contains the ISO numeric country ID (or NaN
        for ocean/unassigned cells).
    country_id:
        The numeric country identifier to select.
    land_mask:
        Binary DataArray (1 = land, 0 = ocean/excluded) on the same grid.

    Returns
    -------
    xr.DataArray
        Float mask (1.0 = land cell belonging to this country, 0.0 otherwise).
    """
    mask = xr.where(country_grid == country_id, 1.0, 0.0)
    return mask * land_mask


def build_region_masks_from_json(
    region_defs: dict,
    template: xr.DataArray,
    land_mask: xr.DataArray,
) -> xr.DataArray:
    """Build stacked region masks from the hyde35_region_defs.json structure.

    Parameters
    ----------
    region_defs:
        Dict mapping region name -> bbox definition (keys: lat_min, lat_max,
        lon_min, lon_max, and optionally subtract_boxes).
    template:
        DataArray with ``lat`` and ``lon`` coordinates defining the grid.
    land_mask:
        Binary DataArray (1 = land) on the same grid.

    Returns
    -------
    xr.DataArray
        DataArray with a ``region`` dimension containing one mask per region.
    """
    masks = []
    names = []
    for name, rdef in region_defs.items():
        bbox = {
            "lat_min": rdef["lat_min"], "lat_max": rdef["lat_max"],
            "lon_min": rdef["lon_min"], "lon_max": rdef["lon_max"],
        }
        subtract = rdef.get("subtract_boxes", None)
        m = build_bbox_mask(template, bbox, land_mask, subtract_boxes=subtract)
        masks.append(m)
        names.append(name)
    stacked = xr.concat(masks, dim="region")
    stacked["region"] = names
    return stacked
