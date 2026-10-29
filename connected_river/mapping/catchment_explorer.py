"""Release-time catchment geometry and spatial-assignment helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# Keep a lighter copy of the boundary geometry for plotting. The detailed raw
# shapefile is not needed every time the notebook draws a national map.
def build_catchment_map_geometry(
    source: str | Path,
    output: str | Path,
    *,
    tolerance_degrees: float = 0.002,
) -> Path:
    """Create a lightweight polygon-only GeoJSON from publisher geometry."""
    source, output = Path(source), Path(output)
    geometry = gpd.read_file(source)
    geometry = geometry[
        geometry["geometry-type"].astype(str).str.endswith("/Catchment")
        & geometry.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        & geometry["id"].notna()
    ][["id", "name", "geometry"]].copy()
    geometry["geometry"] = geometry.geometry.simplify(
        tolerance_degrees, preserve_topology=True
    )
    geometry = geometry.rename(columns={"name": "catchment_name"}).drop_duplicates(
        "id"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry.to_file(output, driver="GeoJSON")
    return output


def build_hydrological_area_map_geometry(
    source: str | Path,
    output: str | Path,
    *,
    tolerance_degrees: float = 0.002,
) -> Path:
    """Prepare simplified hydrological-area geometry for release outputs."""
    source, output = Path(source), Path(output)
    geometry = gpd.read_file(source)[["HA_NUM", "HA_NAME", "geometry"]].copy()
    geometry = geometry.to_crs("EPSG:4326")
    geometry["geometry"] = geometry.geometry.simplify(
        tolerance_degrees, preserve_topology=True
    )
    geometry["id"] = geometry["HA_NUM"].astype("Int64").astype(str)
    geometry = geometry.rename(columns={"HA_NAME": "catchment_name"})
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry[["id", "catchment_name", "geometry"]].to_file(
        output, driver="GeoJSON"
    )
    return output


# Use a spatial join first, then keep unmatched points visible rather than
# forcing them into the nearest area without a clear rule.
def assign_hydrological_areas(
    evidence: pd.DataFrame,
    shapefile: str | Path,
) -> pd.DataFrame:
    """Spatially assign each evidence row to the supplied hydrological areas."""
    frame = evidence.reset_index(drop=True).copy()
    frame["_row_id"] = np.arange(len(frame))
    valid = (
        pd.to_numeric(frame["easting"], errors="coerce").notna()
        & pd.to_numeric(frame["northing"], errors="coerce").notna()
    )
    points = gpd.GeoDataFrame(
        frame.loc[valid, ["_row_id"]].copy(),
        geometry=gpd.points_from_xy(
            frame.loc[valid, "easting"], frame.loc[valid, "northing"]
        ),
        crs="EPSG:27700",
    )
    areas = gpd.read_file(shapefile)[["HA_NUM", "HA_NAME", "geometry"]]
    joined = gpd.sjoin(points, areas, how="left", predicate="within")
    assignment = (
        joined.sort_values("_row_id")
        .drop_duplicates("_row_id")[["_row_id", "HA_NUM", "HA_NAME"]]
    )
    frame = frame.merge(assignment, on="_row_id", how="left", validate="one_to_one")
    frame["map_area_id"] = (
        pd.to_numeric(frame["HA_NUM"], errors="coerce")
        .astype("Int64")
        .astype("string")
    )
    frame["catchment_name"] = frame["HA_NAME"]
    return frame.drop(columns=["_row_id", "HA_NUM", "HA_NAME"])
