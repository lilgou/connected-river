from pathlib import Path

import geopandas as gpd
import pandas as pd

from connected_river.utils.config import resolve_path


def ensure_dir(path: str | Path) -> Path:
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Pick the writer from the file extension so stages can use one small helper
# without hiding the difference between CSV and Parquet outputs.
def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {suffix}")

    return path


# Mirror write_table above and fail clearly on an extension the project does not
# support, rather than guessing how a file should be read.
def read_table(path: str | Path) -> pd.DataFrame:
    path = resolve_path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table format: {suffix}")


def write_geo(gdf: gpd.GeoDataFrame, path: str | Path, layer: str | None = None) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".gpkg":
        gdf.to_file(path, layer=layer or "data", driver="GPKG")
    else:
        gdf.to_file(path)

    return path
