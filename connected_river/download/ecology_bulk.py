"""Bulk downloaders for EA Ecology & Fish Data Explorer datasets.

These functions fetch the official bulk ZIP files from the EA Ecology & Fish Data
Explorer and optionally normalise CSV/XLSX contents into Parquet for downstream
processing. They are intentionally generic because EA occasionally changes the
inner filenames while keeping the public download URLs stable.
"""

from __future__ import annotations

from pathlib import Path

from connected_river.download.http import (
    download_file,
    extract_zip,
    normalise_columns,
    read_tabular,
)
from connected_river.utils.config import resolve_path
from connected_river.utils.io import ensure_dir
from connected_river.utils.manifest import write_manifest_entry

BASE_URL = "https://environment.data.gov.uk/ecology/explorer/downloads"

ECOLOGY_DOWNLOADS = {
    "fish_counts": f"{BASE_URL}/FW_Fish_Counts.zip",
    "fish_lengths": f"{BASE_URL}/FW_Fish_Individual_Lengths.zip",
    "fish_banded_measurements": f"{BASE_URL}/FW_Fish_Banded_Measurements.zip",
    "fish_bulk_measurements": f"{BASE_URL}/FW_Fish_Bulk_Measurements.zip",
    "fish_data_types": f"{BASE_URL}/FW_Fish_Data_Types.zip",
    "fish_sites": f"{BASE_URL}/FW_Fish_Sites.zip",
    "trac_fish_counts": f"{BASE_URL}/TRAC_Fish_Counts.zip",
    "trac_fish_sites": f"{BASE_URL}/TRAC_Fish_Sites.zip",
    "macroinvertebrates": f"{BASE_URL}/INV_OPEN_DATA.zip",
    "macrophytes": f"{BASE_URL}/MACP_OPEN_DATA.zip",
    "diatoms": f"{BASE_URL}/DIAT_OPEN_DATA.zip",
    "biosys_taxon_info": f"{BASE_URL}/OPEN_DATA_TAXON_INFO.zip",
}


# Keep each publisher file as supplied. Cleaning and column changes happen in
# the ingest step, which leaves the original download available for checking.
def download_ecology_bulk(
    kind: str,
    *,
    raw_dir: str | Path = "data/raw/ecology",
    overwrite: bool = False,
    extract: bool = True,
    convert_parquet: bool = True,
) -> dict[str, Path | list[Path]]:
    """Download one EA ecology/fish bulk dataset.

    Parameters
    ----------
    kind:
        One of ``ECOLOGY_DOWNLOADS`` keys, e.g. ``fish_counts`` or
        ``macroinvertebrates``.
    """
    if kind not in ECOLOGY_DOWNLOADS:
        raise ValueError(f"Unknown ecology bulk dataset {kind!r}. Options: {sorted(ECOLOGY_DOWNLOADS)}")
    raw_dir = resolve_path(raw_dir)
    url = ECOLOGY_DOWNLOADS[kind]
    # Keep the publisher filename. The import settings use these stable
    # archive names, so an annual refresh must replace the exact file consumed
    # by the release rather than create a second, friendly-named cache.
    zip_path = raw_dir / Path(url).name
    zip_path = download_file(
        url,
        zip_path,
        overwrite=overwrite,
        dataset=f"ea_ecology_{kind}",
        metadata={"kind": kind},
    )
    result: dict[str, Path | list[Path]] = {"zip": zip_path}
    if not extract:
        return result
    extracted_dir = raw_dir / kind
    extracted = extract_zip(zip_path, extracted_dir, overwrite=overwrite)
    result["extracted"] = extracted
    if convert_parquet:
        parquet_dir = ensure_dir(Path("data/interim/ecology") / kind)
        parquet_files: list[Path] = []
        for file in extracted:
            if file.suffix.lower() not in {".csv", ".txt", ".xlsx", ".xls"}:
                continue
            try:
                df = normalise_columns(read_tabular(file))
            except Exception:
                continue
            out = parquet_dir / f"{file.stem}.parquet"
            df.to_parquet(out, index=False)
            parquet_files.append(out)
        result["parquet"] = parquet_files
        write_manifest_entry(
            "outputs/logs/download_manifest.json",
            dataset=f"ea_ecology_{kind}_parquet",
            source_url=url,
            output_path=parquet_dir,
            metadata={"files": [str(p) for p in parquet_files], "n_files": len(parquet_files)},
        )
    return result


def download_all_core_ecology(
    *,
    raw_dir: str | Path = "data/raw/full_river_health_sources/ea_ecology_fish",
    overwrite: bool = False,
    extract: bool = False,
    convert_parquet: bool = False,
) -> dict[str, dict[str, Path | list[Path]]]:
    """Download the biology modules used by the complete workflow.

    The release ingester reads the ZIP members directly.  Extraction and an
    additional interim Parquet conversion are therefore opt-in, avoiding a
    redundant multi-gigabyte rewrite during every annual update.
    """
    return {
        kind: download_ecology_bulk(
            kind, raw_dir=raw_dir, overwrite=overwrite,
            extract=extract, convert_parquet=convert_parquet,
        )
        for kind in [
            "macroinvertebrates", "fish_counts", "fish_sites",
            "trac_fish_counts", "trac_fish_sites",
            "macrophytes", "diatoms", "biosys_taxon_info",
        ]
    }
