"""Ingest cached EA Water Quality Archive partitions without loading the archive at once."""

from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from connected_river.chemistry_registry import TARGET_DETERMINANDS
from connected_river.harmonise.wq_samples import COLUMN_MAP, harmonise_wq_observations

RAW_COLUMNS = set(COLUMN_MAP)
REQUIRED_RAW_COLUMNS = {
    "id", "samplingPoint.notation", "phenomenonTime",
    "determinand.notation", "determinand.prefLabel", "result",
}
FULL_COLUMNS = [
    "observation_id", "wq_site_id", "wq_site_name", "easting", "northing",
    "region", "area", "sub_area", "site_status", "site_type",
    "sample_datetime", "sample_date", "year", "month", "determinand_id",
    "determinand_name", "determinand_key", "target_determinand_key",
    "is_matching_determinand", "result", "result_qualifier", "result_standardized",
    "result_standardized_loq_half", "result_standardized_below_loq_zero",
    "unit_standardized", "has_valid_coordinates", "has_numeric_result",
]
COMPACT_COLUMNS = [
    "observation_id", "wq_site_id", "wq_site_name", "easting", "northing",
    "sample_datetime", "sample_date", "year",
    "determinand_id", "determinand_name", "determinand_key", "target_determinand_key",
    "is_matching_determinand", "result_qualifier", "result_standardized",
    "result_standardized_loq_half", "result_standardized_below_loq_zero",
    "unit_standardized",
]


def _stable_mean_sql(value: str) -> str:
    """Return an order-independent mean for decimal environmental results."""
    fixed = f"CAST({value} AS DECIMAL(38, 10))"
    return f"CAST(SUM({fixed}) / COUNT({value}) AS DOUBLE)"


def _stable_sample_std_sql(value: str) -> str:
    """Return an order-independent sample SD using fixed-point moments."""
    fixed = f"CAST({value} AS DECIMAL(38, 10))"
    variance = (
        f"(SUM({fixed} * {fixed}) - SUM({fixed}) * SUM({fixed}) "
        f"/ COUNT({value})) / (COUNT({value}) - 1)"
    )
    return (
        f"CASE WHEN COUNT({value}) > 1 THEN SQRT(CAST(GREATEST("
        f"CAST(0 AS DECIMAL(38, 20)), {variance}) AS DOUBLE)) ELSE NULL END"
    )


def _sql_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


def _read_required_columns(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        available = set(pq.ParquetFile(path).schema_arrow.names)
        return pd.read_parquet(path, columns=sorted(RAW_COLUMNS & available))
    return pd.read_csv(
        path, usecols=lambda column: column in RAW_COLUMNS,
        dtype="string", low_memory=False,
    )


def _observation_partition_status(path: Path) -> str:
    """Classify a cache without confusing a valid zero-result query with damage."""
    try:
        if path.suffix.casefold() == ".parquet":
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows == 0:
                columns = set(parquet.schema_arrow.names)
                return (
                    "queried zero observations"
                    if not columns or REQUIRED_RAW_COLUMNS.issubset(columns)
                    else "wrong schema"
                )
            columns = set(parquet.schema_arrow.names)
        elif path.suffix.casefold() == ".csv":
            try:
                columns = set(pd.read_csv(path, nrows=0).columns)
            except pd.errors.EmptyDataError:
                return "queried zero observations"
            if path.stat().st_size == 0:
                return "queried zero observations"
        else:
            return "unsupported format"
    except Exception:
        # Parser/footer exceptions vary across pandas and PyArrow versions; any
        # inspection failure is an acquisition failure, never evidence that a
        # required code/year partition exists.
        return "unreadable"
    return "valid" if REQUIRED_RAW_COLUMNS.issubset(columns) else "wrong schema"


# Summarise observations at site-year level first. This keeps the later
# matching stage small without throwing away the original sample files.
def _site_year_tables(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = connection.sql("""
        SELECT
            wq_site_id,
            CAST(year AS INTEGER) AS year,
            first(wq_site_name) AS wq_site_name,
            median(easting) AS easting,
            median(northing) AS northing,
            first(region) AS region,
            first(area) AS area,
            first(sub_area) AS sub_area,
            first(site_status) AS site_status,
            first(site_type) AS site_type,
            count(observation_id) AS n_observations,
            sum(CASE WHEN has_numeric_result THEN 1 ELSE 0 END) AS n_numeric_observations,
            count(DISTINCT sample_date) AS n_sampling_dates,
            count(DISTINCT month) AS n_months_sampled,
            count(DISTINCT determinand_name) AS n_determinands,
            count(DISTINCT determinand_key) AS n_target_determinands,
            min(sample_datetime) AS date_min,
            max(sample_datetime) AS date_max
        FROM requested_observations
        WHERE has_valid_coordinates AND is_matching_determinand
        GROUP BY wq_site_id, year
    """).df()
    target_counts = connection.sql("""
        SELECT wq_site_id, CAST(year AS INTEGER) AS year, determinand_key,
               count(observation_id) AS observation_count
        FROM requested_observations
        WHERE has_valid_coordinates AND is_matching_determinand
          AND determinand_key IS NOT NULL
        GROUP BY wq_site_id, year, determinand_key
    """).df()
    return summary, target_counts


def _finish_site_year(summary: pd.DataFrame, target_counts: pd.DataFrame) -> pd.DataFrame:
    if target_counts.empty:
        out = summary.copy()
    else:
        wide = target_counts.pivot_table(
            index=["wq_site_id", "year"], columns="determinand_key",
            values="observation_count", aggfunc="sum", fill_value=0,
        ).reset_index()
        out = summary.merge(wide, on=["wq_site_id", "year"], how="left")
    numeric_columns = out.select_dtypes(include="number").columns
    out[numeric_columns] = out[numeric_columns].fillna(0)
    out["has_minimum_temporal_coverage"] = out["n_sampling_dates"] >= 3
    metals = [column for column in out if column in {
        "zinc", "copper", "cadmium", "lead", "nickel", "chromium", "mercury", "arsenic",
    }]
    core = [column for column in out if column in {
        "ammoniacal_n_mg_l", "nitrate_n_mg_l", "orthophosphate_p_mg_l",
        "dissolved_oxygen_mg_l", "dissolved_oxygen_percent", "ph",
    }]
    out["has_metals"] = out[metals].sum(axis=1).gt(0) if metals else False
    out["has_core_chemistry"] = out[core].sum(axis=1).gt(0) if core else False
    return out


def _feature_table(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    value = "result_standardized"
    stable_mean = _stable_mean_sql(value)
    stable_std = _stable_sample_std_sql(value)
    feature_long = connection.sql(f"""
        SELECT
            wq_site_id,
            CAST(year AS INTEGER) AS year,
            determinand_key,
            count(result_standardized) AS observation_count,
            {stable_mean} AS mean,
            median(result_standardized) AS median,
            min(result_standardized) AS minimum,
            max(result_standardized) AS maximum,
            {stable_std} AS std
        FROM requested_observations
        WHERE determinand_key IS NOT NULL AND result_standardized IS NOT NULL
        GROUP BY wq_site_id, year, determinand_key
    """).df()
    if feature_long.empty:
        return pd.DataFrame(columns=["wq_site_id", "year"])
    feature_wide = feature_long.pivot(
        index=["wq_site_id", "year"], columns="determinand_key",
        values=["observation_count", "mean", "median", "minimum", "maximum", "std"],
    )
    feature_wide.columns = [
        f"{determinand}_{stat}" for stat, determinand in feature_wide.columns
    ]
    return feature_wide.reset_index()


# This is the main entry point for turning the downloaded archive partitions
# into the site, year and determinand tables used by the rest of the pipeline.
def build_water_quality_index(
    raw_dir: str | Path, output_dir: str | Path, years: list[int] | tuple[int, ...],
    *, require_target_determinand_completeness: bool = False,
    target_determinand_completeness_years: list[int] | tuple[int, ...] | range | None = None,
) -> dict[str, Path]:
    """Build compact observations and site-year features using bounded memory.

    Raw year/determinand files are harmonised independently. DuckDB then
    performs archive-wide deduplication and aggregation on disk. The explicit
    preceding year is retained only in the observations output for January-June
    six-month survey windows; candidate activity and annual features remain
    restricted to requested integration years.
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_years = {int(year) for year in years}
    if not requested_years:
        raise ValueError("At least one water-quality year is required")
    observation_years = requested_years | {min(requested_years) - 1}
    paths = {
        "observations": output_dir / "wq_observations.parquet",
        "site_year": output_dir / "wq_site_year.parquet",
        "features": output_dir / "wq_site_year_features.parquet",
        "inventory": output_dir / "wq_source_inventory.parquet",
    }
    inventory: list[dict[str, object]] = []
    paths_to_read = sorted(raw_dir.rglob("*.csv")) + sorted(raw_dir.rglob("*.parquet"))
    partition_status = {path: _observation_partition_status(path) for path in paths_to_read}
    expected_partitions: set[tuple[int, str]] = set()
    verified_partitions: set[tuple[int, str]] = set()
    if require_target_determinand_completeness:
        available_partitions: set[tuple[int, str]] = set()
        for path in paths_to_read:
            match = re.match(r"((?:19|20)\d{2})[_-].*[_-](\d{4})$", path.stem)
            if match and partition_status[path] in {"valid", "queried zero observations"}:
                available_partitions.add((int(match.group(1)), match.group(2)))
        required_years = (
            {int(year) for year in target_determinand_completeness_years}
            if target_determinand_completeness_years is not None else observation_years
        )
        expected_partitions = {
            (year, item.notation)
            for year in required_years
            for item in TARGET_DETERMINANDS
        }
        missing_partitions = sorted(expected_partitions - available_partitions)
        if missing_partitions:
            preview = ", ".join(
                f"{year}/{notation}" for year, notation in missing_partitions[:12]
            )
            raise RuntimeError(
                f"Missing {len(missing_partitions)} required target year/determinand "
                f"partitions (first: {preview})"
            )
        verified_partitions = {
            (int(match.group(1)), match.group(2))
            for path in paths_to_read
            if partition_status[path] == "queried zero observations"
            if (match := re.match(r"((?:19|20)\d{2})[_-].*[_-](\d{4})$", path.stem))
            if (int(match.group(1)), match.group(2)) in expected_partitions
        }

    with TemporaryDirectory(prefix="_wq_chunks_", dir=output_dir) as temporary_directory:
        chunk_dir = Path(temporary_directory)
        chunk_number = 0
        for path in paths_to_read:
            file_year = re.match(r"((?:19|20)\d{2})[_-]", path.name)
            if file_year and int(file_year.group(1)) not in observation_years:
                inventory.append({"file": str(path), "rows": 0, "status": "outside requested years"})
                continue
            status = partition_status[path]
            if status != "valid":
                inventory.append({"file": str(path), "rows": 0, "status": status})
                continue
            try:
                raw = _read_required_columns(path)
            except (pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
                inventory.append({"file": str(path), "rows": 0, "status": f"unreadable: {exc}"})
                continue
            if not REQUIRED_RAW_COLUMNS.issubset(raw.columns):
                inventory.append({"file": str(path), "rows": len(raw), "status": "wrong schema"})
                continue
            partition_match = re.match(
                r"((?:19|20)\d{2})[_-].*[_-](\d{4})$", path.stem,
            )
            partition_identity = (
                (int(partition_match.group(1)), partition_match.group(2))
                if partition_match else None
            )
            if partition_identity in expected_partitions:
                expected_year, expected_code = partition_identity
                observed_codes = (
                    raw["determinand.notation"].astype("string").str.strip().str.zfill(4)
                    .dropna().unique().tolist()
                )
                observed_years = (
                    pd.to_datetime(raw["phenomenonTime"], errors="coerce").dt.year
                    .dropna().astype(int).unique().tolist()
                )
                if set(observed_codes) != {expected_code}:
                    inventory.append({
                        "file": str(path), "rows": len(raw),
                        "status": "wrong determinand partition",
                    })
                    continue
                if set(observed_years) != {expected_year}:
                    inventory.append({
                        "file": str(path), "rows": len(raw),
                        "status": "wrong year partition",
                    })
                    continue
                verified_partitions.add(partition_identity)
            source_rows = len(raw)
            observations = harmonise_wq_observations(raw)
            observations = observations[
                observations["year"].isin(observation_years)
                & observations["site_type"].astype("string").str.upper().eq("FRESHWATER - RIVERS")
                & observations["sample_material"].astype("string").str.contains(
                    "RIVER|RUNNING SURFACE WATER", case=False, na=False,
                )
                & ~observations["sample_material"].astype("string").str.contains(
                    "SEDIMENT", case=False, na=False,
                )
                & (
                    observations["determinand_key"].notna()
                    | observations["target_determinand_key"].notna()
                )
            ].copy()
            if not observations.empty:
                observations[[column for column in FULL_COLUMNS if column in observations]].to_parquet(
                    chunk_dir / f"chunk_{chunk_number:05d}.parquet", index=False,
                )
                chunk_number += 1
            inventory.append({"file": str(path), "rows": source_rows, "status": "loaded"})

        if require_target_determinand_completeness:
            invalid_partitions = sorted(expected_partitions - verified_partitions)
            if invalid_partitions:
                preview = ", ".join(
                    f"{year}/{notation}" for year, notation in invalid_partitions[:12]
                )
                raise RuntimeError(
                    f"Invalid {len(invalid_partitions)} required target year/determinand "
                    f"partitions after content verification (first: {preview})"
                )

        if not chunk_number:
            raise RuntimeError(f"No EA water-quality observations found in {raw_dir}")

        full_path = output_dir / "_wq_observations_full.parquet"
        connection = duckdb.connect()
        try:
            connection.execute("PRAGMA threads=4")
            connection.execute("PRAGMA preserve_insertion_order=false")
            connection.execute(f"""
                COPY (
                    SELECT * EXCLUDE (_duplicate_rank)
                    FROM (
                        SELECT *, row_number() OVER (
                            PARTITION BY observation_id ORDER BY sample_datetime
                        ) AS _duplicate_rank
                        FROM read_parquet('{_sql_path(chunk_dir / '*.parquet')}', union_by_name=true)
                    )
                    WHERE observation_id IS NULL OR _duplicate_rank = 1
                ) TO '{_sql_path(full_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            connection.execute(
                f"CREATE VIEW all_observations AS SELECT * FROM read_parquet('{_sql_path(full_path)}')"
            )
            requested_sql = ", ".join(str(year) for year in sorted(requested_years))
            connection.execute(
                f"CREATE VIEW requested_observations AS "
                f"SELECT * FROM all_observations WHERE year IN ({requested_sql})"
            )
            compact_sql = ", ".join(COMPACT_COLUMNS)
            connection.execute(f"""
                COPY (SELECT {compact_sql} FROM all_observations)
                TO '{_sql_path(paths['observations'])}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            summary, target_counts = _site_year_tables(connection)
            site_year = _finish_site_year(summary, target_counts)
            features = _feature_table(connection)
        finally:
            connection.close()
            full_path.unlink(missing_ok=True)

    site_year.to_parquet(paths["site_year"], index=False)
    features.to_parquet(paths["features"], index=False)
    pd.DataFrame(inventory).to_parquet(paths["inventory"], index=False)
    return paths
