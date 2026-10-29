"""Download the public sewage-discharge layer used for route context."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pandas as pd

from connected_river.download.http import session_with_retries
from connected_river.utils.config import resolve_path
from connected_river.utils.manifest import write_manifest_entry

LAYER_URL = (
    "https://services3.arcgis.com/Bb8lfThdhugyc4G3/ArcGIS/rest/services/"
    "Sewage_discharges_Final_treated_effluent/FeatureServer/0"
)
SOURCE_ITEM_ID = "5780673804f14d1fb433788f63b1804f"
SOURCE_DATASET_URL = (
    "https://www.data.gov.uk/dataset/55b8eaa8-60df-48a8-929a-060891b7a109/"
    "consented-discharges-to-controlled-waters-with-conditions1"
)
ARCGIS_PAGE_SIZE = 2_000


def _query_json(session, params: dict[str, object]) -> dict[str, object]:
    response = session.get(f"{LAYER_URL}/query", params=params, timeout=180)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS discharge query failed: {payload['error']}")
    return payload


# Permit records provide possible pressure context, not evidence that a
# discharge occurred on a particular day, so they stay separate from samples.
def download_consented_discharges(
    *,
    raw_dir: str | Path = "data/raw/discharge_permits",
    overwrite: bool = False,
    extract: bool = True,
) -> dict[str, Path | list[Path]]:
    """Download the EA-derived final/treated sewage layer as a portable CSV.

    The current official quarterly archive is a Microsoft Access database.
    The public ArcGIS layer exposes the Tier 1 sewage subset needed here and
    keeps the notebook independent of operating-system database drivers.
    ``extract`` is accepted for compatibility with the common downloader interface.
    """

    del extract
    raw_dir = resolve_path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "consented_sewage_discharges.csv"
    if output_path.is_file() and not overwrite:
        return {"csv": output_path}

    rows: list[dict[str, object]] = []
    with session_with_retries(total=6, backoff_factor=1.0) as session:
        count_payload = _query_json(
            session,
            {"where": "1=1", "returnCountOnly": "true", "f": "json"},
        )
        expected_count = int(count_payload.get("count", 0))
        if expected_count <= 0:
            raise RuntimeError("The public sewage-discharge layer returned no records")

        for offset in range(0, expected_count, ARCGIS_PAGE_SIZE):
            payload = _query_json(
                session,
                {
                    "where": "1=1",
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": 4326,
                    "orderByFields": "OBJECTID_1",
                    "resultOffset": offset,
                    "resultRecordCount": ARCGIS_PAGE_SIZE,
                    "f": "json",
                },
            )
            features = payload.get("features") or []
            if not features:
                break
            for feature in features:
                row = dict(feature.get("attributes") or {})
                geometry = feature.get("geometry") or {}
                row["longitude"] = geometry.get("x")
                row["latitude"] = geometry.get("y")
                rows.append(row)
            time.sleep(0.05)

    table = pd.DataFrame(rows)
    if "OBJECTID_1" not in table:
        raise RuntimeError("The public sewage-discharge layer has no OBJECTID_1 field")
    table = table.drop_duplicates("OBJECTID_1").sort_values("OBJECTID_1")
    if len(table) != expected_count:
        raise RuntimeError(
            "Incomplete public sewage-discharge download: "
            f"expected {expected_count:,} records, received {len(table):,}"
        )

    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.part")
    try:
        table.to_csv(temporary, index=False)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    write_manifest_entry(
        "outputs/logs/download_manifest.json",
        dataset="ea_consented_sewage_discharges",
        source_url=LAYER_URL,
        output_path=output_path,
        metadata={
            "records": len(table),
            "arcgis_item_id": SOURCE_ITEM_ID,
            "original_dataset_url": SOURCE_DATASET_URL,
            "publisher_attribution": "Environment Agency",
            "host": "The Rivers Trust ArcGIS Online",
            "licence": "Environment Agency Conditional Licence",
        },
    )
    return {"csv": output_path}
