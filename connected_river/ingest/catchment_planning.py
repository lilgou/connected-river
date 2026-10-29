"""Create river-only WFD status, objective and pressure tables from CDE exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _river_chunks(path: Path) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        if "Water Body Type" in chunk:
            chunk = chunk[chunk["Water Body Type"].astype("string").str.contains("River", case=False, na=False)]
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _join_distinct(values: pd.Series) -> str:
    return "; ".join(sorted(set(values.dropna().astype(str))))


def _reason_summary(reasons: pd.DataFrame) -> pd.DataFrame:
    """Keep regulatory RNAGs distinct from Reasons for Deterioration.

    The Environment Agency export places both record types in ``rnags.csv``.
    Combining them overstates the number of Reasons for Not Achieving Good and
    obscures whether a row describes current failure or deterioration.
    """
    columns = [
        "Water Body ID", "rnag_count", "rfd_count", "attributed_reason_count",
        "sectors", "activities", "pressure_tier_3", "rfd_sectors",
        "rfd_activities", "rfd_pressure_tier_3",
    ]
    if reasons.empty:
        return pd.DataFrame(columns=columns)
    x = reasons.copy()
    reason_type = x.get(
        "Reason Type", pd.Series("RNAG", index=x.index, dtype="string")
    ).astype("string").str.strip().str.upper()
    x["_reason_type"] = reason_type
    x["_reason_id"] = x["ID"].astype("string")

    counts = (
        x.dropna(subset=["Water Body ID", "_reason_id"])
        .drop_duplicates(["Water Body ID", "_reason_type", "_reason_id"])
        .pivot_table(
            index="Water Body ID",
            columns="_reason_type",
            values="_reason_id",
            aggfunc="nunique",
            fill_value=0,
        )
        .reset_index()
    )
    for source, target in [("RNAG", "rnag_count"), ("RFD", "rfd_count")]:
        values = counts.get(source, pd.Series(0, index=counts.index))
        counts[target] = pd.to_numeric(values, errors="coerce").fillna(0).astype(int)
    counts["attributed_reason_count"] = counts["rnag_count"] + counts["rfd_count"]
    counts = counts[["Water Body ID", "rnag_count", "rfd_count", "attributed_reason_count"]]

    def category_summary(kind: str, prefix: str) -> pd.DataFrame:
        subset = x[x["_reason_type"].eq(kind)]
        if subset.empty:
            return pd.DataFrame(columns=[
                "Water Body ID", f"{prefix}sectors", f"{prefix}activities",
                f"{prefix}pressure_tier_3",
            ])
        return subset.groupby("Water Body ID", as_index=False).agg(
            **{
                f"{prefix}sectors": ("Business Sector", _join_distinct),
                f"{prefix}activities": ("Activity", _join_distinct),
                f"{prefix}pressure_tier_3": ("Pressure Tier 3", _join_distinct),
            }
        )

    rnag = category_summary("RNAG", "").rename(columns={
        "sectors": "sectors", "activities": "activities",
        "pressure_tier_3": "pressure_tier_3",
    })
    rfd = category_summary("RFD", "rfd_")
    return counts.merge(rnag, on="Water Body ID", how="left").merge(
        rfd, on="Water Body ID", how="left"
    )[columns]


# Reduce the planning tables to the water-body fields needed later, while still
# retaining the source categories used to describe local context.
def build_catchment_planning_index(raw_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    raw_root, output_dir = Path(raw_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classifications = _river_chunks(raw_root / "catchment_classifications/classifications.csv")
    classifications["Year"] = pd.to_numeric(classifications["Year"], errors="coerce")
    classifications["Cycle"] = pd.to_numeric(classifications["Cycle"], errors="coerce")
    latest = classifications.sort_values(["Cycle", "Year"]).drop_duplicates(["Water Body ID", "Classification Level", "Classification Item"], keep="last")
    headline = latest[latest["Classification Item"].isin(["Overall Water Body", "Ecological", "Chemical"])].copy()
    status_summary = headline.pivot_table(
        index="Water Body ID", columns="Classification Item", values="Status", aggfunc="first",
    ).rename(columns={"Overall Water Body": "overall_status", "Ecological": "ecological_status", "Chemical": "chemical_status"}).reset_index()
    status_years = headline.pivot_table(
        index="Water Body ID", columns="Classification Item", values="Year", aggfunc="max",
    ).rename(columns={"Overall Water Body": "overall_status_year", "Ecological": "ecological_status_year", "Chemical": "chemical_status_year"}).reset_index()
    status_summary = status_summary.merge(status_years, on="Water Body ID", how="outer")
    objectives = _river_chunks(raw_root / "catchment_objectives/objectives.csv")
    rnags = _river_chunks(raw_root / "catchment_rnags/rnags.csv")
    protected = pd.read_csv(raw_root / "catchment_protected_areas/protected_areas.csv", low_memory=False)
    pressure_summary = _reason_summary(rnags)
    paths = {}
    for name, table in {"classifications": classifications, "latest_classifications": latest, "status_summary": status_summary, "objectives": objectives, "rnags": rnags, "pressure_summary": pressure_summary, "protected_areas": protected}.items():
        path = output_dir / f"wfd_{name}.parquet"
        table.to_parquet(path, index=False)
        paths[name] = path
    return paths
