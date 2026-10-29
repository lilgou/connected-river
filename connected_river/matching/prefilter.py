"""Reusable row filters for matching inputs and candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd

FilterOperator = Literal[
    "eq", "ne", "gt", "ge", "lt", "le", "in", "not_in", "is_null", "not_null"
]
FILTER_OPERATORS = frozenset(
    {"eq", "ne", "gt", "ge", "lt", "le", "in", "not_in", "is_null", "not_null"}
)


@dataclass(frozen=True)
class TableFilter:
    """One declarative filter without arbitrary code execution."""

    column: str
    operator: FilterOperator
    value: object = None

    def expression(self) -> str:
        if self.operator in {"is_null", "not_null"}:
            return f"{self.column}:{self.operator}"
        return f"{self.column}:{self.operator}:{json.dumps(self.value, sort_keys=True)}"

    def as_dict(self) -> dict[str, object]:
        return {
            "column": self.column,
            "operator": self.operator,
            "value": self.value,
            "expression": self.expression(),
        }


def parse_filter(expression: str) -> TableFilter:
    """Parse ``COLUMN:OPERATOR:JSON_VALUE`` into a validated filter.

    JSON values make booleans, numbers and lists unambiguous. Unquoted text is
    retained as a string for convenient commands such as
    ``network_direction:eq:wq_upstream_of_bio``.
    """

    parts = str(expression).split(":", 2)
    if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(
            "Filters must use COLUMN:OPERATOR:VALUE, for example "
            "`has_metals:eq:true`"
        )
    column = parts[0].strip()
    operator = parts[1].strip().lower()
    if operator not in FILTER_OPERATORS:
        allowed = ", ".join(sorted(FILTER_OPERATORS))
        raise ValueError(f"Unsupported filter operator {operator!r}; choose from: {allowed}")
    if operator in {"is_null", "not_null"}:
        if len(parts) == 3 and parts[2].strip():
            raise ValueError(f"Filter operator {operator!r} does not accept a value")
        return TableFilter(column=column, operator=operator)
    if len(parts) != 3 or not parts[2].strip():
        raise ValueError(f"Filter operator {operator!r} requires a value")
    raw_value = parts[2].strip()
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    if operator in {"in", "not_in"} and not isinstance(value, list):
        raise ValueError(f"Filter operator {operator!r} requires a JSON list value")
    return TableFilter(column=column, operator=operator, value=value)


def _comparison_mask(series: pd.Series, rule: TableFilter) -> pd.Series:
    operator = rule.operator
    if operator == "is_null":
        return series.isna()
    if operator == "not_null":
        return series.notna()
    if operator == "in":
        return series.isin(rule.value)
    if operator == "not_in":
        return series.notna() & ~series.isin(rule.value)
    if operator in {"gt", "ge", "lt", "le"}:
        if isinstance(rule.value, bool) or not isinstance(rule.value, (int, float)):
            raise ValueError(f"Filter operator {operator!r} requires a numeric value")
        if not math.isfinite(float(rule.value)):
            raise ValueError("Numeric filter values must be finite")
        numeric = pd.to_numeric(series, errors="coerce")
        comparisons = {
            "gt": numeric.gt,
            "ge": numeric.ge,
            "lt": numeric.lt,
            "le": numeric.le,
        }
        return comparisons[operator](float(rule.value)).fillna(False)
    if operator == "eq":
        return series.eq(rule.value).fillna(False)
    if operator == "ne":
        return series.notna() & series.ne(rule.value).fillna(False)
    raise AssertionError(f"Unhandled filter operator: {operator}")


# Filters are applied before ranking, so they change the available candidate
# pool rather than merely changing the order of otherwise identical matches.
def apply_table_filters(
    frame: pd.DataFrame,
    filters: Iterable[str | TableFilter] | None,
    *,
    label: str = "table",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the AND filters and return the retained rows plus a summary."""

    parsed = [parse_filter(item) if isinstance(item, str) else item for item in filters or []]
    rows_before = len(frame)
    retained = frame.copy()
    steps: list[dict[str, object]] = []
    for rule in parsed:
        if rule.column not in retained:
            raise KeyError(f"{label} filter refers to missing column {rule.column!r}")
        step_before = len(retained)
        mask = _comparison_mask(retained[rule.column], rule)
        retained = retained.loc[mask].copy()
        steps.append(
            {
                **rule.as_dict(),
                "rows_before": step_before,
                "rows_after": len(retained),
                "rows_excluded": step_before - len(retained),
            }
        )
    rows_after = len(retained)
    audit = {
        "label": label,
        "logic": "AND",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_excluded": rows_before - rows_after,
        "retained_fraction": rows_after / rows_before if rows_before else 1.0,
        "filters": [rule.as_dict() for rule in parsed],
        "steps": steps,
    }
    return retained, audit
