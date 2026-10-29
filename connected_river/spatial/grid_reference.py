"""British National Grid reference parsing for open monitoring assets."""

from __future__ import annotations

import re

import pandas as pd


# Return missing coordinates for incomplete references instead of inventing
# precision by padding a location beyond what the source supplied.
def ngr_to_easting_northing(value: object) -> tuple[float | None, float | None]:
    if value is None or pd.isna(value):
        return None, None
    text = re.sub(r"\s+", "", str(value)).upper()
    match = re.fullmatch(r"([A-HJ-Z]{2})(\d{2,10})", text)
    if not match or len(match.group(2)) % 2:
        return None, None
    letters, digits = match.groups()
    alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    l1, l2 = alphabet.index(letters[0]), alphabet.index(letters[1])
    e100km = ((l1 - 2) % 5) * 5 + (l2 % 5)
    n100km = 19 - (l1 // 5) * 5 - l2 // 5
    if not (0 <= e100km <= 6 and 0 <= n100km <= 12):
        return None, None
    half = len(digits) // 2
    east = int(digits[:half].ljust(5, "0"))
    north = int(digits[half:].ljust(5, "0"))
    return float(e100km * 100000 + east), float(n100km * 100000 + north)
