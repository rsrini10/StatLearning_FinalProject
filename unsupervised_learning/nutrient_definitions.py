"""
Column rules aligned with R/nutrient_definitions.R.

Micronutrient-only sets exclude macronutrients, total SFA/MUFA/PUFA, and
individual fatty-acid chain columns (prefixes MUFA / PUFA / SFA).
"""

from __future__ import annotations

import re
from typing import Iterable

# Exact names: macronutrients / direct energy carriers (same list as R macro_exact).
MACRO_EXACT: frozenset[str] = frozenset(
    {
        "Total lipid (fat)",
        "Total Sugars",
        "Carbohydrate, by difference",
        "Protein",
        "Water",
        "Fiber, total dietary",
        "Alcohol, ethyl",
        "Cholesterol",
        "Fatty acids, total saturated",
        "Fatty acids, total monounsaturated",
        "Fatty acids, total polyunsaturated",
    }
)

_FATTY_DETAIL = re.compile(r"^(MUFA |PUFA |SFA )")


def is_macro_or_fatty_acid_detail(colnm: str) -> bool:
    if colnm in MACRO_EXACT:
        return True
    return bool(_FATTY_DETAIL.match(colnm))


def micronutrient_column_names(
    column_names: Iterable[str],
    *,
    y_col: str | None = "Energy",
    id_cols: frozenset[str] | None = None,
) -> list[str]:
    """
    Names kept for micronutrient-only analyses (matches R micronutrient_predictor_names).

    id_cols default: empty string and Food_Name. If y_col is None, no response column is dropped.

    Columns named like ``Unnamed: 0`` (pandas default when CSV has a blank header) are skipped,
    matching R's leading ``""`` column from ``read.csv(..., check.names = FALSE)``.
    """
    if id_cols is None:
        id_cols = frozenset({"", "Food_Name"})
    drop_ids = set(id_cols)
    if y_col is not None:
        drop_ids.add(y_col)
    out: list[str] = []
    for nm in column_names:
        if nm in drop_ids:
            continue
        # pandas: first column "" becomes "Unnamed: 0"; R keeps "" in id_cols
        if str(nm).startswith("Unnamed:"):
            continue
        if is_macro_or_fatty_acid_detail(nm):
            continue
        out.append(nm)
    return out
