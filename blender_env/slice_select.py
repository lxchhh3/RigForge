"""Pure bbox-slice vertex selection — the viewport-free region picker.

Given each vertex's scalar coordinate along one axis (e.g. world Z) and a
normalized slice range of that axis' bounding box, return the indices inside
the slice. "Bottom 15%" = (0.0, 0.15); "top ring" = (0.85, 1.0).

bpy-free on purpose: unit-tested in the RigForge env, imported verbatim by the
Blender script (`geometry_edit.py`) so the selection math is identical in both
places.
"""
from __future__ import annotations

from typing import Sequence


def select_slice_indices(
    coords: Sequence[float], from_pct: float, to_pct: float
) -> set[int]:
    """Indices of vertices whose normalized position along the axis lies within
    [from_pct, to_pct] (inclusive).

    Normalization maps the axis bounding box [min, max] onto [0, 1]. The range
    is order-insensitive: (0.15, 0.0) behaves like (0.0, 0.15). A degenerate
    axis (all coords equal) maps every vertex to position 0.0, so a range that
    includes 0 selects everything and one that doesn't selects nothing.
    """
    if not coords:
        return set()

    lo, hi = (from_pct, to_pct) if from_pct <= to_pct else (to_pct, from_pct)

    cmin = min(coords)
    cmax = max(coords)
    span = cmax - cmin

    selected: set[int] = set()
    for i, c in enumerate(coords):
        pos = 0.0 if span == 0 else (c - cmin) / span
        if lo <= pos <= hi:
            selected.add(i)
    return selected
