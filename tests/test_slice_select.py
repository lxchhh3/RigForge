"""Tests for the pure bbox-slice vertex selector.

This is the viewport-free region selector: given a list of per-vertex scalar
coordinates along one axis and a normalized slice range of that axis' bounding
box, return the indices inside the slice. No bpy — unit-tested in the RigForge
env and imported verbatim by the Blender script.
"""
from __future__ import annotations

from blender_env.slice_select import select_slice_indices


def test_bottom_slice():
    coords = [float(i) for i in range(11)]  # 0..10 -> normalized 0.0..1.0
    sel = select_slice_indices(coords, 0.0, 0.15)
    assert sel == {0, 1}  # 0.0 and 0.1 are <= 0.15


def test_top_slice():
    coords = [float(i) for i in range(11)]
    sel = select_slice_indices(coords, 0.85, 1.0)
    assert sel == {9, 10}  # 0.9 and 1.0


def test_middle_band_excludes_ends():
    coords = [float(i) for i in range(11)]
    sel = select_slice_indices(coords, 0.4, 0.6)
    assert sel == {4, 5, 6}


def test_bounds_are_inclusive():
    coords = [0.0, 5.0, 10.0]  # normalized 0.0, 0.5, 1.0
    assert select_slice_indices(coords, 0.0, 0.5) == {0, 1}
    assert select_slice_indices(coords, 0.5, 1.0) == {1, 2}


def test_unordered_coords_use_index_positions():
    coords = [10.0, 0.0, 5.0]  # max=10@0, min=0@1, mid=5@2
    sel = select_slice_indices(coords, 0.0, 0.15)
    assert sel == {1}  # only the global-min vertex


def test_degenerate_all_equal_includes_when_zero_in_range():
    coords = [5.0, 5.0, 5.0]
    # All collapse to normalized 0.0; a range touching 0 selects everything.
    assert select_slice_indices(coords, 0.0, 0.15) == {0, 1, 2}
    # A range not touching 0 selects nothing.
    assert select_slice_indices(coords, 0.5, 0.6) == set()


def test_empty_coords():
    assert select_slice_indices([], 0.0, 0.15) == set()


def test_swapped_range_is_normalized():
    """A caller passing (to, from) shouldn't silently select nothing."""
    coords = [float(i) for i in range(11)]
    assert select_slice_indices(coords, 0.15, 0.0) == {0, 1}
