"""Tests for the Unity <-> FBX local-transform math (bone-move feature).

The FE shows the modder the same numbers Unity's Inspector shows for a bone
(position / rotation / scale in the parent's frame). Unity imports FBX by
mirroring the X axis, so a Unity-side value has to be converted before it is
written into the FBX's `Lcl Translation` / `Lcl Rotation` / `Lcl Scaling`:

    position  (x, y, z)  ->  (-x, y, z)
    rotation  R_unity    ->  M . R_unity . M      (M = diag(-1, 1, 1))
    scale     unchanged

Rotation is where the conventions bite: Unity's Euler order is Z-X-Y
(applied z, then x, then y), FBX's default `eEulerXYZ` is X-Y-Z. Both are
converted through a rotation matrix so the composite case is exact, not just
the single-axis one.
"""
from __future__ import annotations

import math
import random

import pytest

from rigforge.ascii_fbx.sections import _local_matrix
from rigforge.ascii_fbx.transform import (
    LocalTransform,
    convert_local,
    fbx_euler_to_matrix,
    fbx_to_unity,
    matrix_to_fbx_euler,
    matrix_to_unity_euler,
    mirror_x,
    unity_euler_to_matrix,
    unity_to_fbx,
)


def _close_mat(a, b, tol=1e-9) -> bool:
    return all(abs(a[i][j] - b[i][j]) <= tol for i in range(3) for j in range(3))


def _close_vec(a, b, tol=1e-9) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _rot_x(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def _rot_y(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _rot_z(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


# --- FBX Euler XYZ <-> matrix --------------------------------------------------


def test_fbx_euler_matches_sections_local_matrix():
    """The rotation block must agree with sections._local_matrix (the matrix
    the world_xyz fallback walk already trusts), so both codepaths share one
    definition of "FBX Euler XYZ"."""
    rng = random.Random(1)
    for _ in range(50):
        angles = tuple(rng.uniform(-180, 180) for _ in range(3))
        m = fbx_euler_to_matrix(*angles)
        ref = _local_matrix((0.0, 0.0, 0.0), angles, (1.0, 1.0, 1.0))
        ref3 = tuple(tuple(ref[i][j] for j in range(3)) for i in range(3))
        assert _close_mat(m, ref3), angles


def test_fbx_euler_round_trip_preserves_rotation():
    rng = random.Random(2)
    for _ in range(100):
        angles = tuple(rng.uniform(-180, 180) for _ in range(3))
        m = fbx_euler_to_matrix(*angles)
        back = matrix_to_fbx_euler(m)
        assert _close_mat(fbx_euler_to_matrix(*back), m, tol=1e-8), (angles, back)


def test_fbx_euler_round_trip_returns_same_angles_in_principal_range():
    """For pitch inside (-90, 90) the decomposition is unique, so the exact
    input angles come back (not just an equivalent rotation)."""
    rng = random.Random(3)
    for _ in range(100):
        angles = (rng.uniform(-179, 179), rng.uniform(-89, 89), rng.uniform(-179, 179))
        back = matrix_to_fbx_euler(fbx_euler_to_matrix(*angles))
        assert _close_vec(back, angles, tol=1e-7), (angles, back)


def test_fbx_euler_gimbal_lock_still_round_trips():
    for ry in (90.0, -90.0):
        angles = (35.0, ry, -20.0)
        m = fbx_euler_to_matrix(*angles)
        back = matrix_to_fbx_euler(m)
        assert _close_mat(fbx_euler_to_matrix(*back), m, tol=1e-8), back


# --- Unity Euler ZXY <-> matrix ------------------------------------------------


def test_unity_euler_single_axes_match_standard_rotations():
    """Unity's Euler(x, y, z) applies z, then x, then y. Single-axis inputs
    must equal the plain rotation matrices about that axis."""
    assert _close_mat(unity_euler_to_matrix(30, 0, 0), _rot_x(30))
    assert _close_mat(unity_euler_to_matrix(0, 30, 0), _rot_y(30))
    assert _close_mat(unity_euler_to_matrix(0, 0, 30), _rot_z(30))


def test_unity_euler_order_is_z_then_x_then_y():
    """Composite check against explicit matrix products: R = Ry . Rx . Rz."""
    expected = _mul(_rot_y(40), _mul(_rot_x(30), _rot_z(50)))
    assert _close_mat(unity_euler_to_matrix(30, 40, 50), expected)


def test_unity_euler_round_trip_preserves_rotation():
    rng = random.Random(4)
    for _ in range(100):
        angles = tuple(rng.uniform(-180, 180) for _ in range(3))
        m = unity_euler_to_matrix(*angles)
        back = matrix_to_unity_euler(m)
        assert _close_mat(unity_euler_to_matrix(*back), m, tol=1e-8), (angles, back)


def test_unity_euler_round_trip_returns_same_angles_in_principal_range():
    rng = random.Random(5)
    for _ in range(100):
        angles = (rng.uniform(-89, 89), rng.uniform(-179, 179), rng.uniform(-179, 179))
        back = matrix_to_unity_euler(unity_euler_to_matrix(*angles))
        assert _close_vec(back, angles, tol=1e-7), (angles, back)


def test_unity_euler_gimbal_lock_still_round_trips():
    for rx in (90.0, -90.0):
        angles = (rx, 25.0, -70.0)
        m = unity_euler_to_matrix(*angles)
        back = matrix_to_unity_euler(m)
        assert _close_mat(unity_euler_to_matrix(*back), m, tol=1e-8), back


# --- mirroring -----------------------------------------------------------------


def test_mirror_x_is_an_involution():
    m = fbx_euler_to_matrix(12, -34, 56)
    assert _close_mat(mirror_x(mirror_x(m)), m)


def test_mirror_x_negates_cross_terms_only():
    m = fbx_euler_to_matrix(12, -34, 56)
    mm = mirror_x(m)
    for i in range(3):
        for j in range(3):
            sign = -1 if (i == 0) != (j == 0) else 1
            assert abs(mm[i][j] - sign * m[i][j]) < 1e-12


# --- Unity <-> FBX local transforms ---------------------------------------------


def test_unity_to_fbx_position_negates_x():
    t = unity_to_fbx(LocalTransform(position=(0.1, 0.2, 0.3)))
    assert _close_vec(t.position, (-0.1, 0.2, 0.3))
    assert t.rotation is None and t.scale is None


def test_unity_to_fbx_single_axis_rotations():
    """X rotations keep their sign; Y and Z rotations flip (mirroring across
    the YZ plane reverses the sense of rotation about axes in that plane)."""
    assert _close_vec(unity_to_fbx(LocalTransform(rotation=(90, 0, 0))).rotation, (90, 0, 0))
    assert _close_vec(unity_to_fbx(LocalTransform(rotation=(0, 90, 0))).rotation, (0, -90, 0))
    assert _close_vec(unity_to_fbx(LocalTransform(rotation=(0, 0, 90))).rotation, (0, 0, -90))


def test_unity_to_fbx_composite_rotation_is_the_mirrored_matrix():
    """Not just single axes: a composite Unity rotation converts to whichever
    FBX Euler triple reproduces M . R_unity . M exactly."""
    rng = random.Random(6)
    for _ in range(50):
        angles = tuple(rng.uniform(-180, 180) for _ in range(3))
        t = unity_to_fbx(LocalTransform(rotation=angles))
        expected = mirror_x(unity_euler_to_matrix(*angles))
        assert _close_mat(fbx_euler_to_matrix(*t.rotation), expected, tol=1e-8), angles


def test_unity_to_fbx_scale_passes_through():
    t = unity_to_fbx(LocalTransform(scale=(1.5, 2.0, 0.5)))
    assert t.scale == (1.5, 2.0, 0.5)


def test_fbx_to_unity_is_inverse_of_unity_to_fbx():
    rng = random.Random(7)
    for _ in range(50):
        t = LocalTransform(
            position=tuple(rng.uniform(-1, 1) for _ in range(3)),
            rotation=(rng.uniform(-89, 89), rng.uniform(-179, 179), rng.uniform(-179, 179)),
            scale=tuple(rng.uniform(0.5, 2) for _ in range(3)),
        )
        back = fbx_to_unity(unity_to_fbx(t))
        assert _close_vec(back.position, t.position, tol=1e-9)
        assert _close_mat(
            unity_euler_to_matrix(*back.rotation), unity_euler_to_matrix(*t.rotation), tol=1e-8,
        )
        assert _close_vec(back.scale, t.scale, tol=1e-9)


def test_fbx_to_unity_on_real_hat_root_values():
    """GROUND TRUTH from Unity 2022.3's own importer (batch-mode probe,
    2026-09-08, retargeted_20260907_001630_SF_Manuka.fbx): the hat bone's FBX
    Lcl Rotation (5.19, -0.09, 29.93) imports as localEulerAngles
    (4.54, -2.52, -30.03), quaternion (0.04396, -0.01097, -0.25798, 0.96509),
    localPosition (0.0372, 0.0787, 0.0527). A composite rotation, so this pins
    the Euler ORDER (FBX X-then-Y-then-Z) as well as the X mirror."""
    fbx = LocalTransform(
        position=(-0.0371849983930588, 0.0787041112780571, 0.0526540018618107),
        rotation=(5.19242976224099, -0.086554311606837, 29.9282018862709),
        scale=(0.999999940395355, 0.999999940395355, 0.999999940395355),
    )
    u = fbx_to_unity(fbx)
    assert _close_vec(u.position, (0.0371849983930588, 0.0787041112780571, 0.0526540018618107))
    assert _close_vec(u.rotation, (4.54, -2.52, -30.03), tol=0.006)
    back = unity_to_fbx(u)
    assert _close_mat(
        fbx_euler_to_matrix(*back.rotation), fbx_euler_to_matrix(*fbx.rotation), tol=1e-8,
    )


def test_fbx_to_unity_matches_unity_importer_on_second_bone():
    """Second Unity ground-truth point from the same probe: E#5_Hairpin's FBX
    Lcl Rotation (-8.35, -0.09, 4.74) imports as (-8.31, 0.78, -4.79)."""
    u = fbx_to_unity(LocalTransform(rotation=(-8.35, -0.09, 4.74)))
    assert _close_vec(u.rotation, (-8.31, 0.78, -4.79), tol=0.02)


def test_unity_to_fbx_matches_what_unity_read_back_from_the_bake():
    """Round trip through the real bake: the modder's Unity values for two
    moved bones were written to FBX and Unity's importer read them back
    bit-for-bit (probe 2026-09-08). Pin the FBX-side numbers it produced."""
    hairpin = unity_to_fbx(LocalTransform(position=(0.0117, 0.1162, 0.0329), rotation=(0.90, 11.66, 0.0)))
    assert _close_vec(hairpin.position, (-0.0117, 0.1162, 0.0329))
    assert _close_vec(hairpin.rotation, (0.90, -11.66, 0.0), tol=1e-6)
    headacc = unity_to_fbx(LocalTransform(position=(0.0313, 0.1072, -0.0175), rotation=(1.15, -0.12, -1.53)))
    assert _close_vec(headacc.position, (-0.0313, 0.1072, -0.0175))
    assert _close_vec(headacc.rotation, (1.1528, 0.0893, 1.5297), tol=1e-3)


# GROUND TRUTH from Unity 2022.3.22f1's own importer (batch-mode probe,
# 2026-10-14): GG_headset.fbx re-written with each FBX Lcl Rotation on the
# left, imported, and read back with TransformUtils.GetInspectorRotation —
# the numbers the modder sees in the Inspector. Not localEulerAngles: for a
# fresh import (Euler hint 0,0,0) the Inspector shows, among the equivalent
# Euler triples, the one closest to (0, 0, 0); at X = +-90 it keeps Y = 0 and
# puts the twist into Z (R15). Unity computes in float32, so X = -90.0000093
# shows as -89.98 (R01, R19) — hence the 0.025 deg tolerance.
UNITY_INSPECTOR_GROUND_TRUTH = [
    ((-90.0000093346673, 0.0, 0.0), (-89.98, 0.0, 0.0)),  # R01 GG_headset as Blender exported it
    ((-90.0, 0.0, 0.0), (-90.0, 0.0, 0.0)),  # R02
    ((90.0, 0.0, 0.0), (90.0, 0.0, 0.0)),  # R03
    ((-76.74, 0.0, 0.0), (-76.74, 0.0, 0.0)),  # R04
    ((-103.26, 0.0, 0.0), (-103.26, 0.0, 0.0)),  # R05
    ((100.0, 0.0, 0.0), (100.0, 0.0, 0.0)),  # R06
    ((0.0, 180.0, 0.0), (0.0, -180.0, 0.0)),  # R07
    ((0.0, 170.0, 0.0), (0.0, -170.0, 0.0)),  # R08
    ((180.0, 0.0, 0.0), (-180.0, 0.0, 0.0)),  # R09
    ((30.0, 120.0, -60.0), (115.905, -7.63098145, 97.631)),  # R10
    ((-150.0, 45.0, 170.0), (36.78, -139.868, -8.818)),  # R11
    ((45.0, 90.0, 0.0), (45.0, -90.0, 0.0)),  # R12
    ((-75.3524, -34.707, -2.7771), (-76.74, 25.0, 10.0)),  # R13
    ((10.0, 20.0, 30.0), (-1.033, -22.246, -28.029)),  # R14
    ((-90.0, 30.0, 0.0), (-90.0, 0.0, -30.0)),  # R15
    ((0.0, 0.0, 135.0), (0.0, 0.0, -135.0)),  # R16
    ((-170.0, -10.0, 95.0), (-8.93, 169.036, -83.264)),  # R17
    ((120.0, -40.0, -160.0), (-44.739, 122.631, 21.645)),  # R18
    ((-90.0000093346673, 0.0, 180.0), (89.98, 0.0, -180.0)),  # R19
    ((89.9, 15.0, -15.0), (75.026, 89.626, 104.639)),  # R20
    ((-141.542, 72.931, 54.735), (165.397, -76.26001, 14.3390045)),  # R21
    ((158.527, -82.399, -87.921), (69.242, 110.323, 21.899)),  # R22
    ((84.261, 57.042, -70.924), (23.859, 86.59, 145.793)),  # R23
    ((66.324, -37.197, 99.906), (4.68, -71.28, -128.065)),  # R24
    ((-137.376, -99.624, 144.527), (7.498, 82.873, 5.616)),  # R25
    ((-51.108, -86.255, 109.541), (58.295, 85.525, -6.726)),  # R26
    ((47.399, -126.107, 18.48), (60.63, 144.421, 22.387)),  # R27
    ((59.04, -120.619, 54.611), (120.959, 59.3799973, -53.8179932)),  # R28
    ((-135.711, -58.656, -150.045), (65.542, 154.072, 38.854)),  # R29
    ((-106.408, 172.08, -34.677), (-54.196, -61.429, -105.577)),  # R30
]


@pytest.mark.parametrize("fbx_rot, inspector", UNITY_INSPECTOR_GROUND_TRUTH)
def test_fbx_to_unity_shows_the_numbers_unitys_inspector_shows(fbx_rot, inspector):
    u = fbx_to_unity(LocalTransform(rotation=fbx_rot))
    assert _close_vec(u.rotation, inspector, tol=0.025), (fbx_rot, u.rotation, inspector)
    # Same rotation, whichever triple is shown. 2e-6, not 1e-8: an X within
    # ~6e-5 deg of +-90 is shown as exact gimbal lock, as Unity does (R01, R19);
    # the FE rounds to 1e-4 deg anyway.
    assert _close_mat(
        fbx_euler_to_matrix(*unity_to_fbx(u).rotation), fbx_euler_to_matrix(*fbx_rot), tol=2e-6,
    )


def test_editing_only_x_of_a_blender_minus_90_mesh_tilts_the_way_unity_does():
    """kevin's GG headset (2026-10-14): the panel opened it as (-90,-180,-180),
    so typing X = -76.74 wrote -103.26 — 13.26 deg the wrong way. Opened as
    Unity shows it, the same edit writes -76.74, as it would in Unity."""
    shown = fbx_to_unity(LocalTransform(rotation=(-90.0000093346673, 0.0, 0.0))).rotation
    edited = (-76.74, shown[1], shown[2])
    assert _close_vec(unity_to_fbx(LocalTransform(rotation=edited)).rotation, (-76.74, 0.0, 0.0), tol=1e-6)


def test_convert_local_frames():
    t = LocalTransform(position=(1.0, 2.0, 3.0))
    assert convert_local(t, "fbx") == t
    assert _close_vec(convert_local(t, "unity").position, (-1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        convert_local(t, "blender")


def test_local_transform_is_empty():
    assert LocalTransform().is_empty()
    assert not LocalTransform(scale=(1, 1, 1)).is_empty()
