"""Local-transform math for bone moves: Unity <-> FBX frames.

A bone move in RigForge is a *rest-pose* edit: the Model node's
`Lcl Translation` / `Lcl Rotation` / `Lcl Scaling` change while the skin
clusters' bind pose stays put, so every vertex weighted to that bone follows
it. That is exactly what dragging the bone in Unity's editor does, and Blender
shows the same result (it evaluates the mesh against the bind pose too).

The modder types the numbers Unity's Inspector shows. Unity imports FBX by
mirroring the X axis (right-handed -> left-handed, Y up, Z forward kept), so:

    position   (x, y, z)   ->  (-x, y, z)
    rotation   R_unity     ->  M . R_unity . M         with M = diag(-1, 1, 1)
    scale      unchanged

and Euler conventions differ: Unity applies Z, then X, then Y (R = Ry.Rx.Rz);
FBX's default `eEulerXYZ` applies X, then Y, then Z (R = Rz.Ry.Rx). Both go
through a rotation matrix so composite rotations convert exactly.

Everything here is pure Python (no numpy) and frame-explicit. Nothing writes
FBX bytes — that is `edits.set_model_local_transform_edits`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

Vec3 = tuple[float, float, float]
Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

FRAMES = ("unity", "fbx")

_EPS = 1e-9


@dataclass(frozen=True)
class LocalTransform:
    """A (partial) local transform. `None` means "leave that component
    alone". Rotation is Euler degrees in the frame's own convention."""
    position: Optional[Vec3] = None
    rotation: Optional[Vec3] = None
    scale: Optional[Vec3] = None

    def is_empty(self) -> bool:
        return self.position is None and self.rotation is None and self.scale is None

    def to_dict(self) -> dict:
        return {
            "position": list(self.position) if self.position is not None else None,
            "rotation": list(self.rotation) if self.rotation is not None else None,
            "scale": list(self.scale) if self.scale is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LocalTransform":
        return cls(
            position=_vec3_or_none(d.get("position")),
            rotation=_vec3_or_none(d.get("rotation")),
            scale=_vec3_or_none(d.get("scale")),
        )


def _vec3_or_none(v) -> Optional[Vec3]:
    if v is None:
        return None
    if len(v) != 3:
        raise ValueError(f"expected 3 components, got {len(v)}: {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


# --- elementary rotations (column-vector convention) ---------------------------


def _rot_x(deg: float) -> Mat3:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def _rot_y(deg: float) -> Mat3:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _rot_z(deg: float) -> Mat3:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _clamp1(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _norm_deg(d: float) -> float:
    """Wrap into [-180, 180)."""
    return ((d + 180.0) % 360.0) - 180.0


# --- FBX eEulerXYZ --------------------------------------------------------------


def fbx_euler_to_matrix(rx: float, ry: float, rz: float) -> Mat3:
    """FBX default rotation order: X applied first, then Y, then Z, i.e.
    R = Rz . Ry . Rx (same definition as sections._local_matrix)."""
    return _mul(_rot_z(rz), _mul(_rot_y(ry), _rot_x(rx)))


def matrix_to_fbx_euler(m: Mat3) -> Vec3:
    """Inverse of `fbx_euler_to_matrix`. At gimbal lock (|ry| = 90) the X
    angle is fixed to 0 and Z absorbs the remaining rotation."""
    sy = _clamp1(-m[2][0])
    ry = math.asin(sy)
    if abs(math.cos(ry)) > _EPS:
        rx = math.atan2(m[2][1], m[2][2])
        rz = math.atan2(m[1][0], m[0][0])
    else:
        rx = 0.0
        rz = math.atan2(-m[0][1], m[1][1])
    return (_norm_deg(math.degrees(rx)), _norm_deg(math.degrees(ry)), _norm_deg(math.degrees(rz)))


# --- Unity ZXY -------------------------------------------------------------------


def unity_euler_to_matrix(x: float, y: float, z: float) -> Mat3:
    """Unity's Quaternion.Euler(x, y, z): rotate about Z, then X, then Y, i.e.
    R = Ry . Rx . Rz."""
    return _mul(_rot_y(y), _mul(_rot_x(x), _rot_z(z)))


def matrix_to_unity_euler(m: Mat3) -> Vec3:
    """Inverse of `unity_euler_to_matrix`. At gimbal lock (|x| = 90) the Z
    angle is fixed to 0 and Y absorbs the remaining rotation."""
    sx = _clamp1(-m[1][2])
    x = math.asin(sx)
    if abs(math.cos(x)) > _EPS:
        y = math.atan2(m[0][2], m[2][2])
        z = math.atan2(m[1][0], m[1][1])
    else:
        z = 0.0
        y = math.atan2(-m[2][0], m[0][0])
    return (_norm_deg(math.degrees(x)), _norm_deg(math.degrees(y)), _norm_deg(math.degrees(z)))


# Unity works in float32: an X within ~0.02 deg of +-90 is gimbal-locked there
# (a Blender export's -90.0000093 shows as -89.98). This catches those, and
# exact +-90, without touching any angle a modder would type.
_INSPECTOR_GIMBAL_COS = 1e-6


def _inspector_deg(d: float) -> float:
    """Wrap into [-180, 180) the way the Inspector shows it (180 reads -180),
    and drop float dust so the FE never shows -0."""
    n = _norm_deg(d)
    if n > 180.0 - 1e-6:
        n -= 360.0
    return 0.0 if abs(n) < 1e-9 else n


def matrix_to_unity_inspector_euler(m: Mat3) -> Vec3:
    """The Euler triple Unity's Inspector shows for rotation `m` on a freshly
    imported node (its Euler hint is (0, 0, 0)) — not `localEulerAngles`.

    Every rotation has two Z-X-Y Euler triples, (x, y, z) and
    (180 - x, y + 180, z + 180); the Inspector shows the one closest to the
    hint, so FBX X = -103.26 reads (-103.26, 0, 0), not (-76.74, 180, 180).
    At gimbal lock it keeps Y at the hint and puts the twist into Z. Pinned
    against Unity 2022.3's importer in tests/test_transform.py.
    """
    x = math.asin(_clamp1(-m[1][2]))
    if abs(math.cos(x)) <= _INSPECTOR_GIMBAL_COS:
        z = math.atan2(-m[0][1], m[0][0])
        return (_inspector_deg(math.degrees(x)), 0.0, _inspector_deg(math.degrees(z)))
    x = math.degrees(x)
    y = math.degrees(math.atan2(m[0][2], m[2][2]))
    z = math.degrees(math.atan2(m[1][0], m[1][1]))
    candidates = [
        tuple(_inspector_deg(a) for a in t)
        for t in ((x, y, z), (180.0 - x, y + 180.0, z + 180.0))
    ]
    return min(candidates, key=lambda t: sum(a * a for a in t))  # type: ignore[return-value]


# --- mirroring -------------------------------------------------------------------


def mirror_x(m: Mat3) -> Mat3:
    """Conjugate a rotation by the X-axis reflection: M . R . M with
    M = diag(-1, 1, 1). Flips the sign of every entry that mixes X with Y/Z."""
    return tuple(  # type: ignore[return-value]
        tuple((-m[i][j] if (i == 0) != (j == 0) else m[i][j]) for j in range(3))
        for i in range(3)
    )


# --- affine 4x4 (attach a rigid garment to a bone) -------------------------------

Mat4 = tuple


def affine_inverse(m: Mat4) -> Mat4:
    """Inverse of a 4x4 affine matrix (last row 0,0,0,1)."""
    a = [[m[i][j] for j in range(3)] for i in range(3)]
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
           - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    if abs(det) < 1e-12:
        raise ValueError("matrix is singular (zero scale on some axis)")
    inv = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            r = [x for x in range(3) if x != j]
            c = [x for x in range(3) if x != i]
            minor = a[r[0]][c[0]] * a[r[1]][c[1]] - a[r[0]][c[1]] * a[r[1]][c[0]]
            inv[i][j] = (-1) ** (i + j) * minor / det
    t = [-sum(inv[i][k] * m[k][3] for k in range(3)) for i in range(3)]
    return tuple(  # type: ignore[return-value]
        [tuple(inv[i]) + (t[i],) for i in range(3)] + [(0.0, 0.0, 0.0, 1.0)]
    )


def affine_to_fbx_local(m: Mat4) -> LocalTransform:
    """Split a 4x4 affine matrix into FBX Lcl T / R (eEulerXYZ degrees) / S —
    the inverse of sections._local_matrix (M = T . R . S). Raises ValueError
    on shear, which T/R/S cannot express."""
    cols = [[m[i][j] for i in range(3)] for j in range(3)]
    scale = [math.sqrt(sum(v * v for v in c)) for c in cols]
    if min(scale) < 1e-12:
        raise ValueError("matrix has a zero-scale axis")
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
           - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if det < 0:  # a mirror: carry it on X
        scale[0] = -scale[0]
    r = tuple(tuple(m[i][j] / scale[j] for j in range(3)) for i in range(3))
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-6:
                raise ValueError("matrix has shear; it is not expressible as T . R . S")
    return LocalTransform(
        position=(m[0][3], m[1][3], m[2][3]),
        rotation=matrix_to_fbx_euler(r),  # type: ignore[arg-type]
        scale=(scale[0], scale[1], scale[2]),
    )


# --- frame conversion -------------------------------------------------------------


def unity_to_fbx(t: LocalTransform) -> LocalTransform:
    """Unity Inspector values (parent-local) -> FBX Lcl properties."""
    pos = (-t.position[0], t.position[1], t.position[2]) if t.position is not None else None
    rot = (
        matrix_to_fbx_euler(mirror_x(unity_euler_to_matrix(*t.rotation)))
        if t.rotation is not None else None
    )
    return LocalTransform(position=pos, rotation=rot, scale=t.scale)


def fbx_to_unity(t: LocalTransform) -> LocalTransform:
    """FBX Lcl properties -> the values Unity's Inspector will show."""
    pos = (-t.position[0], t.position[1], t.position[2]) if t.position is not None else None
    rot = (
        matrix_to_unity_inspector_euler(mirror_x(fbx_euler_to_matrix(*t.rotation)))
        if t.rotation is not None else None
    )
    return LocalTransform(position=pos, rotation=rot, scale=t.scale)


def convert_local(t: LocalTransform, frame: str) -> LocalTransform:
    """Bring a transform expressed in `frame` ("unity" | "fbx") into the FBX
    frame the edit layer writes."""
    if frame == "fbx":
        return t
    if frame == "unity":
        return unity_to_fbx(t)
    raise ValueError(f"unknown transform frame {frame!r}; expected one of {FRAMES}")
