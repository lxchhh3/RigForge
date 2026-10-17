"""Subprocess wrapper around a headless Blender, mirroring `ascii_fbx.convert`.

Blender is driven as a geometry *library*: the main RigForge env never imports
`bpy`. We shell out to `blender --background --python blender_env/geometry_edit.py`
exactly the way `convert.py` shells out to the fbx_env conda Python.

Two operations in Phase 0 (neither edits geometry):
- `roundtrip_noop(src, out)` — import a binary FBX and re-export it with the
  pinned exporter settings; return the measurement report.
- `inspect(src)`            — import and measure only (no export); return the
  report. Used to re-measure a previously exported file.

Failure handling note: a Python exception inside a Blender script does NOT, by
default, set a non-zero process exit code. So we cannot trust the exit code
alone — every op writes a JSON report file, and we treat a missing/garbage
report as failure (surfacing Blender's stdout/stderr). The script also calls
`sys.exit(1)` on error as belt-and-suspenders.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Blender 4.3.0 (the team-preferred version). Both the exe and the script dir
# are overridable via env vars (same pattern as RIGFORGE_FBX_ENV_PYTHON).
_DEFAULT_BLENDER_EXE = Path(
    "D:/2files/tools/blender-4.3.0-windows-x64/blender.exe"
)
_DEFAULT_SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "blender_env"

ENV_BLENDER_EXE = "RIGFORGE_BLENDER_EXE"
ENV_SCRIPT_DIR = "RIGFORGE_BLENDER_SCRIPT_DIR"

# Headless Blender + FBX import/export is slow to boot (addon registration +
# scene build). A single clothing round-trip is comfortably under this.
_DEFAULT_TIMEOUT_S = 300


class BlenderError(RuntimeError):
    """Raised when the Blender backend fails or is misconfigured."""


@dataclass(frozen=True)
class BlenderConfig:
    blender_exe: Path
    script_dir: Path

    @classmethod
    def from_env(cls) -> "BlenderConfig":
        exe = Path(os.environ.get(ENV_BLENDER_EXE, str(_DEFAULT_BLENDER_EXE)))
        sd = Path(os.environ.get(ENV_SCRIPT_DIR, str(_DEFAULT_SCRIPT_DIR)))
        return cls(blender_exe=exe, script_dir=sd)

    def script(self, name: str) -> Path:
        return self.script_dir / name


def _resolve_config(config: Optional[BlenderConfig]) -> BlenderConfig:
    cfg = config or BlenderConfig.from_env()
    if not cfg.blender_exe.exists():
        raise BlenderError(
            f"Blender executable not found at {cfg.blender_exe}. "
            f"Set ${ENV_BLENDER_EXE} or pass BlenderConfig(blender_exe=...)."
        )
    if not cfg.script_dir.exists():
        raise BlenderError(
            f"Blender script dir not found at {cfg.script_dir}. "
            f"Set ${ENV_SCRIPT_DIR} or pass BlenderConfig(script_dir=...)."
        )
    return cfg


def _blender_cmd(config: BlenderConfig, script_name: str, script_args: list[str]) -> list[str]:
    """Build the headless Blender argv. Blender's own flags precede `--`;
    everything after `--` is passed through to the script untouched."""
    script_path = config.script(script_name)
    return [
        str(config.blender_exe),
        "--background",
        "--factory-startup",  # ignore user prefs/addons for reproducibility
        "--python",
        str(script_path),
        "--",
        *script_args,
    ]


def _run_blender(
    config: BlenderConfig,
    script_name: str,
    script_args: list[str],
    *,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess:
    script_path = config.script(script_name)
    if not script_path.exists():
        raise BlenderError(f"Blender script missing: {script_path}")

    cmd = _blender_cmd(config, script_name, script_args)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=False, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise BlenderError(
            f"{script_name} timed out after {timeout}s\n--- cmd: {cmd}"
        ) from exc

    proc.stdout = (proc.stdout or b"").decode("utf-8", errors="replace")  # type: ignore[assignment]
    proc.stderr = (proc.stderr or b"").decode("utf-8", errors="replace")  # type: ignore[assignment]
    return proc


def _read_report(report_path: Path, proc: subprocess.CompletedProcess, script_name: str) -> dict:
    """Read + parse the JSON report a Blender op writes. A missing or invalid
    report is the real failure signal (Blender exit codes are unreliable)."""
    if not report_path.exists():
        raise BlenderError(
            f"{script_name} produced no report at {report_path} "
            f"(exit={proc.returncode})\n"
            f"--- stdout:\n{proc.stdout}\n--- stderr:\n{proc.stderr}"
        )
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise BlenderError(
            f"{script_name} wrote an unreadable report at {report_path}: {exc}\n"
            f"--- stdout:\n{proc.stdout}\n--- stderr:\n{proc.stderr}"
        ) from exc


def roundtrip_noop(
    src: Path,
    out: Path,
    *,
    config: Optional[BlenderConfig] = None,
    report_path: Optional[Path] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> dict:
    """Import `src` (binary FBX) into a fresh Blender scene and re-export it to
    `out` with the pinned FBX exporter settings — no geometry change. Returns
    the measurement report (per-mesh verts/shape-keys/weights + armatures)."""
    cfg = _resolve_config(config)
    src = Path(src).resolve()
    out = Path(out).resolve()
    if not src.is_file():
        raise BlenderError(f"input FBX not found: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path).resolve() if report_path else out.with_suffix(".report.json")

    proc = _run_blender(
        cfg, "geometry_edit.py",
        ["--op", "noop", "--in", str(src), "--out", str(out), "--report", str(report_path)],
        timeout=timeout,
    )
    report = _read_report(report_path, proc, "geometry_edit.py")
    if not out.exists():
        raise BlenderError(
            f"roundtrip_noop reported success but {out} is missing\n"
            f"--- stdout:\n{proc.stdout}\n--- stderr:\n{proc.stderr}"
        )
    head = out.read_bytes()[:23]
    if not head.startswith(b"Kaydara FBX Binary"):
        raise BlenderError(f"roundtrip_noop produced a non-binary FBX at {out} (head={head!r})")
    return report


def translate_slice(
    src: Path,
    out: Path,
    *,
    mesh: str,
    axis: str = "z",
    from_pct: float = 0.0,
    to_pct: float = 0.15,
    delta_frac: float = -0.3,
    preview_dir: Optional[Path] = None,
    config: Optional[BlenderConfig] = None,
    report_path: Optional[Path] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> dict:
    """Translate a normalized bbox slice of `mesh`'s vertices along `axis` by
    `delta_frac * axis_extent`, writing the edited clothing to `out`.

    This is the vertex-count-preserving "lower-soles / shorten-cuff" op: the
    bottom `(from_pct, to_pct)` band of the mesh along `axis` is shifted by a
    signed fraction of that axis' extent (negative = down/in). Vertex count and
    skin-cluster indexing are never touched.

    `preview_dir` (optional) gets `before.png` / `after.png` side renders.

    Returns the report dict: `before` / `after` measurements (for the
    preservation check), `edit` (slice + move details incl. `moved_count`), and
    `previews` (rendered PNG paths or per-render error strings).
    """
    cfg = _resolve_config(config)
    src = Path(src).resolve()
    out = Path(out).resolve()
    if not src.is_file():
        raise BlenderError(f"input FBX not found: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path).resolve() if report_path else out.with_suffix(".report.json")

    script_args = [
        "--op", "translate_slice",
        "--in", str(src),
        "--out", str(out),
        "--report", str(report_path),
        "--mesh", mesh,
        "--axis", axis,
        "--from-pct", repr(float(from_pct)),
        "--to-pct", repr(float(to_pct)),
        "--delta-frac", repr(float(delta_frac)),
    ]
    if preview_dir is not None:
        preview_dir = Path(preview_dir).resolve()
        preview_dir.mkdir(parents=True, exist_ok=True)
        script_args += ["--preview-dir", str(preview_dir)]

    proc = _run_blender(cfg, "geometry_edit.py", script_args, timeout=timeout)
    report = _read_report(report_path, proc, "geometry_edit.py")
    if not out.exists():
        raise BlenderError(
            f"translate_slice reported success but {out} is missing\n"
            f"--- stdout:\n{proc.stdout}\n--- stderr:\n{proc.stderr}"
        )
    head = out.read_bytes()[:23]
    if not head.startswith(b"Kaydara FBX Binary"):
        raise BlenderError(f"translate_slice produced a non-binary FBX at {out} (head={head!r})")
    return report


def scale_slice(
    src: Path,
    out: Path,
    *,
    mesh: str,
    axis: str = "z",
    from_pct: float = 0.0,
    to_pct: float = 0.3,
    factor: float = 0.5,
    preview_dir: Optional[Path] = None,
    config: Optional[BlenderConfig] = None,
    report_path: Optional[Path] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> dict:
    """Squash a normalized bbox slice of `mesh` along `axis` toward the top of
    the slice by `factor` (0<factor<1 shortens), writing to `out`.

    This is the clean "cut sole height" op: the band's top plane is the pivot
    and stays put (no tear with the rest of the mesh), while everything below
    rises proportionally — so the region gets shorter, not stretched into a
    spike the way a raw slice *translate* does. Vertex-count-preserving.

    Returns the report dict (`before`/`after` measurements, `edit` with
    `pivot`/`height_removed`/`moved_count`, `previews`).
    """
    cfg = _resolve_config(config)
    src = Path(src).resolve()
    out = Path(out).resolve()
    if not src.is_file():
        raise BlenderError(f"input FBX not found: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path).resolve() if report_path else out.with_suffix(".report.json")

    script_args = [
        "--op", "scale_slice",
        "--in", str(src),
        "--out", str(out),
        "--report", str(report_path),
        "--mesh", mesh,
        "--axis", axis,
        "--from-pct", repr(float(from_pct)),
        "--to-pct", repr(float(to_pct)),
        "--factor", repr(float(factor)),
    ]
    if preview_dir is not None:
        preview_dir = Path(preview_dir).resolve()
        preview_dir.mkdir(parents=True, exist_ok=True)
        script_args += ["--preview-dir", str(preview_dir)]

    proc = _run_blender(cfg, "geometry_edit.py", script_args, timeout=timeout)
    report = _read_report(report_path, proc, "geometry_edit.py")
    if not out.exists():
        raise BlenderError(
            f"scale_slice reported success but {out} is missing\n"
            f"--- stdout:\n{proc.stdout}\n--- stderr:\n{proc.stderr}"
        )
    head = out.read_bytes()[:23]
    if not head.startswith(b"Kaydara FBX Binary"):
        raise BlenderError(f"scale_slice produced a non-binary FBX at {out} (head={head!r})")
    return report


def bisect_cut(
    src: Path,
    out: Path,
    *,
    mesh: str,
    axis: str = "z",
    cut_pct: float = 0.15,
    preview_dir: Optional[Path] = None,
    textures: Optional[dict] = None,
    shading: str = "solid",
    config: Optional[BlenderConfig] = None,
    report_path: Optional[Path] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> dict:
    """Real cut: bisect `mesh` by a world-`axis` plane at `cut_pct` of its
    extent, delete the slab below, cap the opening, write to `out`.

    Removes height with NO texture compression (geometry is removed, not
    squashed). Unlike translate/scale this CHANGES the vertex count, so the
    integrity question is whether the new cut-edge verts kept skin weights —
    the report's `edit.unweighted_verts_after` is the empirical answer (0 =
    every surviving vert is still weighted).

    Returns the report dict (`edit` carries verts_before/after, verts_removed,
    weight-entry + unweighted-vert counts; `previews` the rendered PNGs).
    """
    cfg = _resolve_config(config)
    src = Path(src).resolve()
    out = Path(out).resolve()
    if not src.is_file():
        raise BlenderError(f"input FBX not found: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path).resolve() if report_path else out.with_suffix(".report.json")

    script_args = [
        "--op", "bisect_cut",
        "--in", str(src),
        "--out", str(out),
        "--report", str(report_path),
        "--mesh", mesh,
        "--axis", axis,
        "--cut-pct", repr(float(cut_pct)),
        "--shading", shading,
    ]
    if preview_dir is not None:
        preview_dir = Path(preview_dir).resolve()
        preview_dir.mkdir(parents=True, exist_ok=True)
        script_args += ["--preview-dir", str(preview_dir)]
    if textures:
        tex_json = out.with_suffix(".textures.json")
        tex_json.write_text(json.dumps(textures), encoding="utf-8")
        script_args += ["--textures", str(tex_json)]

    proc = _run_blender(cfg, "geometry_edit.py", script_args, timeout=timeout)
    report = _read_report(report_path, proc, "geometry_edit.py")
    if not out.exists():
        raise BlenderError(
            f"bisect_cut reported success but {out} is missing\n"
            f"--- stdout:\n{proc.stdout}\n--- stderr:\n{proc.stderr}"
        )
    head = out.read_bytes()[:23]
    if not head.startswith(b"Kaydara FBX Binary"):
        raise BlenderError(f"bisect_cut produced a non-binary FBX at {out} (head={head!r})")
    return report


def bone_report(
    src: Path,
    *,
    bones: list[str],
    preview_dir: Optional[Path] = None,
    frame_mesh: Optional[str] = None,
    config: Optional[BlenderConfig] = None,
    report_path: Optional[Path] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> dict:
    """Import `src` and report where the named bones sit (no export).

    For each requested bone the report carries `head_world` (the armature's
    REST position — Blender builds it from the skin bind pose) and
    `pose_head_world` (the bone's Model transform applied on top, which is
    what the viewport, a render, and Unity all show). A rest-pose bone move
    therefore shows up in `pose_head_world`.

    `preview_dir` renders `front.png` / `side.png` framed on `frame_mesh`
    (default: the largest mesh) so the move can be eyeballed. Used by the
    bone-move proof (`training/smoke_bone_transform.py`).
    """
    cfg = _resolve_config(config)
    src = Path(src).resolve()
    if not src.is_file():
        raise BlenderError(f"input FBX not found: {src}")
    report_path = (
        Path(report_path).resolve() if report_path
        else src.with_suffix(".bones.json")
    )
    script_args = [
        "--in", str(src),
        "--report", str(report_path),
        "--bones", ",".join(bones),
    ]
    if frame_mesh:
        script_args += ["--frame-mesh", frame_mesh]
    if preview_dir is not None:
        preview_dir = Path(preview_dir).resolve()
        preview_dir.mkdir(parents=True, exist_ok=True)
        script_args += ["--preview-dir", str(preview_dir)]
    proc = _run_blender(cfg, "bone_report.py", script_args, timeout=timeout)
    return _read_report(report_path, proc, "bone_report.py")


def inspect(
    src: Path,
    *,
    config: Optional[BlenderConfig] = None,
    report_path: Optional[Path] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> dict:
    """Import `src` and measure it; no export. Returns the measurement report."""
    cfg = _resolve_config(config)
    src = Path(src).resolve()
    if not src.is_file():
        raise BlenderError(f"input FBX not found: {src}")
    report_path = (
        Path(report_path).resolve() if report_path
        else src.with_suffix(".inspect.json")
    )

    proc = _run_blender(
        cfg, "geometry_edit.py",
        ["--op", "inspect", "--in", str(src), "--report", str(report_path)],
        timeout=timeout,
    )
    return _read_report(report_path, proc, "geometry_edit.py")
