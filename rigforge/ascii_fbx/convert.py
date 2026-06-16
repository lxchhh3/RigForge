"""Subprocess wrapper around the fbx_env conda toolchain.

The FBX SDK Python bindings live in an isolated `fbx_env` conda env (see PLAN.md
"Converter Toolchain"). RigForge's main env stays SDK-free; all binary↔ASCII
conversions and structural diffs go through subprocess calls to that env.

Three operations:
- `bin_to_ascii(src, dst)`  invokes fbx_bin2ascii.py
- `ascii_to_bin(src, dst)`  invokes fbx_ascii2bin.py
- `compare(left, right)`    invokes fbx_compare.py and returns a structured result

All operations hard-fail (raise `ConverterError`) on non-zero exit codes other
than the documented success/drift cases of fbx_compare.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# The fbx_env conda PYTHON stays machine-specific (it carries the FBX SDK
# bindings). The converter SCRIPTS (fbx_bin2ascii / fbx_ascii2bin / fbx_compare)
# now live in-repo under fbx_env/ so they're version-controlled — notably the
# DeformPercent re-apply fix in fbx_ascii2bin.py. Both are overridable via
# $RIGFORGE_FBX_ENV_PYTHON / $RIGFORGE_FBX_TOOLCHAIN_DIR.
_DEFAULT_FBX_ENV_PYTHON = Path("D:/2files/env_build/conda/conda/envs/fbx_env/python.exe")
_DEFAULT_TOOLCHAIN_DIR = Path(__file__).resolve().parent.parent.parent / "fbx_env"

ENV_PYTHON = "RIGFORGE_FBX_ENV_PYTHON"
ENV_TOOLCHAIN = "RIGFORGE_FBX_TOOLCHAIN_DIR"


class ConverterError(RuntimeError):
    """Raised when the fbx_env toolchain fails or is misconfigured."""


@dataclass(frozen=True)
class ConverterConfig:
    fbx_env_python: Path
    toolchain_dir: Path

    @classmethod
    def from_env(cls) -> "ConverterConfig":
        py = Path(os.environ.get(ENV_PYTHON, str(_DEFAULT_FBX_ENV_PYTHON)))
        tc = Path(os.environ.get(ENV_TOOLCHAIN, str(_DEFAULT_TOOLCHAIN_DIR)))
        return cls(fbx_env_python=py, toolchain_dir=tc)

    def script(self, name: str) -> Path:
        return self.toolchain_dir / name


@dataclass(frozen=True)
class CompareResult:
    """Outcome of an fbx_compare.py invocation.

    `identical` is True only when fbx_compare exits 0 (no schema/structural drift).
    `raw_output` is the full stdout+stderr text — keep it for debugging / manifests.
    """
    identical: bool
    raw_output: str
    drift_count: int  # parsed from "RESULT: N field(s) drifted" line; 0 if identical
    schema_match: bool


def _resolve_config(config: Optional[ConverterConfig]) -> ConverterConfig:
    cfg = config or ConverterConfig.from_env()
    if not cfg.fbx_env_python.exists():
        raise ConverterError(
            f"fbx_env Python not found at {cfg.fbx_env_python}. "
            f"Set ${ENV_PYTHON} or pass ConverterConfig(fbx_env_python=...)."
        )
    if not cfg.toolchain_dir.exists():
        raise ConverterError(
            f"FBX toolchain dir not found at {cfg.toolchain_dir}. "
            f"Set ${ENV_TOOLCHAIN} or pass ConverterConfig(toolchain_dir=...)."
        )
    return cfg


def _run_script(
    script_name: str,
    args: list[str],
    *,
    config: ConverterConfig,
    allowed_exit_codes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess:
    script_path = config.script(script_name)
    if not script_path.exists():
        raise ConverterError(f"converter script missing: {script_path}")

    cmd = [str(config.fbx_env_python), "-u", str(script_path), *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=False,  # bytes — decode with errors='replace' to survive cp1252 quirks
        check=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""

    if proc.returncode not in allowed_exit_codes:
        raise ConverterError(
            f"{script_name} exited {proc.returncode}\n"
            f"--- cmd: {cmd}\n"
            f"--- stdout:\n{stdout}\n"
            f"--- stderr:\n{stderr}"
        )

    # Attach decoded text back onto the CompletedProcess for caller convenience.
    proc.stdout = stdout  # type: ignore[assignment]
    proc.stderr = stderr  # type: ignore[assignment]
    return proc


def bin_to_ascii(
    src: Path,
    dst: Path,
    *,
    config: Optional[ConverterConfig] = None,
) -> Path:
    """Convert a binary FBX at `src` to an ASCII FBX at `dst`.

    Returns the absolute path to the produced ASCII file.
    """
    cfg = _resolve_config(config)
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    if not src.is_file():
        raise ConverterError(f"input FBX not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    _run_script("fbx_bin2ascii.py", [str(src), str(dst)], config=cfg)

    # Sanity: output should start with the ASCII FBX comment.
    if not dst.exists():
        raise ConverterError(f"bin_to_ascii claimed success but {dst} is missing")
    head = dst.read_bytes()[:32]
    if head.startswith(b"Kaydara FBX Binary"):
        raise ConverterError(f"bin_to_ascii produced a binary file at {dst}")
    return dst


def _ascii_cache_key(src: Path) -> str:
    """Short, stable key identifying a source FBX by (resolved path, size,
    mtime). Cheap — no full-content read. If the file is edited the size or
    mtime changes, so the key changes and we reconvert."""
    import hashlib

    st = src.stat()
    raw = f"{src}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def bin_to_ascii_cached(
    src: Path,
    cache_dir: Path,
    *,
    config: Optional[ConverterConfig] = None,
) -> Path:
    """Return the ASCII form of `src`, converting through a shared, on-disk,
    content-addressed cache so repeated calls for the same source return the
    SAME ASCII file (and therefore the same FBX node ids).

    Why this exists: the FBX SDK derives node ids from memory pointers, so two
    independent `bin_to_ascii` runs over one clothing file disagree on every
    model_id. The FE outliner (inspect) and the pipeline (Phase A) both need
    those ids to line up — otherwise every clothing-side drop the FE sends is a
    silent no-op. Routing both through this one cache guarantees they see
    identical ids, within a run and across a BE restart (the cache is on disk,
    keyed by source identity).

    An already-ASCII `src` is returned verbatim (no conversion, no cache entry)
    — both callers then read the original file, so ids already agree.
    """
    src = Path(src).resolve()
    if not src.is_file():
        raise ConverterError(f"input FBX not found: {src}")

    head = src.read_bytes()[:32]
    if not head.startswith(b"Kaydara FBX Binary"):
        return src  # already ASCII — passthrough

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{src.stem}.{_ascii_cache_key(src)}_ascii.fbx"
    if dst.is_file() and dst.stat().st_size > 0:
        return dst.resolve()
    return bin_to_ascii(src, dst, config=config)


def ascii_to_bin(
    src: Path,
    dst: Path,
    *,
    config: Optional[ConverterConfig] = None,
) -> Path:
    """Convert an ASCII FBX at `src` to a binary FBX at `dst`.

    Returns the absolute path to the produced binary file.
    """
    cfg = _resolve_config(config)
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    if not src.is_file():
        raise ConverterError(f"input FBX not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    _run_script("fbx_ascii2bin.py", [str(src), str(dst)], config=cfg)

    if not dst.exists():
        raise ConverterError(f"ascii_to_bin claimed success but {dst} is missing")
    head = dst.read_bytes()[:23]
    if not head.startswith(b"Kaydara FBX Binary"):
        raise ConverterError(
            f"ascii_to_bin produced a non-binary file at {dst} (head={head!r})"
        )
    return dst


def compare(
    left: Path,
    right: Path,
    *,
    config: Optional[ConverterConfig] = None,
) -> CompareResult:
    """Run fbx_compare.py on two FBX files.

    fbx_compare exits 0 on identical, 1 on drift, 2 on load failure. We treat 0
    and 1 as "ran successfully, may or may not match" and raise on 2.
    """
    cfg = _resolve_config(config)
    left = Path(left).resolve()
    right = Path(right).resolve()
    if not left.is_file():
        raise ConverterError(f"left file not found: {left}")
    if not right.is_file():
        raise ConverterError(f"right file not found: {right}")

    proc = _run_script(
        "fbx_compare.py",
        [str(left), str(right)],
        config=cfg,
        allowed_exit_codes=(0, 1),
    )
    text = (proc.stdout or "") + (proc.stderr or "")  # type: ignore[operator]

    drift_count = _parse_drift_count(text, identical=(proc.returncode == 0))
    schema_match = "schema match: OK" in text

    return CompareResult(
        identical=(proc.returncode == 0),
        raw_output=text,
        drift_count=drift_count,
        schema_match=schema_match,
    )


def _parse_drift_count(text: str, *, identical: bool) -> int:
    """Extract the integer from `RESULT: N field(s) drifted` if present."""
    if identical:
        return 0
    # Look for the result line; if missing, fall back to -1 to flag parsing issue.
    for line in text.splitlines():
        if "field(s) drifted" in line:
            # "RESULT: 3 field(s) drifted"
            parts = line.split()
            for tok in parts:
                if tok.isdigit():
                    return int(tok)
    return -1
