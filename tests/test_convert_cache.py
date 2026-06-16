"""Tests for the shared, content-addressed bin->ASCII cache.

The bug these guard against: inspect and Phase A used to each run their own
bin_to_ascii on the SAME clothing file, into their own directories. The FBX SDK
mints node ids from memory pointers, so two conversions of one file disagree on
every model_id -> the FE's drop ids never matched the pipeline's view and every
clothing-side drop was a silent no-op.

`bin_to_ascii_cached` collapses both call sites onto one on-disk cache keyed by
the source file's identity, so a given clothing always resolves to the SAME
ASCII bytes (hence the same ids) -- within a run AND across a BE restart.

These tests mock the (slow, SDK-backed) bin_to_ascii so they run without the
fbx_env toolchain.
"""
from __future__ import annotations

from pathlib import Path

import rigforge.ascii_fbx.convert as convert
from rigforge.ascii_fbx.convert import bin_to_ascii_cached

_MAGIC = b"Kaydara FBX Binary  \x00"


def _fake_binary(p: Path) -> Path:
    p.write_bytes(_MAGIC + b"\x00" * 64)
    return p


def _install_counting_converter(monkeypatch) -> dict:
    """Replace bin_to_ascii with a counter that writes a unique dummy ASCII
    each call, so a second conversion is detectable (different bytes)."""
    state = {"calls": 0}

    def fake_bin_to_ascii(src, dst, *, config=None):
        state["calls"] += 1
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Embed the call count so re-conversion would yield different bytes.
        dst.write_bytes(f"; ascii conversion #{state['calls']}\n".encode())
        return dst

    monkeypatch.setattr(convert, "bin_to_ascii", fake_bin_to_ascii)
    return state


def test_same_source_converts_once_and_is_stable(tmp_path, monkeypatch):
    state = _install_counting_converter(monkeypatch)
    src = _fake_binary(tmp_path / "clothing.fbx")
    cache_dir = tmp_path / "_fbx_ascii_cache"

    first = bin_to_ascii_cached(src, cache_dir)
    second = bin_to_ascii_cached(src, cache_dir)

    # One underlying conversion total -> ids are stable across both callers.
    assert state["calls"] == 1
    assert first == second
    assert Path(first).read_bytes() == Path(second).read_bytes()


def test_two_callers_share_one_file(tmp_path, monkeypatch):
    """Simulates inspect (caller A) then assemble (caller B) of the same
    clothing: they must resolve to the identical cached ASCII path."""
    state = _install_counting_converter(monkeypatch)
    src = _fake_binary(tmp_path / "clothing.fbx")
    cache_dir = tmp_path / "_fbx_ascii_cache"

    inspect_ascii = bin_to_ascii_cached(src, cache_dir)   # FE outliner
    assemble_ascii = bin_to_ascii_cached(src, cache_dir)  # Phase A

    assert inspect_ascii == assemble_ascii
    assert state["calls"] == 1


def test_survives_simulated_restart(tmp_path, monkeypatch):
    """The cache is on disk, so a fresh process (no warm state) reuses the
    file produced before the restart -- as long as the source is unchanged."""
    state = _install_counting_converter(monkeypatch)
    src = _fake_binary(tmp_path / "clothing.fbx")
    cache_dir = tmp_path / "_fbx_ascii_cache"

    before = bin_to_ascii_cached(src, cache_dir)
    # "restart": new monkeypatched converter instance, same on-disk cache_dir.
    state2 = _install_counting_converter(monkeypatch)
    after = bin_to_ascii_cached(src, cache_dir)

    assert before == after
    assert state2["calls"] == 0  # disk hit; no reconversion after restart


def test_content_change_reconverts(tmp_path, monkeypatch):
    state = _install_counting_converter(monkeypatch)
    src = _fake_binary(tmp_path / "clothing.fbx")
    cache_dir = tmp_path / "_fbx_ascii_cache"

    first = bin_to_ascii_cached(src, cache_dir)
    # Change the source content (and therefore size) -> new cache key.
    src.write_bytes(_MAGIC + b"\x01" * 128)
    second = bin_to_ascii_cached(src, cache_dir)

    assert state["calls"] == 2
    assert first != second


def test_ascii_input_is_passthrough(tmp_path, monkeypatch):
    """An already-ASCII clothing is returned as-is, never converted -- both
    callers read the original file, so ids already matched (the case that
    'worked before')."""
    state = _install_counting_converter(monkeypatch)
    src = tmp_path / "clothing_ascii.fbx"
    src.write_bytes(b"; FBX 7.4.0 project file\n")
    cache_dir = tmp_path / "_fbx_ascii_cache"

    out = bin_to_ascii_cached(src, cache_dir)

    assert Path(out).resolve() == src.resolve()
    assert state["calls"] == 0
