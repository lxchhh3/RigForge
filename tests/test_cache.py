"""Tests for cache key derivation and on-disk store."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from rigforge.cache.key import (
    CacheKeyInputs,
    compute_cache_key,
    compute_cache_key_inputs,
)
from rigforge.cache.store import CacheError, DecisionCache
from rigforge.canonical.decisions import Decision, DecisionSet


# Build a minimal BoneRecord-like stub: cache/key.py only reads
#   name, parent_name, translation_xyz
@dataclass
class StubBone:
    name: str
    parent_name: Optional[str]
    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)


# --- compute_cache_key_inputs ----------------------------------------------


def test_inputs_are_normalized_and_sorted():
    bones = [
        StubBone("J_Bip_Spine", "J_Bip_Hips", (0.0, 0.1, 0.0)),
        StubBone("J_Bip_Hips",  None,         (0.0, 0.9, 0.0)),
    ]
    inputs = compute_cache_key_inputs(
        canonical_schema_version="1.0",
        llm_model_id="mock",
        donor_id="d", target_id="t",
        bones=bones,
    )
    assert list(inputs.normalized_bone_names) == sorted({"jbiphips", "jbipspine"})
    assert ("jbiphips", "jbipspine") in inputs.hierarchy_edges


def test_input_translation_rounded_to_mm():
    bones = [StubBone("Hips", None, (0.0123456, 0.9999, -0.0007))]
    inputs = compute_cache_key_inputs(
        canonical_schema_version="1.0",
        llm_model_id="mock",
        donor_id="d", target_id="t",
        bones=bones,
    )
    sig = inputs.translation_signature[0]
    assert sig == ("hips", 0.012, 1.0, -0.001)


# --- compute_cache_key ------------------------------------------------------


def test_key_is_deterministic():
    bones = [StubBone("Hips", None, (0.0, 0.9, 0.0))]
    inputs = compute_cache_key_inputs(
        canonical_schema_version="1.0", llm_model_id="m",
        donor_id="d", target_id="t", bones=bones,
    )
    assert compute_cache_key(inputs) == compute_cache_key(inputs)


def test_key_changes_with_schema_version():
    bones = [StubBone("Hips", None, (0.0, 0.9, 0.0))]
    common = dict(llm_model_id="m", donor_id="d", target_id="t", bones=bones)
    k1 = compute_cache_key(compute_cache_key_inputs(canonical_schema_version="1.0", **common))
    k2 = compute_cache_key(compute_cache_key_inputs(canonical_schema_version="1.1", **common))
    assert k1 != k2


def test_key_changes_with_llm_id():
    bones = [StubBone("Hips", None, (0.0, 0.9, 0.0))]
    common = dict(canonical_schema_version="1.0", donor_id="d", target_id="t", bones=bones)
    k1 = compute_cache_key(compute_cache_key_inputs(llm_model_id="mock", **common))
    k2 = compute_cache_key(compute_cache_key_inputs(llm_model_id="real", **common))
    assert k1 != k2


def test_key_stable_under_mm_translation_noise():
    """Two rigs that differ only by sub-mm float noise should hash to the
    same key (cache hit across DCC re-saves)."""
    bones_a = [StubBone("Hips", None, (0.000001, 0.900001, 0.000001))]
    bones_b = [StubBone("Hips", None, (0.0,       0.9,      0.0))]
    common = dict(canonical_schema_version="1.0", llm_model_id="m", donor_id="d", target_id="t")
    k1 = compute_cache_key(compute_cache_key_inputs(bones=bones_a, **common))
    k2 = compute_cache_key(compute_cache_key_inputs(bones=bones_b, **common))
    assert k1 == k2


def test_key_changes_with_bone_set_change():
    common = dict(canonical_schema_version="1.0", llm_model_id="m", donor_id="d", target_id="t")
    k1 = compute_cache_key(compute_cache_key_inputs(
        bones=[StubBone("Hips", None)], **common))
    k2 = compute_cache_key(compute_cache_key_inputs(
        bones=[StubBone("Hips", None), StubBone("Spine", "Hips")], **common))
    assert k1 != k2


def test_key_invariant_under_bone_input_order():
    bone_a = StubBone("Hips", None, (0.0, 0.9, 0.0))
    bone_b = StubBone("Spine", "Hips", (0.0, 0.1, 0.0))
    common = dict(canonical_schema_version="1.0", llm_model_id="m", donor_id="d", target_id="t")
    k_ab = compute_cache_key(compute_cache_key_inputs(bones=[bone_a, bone_b], **common))
    k_ba = compute_cache_key(compute_cache_key_inputs(bones=[bone_b, bone_a], **common))
    assert k_ab == k_ba


# --- DecisionCache ----------------------------------------------------------


def test_cache_round_trip(tmp_path: Path):
    cache = DecisionCache(root=tmp_path)
    key = "a" * 64
    ds = DecisionSet(bones=[Decision(model_id=1, role="Hips", verdict="keep")])
    assert not cache.has(key)
    path = cache.put(key, ds)
    assert path.exists()
    assert cache.has(key)
    loaded = cache.get(key)
    assert loaded == ds


def test_cache_get_miss_returns_none(tmp_path: Path):
    cache = DecisionCache(root=tmp_path)
    assert cache.get("b" * 64) is None


def test_cache_rejects_invalid_key(tmp_path: Path):
    cache = DecisionCache(root=tmp_path)
    with pytest.raises(CacheError, match="invalid cache key"):
        cache.path_for("not-a-hex-digest")


def test_cache_clear(tmp_path: Path):
    cache = DecisionCache(root=tmp_path)
    cache.put("a" * 64, DecisionSet(bones=[Decision(model_id=1, role="Hips", verdict="keep")]))
    cache.put("b" * 64, DecisionSet(bones=[Decision(model_id=2, role="Spine", verdict="keep")]))
    assert cache.clear() == 2
    assert cache.clear() == 0
