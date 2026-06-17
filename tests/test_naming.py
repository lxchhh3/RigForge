"""Tests for dictionary-based name translation (morphs + mesh names).

translate_name is deterministic: a curated JP/KR->EN table, numeric/index tail
preserved, misses keep the original (return None), collisions allowed (a smile
spanning mouth+eyes+cheeks can legitimately share a translated name).
"""
from __future__ import annotations

import pytest

from rigforge.naming import load_translation_table, translate_name


# A tiny fixed table so these tests don't depend on the shipped dictionary's
# evolving contents.
TABLE = {
    "まばたき": "Blink",
    "笑い": "Smile",
    "ウィンク": "Wink",
    "髪": "Hair",
}


# --- core lookup -------------------------------------------------------------


def test_exact_hit_translates():
    assert translate_name("まばたき", TABLE) == "Blink"


def test_miss_returns_none():
    # Not in the table -> caller keeps the original name.
    assert translate_name("謎モーフ", TABLE) is None


def test_already_english_returns_none():
    # English/ASCII names aren't keys; nothing to do.
    assert translate_name("Smile", TABLE) is None
    assert translate_name("Big breasts", TABLE) is None


def test_empty_or_none_returns_none():
    assert translate_name("", TABLE) is None
    assert translate_name(None, TABLE) is None


# --- numeric / index tail preservation --------------------------------------


def test_trailing_digit_tail_preserved():
    assert translate_name("まばたき2", TABLE) == "Blink2"


def test_dotted_index_tail_preserved():
    assert translate_name("笑い.001", TABLE) == "Smile.001"


def test_underscore_index_tail_preserved():
    assert translate_name("ウィンク_02", TABLE) == "Wink_02"


def test_tail_on_a_miss_still_returns_none():
    # The base (sans tail) doesn't translate -> None even though a tail exists.
    assert translate_name("謎_03", TABLE) is None


# --- laterality (L/R) suffixes ----------------------------------------------


def test_underscore_lr_suffix_stripped_and_normalized():
    # びっくり-style: base translates, _L/_R laterality re-attached.
    t = {"びっくり": "Surprised"}
    assert translate_name("びっくり_L", t) == "Surprised_L"
    assert translate_name("びっくり_R", t) == "Surprised_R"


def test_dot_lr_suffix_normalized_to_underscore():
    t = {"びっくり": "Surprised"}
    assert translate_name("びっくり.L", t) == "Surprised_L"


def test_japanese_lr_kanji_suffix():
    # 右 = right, 左 = left.
    assert translate_name("ウィンク右", TABLE) == "Wink_R"
    assert translate_name("ウィンク左", TABLE) == "Wink_L"


def test_index_and_laterality_combined():
    # ウィンク２右 -> base + fullwidth index + right.
    assert translate_name("ウィンク２右", TABLE) == "Wink2_R"


# --- full-width digit normalization -----------------------------------------


def test_fullwidth_digit_tail_normalized_to_ascii():
    assert translate_name("ウィンク２", TABLE) == "Wink2"
    assert translate_name("まばたき３", TABLE) == "Blink3"


# --- collisions are allowed (no forced uniqueness) --------------------------


def test_collisions_are_not_resolved():
    # Two distinct originals may translate to the same English name; the
    # function does NOT suffix-disambiguate. (Duplicate morphs that share a
    # function are legitimate.)
    t = {"笑い": "Smile", "にっこり": "Smile"}
    assert translate_name("笑い", t) == "Smile"
    assert translate_name("にっこり", t) == "Smile"


# --- shipped dictionary ------------------------------------------------------


def test_shipped_dictionary_loads_and_has_standard_morphs():
    table = load_translation_table()
    assert isinstance(table, dict)
    # A few standard MMD morphs we rely on being present.
    assert table.get("まばたき") == "Blink"
    assert "あ" in table  # viseme
    # Blender's localized primitive defaults — the most common non-English mesh
    # names in real Booth files (Korean/Japanese Blender users).
    assert table.get("평면") == "Plane"   # KR
    assert table.get("큐브") == "Cube"    # KR
    assert table.get("平面") == "Plane"   # JP
    # values must be ASCII so they're tool-safe everywhere downstream.
    for k, v in table.items():
        assert v.isascii(), f"non-ASCII translation value for {k!r}: {v!r}"
        assert v.strip() == v and v, f"blank/edge-padded value for {k!r}"


def test_shipped_dictionary_covers_standard_mmd_morph_set():
    """The expanded MMD face-morph vocabulary, including compound names that
    rely on the laterality + index peel."""
    table = load_translation_table()
    assert translate_name("瞳右", table) == "Pupil_right"
    assert translate_name("怒る_L", table) == "Angry_L"
    assert translate_name("光_上3", table) == "Highlight_up3"
    assert translate_name("下まつ毛_斜め_L", table) == "Lower_lash_diag_L"
    assert translate_name("やえば2_R", table) == "Fang2_R"
