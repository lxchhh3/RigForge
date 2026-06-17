"""Dictionary-based name translation for blendshape (morph) and mesh names.

Booth assets ship JP/KR/CN morph and mesh names; this module maps them to
readable English via a curated, offline dictionary (`data/name_translation.json`)
-- deterministic, no LLM, no network. Misses keep the original name. A trailing
numeric/index tail (`.001`, `_02`, trailing digits) is preserved so chain order
and per-mesh uniqueness survive translation.

Collisions are allowed by design: distinct morphs that serve one function
(a smile spans mouth + eyes + cheeks) may legitimately translate to the same
English name -- we do NOT suffix-disambiguate, because that would trade away the
readability the translation exists to provide.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional


_DEFAULT_DICT_PATH = Path(__file__).resolve().parent.parent / "data" / "name_translation.json"

# Laterality suffixes peeled off the end and re-attached in normalized `_L`/`_R`
# form, so `びっくり_R` resolves via the `びっくり` entry. Longest keys first so
# `_L`/`.L` win before any shorter match. Japanese 右/左 (right/left) included —
# they're common on mirrored morphs.
_LATERAL_SUFFIXES = (
    ("_L", "_L"), ("_R", "_R"), (".L", "_L"), (".R", "_R"),
    ("右", "_R"), ("左", "_L"),
)

# Trailing index tail: an optional separator (`.`, `_`, or space) then a run of
# ASCII or full-width digits, anchored at end. Full-width digits get normalized
# to ASCII on re-attach (`２` -> `2`).
_INDEX_RE = re.compile(r"^(?P<base>.*?)(?P<sep>[._ ]?)(?P<digits>[0-9０-９]+)$")
_FW_TO_ASCII_DIGITS = {0xFF10 + i: ord("0") + i for i in range(10)}


def load_translation_table(path: Optional[Path] = None) -> dict[str, str]:
    """Load the JP/KR->EN name dictionary -- the flat `entries` map.

    `path` overrides the shipped `data/name_translation.json` (used by tests).
    """
    p = Path(path) if path is not None else _DEFAULT_DICT_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    entries = data.get("entries", {})
    if not isinstance(entries, dict):
        raise ValueError(f"{p}: 'entries' must be a JSON object")
    return entries


def translate_name(name: Optional[str], table: dict[str, str]) -> Optional[str]:
    """Translate `name` to English via `table`, or return None when there's
    nothing to do (no dictionary entry, or the name is already English).

    Tries an exact match first (so a key that literally contains digits/suffixes
    still resolves), then iteratively peels trailing index tails (`まばたき2`,
    `笑い.001`) and laterality suffixes (`びっくり_R`, `ウィンク右`, bare `口角上げL`)
    off the end, normalizing them (full-width `２` -> `2`, `.R`/`右` -> `_R`), and
    looks up the bare base. The normalized suffixes are re-attached in original
    order: `ウィンク２右` -> `Wink2_R`. If that fails, retries on the NFKC-normalized
    name so half-width katakana (`ｳｨﾝｸ` -> `ウィンク`) resolves. A base that isn't
    in the table yields None, so the caller keeps the original name. Never raises.
    """
    if not name:
        return None
    result = _translate_core(name, table)
    if result is not None:
        return result
    # Fallback: normalize compatibility forms (half-width katakana -> full-width,
    # full-width ASCII -> ASCII, ...) and retry. Only reached when the raw name
    # didn't resolve, so well-formed names never pay for it.
    normalized = unicodedata.normalize("NFKC", name)
    if normalized != name:
        return _translate_core(normalized, table)
    return None


def _translate_core(name: str, table: dict[str, str]) -> Optional[str]:
    """Exact match, else peel trailing suffixes and look up the bare base."""
    hit = table.get(name)
    if hit is not None:
        return hit
    base = name
    suffix = ""
    while True:
        peeled = _peel_suffix(base)
        if peeled is None:
            break
        base, normalized = peeled
        suffix = normalized + suffix
    if base != name:
        base_hit = table.get(base)
        if base_hit is not None:
            return f"{base_hit}{suffix}"
    return None


def _peel_suffix(s: str) -> Optional[tuple[str, str]]:
    """Peel ONE trailing laterality or index suffix off `s`. Returns
    `(remaining_base, normalized_suffix)`, or None when nothing peels. The
    caller loops until None to strip stacked suffixes (e.g. index + laterality).
    """
    for raw, norm in _LATERAL_SUFFIXES:
        if s.endswith(raw) and len(s) > len(raw):
            return s[: -len(raw)], norm
    # Bare trailing L/R with no separator (e.g. 口角上げL). Only when preceded by
    # a non-ASCII char, so we never split an English name that happens to end in
    # L/R — that boundary is the MMD-morph convention, not an English word.
    if len(s) >= 2 and s[-1] in ("L", "R") and ord(s[-2]) > 127:
        return s[:-1], f"_{s[-1]}"
    m = _INDEX_RE.match(s)
    if m and m.group("digits"):
        digits = m.group("digits").translate(_FW_TO_ASCII_DIGITS)
        return m.group("base"), f"{m.group('sep')}{digits}"
    return None
