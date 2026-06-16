"""identify_donor target-preference.

A clothing adapted FOR the target avatar can lose donor identification to a
BIGGER curated avatar: the coverage scorer favors whichever avatar's fingerprint
is larger, so one coincidental bone (e.g. a skirt-dynamics bone the bigger
avatar happens to carry) flips the pick. When the target is itself a viable
match (score >= threshold), identify_donor should pick the target so Phase C
takes the reliable name-based merge instead of the cross-avatar path.
"""
from __future__ import annotations

import pytest

from rigforge.avatars.fingerprint import fingerprint
from rigforge.avatars.registry import (
    DEFAULT_DONOR_THRESHOLD,
    DonorIdentificationError,
    identify_donor,
)

# Ten distinct core bones both avatars share, + bones unique to the big avatar.
_SHARED = ["hips", "spine", "chest", "neck", "head",
           "clavicle", "elbow", "knee", "ankle", "toe"]
_MOE_ONLY = ["skirtroot", "tail", "wing", "ribbon", "antenna"]


class _FakeRegistry:
    """identify_donor only needs .fingerprints()."""
    def __init__(self, fps):
        self._fps = fps
    def fingerprints(self):
        return self._fps


def _registry():
    return _FakeRegistry({
        "maya": fingerprint(_SHARED),                 # smaller (target)
        "moe": fingerprint(_SHARED + _MOE_ONLY),      # bigger superset
    })


def test_default_picks_bigger_avatar_on_coincidental_bone():
    reg = _registry()
    # Clothing = the shared core + ONE moe-only bone (a skirt bone).
    clothing = _SHARED + ["skirtroot"]
    donor, score, ranked = identify_donor(clothing, reg)
    assert donor == "moe"                 # the bug we're fixing
    assert dict(ranked)["maya"] >= DEFAULT_DONOR_THRESHOLD  # maya was viable


def test_prefers_target_when_viable():
    reg = _registry()
    clothing = _SHARED + ["skirtroot"]
    donor, score, _ = identify_donor(clothing, reg, target_id="maya")
    assert donor == "maya"
    assert score >= DEFAULT_DONOR_THRESHOLD


def test_falls_back_to_cross_avatar_when_target_not_viable():
    reg = _registry()
    # Clothing matches ONLY moe-unique bones — maya isn't a real match here.
    clothing = _MOE_ONLY + ["hips"]
    donor, _, ranked = identify_donor(clothing, reg, target_id="maya")
    assert donor == "moe"
    assert dict(ranked)["maya"] < DEFAULT_DONOR_THRESHOLD


def test_no_target_is_unchanged_behavior():
    reg = _registry()
    clothing = _SHARED + ["skirtroot"]
    assert identify_donor(clothing, reg)[0] == "moe"


def test_below_threshold_still_raises_without_target():
    reg = _registry()
    clothing = ["totally", "unrelated", "bones"]
    with pytest.raises(DonorIdentificationError):
        identify_donor(clothing, reg)
