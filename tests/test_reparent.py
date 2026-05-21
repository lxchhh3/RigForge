"""Tests for the bone-reparent edit primitive and surrounding wiring.

Covers:
  - `reparent_bone_edits` produces correct byte-range edits on a fixture FBX
  - Round-trip: ASCII -> reparent -> re-parsed view sees the new parent
  - DecisionSet round-trips the new optional `new_parent_role` field
  - hierarchy_consistency passes when the reparent decision fixes the chain
  - hierarchy_consistency still fails when the reparent doesn't fix the chain
  - hierarchy_consistency unaffected when no reparent decisions are present
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from rigforge.ascii_fbx.edits import (
    EditError,
    apply_edits,
    reparent_bone_edits,
)
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract, parse_args
from rigforge.canonical.decisions import Decision, DecisionSet
from rigforge.canonical.schema import CanonicalSchema
from rigforge.canonical.validators import (
    rule_hierarchy_consistency,
)


# --- fixture FBX -------------------------------------------------------------
#
# Three-bone chain: Hips (root) -> Chest -> ShoulderL.
# Goal scenarios: reparent ShoulderL from Chest to a (yet-to-exist) UpperChest,
# OR reparent ShoulderL directly so its kept canonical parent fixes the chain.
#
# FBX parent-child convention (see sections.extract): `C: "OO", child, parent`.
REPARENT_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 100, "Model::Hips", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0.9,0
\t\t}
\t}
\tModel: 101, "Model::Chest", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0.25,0
\t\t}
\t}
\tModel: 102, "Model::UpperChest", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0.1,0
\t\t}
\t}
\tModel: 103, "Model::ShoulderL", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0.12,0.05,0
\t\t}
\t}
}
Connections:  {
\t;Model::Chest, Model::Hips
\tC: "OO",101,100
\t;Model::UpperChest, Model::Chest
\tC: "OO",102,101
\t;Model::ShoulderL, Model::Chest
\tC: "OO",103,101
\t;Model::Hips, Model::RootNode
\tC: "OO",100,0
}
"""


# --- helpers ----------------------------------------------------------------


def _parent_id_of(view, model_id: int) -> Optional[int]:
    bone = view.bones.get(model_id)
    return bone.parent_id if bone else None


# --- reparent_bone_edits primitive ------------------------------------------


def test_reparent_swaps_parent_id_in_connection_line():
    doc = parse(REPARENT_FBX)
    view = extract(doc)

    edits = reparent_bone_edits(view, view.bones[103], new_parent_id=102)
    out = apply_edits(REPARENT_FBX, edits)

    # The ShoulderL parent-connection used to be (103,101); now (103,102).
    assert b'C: "OO",103,101' not in out
    assert b'C: "OO",103,102' in out

    # Other connections are untouched verbatim.
    assert b'C: "OO",101,100' in out
    assert b'C: "OO",102,101' in out
    assert b'C: "OO",100,0' in out


def test_reparent_round_trip_reparses_with_new_parent():
    doc = parse(REPARENT_FBX)
    view = extract(doc)

    edits = reparent_bone_edits(view, view.bones[103], new_parent_id=102)
    out = apply_edits(REPARENT_FBX, edits)

    out_view = extract(parse(out))
    assert _parent_id_of(out_view, 103) == 102
    # The original parent (Chest, 101) no longer claims ShoulderL as child.
    assert 103 not in out_view.bones[101].children_ids
    assert 103 in out_view.bones[102].children_ids
    # Bone count is unchanged — reparent doesn't drop bones.
    assert set(out_view.bones) == set(view.bones)


def test_reparent_to_same_parent_is_noop():
    doc = parse(REPARENT_FBX)
    view = extract(doc)

    edits = reparent_bone_edits(view, view.bones[103], new_parent_id=101)
    # Either zero edits OR edits that splice in the identical bytes — both
    # leave the source unchanged.
    out = apply_edits(REPARENT_FBX, edits)
    assert out == REPARENT_FBX


def test_reparent_bone_with_no_existing_connection_raises():
    """Bone has no OO row for its parent — reparent has nothing to flip and
    must error rather than silently produce a broken EditPlan."""
    src = b"""\
Objects:  {
\tModel: 100, "Model::Hips", "LimbNode" {
\t\tProperties70:  {
\t\t}
\t}
\tModel: 200, "Model::Floating", "LimbNode" {
\t\tProperties70:  {
\t\t}
\t}
}
Connections:  {
\tC: "OO",100,0
}
"""
    doc = parse(src)
    view = extract(doc)
    with pytest.raises(EditError):
        reparent_bone_edits(view, view.bones[200], new_parent_id=100)


def test_reparent_preserves_byte_layout_of_other_lines():
    """The edit must be a single byte-range splice — touching only the
    parent id in the bone's OO line."""
    doc = parse(REPARENT_FBX)
    view = extract(doc)

    edits = reparent_bone_edits(view, view.bones[103], new_parent_id=102)
    # Should produce exactly one edit (just the parent id slot).
    assert len(edits) == 1
    e = edits[0]
    # The edit must lie entirely inside the Connections section body
    # (not the Model body or another connection line).
    conn = doc.root("Connections")
    assert conn is not None
    assert e.start >= conn.body_open
    assert e.end <= conn.body_close


# --- DecisionSet new_parent_role field --------------------------------------


def test_decision_accepts_optional_new_parent_role():
    d = Decision(
        model_id=103, role="Shoulder.L", verdict="keep",
        new_parent_role="UpperChest",
    )
    assert d.new_parent_role == "UpperChest"


def test_decision_new_parent_role_defaults_to_none():
    d = Decision(model_id=103, role="Shoulder.L", verdict="keep")
    assert d.new_parent_role is None


def test_decision_set_round_trips_new_parent_role():
    raw = {
        "bones": [
            {"model_id": 1, "role": "Hips", "verdict": "keep"},
            {"model_id": 103, "role": "Shoulder.L", "verdict": "keep",
             "new_parent_role": "UpperChest"},
        ]
    }
    ds = DecisionSet.model_validate(raw)
    assert ds.bones[0].new_parent_role is None
    assert ds.bones[1].new_parent_role == "UpperChest"


def test_decision_new_parent_role_must_be_string_or_none():
    """Wrong-type value should be rejected by pydantic."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Decision(
            model_id=103, role="Shoulder.L", verdict="keep",
            new_parent_role=42,  # not a string
        )


# --- hierarchy_consistency: post-reparent topology --------------------------
#
# These tests reuse the StubBone / StubView shapes from test_validators.py for
# parity. We can't import them (private fixtures), so we redeclare locally.


@dataclass
class StubBone:
    model_id: int
    name: str
    parent_id: Optional[int]
    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cluster_weight_count: int = 0
    children_ids: list[int] = field(default_factory=list)


@dataclass
class StubView:
    bones: dict[int, StubBone]


@pytest.fixture(scope="module")
def schema() -> CanonicalSchema:
    return CanonicalSchema.load_default()


def _keep(mid: int, role: str, **kw) -> Decision:
    return Decision(model_id=mid, role=role, verdict="keep", **kw)


def test_hierarchy_passes_when_reparent_fixes_chain(schema):
    """Shoulder.L is parented under Spine (wrong: should be Chest).
    A new_parent_role='Chest' decision flips the effective parent chain
    so the validator sees a clean chain."""
    decisions = DecisionSet(bones=[
        _keep(1, "Hips"),
        _keep(2, "Spine"),
        _keep(3, "Chest"),
        _keep(10, "Shoulder.L", new_parent_role="Chest"),
    ])
    view = StubView(bones={
        1:  StubBone(1, "h", None),
        2:  StubBone(2, "s", 1),
        3:  StubBone(3, "c", 2),
        10: StubBone(10, "shL", 2),  # PARENTED UNDER SPINE — wrong on input
    })
    violations = rule_hierarchy_consistency(decisions, view, schema)
    assert violations == [], (
        f"reparent should fix chain; got: {[v.message for v in violations]}"
    )


def test_hierarchy_still_fails_when_reparent_does_not_fix(schema):
    """Same wrong input as above, but the reparent target is itself wrong
    (Hips, which still skips Chest). Validator should still complain."""
    decisions = DecisionSet(bones=[
        _keep(1, "Hips"),
        _keep(2, "Spine"),
        _keep(3, "Chest"),
        _keep(10, "Shoulder.L", new_parent_role="Hips"),  # wrong target
    ])
    view = StubView(bones={
        1:  StubBone(1, "h", None),
        2:  StubBone(2, "s", 1),
        3:  StubBone(3, "c", 2),
        10: StubBone(10, "shL", 2),
    })
    violations = rule_hierarchy_consistency(decisions, view, schema)
    assert violations, "reparent to wrong role must still fail hierarchy"


def test_hierarchy_unchanged_when_no_reparent_present(schema):
    """No new_parent_role anywhere — behavior must match the v1 validator
    exactly (regression guard for the existing tests)."""
    decisions = DecisionSet(bones=[
        _keep(1, "Hips"),
        _keep(2, "Spine"),
        _keep(13, "Hand.L"),
    ])
    view = StubView(bones={
        1:  StubBone(1, "h", None),
        2:  StubBone(2, "s", 1),
        13: StubBone(13, "haL", 2),
    })
    violations = rule_hierarchy_consistency(decisions, view, schema)
    assert violations, "v1 wrong-parent case must still fail"


def test_hierarchy_reparent_fixes_deep_chain(schema):
    """Hand.L parented straight under Spine (skipping arm chain). Reparent
    onto LowerArm.L (when LowerArm.L itself is properly parented) restores a
    legal chain Hand.L -> LowerArm.L -> UpperArm.L (-> Shoulder.L) -> Chest."""
    decisions = DecisionSet(bones=[
        _keep(1, "Hips"),
        _keep(2, "Spine"),
        _keep(3, "Chest"),
        _keep(11, "UpperArm.L"),
        _keep(12, "LowerArm.L"),
        _keep(13, "Hand.L", new_parent_role="LowerArm.L"),
    ])
    view = StubView(bones={
        1:  StubBone(1, "h", None),
        2:  StubBone(2, "s", 1),
        3:  StubBone(3, "c", 2),
        11: StubBone(11, "uaL", 3),
        12: StubBone(12, "laL", 11),
        13: StubBone(13, "haL", 2),  # as-input: Hand.L under Spine
    })
    violations = rule_hierarchy_consistency(decisions, view, schema)
    assert violations == [], (
        f"reparent should restore arm chain; got: "
        f"{[v.message for v in violations]}"
    )


def test_hierarchy_reparent_target_role_missing_fails(schema):
    """new_parent_role names a role that's not assigned to any kept bone in
    the decision set. The reparent target is unresolvable — validator must
    fail (would otherwise silently treat as no-op)."""
    decisions = DecisionSet(bones=[
        _keep(1, "Hips"),
        _keep(2, "Spine"),
        _keep(3, "Chest"),
        # UpperChest is NOT in this decision set.
        _keep(10, "Shoulder.L", new_parent_role="UpperChest"),
    ])
    view = StubView(bones={
        1:  StubBone(1, "h", None),
        2:  StubBone(2, "s", 1),
        3:  StubBone(3, "c", 2),
        10: StubBone(10, "shL", 2),
    })
    violations = rule_hierarchy_consistency(decisions, view, schema)
    assert violations, "reparent to unassigned role must fail"
