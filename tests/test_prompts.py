"""Tests for the prompt builder."""
from __future__ import annotations

import json

import pytest

from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.prompts import build_messages


@pytest.fixture(scope="module")
def schema() -> CanonicalSchema:
    return CanonicalSchema.load_default()


def _make_request(n_bones: int = 2):
    return {
        "donor_id": "maya",
        "target_id": "maya",
        "canonical_schema_version": "1.0",
        "bones": [
            {"model_id": 1, "name": "Hips", "parent_name": None,
             "translation_xyz": [0, 0.9, 0], "child_names": ["Spine"],
             "has_skin_cluster": True, "cluster_weight_count": 100},
            {"model_id": 2, "name": "Spine", "parent_name": "Hips",
             "translation_xyz": [0, 0.1, 0], "child_names": [],
             "has_skin_cluster": True, "cluster_weight_count": 50},
        ][:n_bones],
    }


def test_builds_two_messages_when_no_violations(schema):
    msgs = build_messages(request=_make_request(), schema=schema)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_adds_third_message_when_reprompting(schema):
    msgs = build_messages(
        request=_make_request(),
        schema=schema,
        previous_violations=["unique_role: Hips assigned twice"],
    )
    assert len(msgs) == 3
    assert "Hips assigned twice" in msgs[2]["content"]


def test_repair_hint_for_duplicate_canonical_role(schema):
    """Re-prompt must steer the LLM: when a canonical role lands on multiple
    bones, keep the one that best fits the role and reassign the others —
    don't just shuffle the conflict (which produced the post-reprompt failures
    on azure_virtue / classic_chic)."""
    msgs = build_messages(
        request=_make_request(),
        schema=schema,
        previous_violations=[
            "canonical role 'Hand.L' assigned to 16 kept bones: [1, 2, 3]"
        ],
    )
    repair = msgs[2]["content"].lower()
    assert "unique_role" in repair or "canonical role" in repair
    assert "reassign" in repair or "remap" in repair or "change" in repair


def test_repair_hint_for_missing_required_ancestor(schema):
    """Re-prompt must steer the LLM: when a bone's actual chain is missing a
    required canonical ancestor, the bone is almost certainly mis-labeled.
    The fix is to drop its canonical role (aux/secondary), NOT to invent a
    new ancestor."""
    msgs = build_messages(
        request=_make_request(),
        schema=schema,
        previous_violations=[
            "bone 999 (role=UpperChest) actual chain ['UpperChest', 'Spine', 'Hips'] is missing required ancestors ['Chest']"
        ],
    )
    repair = msgs[2]["content"].lower()
    assert "missing required ancestors" in repair or "missing required" in repair
    assert "aux" in repair or "different role" in repair or "mis-labeled" in repair


def test_system_message_includes_required_roles(schema):
    msgs = build_messages(request=_make_request(), schema=schema)
    sys = msgs[0]["content"]
    for r in schema.required_roles:
        assert r in sys, f"required role {r} missing from system prompt"


def test_system_message_explains_drop_categories(schema):
    msgs = build_messages(request=_make_request(), schema=schema)
    sys = msgs[0]["content"]
    for cat in schema.drop_categories:
        assert cat in sys


def test_user_message_contains_bones_json(schema):
    req = _make_request(n_bones=2)
    msgs = build_messages(request=req, schema=schema)
    user = msgs[1]["content"]
    assert '"model_id": 1' in user or '"model_id":1' in user
    assert "Hips" in user
    assert "Spine" in user
    assert 'donor=' in user.lower() or "donor_id" in user


def test_output_schema_documented_in_system(schema):
    msgs = build_messages(request=_make_request(), schema=schema)
    sys = msgs[0]["content"]
    assert "model_id" in sys
    assert "verdict" in sys
    assert "keep" in sys and "drop" in sys


def test_system_message_documents_new_parent_role(schema):
    """v2 reparent: the system prompt must explain `new_parent_role` so the
    model can flag chain-topology mismatches without us hard-failing."""
    msgs = build_messages(request=_make_request(), schema=schema)
    sys = msgs[0]["content"]
    assert "new_parent_role" in sys


def test_system_message_disambiguates_terminal_arm_bone(schema):
    """classic_chic regression: LLM labeled `Left wrist` as LowerArm.L because
    the chain shoulder→arm→elbow→wrist looks like 4 lower-arm bones to it.
    Prompt must call out that the terminal arm bone (wrist / hand / palm /
    paw) is Hand.L/.R, not LowerArm.L/.R."""
    msgs = build_messages(request=_make_request(), schema=schema)
    sys = msgs[0]["content"].lower()
    assert "wrist" in sys, "prompt must mention 'wrist' to disambiguate Maya-style chains"
    assert "hand.l" in sys and "hand.r" in sys
    # Must clearly state that the terminal bone is Hand, not LowerArm
    assert "terminal" in sys or "deepest" in sys or "last" in sys


def test_system_message_documents_accessory_for_structural_parents(schema):
    """v2.1: Accessory.* is the LLM's role for kept structural-only parent
    bones (ribbon roots, jacket roots, decorative chain anchors). Prompt must
    spell this out, since the school_uniform regression showed the model would
    rather drop such bones than use a non-canonical label."""
    msgs = build_messages(request=_make_request(), schema=schema)
    sys = msgs[0]["content"]
    assert "Accessory" in sys, "Accessory secondary role must appear in prompt"
    sys_low = sys.lower()
    # Must connect "children carry weights" to the Accessory role
    assert "structural" in sys_low or "orphan" in sys_low
    assert "ribbon" in sys_low or "jacket" in sys_low


def test_repair_hint_for_drop_safety_violation(schema):
    """Re-prompt must tell the LLM: when drop_safety fires, flip verdict to
    keep + use Accessory role. Not just retry the same drop."""
    msgs = build_messages(
        request=_make_request(),
        schema=schema,
        previous_violations=[
            "bone 7 (Ribon_Root) marked drop but its subtree carries 260 cluster weight bindings"
        ],
    )
    repair = msgs[2]["content"]
    assert "Accessory" in repair, "repair hint must steer LLM toward Accessory"
    assert "drop_safety" in repair.lower() or "subtree carries" in repair.lower()


def test_system_message_warns_against_optional_role_overreach(schema):
    """school_uniform regression: LLM labeled `Jakit_Root` (a jacket aux bone
    under Spine) as UpperChest because UpperChest is optional and 'nearby'.
    Prompt must forbid promoting accessory bones into optional canonical
    roles when no real spine bone fits the slot."""
    msgs = build_messages(request=_make_request(), schema=schema)
    sys = msgs[0]["content"]
    assert "UpperChest" in sys, "must explicitly mention UpperChest as the trap case"
    sys_low = sys.lower()
    # Must convey 'optional canonical roles can be left unfilled'
    assert any(p in sys_low for p in ("leave unfilled", "leave it unfilled",
                                       "leave the role unfilled", "may be omitted",
                                       "may be skipped", "do not assign")), \
        "must tell the LLM optional canonical roles can be left unassigned"
    # Must warn against promoting attachment/accessory bones
    assert any(w in sys_low for w in ("accessory", "attachment", "decorative", "jacket", "ribbon"))
