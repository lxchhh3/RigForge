"""Canonical bone schema loader.

Loads `data/canonical_schema.json` (the frozen v1 IR) and exposes typed
accessors. Validators (see validators.py) consume this. The cache key (see
cache/key.py) includes `version` so a schema bump invalidates all cached
LLM decisions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


_DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "canonical_schema.json"


class RoleDef(BaseModel):
    parent: Optional[str] = None
    category: str
    optional: bool = False


class SecondaryPattern(BaseModel):
    prefix: str
    suffixes: list[str]
    indexed: bool = True


class CanonicalSchema(BaseModel):
    """The frozen schema bundle. Pydantic enforces JSON shape at load time."""
    version: str
    description: str = ""
    roles: dict[str, RoleDef]
    spine_chain: list[str]
    required_roles: list[str]
    secondary_role_patterns: list[SecondaryPattern] = Field(default_factory=list)
    verdict_values: list[str]
    drop_categories: list[str]
    special_role_tokens: dict[str, str] = Field(default_factory=dict)
    excluded_in_v1: list[str] = Field(default_factory=list)

    # --- derived structures (computed lazily) -------------------------------

    def role_names(self) -> set[str]:
        return set(self.roles.keys())

    def is_canonical(self, role: str) -> bool:
        return role in self.roles

    def is_secondary(self, role: str) -> bool:
        if role in self.roles:
            return False
        for p in self.secondary_role_patterns:
            if _matches_secondary(p, role):
                return True
        return False

    def is_special_token(self, role: str) -> bool:
        return role in self.special_role_tokens

    def lateral_pair_of(self, role: str) -> Optional[str]:
        """Given an `.L` role, return its `.R` peer (or vice versa). None if
        the role isn't a lateral one."""
        if role.endswith(".L"):
            peer = role[:-2] + ".R"
        elif role.endswith(".R"):
            peer = role[:-2] + ".L"
        else:
            return None
        if peer in self.roles or self.is_secondary(peer):
            return peer
        return None

    def parent_of(self, role: str) -> Optional[str]:
        rdef = self.roles.get(role)
        return rdef.parent if rdef else None

    def ancestor_chain(self, role: str) -> list[str]:
        """Return [role, parent, grandparent, ...] up to the root canonical
        bone. Empty if `role` isn't canonical."""
        out: list[str] = []
        cur: Optional[str] = role
        while cur is not None and cur in self.roles:
            out.append(cur)
            cur = self.roles[cur].parent
        return out

    @classmethod
    def load_default(cls) -> "CanonicalSchema":
        return cls.load(_DEFAULT_SCHEMA_PATH)

    @classmethod
    def load(cls, path: Path) -> "CanonicalSchema":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


def _matches_secondary(pattern: SecondaryPattern, role: str) -> bool:
    """Secondary role string matches `<prefix><suffix>[.<index>]?`."""
    for sfx in pattern.suffixes:
        prefix_part = pattern.prefix + sfx
        if not role.startswith(prefix_part):
            continue
        rest = role[len(prefix_part):]
        if not pattern.indexed:
            return rest == ""
        # indexed: optional ".<digits>" tail (zero-padded or not)
        if rest == "":
            return True  # allow unindexed first instance
        if re.match(r"^\.\d+$", rest):
            return True
    return False
