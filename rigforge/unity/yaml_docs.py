"""Unity's text-serialised YAML (`.unity` scenes, `.prefab`s) -> documents.

Each Unity object is one YAML document headed `--- !u!<classID> &<fileID>`,
optionally followed by `stripped` (a placeholder for an object that lives in
a prefab this file instances). The body is plain YAML 1.1, so it is parsed
with PyYAML's *base* loader: every scalar stays a string. That matters —
YAML 1.1 reads a GameObject named `On` as True and `001` as 1 — and numbers
are converted by the reader that knows which field it is reading.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import yaml

try:  # libyaml: ~50x faster on a 3 MB avatar prefab
    _Loader = yaml.CBaseLoader
except AttributeError:  # pragma: no cover - pure-python fallback
    _Loader = yaml.BaseLoader


class UnityYamlError(ValueError):
    """The text is not a Unity YAML asset, or a property path is malformed."""


@dataclass
class UnityDoc:
    class_id: int
    file_id: int
    stripped: bool
    type_name: str      # the document's single top-level key, e.g. "Transform"
    data: dict          # the body under that key (strings / lists / dicts)


_HEADER = re.compile(r"^--- !u!(\d+) &(-?\d+)( stripped)?[ \t]*$", re.M)


def parse_unity_yaml(text: str) -> list[UnityDoc]:
    """Split a Unity YAML asset into its object documents."""
    heads = list(_HEADER.finditer(text))
    if not heads:
        raise UnityYamlError("no Unity object documents (--- !u!N &id) found")
    # Replace every header with a bare document marker (the `!u!` tag needs
    # the %TAG directive and carries nothing the base loader keeps), then
    # load all bodies in one libyaml pass.
    bodies = _HEADER.sub("---", text[heads[0].start():])
    try:
        loaded = list(yaml.load_all(bodies, Loader=_Loader))
    except yaml.YAMLError as exc:
        raise UnityYamlError(f"YAML parse failed: {exc}") from exc
    if len(loaded) != len(heads):
        raise UnityYamlError(
            f"{len(heads)} document headers but {len(loaded)} documents parsed"
        )
    docs: list[UnityDoc] = []
    for m, body in zip(heads, loaded):
        if isinstance(body, dict) and len(body) == 1:
            (type_name, data), = body.items()
        else:
            type_name, data = "", body
        docs.append(UnityDoc(
            class_id=int(m.group(1)),
            file_id=int(m.group(2)),
            stripped=bool(m.group(3)),
            type_name=type_name,
            data=data if isinstance(data, dict) else {},
        ))
    return docs


_INDEX = re.compile(r"^data\[(\d+)\]$")


def set_property_path(data: dict, path: str, value: Any) -> None:
    """Apply one prefab override (`propertyPath: value`) to an object body.

    Paths are dot-joined field names; arrays appear as `<field>.Array.data[i]`
    and `<field>.Array.size`. Missing parents are created; an index past the
    end extends the list (Unity only writes the overridden elements)."""
    parts = path.split(".")
    if not parts or any(p == "" for p in parts):
        raise UnityYamlError(f"bad property path {path!r}")
    node: Any = data
    i = 0
    while i < len(parts):
        part = parts[i]
        last = i == len(parts) - 1
        if part == "Array":
            if i + 1 >= len(parts):
                raise UnityYamlError(f"bad property path {path!r}")
            sub = parts[i + 1]
            if not isinstance(node, list):
                raise UnityYamlError(f"{path!r}: Array on a non-list")
            if sub == "size":
                n = int(value)
                del node[n:]
                node.extend([None] * (n - len(node)))
                return
            m = _INDEX.match(sub)
            if not m:
                raise UnityYamlError(f"bad array step {sub!r} in {path!r}")
            idx = int(m.group(1))
            if idx >= len(node):
                fill = "0" if all(isinstance(x, str) or x is None for x in node) else None
                node.extend([fill] * (idx + 1 - len(node)))
            if i + 1 == len(parts) - 1:
                node[idx] = value
                return
            if not isinstance(node[idx], (dict, list)):
                node[idx] = {}
            node = node[idx]
            i += 2
            continue
        if not isinstance(node, dict):
            raise UnityYamlError(f"{path!r}: field {part!r} on a non-mapping")
        if last:
            node[part] = value
            return
        nxt = parts[i + 1]
        if nxt == "Array":
            if not isinstance(node.get(part), list):
                node[part] = []
        elif not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
        i += 1


def file_ref(value: Any) -> tuple[int, str]:
    """`{fileID: N, guid: G, type: T}` -> (N, G); guid '' when local."""
    if not isinstance(value, dict):
        return 0, ""
    try:
        fid = int(value.get("fileID", 0) or 0)
    except (TypeError, ValueError):
        fid = 0
    return fid, str(value.get("guid", "") or "")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "UnityDoc",
    "UnityYamlError",
    "as_float",
    "file_ref",
    "parse_unity_yaml",
    "set_property_path",
]
