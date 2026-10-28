"""Object names inside an imported model, from Unity's import artifact.

A prefab or scene override on an object INSIDE an imported FBX names it only
by fileID (e.g. kevin's Toast: Armature_Toast's position, EC_Toast's 'Eat'
weight). Unity keeps no fileID -> node table in the .meta, and the ids are not
a reproducible hash of the node path (Blender-MHWilds lesson #359), but the
import result in `Library/Artifacts/<xx>/<hash>` is a binary SerializedFile
whose object table is keyed by exactly those fileIDs (as pathIDs). GameObjects
carry their name; every component carries a pointer to its GameObject.

Layout read here (SerializedFile v22+, editor artifacts): a big-endian header
(version, metadata size, data offset), then metadata in the file's endianness
(unity version, platform, type-tree flag, types, object table). Each editor
object starts with a 40-byte header (m_ObjectHideFlags + three PPtrs); a
GameObject follows with m_Component, m_Layer, m_Name; a component with
m_GameObject.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional

MASK = 0x7FFF_FFFF_FFFF_FFFF
MODEL_ROOT_GO = 919132149155446097
_EDITOR_HEADER = 40
_GAME_OBJECT = 1
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


def _objects(data: bytes) -> tuple[str, dict[int, tuple[int, int, int]]]:
    """(endian, {pathID: (classID, absolute offset, size)})."""
    if len(data) < 48:
        raise ValueError("too short for a SerializedFile")
    version = struct.unpack(">I", data[8:12])[0]
    if version < 22 or version > 40:
        raise ValueError(f"unsupported SerializedFile version {version}")
    e = "<" if data[16] == 0 else ">"
    meta_size, file_size, data_offset = struct.unpack(">IQQ", data[20:40])
    if file_size != len(data) or data_offset > len(data):
        raise ValueError("header sizes do not match the file")
    pos = 48
    end = data.index(b"\0", pos)
    pos = end + 1
    pos += 4                                   # target platform
    type_tree = data[pos] != 0
    pos += 1
    (ntypes,) = struct.unpack_from(e + "i", data, pos)
    pos += 4
    classes: list[int] = []
    for _ in range(ntypes):
        (cls,) = struct.unpack_from(e + "i", data, pos)
        pos += 4 + 1 + 2                       # classID, isStripped, scriptTypeIndex
        if cls == 114:
            pos += 16                          # script id
        pos += 16                              # old type hash
        if type_tree:
            nnodes, sbuf = struct.unpack_from(e + "ii", data, pos)
            pos += 8 + nnodes * 32 + sbuf
            (ndeps,) = struct.unpack_from(e + "i", data, pos)
            pos += 4 + 4 * ndeps
        classes.append(cls)
    (nobj,) = struct.unpack_from(e + "i", data, pos)
    pos += 4
    out: dict[int, tuple[int, int, int]] = {}
    for _ in range(nobj):
        pos = (pos + 3) & ~3
        pid, start, size, tidx = struct.unpack_from(e + "qqIi", data, pos)
        pos += 24
        if not 0 <= tidx < len(classes):
            raise ValueError("object table names an unknown type")
        out[pid] = (classes[tidx], data_offset + start, size)
    return e, out


def read_object_names(path: Path) -> dict[int, tuple[int, str]]:
    """{pathID: (classID, GameObject name)} for every GameObject and every
    component (named after its GameObject)."""
    data = Path(path).read_bytes()
    try:
        e, objs = _objects(data)
    except (struct.error, IndexError) as exc:
        raise ValueError(f"{path}: not a readable SerializedFile ({exc})") from exc
    names: dict[int, tuple[int, str]] = {}
    for pid, (cls, off, size) in objs.items():
        if cls != _GAME_OBJECT:
            continue
        p = off + _EDITOR_HEADER
        (n,) = struct.unpack_from(e + "i", data, p)
        p += 4 + 12 * n + 4                    # components, m_Layer
        (ln,) = struct.unpack_from(e + "i", data, p)
        p += 4
        names[pid] = (cls, data[p:p + ln].decode("utf-8", "replace"))
    for pid, (cls, off, size) in objs.items():
        if cls == _GAME_OBJECT or size < _EDITOR_HEADER + 12:
            continue
        _fid, go = struct.unpack_from(e + "iq", data, off + _EDITOR_HEADER)
        if go in names:
            names[pid] = (cls, names[go][1])
    return names


_cache: dict[tuple[str, float], Optional[Path]] = {}


def find_model_artifact(project_root: Path, model: Path) -> Optional[Path]:
    """The import artifact of `model`: the artifact holding Unity's fixed
    model-root GameObject named after the file. (A prefab built on the
    model also names objects after its nodes, but under its own ids.)"""
    model = Path(model)
    try:
        key = (str(model.resolve()), model.stat().st_mtime)
    except OSError:
        return None
    if key in _cache:
        return _cache[key]
    needle = model.stem.encode("utf-8")
    found: Optional[Path] = None
    base = Path(project_root) / "Library" / "Artifacts"
    if base.is_dir():
        candidates = []
        for dirpath, _dirs, files in os.walk(base):
            for f in files:
                p = Path(dirpath) / f
                try:
                    if p.stat().st_size > _MAX_ARTIFACT_BYTES:
                        continue
                    if needle in p.read_bytes():
                        candidates.append(p)
                except OSError:
                    continue
        # Newest import first: an old artifact of the same file can linger.
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in candidates:
            try:
                names = read_object_names(p)
            except ValueError:
                continue
            if names.get(MODEL_ROOT_GO) == (_GAME_OBJECT, model.stem):
                found = p
                break
    _cache[key] = found
    return found


def model_object_names(project_root: Path, model: Path) -> dict[int, tuple[int, str]]:
    """{fileID & MASK: (classID, node name)} for the objects of `model`'s
    import — masked, because a nested prefab id loses the sign bit when it is
    composed (`(instance ^ inner) & MASK`)."""
    art = find_model_artifact(project_root, model)
    if art is None:
        return {}
    return {pid & MASK: v for pid, v in read_object_names(art).items()}


__all__ = ["find_model_artifact", "model_object_names", "read_object_names"]
