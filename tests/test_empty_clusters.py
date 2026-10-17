"""A garment's weightless skin clusters must not reach the merged file.

A MochiFitter export (save_08's Riptide_Watch) skins its one mesh to ALL 174
bones of the fitted scene, though only `Left elbow` carries weight. Each empty
cluster still holds a TransformLink: that bone's bind pose in the save_08
scene, where the jacket hem and the tail sat elsewhere. Merged and repointed
onto the target's bones, those links overrode the bones' rest in Blender: 27
Tail.*/Jacket_* bones moved up to 538 mm. An empty cluster deforms nothing,
so Phase C drops it from the clothing before the splice.
"""
from __future__ import annotations

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.registry import AvatarRegistry
from rigforge.canonical.decisions import DecisionSet
from rigforge.pipeline.edit_plan import EditPlan
from rigforge.pipeline.phase_c import run_phase_c


def _link(t: tuple[float, float, float]) -> str:
    return "1,0,0,0,0,1,0,0,0,0,1,0,%s,%s,%s,1" % t


def _cluster(cid: int, bone: str, t, verts: list[int]) -> str:
    body = ""
    if verts:
        n = len(verts)
        body = (f"\t\tIndexes: *{n} {{\n\t\t\ta: {','.join(map(str, verts))}\n\t\t}}\n"
                f"\t\tWeights: *{n} {{\n\t\t\ta: {','.join('1' for _ in verts)}\n\t\t}}\n")
    return (f'\tDeformer: {cid}, "SubDeformer::{bone}", "Cluster" {{\n{body}'
            f"\t\tTransformLink: *16 {{\n\t\t\ta: {_link(t)}\n\t\t}}\n\t}}\n")


def _fbx(objects: str, conns: list[tuple[int, int]]) -> bytes:
    c = "".join(f'\tC: "OO",{a},{b}\n' for a, b in conns)
    return f"; FBX 7.4.0 project file\nObjects:  {{\n{objects}}}\nConnections:  {{\n{c}}}\n".encode()


def _mesh(mid: int, name: str) -> str:
    return (f'\tGeometry: {mid + 1}, "Geometry::{name}", "Mesh" {{\n\t}}\n'
            f'\tModel: {mid}, "Model::{name}", "Mesh" {{\n\t}}\n'
            f'\tDeformer: {mid + 2}, "Deformer::{name}", "Skin" {{\n\t}}\n')


def _mesh_conns(mid: int) -> list[tuple[int, int]]:
    return [(mid, 0), (mid + 1, mid), (mid + 2, mid + 1)]


# Target: Armature > Hips > Tail, with a Body mesh weighted on both (Tail's
# bind pose at y 0.68).
_TARGET = _fbx(
    '\tModel: 1, "Model::Armature", "Null" {\n\t}\n'
    '\tModel: 10, "Model::Hips", "LimbNode" {\n\t}\n'
    '\tModel: 11, "Model::Tail", "LimbNode" {\n\t}\n'
    + _mesh(20, "Body")
    + _cluster(23, "Hips", (0, 0.81, 0.04), [0, 1])
    + _cluster(24, "Tail", (0, 0.68, -0.15), [2]),
    [(1, 0), (10, 1), (11, 10), *_mesh_conns(20), (23, 22), (24, 22), (10, 23), (11, 24)],
)


def _clothing(*, watch_tail_weights: list[int], strap_only_empty: bool = False) -> bytes:
    """A MochiFitter-style watch: weighted on Hips_MFBase only, plus an EMPTY
    cluster on Tail whose bind pose is the fitting scene's (y 0.49)."""
    objects = (
        '\tModel: 101, "Model::Armature", "Null" {\n\t}\n'
        '\tModel: 110, "Model::Hips_MFBase", "LimbNode" {\n\t}\n'
        '\tModel: 111, "Model::Tail", "LimbNode" {\n\t}\n'
        + _mesh(120, "Watch")
        + _cluster(123, "Hips_MFBase", (0, 0.81, 0.04), [0, 1, 2])
        + _cluster(124, "Tail", (0, 0.49, -0.27), watch_tail_weights)
    )
    conns = [(101, 0), (110, 101), (111, 110), *_mesh_conns(120),
             (123, 122), (124, 122), (110, 123), (111, 124)]
    if strap_only_empty:
        # A second mesh whose skin has NO weighted cluster at all.
        objects += _mesh(130, "Strap") + _cluster(133, "Tail", (0, 0.49, -0.27), [])
        conns += [*_mesh_conns(130), (133, 132), (111, 133)]
    return _fbx(objects, conns)


def _merge(clothing: bytes):
    result = run_phase_c(
        clothing_ascii=clothing,
        clothing_view=extract(parse(clothing)),
        donor_id="maya", target_id="maya",
        registry=AvatarRegistry.load_default(),
        edit_plan=EditPlan(), decisions=DecisionSet(bones=[]),
        target_ascii=_TARGET,
    )
    return result, extract(parse(result.merged_ascii))


def _clusters_on(view, bone_name: str):
    bone = next(b for b in view.limb_bones() if b.name == bone_name)
    return [view.clusters[c] for c in bone.cluster_ids if c in view.clusters]


def test_empty_clothing_cluster_does_not_reach_the_merge():
    result, merged = _merge(_clothing(watch_tail_weights=[]))

    # Tail keeps only the target's own cluster: the watch's empty one (and its
    # fitting-scene bind pose) is gone.
    assert [c.cluster_id for c in _clusters_on(merged, "Tail")] == [24]
    # The watch's weighted cluster survives, repointed onto the target's Hips.
    hips = _clusters_on(merged, "Hips")
    assert sorted(c.weight_count for c in hips) == [2, 3]
    assert "Watch" in {b.name for b in merged.bones.values()}
    assert any("empty_clusters: dropped 1" in n for n in result.notes)


def test_weighted_clothing_cluster_is_kept():
    _, merged = _merge(_clothing(watch_tail_weights=[1]))

    assert sorted(c.weight_count for c in _clusters_on(merged, "Tail")) == [1, 1]


def test_skin_with_no_weighted_cluster_is_left_alone():
    """A skin whose clusters are ALL empty (a mesh nobody weighted) keeps them:
    dropping every cluster would turn it from skinned into unskinned."""
    _, merged = _merge(_clothing(watch_tail_weights=[], strap_only_empty=True))

    tail = _clusters_on(merged, "Tail")
    # The target's cluster + the Strap's lone empty one; the Watch's empty one went.
    assert sorted(c.weight_count for c in tail) == [0, 1]
    assert len(tail) == 2
