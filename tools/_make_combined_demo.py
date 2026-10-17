"""One-off: produce a single FBX with clothing drops + avatar drops + the
armature-rotation fix, so all three can be eyeballed together. Cache-hits
Phase B (no LLM call)."""
from pathlib import Path

from rigforge.ascii_fbx.convert import bin_to_ascii_cached
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.registry import AvatarRegistry
from rigforge.cache.store import DecisionCache
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.anthropic import AnthropicLLMClient
from rigforge.pipeline.orchestrator import assemble

CLOTHING = Path("D:/2files/unity_proj/MochiFitter_VCC/Assets/OutfitRetargetingSystem/"
                "Outputs/retargeted_20260615_233505_FBX_MANUKA.fbx")
OUT = Path("Output/_fixtest/FBX_MANUKA__maya_FINAL.fbx")
CACHE_DIR = OUT.parent / "_fbx_ascii_cache"

# Clothing meshes to drop — read ids exactly the way the FE/inspect does.
cview = extract(parse(bin_to_ascii_cached(CLOTHING, CACHE_DIR).read_bytes()))
cmesh = {b.name: b.model_id for b in cview.bones.values() if b.type_class == "Mesh"}

# Resolve a blendshape channel id by (owner mesh name, channel name).
from rigforge.ascii_fbx.edits import iter_oo_connections
_oo = list(iter_oo_connections(cview))
_bs2geo = {s: d for s, d in _oo if s in cview.blend_shapes and d in cview.geometries}
_geo_owner = cview.geometry_owner_model
_mesh_name = {b.model_id: b.name for b in cview.bones.values() if b.type_class == "Mesh"}
def _chan(owner_mesh, chan_name):
    hits = [ch.channel_id for ch in cview.blend_shape_channels.values()
            if ch.name == chan_name
            and _mesh_name.get(_geo_owner.get(_bs2geo.get(ch.blend_shape_id))) == owner_mesh]
    assert len(hits) == 1, f"{owner_mesh}/{chan_name}: {len(hits)} matches"
    return hits[0]
# Exactly the user's override: C_Shoes High heeled = 62; nothing else touched.
blend_shape_channel_overrides = {
    _chan("C_Shoes", "High heeled"): 62.0,
}
print("overrides:", blend_shape_channel_overrides)
# The user's actual 9 deselections (recovered from the FE persistence).
CLOTH_DROP = ["A_Skirt_Frill", "B_Balloon_shorts", "B_Beret", "B_Earring",
              "B_Socks", "D_Cat_mimi", "D_Cat_Tail", "D_Rabbit", "D_Tail_tag"]
missing = [n for n in CLOTH_DROP if n not in cmesh]
assert not missing, f"clothing meshes not found: {missing}"
drop_mesh_ids = {cmesh[n] for n in CLOTH_DROP}

registry = AvatarRegistry.load_default()
mview = registry.get("maya").load_ascii_view()
tmesh = {b.name: b.model_id for b in mview.bones.values() if b.type_class == "Mesh"}
TARGET_DROP = ["Hat", "Cloth", "Shoes"]
target_drop_mesh_ids = {tmesh[n] for n in TARGET_DROP if n in tmesh}

print("clothing drops:", {n: cmesh.get(n) for n in CLOTH_DROP})
print("avatar drops  :", {n: tmesh.get(n) for n in TARGET_DROP})

run = assemble(
    clothing_fbx=CLOTHING,
    target_id="maya",
    out_fbx=OUT,
    registry=registry,
    schema=CanonicalSchema.load_default(),
    llm_client=AnthropicLLMClient.from_key_file(None),
    cache=DecisionCache(root=Path("cache")),
    ascii_cache_dir=CACHE_DIR,
    drop_mesh_ids=drop_mesh_ids,
    target_drop_mesh_ids=target_drop_mesh_ids,
    blend_shape_channel_overrides=blend_shape_channel_overrides,
)
print("DONE:", run.output_fbx, "cache_hit=", run.cache_hit)
for n in run.notes:
    if "drop" in n or "armature" in n:
        print("NOTE:", n)
