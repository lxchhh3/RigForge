"""Talk to the blender-mcp socket (port 9876) to read the live scene state."""
import json
import socket
import sys


def send(cmd, timeout=20):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(("127.0.0.1", 9876))
    s.sendall(json.dumps(cmd).encode("utf-8"))
    buf = b""
    while True:
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        buf += chunk
        try:
            json.loads(buf.decode("utf-8"))
            break
        except json.JSONDecodeError:
            continue
    s.close()
    return buf.decode("utf-8", errors="replace")


_REIMPORT_CODE = r'''
import bpy
path = r"PATH_PLACEHOLDER"
before = set(bpy.data.objects.keys())
bpy.ops.import_scene.fbx(filepath=path)
new = [o for o in bpy.data.objects if o.name not in before]
res = []
for o in new:
    if o.type == 'MESH' and o.data.shape_keys:
        for kb in o.data.shape_keys.key_blocks:
            ln = kb.name.lower()
            if 'high heel' in ln or 'big breast' in ln:
                res.append("%s :: %s = %s" % (o.name, kb.name, round(kb.value, 4)))
# cleanup: remove the freshly imported objects + mesh/armature data
meshes, arms = set(), set()
for o in new:
    if o.type == 'MESH':
        meshes.add(o.data)
    elif o.type == 'ARMATURE':
        arms.add(o.data)
    bpy.data.objects.remove(o, do_unlink=True)
for m in meshes:
    try: bpy.data.meshes.remove(m)
    except Exception: pass
for a in arms:
    try: bpy.data.armatures.remove(a)
    except Exception: pass
print("FRESH import of FINAL.fbx — shape-key values Blender assigns:\n" + ("\n".join(sorted(set(res))) or "none"))
'''

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "scene"
    if what == "verify":
        p = sys.argv[2]
        code = r'''
import bpy
path = r"%s"
before = set(bpy.data.objects.keys())
bpy.ops.import_scene.fbx(filepath=path)
new = [o for o in bpy.data.objects if o.name not in before]
meshes = sorted(o.name for o in new if o.type == 'MESH')
nonzero = []
faceflat = True
for o in new:
    if o.type == 'MESH' and o.data.shape_keys:
        for kb in o.data.shape_keys.key_blocks:
            if abs(kb.value) > 1e-4:
                nonzero.append("%%s :: %%s = %%.3f" %% (o.name, kb.name, kb.value))
# cleanup
md, ad = set(), set()
for o in new:
    if o.type == 'MESH': md.add(o.data)
    elif o.type == 'ARMATURE': ad.add(o.data)
    bpy.data.objects.remove(o, do_unlink=True)
for m in md:
    try: bpy.data.meshes.remove(m)
    except Exception: pass
for a in ad:
    try: bpy.data.armatures.remove(a)
    except Exception: pass
print("MESHES (%%d): %%s" %% (len(meshes), ", ".join(meshes)))
print("NON-ZERO shape keys (%%d):" %% len(nonzero))
for s in sorted(nonzero):
    print("  " + s)
''' % p
        print(send({"type": "execute_code", "params": {"code": code}}, timeout=60))
        sys.exit(0)
    if what == "reimport":
        p = sys.argv[2] if len(sys.argv) > 2 else r"D:/2files/models/vrc/CC/Blender-MHWilds/RigForge/Output/_fixtest/FBX_MANUKA__maya_FINAL.fbx"
        print(send({"type": "execute_code", "params": {"code": _REIMPORT_CODE.replace("PATH_PLACEHOLDER", p)}}, timeout=60))
        sys.exit(0)
    if what == "scene":
        print(send({"type": "get_scene_info"}))
    elif what == "shapekeys":
        code = r'''
import bpy
lines = []
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.data.shape_keys:
        for kb in obj.data.shape_keys.key_blocks:
            n = kb.name.lower()
            if 'high heel' in n or 'big breast' in n:
                lines.append("%s :: %s = %s (min=%s max=%s)" % (
                    obj.name, kb.name, round(kb.value, 4), kb.slider_min, kb.slider_max))
report = "\n".join(lines) if lines else "NO high-heel / big-breast shape keys found"
with open(r"D:/2files/models/vrc/CC/Blender-MHWilds/RigForge/Output/_blender_dump.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(report)
'''
        print(send({"type": "execute_code", "params": {"code": code}}))
