"""Geometry subsystem — taste-level mesh edits via a headless Blender backend.

RigForge's core never touches vertex positions (see ARCHITECTURE.md "Out of
scope"); this package is the deliberate, isolated exception. It drives Blender
as a headless geometry *library* over the same subprocess pattern the FBX SDK
uses (`rigforge.ascii_fbx.convert`), so the main env stays Blender-free.

Phase 0 scope: prove a no-op Blender round-trip preserves a clothing FBX
(verts, skin weights, shape keys, orientation) well enough to keep flowing
through the existing assemble pipeline. Nothing here edits geometry yet.
"""
