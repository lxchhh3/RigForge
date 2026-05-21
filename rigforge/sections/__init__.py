"""Cross-document section merging (v2).

Materials + BlendShape channels are merged across donor + target by name. The
package emits TextEdit-shaped operations against the TARGET document, kept in
a `MergePlan`. The integration pass (pipeline) converts those into proper
`TextEdit` instances and composes them with the rest of Phase C's edits.

This package is intentionally decoupled from `rigforge.ascii_fbx.edits` so
the merge logic can be developed and tested in isolation.
"""
from .merge import (
    MergeEdit,
    MergePlan,
    merge_blendshapes,
    merge_materials,
)

__all__ = [
    "MergeEdit",
    "MergePlan",
    "merge_blendshapes",
    "merge_materials",
]
