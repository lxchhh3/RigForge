"""Chained-compose id offsets must stay inside FBX's signed 64-bit ids.

Each garment's ids are shifted clear of everything already in the running
merge. The old rule took a power of 10 with one SPARE digit above the max id
(2.2e12 -> 1e14), so every garment added two digits: 1e14, 1e16, 1e18, 1e20.
The 4th garment of kevin's save_08 compose (Riptide_Watch) got Model id
100000002729049483520 > 2**63-1; the merged ASCII still held it, and the
binary conversion dropped it without a word.
"""
from __future__ import annotations

import pytest

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.pipeline.phase_c import PhaseCError, _compute_safe_offset

_INT64_MAX = 2**63 - 1
# Real magnitudes from the save_08 run: maya's largest id, then the four
# garments' largest source ids (RabbitGear, SimpleLeotard, Chrome_Maya, Watch).
_MAYA_MAX = 2_226_623_424_832
_GARMENT_MAX = [2_722_083_983_504, 1_976_796_889_664, 2_374_835_510_944, 2_729_051_005_216]


def _view(*ids: int):
    models = "".join(f'\tModel: {i}, "Model::m{i}", "Mesh" {{\n\t}}\n' for i in ids)
    return extract(parse(
        f"; FBX 7.4.0 project file\nObjects:  {{\n{models}}}\nConnections:  {{\n}}\n".encode()
    ))


def test_offset_clears_every_target_id_by_a_small_step():
    offset = _compute_safe_offset(_view(17, _MAYA_MAX))
    assert offset > _MAYA_MAX
    # Rounded up at the max id's leading digit, not pushed a whole digit past it.
    assert offset == 3_000_000_000_000


def test_four_garment_chain_stays_inside_int64():
    """The save_08 chain: every garment's shifted ids fit in an FBX id, with
    room for many more garments after them."""
    running_max = _MAYA_MAX
    for garment_max in _GARMENT_MAX * 3:   # 12 garments
        offset = _compute_safe_offset(_view(running_max), _view(garment_max))
        assert offset > running_max
        running_max = offset + garment_max
    assert running_max < 10**15 < _INT64_MAX


def test_offset_refuses_ids_past_int64():
    """When shifted ids cannot fit, fail loudly instead of writing a file the
    binary conversion silently drops the garment from."""
    with pytest.raises(PhaseCError, match="64-bit"):
        _compute_safe_offset(_view(9_000_000_000_000_000_000), _view(2_729_051_005_216))
