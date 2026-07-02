# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PR1 single-rank guard in RuntimeV2Runner._build_topology.

The guard rejects any parallel_config that would form a multi-rank execution
group (tp>1, sp>1, cfg>1, or num_gpus!=1) and raises NotImplementedError with
a clear message. The single-rank (num_gpus=1, tp=1, sp=1, cfg=1) path must not
raise.

These tests call _build_topology directly (a static method), so they never touch
worker pools, GPU processes, or model checkpoints.
"""

import pytest
from types import SimpleNamespace

from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def _make_od_config(*, tp=1, sp=1, cfg=1, num_gpus=1):
    """Construct a minimal od_config SimpleNamespace matching _build_topology's expectations."""
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tp,
        sequence_parallel_size=sp,
        cfg_parallel_size=cfg,
        world_size=tp * sp * cfg,
    )
    return SimpleNamespace(
        parallel_config=parallel_config,
        num_gpus=num_gpus,
    )


# --------------------------------------------------------------------------- #
# Guard: multi-rank configs must raise                                         #
# --------------------------------------------------------------------------- #

def test_guard_raises_on_sp_gt_1():
    od_config = _make_od_config(sp=2, num_gpus=2)
    with pytest.raises(NotImplementedError, match="runtime_v2 PR1 supports single-rank"):
        RuntimeV2Runner._build_topology(od_config)


def test_guard_raises_on_tp_gt_1():
    od_config = _make_od_config(tp=2, num_gpus=2)
    with pytest.raises(NotImplementedError, match="runtime_v2 PR1 supports single-rank"):
        RuntimeV2Runner._build_topology(od_config)


def test_guard_raises_on_cfg_gt_1():
    od_config = _make_od_config(cfg=2, num_gpus=2)
    with pytest.raises(NotImplementedError, match="runtime_v2 PR1 supports single-rank"):
        RuntimeV2Runner._build_topology(od_config)


def test_guard_raises_on_num_gpus_2():
    """num_gpus=2 with SP1/TP1/CFG1 must still be rejected."""
    od_config = _make_od_config(num_gpus=2)
    with pytest.raises(NotImplementedError, match="Multi-rank groups require the artifact codec"):
        RuntimeV2Runner._build_topology(od_config)


def test_guard_message_includes_degrees():
    """Error message must embed the actual tp/sp/cfg/num_gpus values."""
    od_config = _make_od_config(tp=2, sp=4, cfg=1, num_gpus=8)
    with pytest.raises(NotImplementedError, match=r"tp=2 sp=4 cfg=1 num_gpus=8"):
        RuntimeV2Runner._build_topology(od_config)


# --------------------------------------------------------------------------- #
# Happy path: single-rank config must NOT raise                                #
# --------------------------------------------------------------------------- #

def test_single_rank_does_not_raise():
    """num_gpus=1, tp=1, sp=1, cfg=1 must succeed and return a topology."""
    od_config = _make_od_config(num_gpus=1, tp=1, sp=1, cfg=1)
    topology = RuntimeV2Runner._build_topology(od_config)
    assert len(topology.groups) == 1
    assert len(topology.workers) == 1
