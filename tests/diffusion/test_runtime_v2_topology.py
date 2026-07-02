# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm_omni.diffusion.runtime_v2.protocol import ExecutionGroupSpec, ParallelSpec, TaskKind
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology, WorkerSpec

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def test_single_group_topology_two_gpus():
    ps = ParallelSpec(tp=2)
    topo = RuntimeTopology.single_group(num_gpus=2, parallel_spec=ps)

    assert len(topo.workers) == 2
    assert topo.workers[0] == WorkerSpec(worker_rank=0, device_id=0)
    assert topo.workers[1] == WorkerSpec(worker_rank=1, device_id=1)

    g = topo.get_group("g0")
    assert g.group_id == "g0"
    assert g.ranks == (0, 1)
    assert g.parallel_spec is ps
    assert TaskKind.DIT_STEP_CHUNK in g.supported_task_kinds
    # all TaskKind values must be present
    for kind in TaskKind:
        assert kind in g.supported_task_kinds, f"{kind} missing from supported_task_kinds"

    assert topo.get_group_leader("g0") == 0
