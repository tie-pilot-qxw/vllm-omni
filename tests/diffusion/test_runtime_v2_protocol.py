# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from vllm_omni.diffusion.runtime_v2.protocol import (
    TaskKind, TaskStatus, ParallelSpec, ExecutionGroupSpec, StepRange,
    ArtifactKind, ArtifactLayout, ArtifactHandle, InferenceTask,
    RequestExecutionPlan, WorkerEvent, WorkerEventKind,
)
pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]

def test_inference_task_is_hashable_by_id_and_carries_step_range():
    h = ArtifactHandle(request_id="r", artifact_id="state", kind=ArtifactKind.REQUEST_STATE,
                       layout=ArtifactLayout.WORKER_LOCAL, codec_id="state")
    t = InferenceTask(task_id="r:dit:0", request_id="r", kind=TaskKind.DIT_STEP_CHUNK,
                      group_id="g0", parallel_spec=ParallelSpec(sp=4),
                      dependencies=("r:prep",), inputs=(h,), outputs=(h,),
                      step_range=StepRange(0, 5))
    assert t.kind is TaskKind.DIT_STEP_CHUNK
    assert t.step_range.end == 5
    assert t.parallel_spec.sp == 4
    assert "r:prep" in t.dependencies

def test_plan_terminal_ids_resolve_to_tasks():
    fin = InferenceTask(task_id="r:fin", request_id="r", kind=TaskKind.FINALIZE,
                        group_id="g0", parallel_spec=ParallelSpec())
    plan = RequestExecutionPlan(request_id="r", tasks={fin.task_id: fin},
                                terminal_task_ids=(fin.task_id,))
    assert all(tid in plan.tasks for tid in plan.terminal_task_ids)

def test_reshard_kind_exists_for_protocol_stability():
    # PR1 does not dispatch RESHARD, but the symbol must exist so later PRs do not churn the protocol.
    assert TaskKind.RESHARD.value == "reshard"
