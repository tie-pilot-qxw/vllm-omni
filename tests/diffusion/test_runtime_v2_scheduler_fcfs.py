# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from types import SimpleNamespace
from vllm_omni.diffusion.runtime_v2.protocol import (
    InferenceTask, RequestExecutionPlan, TaskKind, ParallelSpec,
    ArtifactHandle, ArtifactKind, ArtifactLayout, ArtifactValue, WorkerLocalArtifactRef,
    WorkerEvent, WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.scheduler import GlobalScheduler, InMemoryArtifactStore
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology
from vllm_omni.diffusion.runtime_v2.policies.fcfs import FCFSSchedulerPolicy
pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]

class _EchoPool:
    """Mirrors the real worker pool: publishes outputs on TASK_LAUNCH_END (deps), then
    TASK_EXEC_END; materializes terminal outputs via fetch_artifacts."""
    def __init__(self): self._q = []; self._values = {}
    def dispatch(self, task, inline_inputs, release_after_exec_artifact_ids=()):
        refs = tuple(WorkerLocalArtifactRef(handle=h, group_id=task.group_id or "g0", worker_rank=0)
                     for h in task.outputs)
        for h in task.outputs:
            self._values[(task.request_id, h.artifact_id)] = f"value:{h.artifact_id}"
        self._q.append(WorkerEvent(event_id=task.task_id + ":le", task_id=task.task_id,
            request_id=task.request_id, group_id=task.group_id or "g0", worker_rank=0,
            kind=WorkerEventKind.TASK_LAUNCH_END, timestamp_ns=0,
            metadata={"published_outputs": refs}))
        self._q.append(WorkerEvent(event_id=task.task_id + ":ee", task_id=task.task_id,
            request_id=task.request_id, group_id=task.group_id or "g0", worker_rank=0,
            kind=WorkerEventKind.TASK_EXEC_END, timestamp_ns=0, metadata={}))
    def poll(self, timeout_s=0.0):
        out, self._q = self._q, []; return out
    def fetch_artifacts(self, request_id, group_id, artifact_ids):
        # duck-typed FetchArtifactsResult (real type: multiproc_worker.FetchArtifactsResult)
        arts = tuple(ArtifactValue(
            handle=ArtifactHandle(request_id=request_id, artifact_id=aid,
                                  kind=ArtifactKind.OUTPUT, layout=ArtifactLayout.WORKER_LOCAL),
            value=self._values.get((request_id, aid))) for aid in artifact_ids)
        return SimpleNamespace(request_id=request_id, worker_rank=0, artifacts=arts, error=None)
    def evict_request(self, request_id): pass
    def pump_migrations(self): pass
    def has_pending_migrations(self): return False

class _TwoTaskCompiler:
    def compile_request(self, request):
        prep_out = ArtifactHandle(request_id="r", artifact_id="state", kind=ArtifactKind.REQUEST_STATE,
                                  layout=ArtifactLayout.WORKER_LOCAL)
        prep = InferenceTask(task_id="r:prep", request_id="r", kind=TaskKind.DIT_PREPARE,
                             group_id="g0", parallel_spec=ParallelSpec(), outputs=(prep_out,))
        fin_out = ArtifactHandle(request_id="r", artifact_id="out", kind=ArtifactKind.OUTPUT,
                                 layout=ArtifactLayout.WORKER_LOCAL)
        fin = InferenceTask(task_id="r:fin", request_id="r", kind=TaskKind.FINALIZE,
                            group_id="g0", parallel_spec=ParallelSpec(),
                            dependencies=("r:prep",), inputs=(prep_out,), outputs=(fin_out,))
        return RequestExecutionPlan(request_id="r", tasks={"r:prep": prep, "r:fin": fin},
                                    terminal_task_ids=("r:fin",))

def test_fcfs_runs_two_task_plan_to_finished():
    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    sched = GlobalScheduler(topology=topo, worker_pool=_EchoPool(),
                            compiler=_TwoTaskCompiler(), artifact_store=InMemoryArtifactStore(),
                            policy=FCFSSchedulerPolicy(topo))
    rid = sched.submit_request(object())
    for _ in range(20):
        status, output = sched.get_request_status(rid)
        if status == "finished":
            assert output is not None        # terminal output materialized via fetch_artifacts
            break
        sched.poll_once(timeout_s=0.0)
    assert sched.get_request_status(rid)[0] == "finished"


class _RecordingPool:
    """A worker pool that RECORDS dispatches but never completes them.

    Unlike _EchoPool, it emits NO worker events, so a dispatched request stays
    the group's active request in the FCFS policy until aborted -- exactly the
    state needed to prove abort frees the active slot and promotes the next
    request. ``dispatched_request_ids`` is the ordered list of request ids the
    scheduler dispatched a task for.
    """

    def __init__(self):
        self.dispatched_request_ids = []

    def dispatch(self, task, inline_inputs, release_after_exec_artifact_ids=()):
        self.dispatched_request_ids.append(task.request_id)

    def poll(self, timeout_s=0.0):
        return []

    def evict_request(self, request_id):
        pass

    def pump_migrations(self):
        pass

    def has_pending_migrations(self):
        return False


class _SingleTaskCompiler:
    """Compiles a one-task plan pinned to group g0, keyed by the request's id.

    The request passed to submit_request carries a ``request_id`` attribute
    (a SimpleNamespace), so distinct requests get distinct plans in group g0.
    """

    def compile_request(self, request):
        rid = request.request_id
        out = ArtifactHandle(request_id=rid, artifact_id="out", kind=ArtifactKind.OUTPUT,
                             layout=ArtifactLayout.WORKER_LOCAL)
        task = InferenceTask(task_id=f"{rid}:t", request_id=rid, kind=TaskKind.DIT_PREPARE,
                             group_id="g0", parallel_spec=ParallelSpec(), outputs=(out,))
        return RequestExecutionPlan(request_id=rid, tasks={f"{rid}:t": task},
                                    terminal_task_ids=(f"{rid}:t",))


def _make_scheduler_with_recording_pool():
    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    policy = FCFSSchedulerPolicy(topo)
    sched = GlobalScheduler(topology=topo, worker_pool=_RecordingPool(),
                            compiler=_SingleTaskCompiler(), artifact_store=InMemoryArtifactStore(),
                            policy=policy)
    return sched, policy


def test_abort_active_request_frees_group_slot_and_promotes_next():
    """Aborting the active request must free its FCFS group slot so the next
    request is dispatched (NOT parked forever)."""
    sched, policy = _make_scheduler_with_recording_pool()

    # Submit A -> becomes the active request in group g0 and is dispatched.
    sched.submit_request(SimpleNamespace(request_id="A"))
    assert policy.active_request_by_group.get("g0") == "A"
    assert sched.worker_pool.dispatched_request_ids == ["A"]

    # Submit B -> group g0 is busy with A, so B is PARKED (not dispatched).
    sched.submit_request(SimpleNamespace(request_id="B"))
    assert policy.active_request_by_group.get("g0") == "A"
    assert list(policy.pending_requests_by_group.get("g0", [])) == ["B"]
    assert sched.worker_pool.dispatched_request_ids == ["A"]  # B not yet dispatched

    # Abort A -> its slot is freed and B is promoted to active + dispatched.
    sched.abort_request("A")
    assert policy.active_request_by_group.get("g0") == "B"
    assert not list(policy.pending_requests_by_group.get("g0", []))
    assert sched.worker_pool.dispatched_request_ids == ["A", "B"]
    # A's controller state is retired.
    assert "A" not in sched.plans


def test_abort_active_request_evicts_before_dispatching_promoted_request():
    """Abort must queue the aborted request's eviction BEFORE dispatching the
    promoted next request. Worker command pipes are FIFO, so evicting first frees
    the aborted request's large local artifacts before the next request runs
    (avoiding an OOM). Regression guard on the ordering of evict_request vs
    dispatch during abort."""
    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    policy = FCFSSchedulerPolicy(topo)

    calls: list = []

    class _OrderPool:
        def dispatch(self, task, inline_inputs, release_after_exec_artifact_ids=()):
            calls.append(("dispatch", task.request_id))

        def poll(self, timeout_s=0.0):
            return []

        def evict_request(self, request_id):
            calls.append(("evict", request_id))

        def pump_migrations(self):
            pass

        def has_pending_migrations(self):
            return False

    sched = GlobalScheduler(
        topology=topo,
        worker_pool=_OrderPool(),
        compiler=_SingleTaskCompiler(),
        artifact_store=InMemoryArtifactStore(),
        policy=policy,
    )
    sched.submit_request(SimpleNamespace(request_id="A"))  # active + dispatched
    sched.submit_request(SimpleNamespace(request_id="B"))  # queued behind A
    calls.clear()  # isolate the abort sequence

    sched.abort_request("A")

    assert ("evict", "A") in calls, "aborted request must be evicted"
    assert ("dispatch", "B") in calls, "promoted request must be dispatched"
    # FIFO: the aborted request's eviction must be queued before the promoted
    # dispatch so the worker frees A's artifacts before running B.
    assert calls.index(("evict", "A")) < calls.index(("dispatch", "B"))


def test_abort_unknown_request_is_noop():
    """Aborting an unknown / already-finished id must be a no-op (no raise)."""
    sched, policy = _make_scheduler_with_recording_pool()
    # Never-seen id.
    sched.abort_request("ghost")
    assert not policy.active_request_by_group
    assert sched.worker_pool.dispatched_request_ids == []

    # A request that already completed its lifecycle (submitted then released):
    sched.submit_request(SimpleNamespace(request_id="A"))
    sched.release_request("A")
    dispatched_before = list(sched.worker_pool.dispatched_request_ids)
    sched.abort_request("A")  # must not raise or re-dispatch
    assert sched.worker_pool.dispatched_request_ids == dispatched_before


class _StartRecordingPool:
    """Records the timeout GlobalScheduler.start forwards to worker_pool.start."""
    def __init__(self):
        self.start_calls = []
    def start(self, timeout_s=600.0):
        self.start_calls.append(timeout_s)


def _make_start_scheduler():
    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    pool = _StartRecordingPool()
    sched = GlobalScheduler(
        topology=topo,
        worker_pool=pool,
        compiler=SimpleNamespace(),
        artifact_store=InMemoryArtifactStore(),
        policy=FCFSSchedulerPolicy(topo),
    )
    return sched, pool


def test_global_scheduler_start_forwards_timeout():
    """start(timeout_s=X) must forward X to worker_pool.start (so the runner can
    pass od_config.stage_init_timeout for a slow checkpoint)."""
    sched, pool = _make_start_scheduler()
    sched.start(timeout_s=123.0)
    assert pool.start_calls == [123.0]


def test_global_scheduler_start_without_timeout_uses_pool_default():
    """start() with no timeout must leave the worker pool on its own default."""
    sched, pool = _make_start_scheduler()
    sched.start()
    assert pool.start_calls == [600.0]
