# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU construct/import tests for the runtime_v2 MultiprocWorkerPool.

Process spawn + model load require GPUs and a model checkpoint, so those are
covered by the GPU smoke test. These tests only verify that the module
imports, the pool constructs without ``.start()``, the artifact-fetch API and
the migration no-op shims are present, and the artifact (de)serialization
helpers round-trip -- all on CPU with no subprocess and no fake worker.
"""

import contextlib
import os
import signal
from unittest.mock import Mock, patch

import pytest

from dataclasses import fields

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.runtime_v2.multiproc_worker import (
    FetchArtifactsResult,
    MultiprocWorkerPool,
    _clone_diffusion_output_for_transport,
    _deserialize_artifact_value,
    _serialize_artifact_value,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ArtifactValue,
    ParallelSpec,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def _minimal_od_config() -> Mock:
    od_config = Mock()
    od_config.master_port = 30005
    return od_config


def test_pool_constructs_without_start():
    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    pool = MultiprocWorkerPool(topology=topo, od_config=_minimal_od_config())

    # Construction must not spawn any workers (that needs GPUs + a checkpoint).
    assert pool.worker_handles == {}
    # Topology was serialized into a plain payload at construct time.
    assert pool._execution_groups_payload == [
        {"group_id": "g0", "ranks": [0], "tp": 1, "sp": 1, "cfg": 1, "ulysses_degree": 1, "ring_degree": 1}
    ]


def test_pool_exposes_fetch_and_migration_api():
    topo = RuntimeTopology.single_group(num_gpus=2, parallel_spec=ParallelSpec(tp=2))
    pool = MultiprocWorkerPool(topology=topo, od_config=_minimal_od_config())

    # Scheduler terminal-output path depends on all three fetch entry points.
    assert callable(pool.fetch_artifacts)
    assert callable(pool.start_fetch_artifacts)
    assert callable(pool.poll_fetch_artifacts)
    # Core dispatch/poll/evict/lifecycle surface.
    for name in ("dispatch", "poll", "evict_request", "start", "shutdown"):
        assert callable(getattr(pool, name))

    # Migration is stripped for PR1: the shims must exist and be no-ops.
    assert pool.has_pending_migrations() is False
    pool.pump_migrations()  # no raise, no side effects without workers


def test_fetch_artifacts_result_defaults():
    result = FetchArtifactsResult(request_id="req-1", worker_rank=0)
    assert result.fetch_id == ""
    assert result.artifacts == ()
    assert result.error is None


def test_clone_for_transport_preserves_all_fields():
    """The SHM-transport clone must carry EVERY DiffusionOutput field.

    Previously the clone enumerated only a subset (output/trajectory_timesteps/
    trajectory_latents/trajectory_decoded/error/post_process_func/custom_output),
    silently dropping trajectory_log_probs, error_status_code, error_type,
    aborted, abort_message, finished, chunk_index, total_chunks, stage_durations
    and peak_memory_mb -- so the SHM path did NOT match the pickle path. Build a
    DiffusionOutput with every field set to a NON-default value and assert the
    clone reproduces all of them.
    """
    original = DiffusionOutput(
        output=torch.zeros(2, 3),
        trajectory_timesteps=torch.ones(4),
        trajectory_latents=torch.ones(2, 2),
        trajectory_log_probs=torch.ones(3),
        trajectory_decoded=[],
        error="boom",
        error_status_code=503,
        error_type="service_unavailable",
        aborted=True,
        abort_message="user aborted",
        custom_output={"k": "v"},
        finished=False,
        chunk_index=7,
        total_chunks=9,
        stage_durations={"dit": 1.5},
        peak_memory_mb=123.0,
    )

    clone = _clone_diffusion_output_for_transport(original)

    # Every dataclass field must survive the clone (no silent drops). Exclude
    # to_cpu, which is a construction-time directive rather than carried state.
    for f in fields(DiffusionOutput):
        if f.name == "to_cpu":
            continue
        orig_val = getattr(original, f.name)
        clone_val = getattr(clone, f.name)
        if isinstance(orig_val, torch.Tensor):
            assert torch.equal(clone_val, orig_val), f"tensor field {f.name} not preserved"
        else:
            assert clone_val == orig_val, f"field {f.name} not preserved: {clone_val!r} != {orig_val!r}"

    # Sanity spot-checks on the fields that used to be dropped.
    assert clone.trajectory_log_probs is not None
    assert clone.error_status_code == 503
    assert clone.error_type == "service_unavailable"
    assert clone.aborted is True
    assert clone.abort_message == "user aborted"
    assert clone.finished is False
    assert clone.chunk_index == 7
    assert clone.total_chunks == 9
    assert clone.peak_memory_mb == 123.0


def test_serialize_round_trip_pickle_transport():
    handle = ArtifactHandle(
        request_id="req-1",
        artifact_id="latent",
        kind=ArtifactKind.TENSOR,
        layout=ArtifactLayout.WORKER_LOCAL,
    )
    original = ArtifactValue(handle=handle, value={"data": [1, 2, 3]})

    serialized = _serialize_artifact_value(original)
    assert serialized.transport == "pickle"
    assert serialized.handle == handle

    restored = _deserialize_artifact_value(serialized)
    assert restored.handle == handle
    assert restored.value == {"data": [1, 2, 3]}


# ----------------------------------------------------------------------------
# prefer_shm_output serialization: large tensors nested inside output (a dict/
# tuple) or in trajectory_timesteps/log_probs must ride the SHM transport, and
# the receiver's unpack must unlink every segment (no /dev/shm leak). The old
# top-level-only handle check pickled these full tensors through the queue AND
# orphaned the SHM segments pack_diffusion_output_shm had already created.
# ----------------------------------------------------------------------------

# 1.2 MB > the 1 MB _SHM_TENSOR_THRESHOLD, so packing actually creates a segment.
_LARGE_NUMEL = 300_000


def _shm_names() -> set[str]:
    try:
        return set(os.listdir("/dev/shm"))
    except FileNotFoundError:  # pragma: no cover - non-Linux fallback
        return set()


def _unlink_shm_by_name(name: str) -> None:
    """Best-effort unlink of a POSIX SHM segment by name (test cleanup)."""
    from multiprocessing import shared_memory

    with contextlib.suppress(FileNotFoundError):
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()


def _output_handle(request_id: str) -> ArtifactHandle:
    return ArtifactHandle(
        request_id=request_id,
        artifact_id="out",
        kind=ArtifactKind.OUTPUT,  # prefer_shm_output only fires for OUTPUT kind
        layout=ArtifactLayout.HOST,
    )


@pytest.mark.parametrize(
    "build, check",
    [
        pytest.param(
            lambda t: DiffusionOutput(output={"image": t}),
            lambda out, t: torch.testing.assert_close(out.output["image"], t),
            id="nested-dict-output",
        ),
        pytest.param(
            lambda t: DiffusionOutput(output=(t,)),
            lambda out, t: torch.testing.assert_close(out.output[0], t),
            id="tuple-output",
        ),
        pytest.param(
            lambda t: DiffusionOutput(output=torch.zeros(2), trajectory_timesteps=t),
            lambda out, t: torch.testing.assert_close(out.trajectory_timesteps, t),
            id="trajectory-timesteps",
        ),
    ],
)
def test_serialize_shm_transport_round_trips_without_leak(build, check):
    tensor = torch.arange(_LARGE_NUMEL, dtype=torch.float32)
    original = ArtifactValue(handle=_output_handle("req-shm"), value=build(tensor))

    before = _shm_names()
    serialized = _serialize_artifact_value(original, prefer_shm_output=True)
    # The segments THIS serialize created -- captured by name right after the
    # pack so the leak/transport assertions compare only our own segments and a
    # concurrent unrelated segment can't flake them (shared /dev/shm namespace).
    created = _shm_names() - before
    try:
        # The nested/trajectory large tensor is detected -> real SHM transport,
        # NOT a silent fallback to pickling the whole tensor through the queue.
        assert serialized.transport == "shm"
        assert created, "expected pack to create at least one /dev/shm segment"

        # Receiver side unpacks the handles and unlinks the segments.
        restored = _deserialize_artifact_value(serialized, unpack_shm=True)
        assert isinstance(restored.value, DiffusionOutput)
        check(restored.value, tensor)

        # Every segment this serialize created was unlinked by the receiver.
        assert created & _shm_names() == set()
    finally:
        # If an assertion above fired before the receiver unlinked them, clean up
        # our segments so a failure can't pollute the session's /dev/shm.
        for name in created & _shm_names():
            _unlink_shm_by_name(name)


def test_serialize_small_output_stays_pickle_and_creates_no_segment():
    # Everything is below threshold: no handles, so the SHM path is skipped and
    # the value is pickled inline -- and crucially no segment is orphaned.
    original = ArtifactValue(
        handle=_output_handle("req-small"),
        value=DiffusionOutput(output={"image": torch.zeros(2, 3)}),
    )

    before = _shm_names()
    serialized = _serialize_artifact_value(original, prefer_shm_output=True)
    # Measured immediately after serialize (tight window) and by-name, so an
    # unrelated concurrent segment does not make this flake.
    created = _shm_names() - before
    try:
        assert serialized.transport == "pickle"
        assert created == set()

        restored = _deserialize_artifact_value(serialized, unpack_shm=True)
        torch.testing.assert_close(restored.value.output["image"], torch.zeros(2, 3))
    finally:
        for name in created:
            _unlink_shm_by_name(name)


def test_worker_installs_parent_death_signal():
    """The runtime_v2 GPU worker must tie its lifetime to the scheduler proc:
    register a SIGTERM handler AND arm PR_SET_PDEATHSIG(SIGTERM), so it dies with
    the scheduler instead of holding GPU memory / hanging mid-collective."""
    from vllm_omni.diffusion.runtime_v2.multiproc_worker import _WorkerProcessRuntime

    rt = object.__new__(_WorkerProcessRuntime)
    with (
        patch("signal.signal") as sig_signal,
        patch("vllm_omni.engine.stage_init_utils.set_death_signal") as set_death,
    ):
        rt._install_parent_death_signal()

    assert any(c.args and c.args[0] == signal.SIGTERM for c in sig_signal.call_args_list)
    set_death.assert_called_once_with(signal.SIGTERM)


def test_discard_fetch_result_shm_unlinks_stale_segment():
    """A stale fetch result (request aborted before draining) must have its
    packed POSIX-SHM segment unlinked, not leaked until worker exit."""
    tensor = torch.arange(_LARGE_NUMEL, dtype=torch.float32)
    original = ArtifactValue(handle=_output_handle("req-stale"), value=DiffusionOutput(output=tensor))

    before = _shm_names()
    sav = _serialize_artifact_value(original, prefer_shm_output=True)
    assert sav.transport == "shm"
    created = _shm_names() - before
    assert created, "expected pack to create a /dev/shm segment"
    try:
        result = FetchArtifactsResult(request_id="req-stale", worker_rank=0, artifacts=(sav,))
        MultiprocWorkerPool._discard_fetch_result_shm(result)
        # The dropped result's segment was unlinked (leak closed).
        assert created & _shm_names() == set()
    finally:
        for name in created & _shm_names():
            _unlink_shm_by_name(name)


def test_discard_fetch_result_shm_ignores_pickle_transport():
    """A pickle-transport artifact has no SHM; discarding it must be a harmless
    no-op (no raise, no attempt to open a segment)."""
    original = ArtifactValue(handle=_output_handle("req-pickle"), value={"data": [1, 2, 3]})
    sav = _serialize_artifact_value(original)  # prefer_shm_output=False -> pickle
    assert sav.transport == "pickle"
    result = FetchArtifactsResult(request_id="req-pickle", worker_rank=0, artifacts=(sav,))
    MultiprocWorkerPool._discard_fetch_result_shm(result)  # no raise


def test_discard_fetch_unlinks_completed_result_shm():
    """discard_fetch (abort/cleanup) on an ALREADY-completed fetch must unlink its
    packed POSIX-SHM segment, not just drop the bookkeeping (leak until exit)."""
    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    pool = MultiprocWorkerPool(topology=topo, od_config=_minimal_od_config())

    tensor = torch.arange(_LARGE_NUMEL, dtype=torch.float32)
    original = ArtifactValue(handle=_output_handle("req-abort"), value=DiffusionOutput(output=tensor))
    before = _shm_names()
    sav = _serialize_artifact_value(original, prefer_shm_output=True)
    assert sav.transport == "shm"
    created = _shm_names() - before
    assert created, "expected a /dev/shm segment"
    try:
        # No _inflight_fetches entry -> leader_rank is None -> the rank-drain is
        # skipped; only the completed-result unlink path is exercised.
        pool._completed_fetches["fetch-1"] = FetchArtifactsResult(
            request_id="req-abort", worker_rank=0, artifacts=(sav,)
        )
        pool.discard_fetch("fetch-1")
        assert created & _shm_names() == set()
        assert "fetch-1" not in pool._completed_fetches
    finally:
        for name in created & _shm_names():
            _unlink_shm_by_name(name)


def test_shutdown_unlinks_stranded_completed_fetch_shm():
    """A completed-but-never-drained fetch (e.g. last request aborted) must have
    its SHM reclaimed at shutdown -- the segments outlive the workers."""
    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    pool = MultiprocWorkerPool(topology=topo, od_config=_minimal_od_config())

    tensor = torch.arange(_LARGE_NUMEL, dtype=torch.float32)
    original = ArtifactValue(handle=_output_handle("req-x"), value=DiffusionOutput(output=tensor))
    before = _shm_names()
    sav = _serialize_artifact_value(original, prefer_shm_output=True)
    created = _shm_names() - before
    assert created, "expected a /dev/shm segment"
    try:
        pool._completed_fetches["fetch-x"] = FetchArtifactsResult(
            request_id="req-x", worker_rank=0, artifacts=(sav,)
        )
        pool.shutdown()  # no workers spawned -> exercises only the cleanup path
        assert created & _shm_names() == set()
    finally:
        for name in created & _shm_names():
            _unlink_shm_by_name(name)


def test_shutdown_unlinks_stranded_result_queue_shm():
    """A late fetch result the reader queued but no poll ever drained (e.g. the
    last request was aborted, so its rank is never polled) must have its SHM
    reclaimed at shutdown, not dropped by _result_queues.clear()."""
    import queue as _queue

    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    pool = MultiprocWorkerPool(topology=topo, od_config=_minimal_od_config())

    tensor = torch.arange(_LARGE_NUMEL, dtype=torch.float32)
    original = ArtifactValue(handle=_output_handle("req-q"), value=DiffusionOutput(output=tensor))
    before = _shm_names()
    sav = _serialize_artifact_value(original, prefer_shm_output=True)
    created = _shm_names() - before
    assert created, "expected a /dev/shm segment"
    try:
        rank_queue: _queue.Queue = _queue.Queue()
        rank_queue.put(FetchArtifactsResult(request_id="req-q", worker_rank=0, artifacts=(sav,)))
        pool._result_queues[0] = rank_queue
        pool.shutdown()  # no workers/reader spawned -> exercises the drain+unlink path
        assert created & _shm_names() == set()
    finally:
        for name in created & _shm_names():
            _unlink_shm_by_name(name)
