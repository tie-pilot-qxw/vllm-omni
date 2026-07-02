# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU wire smoke for the flag-gated runtime_v2 engine entry point.

These tests only verify *construction-time wiring*: that the
``enable_runtime_v2`` flag selects between the legacy
(scheduler + executor + execute_fn) path and the runtime_v2
(``RuntimeV2Runner``) path, that the legacy warmup ``_dummy_run`` is skipped
under the flag, and that the runtime_v2 path leaves the legacy scheduler unset.

Spawning workers + loading a model needs GPUs + a checkpoint (covered by the
GPU smoke test), so the GPU-touching bits (worker-pool start, dummy run,
executor construction) are stubbed with ``unittest.mock`` -- this is a wiring
unit test, not a fake backend.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def _od_config(**overrides) -> OmniDiffusionConfig:
    kwargs = dict(model_class_name="QwenImagePipeline")
    kwargs.update(overrides)
    return OmniDiffusionConfig(**kwargs)


@contextlib.contextmanager
def _stub_process_funcs():
    """Stub the model pre/post-process getters.

    These read a model checkpoint off disk (which a CPU wiring test has no
    access to); they run unconditionally at the very top of
    ``DiffusionEngine.__init__`` before the runtime_v2 branch, so they must be
    neutralized for both the legacy and runtime_v2 construction paths.
    """
    with (
        patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func", return_value=None),
        patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_action_post_process_func", return_value=None),
        patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func", return_value=None),
    ):
        yield


def test_runner_module_imports():
    # The ported runner module must import on CPU with no torch/worker deps.
    import vllm_omni.diffusion.runtime_v2.runner as runner_mod

    assert hasattr(runner_mod, "RuntimeV2Runner")


def test_flag_off_builds_legacy_path():
    """enable_runtime_v2=False -> legacy scheduler/executor, no runner."""
    od_config = _od_config(enable_runtime_v2=False)

    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    with (
        _stub_process_funcs(),
        patch("vllm_omni.diffusion.executor.abstract.DiffusionExecutor.get_class") as get_class,
        patch.object(DiffusionEngine, "_dummy_run") as dummy_run,
    ):
        # Stub the executor class so no GPU process is spawned.
        get_class.return_value = MagicMock()

        engine = DiffusionEngine(od_config)
        try:
            assert engine.enable_runtime_v2 is False
            assert engine.runtime_v2_runner is None
            assert engine.scheduler is not None
            assert engine.executor is not None
            assert engine.execute_fn is not None
            # Legacy path still warms up via _dummy_run.
            dummy_run.assert_called_once()
        finally:
            engine.close()


def test_flag_on_builds_scheduler_proc_client_and_skips_warmup():
    """enable_runtime_v2=True -> scheduler-proc manager + client, no in-thread runner, no _dummy_run.

    The RuntimeV2Runner (scheduler + worker pool) now lives in its OWN process
    (RuntimeV2SchedulerProc), spawned by RuntimeV2SchedulerProcManager. The
    engine only holds a ZMQ client to it -- there is no in-process
    ``runtime_v2_runner`` attribute any more. We mock the manager (it would spawn
    a real subprocess + load a model) and the client, so this stays a CPU wiring
    test.
    """
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    od_config = _od_config(enable_runtime_v2=True)

    fake_manager = MagicMock(name="RuntimeV2SchedulerProcManager")
    fake_manager.addresses.inputs = ["ipc:///tmp/fake-in"]
    fake_manager.addresses.outputs = ["ipc:///tmp/fake-out"]
    fake_client = MagicMock(name="RuntimeV2SchedulerClient")

    with (
        _stub_process_funcs(),
        # The manager would spawn the scheduler subprocess (which builds the
        # RuntimeV2Runner + GPU worker pool and loads a model); the client owns
        # ZMQ sockets. Mock both so no process/socket is created.
        patch(
            "vllm_omni.diffusion.runtime_v2.scheduler_proc.RuntimeV2SchedulerProcManager",
            return_value=fake_manager,
        ) as manager_cls,
        patch(
            "vllm_omni.diffusion.runtime_v2.scheduler_client.RuntimeV2SchedulerClient.from_addresses",
            return_value=fake_client,
        ) as client_from_addresses,
        patch.object(DiffusionEngine, "_dummy_run") as dummy_run,
        patch("vllm_omni.diffusion.executor.abstract.DiffusionExecutor.get_class") as get_class,
    ):
        get_class.return_value = MagicMock()

        engine = DiffusionEngine(od_config)
        try:
            assert engine.enable_runtime_v2 is True
            # The scheduler now lives in a separate process: the engine holds a
            # proc manager + client, and the legacy in-process runner is None.
            assert engine.runtime_v2_runner is None
            assert engine._rv2_proc_manager is fake_manager
            assert engine._rv2_client is fake_client
            # The manager was constructed once (spawns the scheduler proc) and the
            # client was wired to the manager's freshly-allocated addresses.
            manager_cls.assert_called_once()
            client_from_addresses.assert_called_once()
            _, client_kwargs = client_from_addresses.call_args
            assert client_kwargs["request_address"] == "ipc:///tmp/fake-in"
            assert client_kwargs["response_address"] == "ipc:///tmp/fake-out"
            assert client_kwargs["proc_manager"] is fake_manager
            # Legacy scheduling surface is intentionally unset so a stray
            # reference fails loudly instead of silently using the wrong path.
            assert engine.scheduler is None
            assert engine.executor is None
            assert engine.execute_fn is None
            # Warmup must be skipped under the flag (lazy warmup for PR1).
            dummy_run.assert_not_called()
        finally:
            engine.close()


def _build_flag_on_engine(fake_manager, fake_client, **od_overrides):
    """Construct a runtime_v2 DiffusionEngine with the proc/client mocked.

    Returns an engine whose scheduler proc manager + client are the provided
    mocks, so no real subprocess/socket is created. Caller owns ``engine.close()``.
    """
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    od_config = _od_config(enable_runtime_v2=True, **od_overrides)
    with (
        _stub_process_funcs(),
        patch(
            "vllm_omni.diffusion.runtime_v2.scheduler_proc.RuntimeV2SchedulerProcManager",
            return_value=fake_manager,
        ),
        patch(
            "vllm_omni.diffusion.runtime_v2.scheduler_client.RuntimeV2SchedulerClient.from_addresses",
            return_value=fake_client,
        ),
        patch.object(DiffusionEngine, "_dummy_run"),
        patch("vllm_omni.diffusion.executor.abstract.DiffusionExecutor.get_class") as get_class,
    ):
        get_class.return_value = MagicMock()
        return DiffusionEngine(od_config)


def _fake_manager_and_client():
    fake_manager = MagicMock(name="RuntimeV2SchedulerProcManager")
    fake_manager.addresses.inputs = ["ipc:///tmp/fake-in"]
    fake_manager.addresses.outputs = ["ipc:///tmp/fake-out"]
    fake_client = MagicMock(name="RuntimeV2SchedulerClient")
    # add_request is awaited by _add_request_runtime_v2; make it an async no-op.
    fake_client.add_request = AsyncMock()
    return fake_manager, fake_client


def test_is_backend_dead_runtime_v2_client_engine_dead():
    """runtime_v2: is_backend_dead() is True when the scheduler client reports dead."""
    fake_manager, fake_client = _fake_manager_and_client()
    # Healthy proc so only the client's engine_dead flag decides.
    fake_manager.proc.is_alive.return_value = True
    fake_client.engine_dead = False

    engine = _build_flag_on_engine(fake_manager, fake_client)
    try:
        assert engine.is_backend_dead() is False
        fake_client.engine_dead = True
        assert engine.is_backend_dead() is True
    finally:
        engine.close()


def test_is_backend_dead_runtime_v2_proc_not_alive():
    """runtime_v2: is_backend_dead() is True when the scheduler proc is not alive."""
    fake_manager, fake_client = _fake_manager_and_client()
    fake_client.engine_dead = False
    fake_manager.proc.is_alive.return_value = True

    engine = _build_flag_on_engine(fake_manager, fake_client)
    try:
        assert engine.is_backend_dead() is False
        # Proc died silently (SIGKILL/segfault) without the client seeing it yet.
        fake_manager.proc.is_alive.return_value = False
        assert engine.is_backend_dead() is True
    finally:
        engine.close()


def test_is_backend_dead_legacy_keys_off_executor():
    """legacy: is_backend_dead() keys off the executor's _closed / is_failed flags."""
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    od_config = _od_config(enable_runtime_v2=False)
    fake_executor = MagicMock(name="MultiprocDiffusionExecutor")
    fake_executor._closed = False
    fake_executor.is_failed = False

    with (
        _stub_process_funcs(),
        patch("vllm_omni.diffusion.executor.abstract.DiffusionExecutor.get_class") as get_class,
        patch.object(DiffusionEngine, "_dummy_run"),
    ):
        get_class.return_value = lambda _cfg: fake_executor
        engine = DiffusionEngine(od_config)
        try:
            # Healthy executor -> not dead.
            assert engine.is_backend_dead() is False
            # Worker crash flips is_failed (and later _closed) from the monitor.
            fake_executor.is_failed = True
            assert engine.is_backend_dead() is True
            fake_executor.is_failed = False
            fake_executor._closed = True
            assert engine.is_backend_dead() is True
        finally:
            engine.close()


def test_add_request_runtime_v2_rejects_kv_sender_info():
    """runtime_v2 (PR1) must fail loudly (synchronously) when kv_sender_info is set.

    The runtime_v2 QwenPrepareExecutor copies kv_sender_info onto the state but
    never receives the upstream KV, so a multi-stage request would silently run
    wrong. _add_request_runtime_v2 (the earliest frontend entry) rejects it with
    NotImplementedError before the request crosses into the scheduler proc.
    """
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    fake_manager, fake_client = _fake_manager_and_client()

    async def _run() -> None:
        engine = _build_flag_on_engine(fake_manager, fake_client)
        try:
            engine.main_loop = asyncio.get_running_loop()
            request = OmniDiffusionRequest(
                prompt={"prompt": "a small test image"},
                sampling_params=OmniDiffusionSamplingParams(height=64, width=64, num_inference_steps=1),
                request_id="req-kv-1",
                kv_sender_info={"engine_id": "upstream", "some": "info"},
            )
            with pytest.raises(NotImplementedError, match="kv_sender_info"):
                await engine._add_request_runtime_v2(request)
            # The request must NOT have been sent to the scheduler proc.
            fake_client.add_request.assert_not_called()

            # A request WITHOUT kv_sender_info is accepted (sent to the proc).
            ok_request = OmniDiffusionRequest(
                prompt={"prompt": "a small test image"},
                sampling_params=OmniDiffusionSamplingParams(height=64, width=64, num_inference_steps=1),
                request_id="req-ok-1",
            )
            await engine._add_request_runtime_v2(ok_request)
            fake_client.add_request.assert_awaited_once()
        finally:
            engine.close()

    asyncio.run(_run())


def test_add_request_runtime_v2_rejects_lora_request():
    """runtime_v2 (PR1) must fail loudly (synchronously) when a lora_request is set.

    The runtime_v2 worker path calls the task executor directly and never
    activates worker.lora_manager (which the legacy DiffusionWorker does before
    every forward), so a LoRA request would silently run with the base /
    previously-active adapter. _add_request_runtime_v2 rejects it -- like
    kv_sender_info -- before the request crosses into the scheduler proc.
    """
    from vllm.lora.request import LoRARequest

    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    fake_manager, fake_client = _fake_manager_and_client()

    async def _run() -> None:
        engine = _build_flag_on_engine(fake_manager, fake_client)
        try:
            engine.main_loop = asyncio.get_running_loop()
            request = OmniDiffusionRequest(
                prompt={"prompt": "a small test image"},
                sampling_params=OmniDiffusionSamplingParams(
                    height=64,
                    width=64,
                    num_inference_steps=1,
                    lora_request=LoRARequest("adapter", 1, "/fake/lora/path"),
                ),
                request_id="req-lora-1",
            )
            with pytest.raises(NotImplementedError, match="lora_request"):
                await engine._add_request_runtime_v2(request)
            # The request must NOT have been sent to the scheduler proc.
            fake_client.add_request.assert_not_called()

            # A request WITHOUT a lora_request is accepted (sent to the proc).
            ok_request = OmniDiffusionRequest(
                prompt={"prompt": "a small test image"},
                sampling_params=OmniDiffusionSamplingParams(height=64, width=64, num_inference_steps=1),
                request_id="req-ok-lora",
            )
            await engine._add_request_runtime_v2(ok_request)
            fake_client.add_request.assert_awaited_once()
        finally:
            engine.close()

    asyncio.run(_run())


def test_add_request_runtime_v2_rolls_back_on_send_failure():
    """If the ZMQ send raises (e.g. _send_request_nowait times out because the
    proc is not draining), the request must NOT stay registered: leaving it in
    _out_queue / _runtime_v2_inflight makes the drain loop poll forever for a
    terminal that can never arrive and retains the future."""
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    fake_manager, fake_client = _fake_manager_and_client()
    fake_client.add_request = AsyncMock(side_effect=RuntimeError("send timed out"))

    async def _run() -> None:
        engine = _build_flag_on_engine(fake_manager, fake_client)
        try:
            engine.main_loop = asyncio.get_running_loop()
            request = OmniDiffusionRequest(
                prompt={"prompt": "a small test image"},
                sampling_params=OmniDiffusionSamplingParams(height=64, width=64, num_inference_steps=1),
                request_id="req-send-fail",
            )
            with pytest.raises(RuntimeError, match="send timed out"):
                await engine._add_request_runtime_v2(request)
            # Rolled back on both structures -> no phantom in-flight request.
            assert "req-send-fail" not in engine._out_queue
            assert "req-send-fail" not in engine._runtime_v2_inflight
        finally:
            engine.close()

    asyncio.run(_run())


def test_add_request_runtime_v2_rolls_back_on_cancellation():
    """If the awaiting task is CANCELLED mid-send (CancelledError, a BaseException
    on 3.11+), the registration must still be rolled back -- a bare
    `except Exception` would miss it and leak the future / inflight entry."""
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    fake_manager, fake_client = _fake_manager_and_client()
    fake_client.add_request = AsyncMock(side_effect=asyncio.CancelledError())

    async def _run() -> None:
        engine = _build_flag_on_engine(fake_manager, fake_client)
        try:
            engine.main_loop = asyncio.get_running_loop()
            request = OmniDiffusionRequest(
                prompt={"prompt": "a small test image"},
                sampling_params=OmniDiffusionSamplingParams(height=64, width=64, num_inference_steps=1),
                request_id="req-cancel",
            )
            with pytest.raises(asyncio.CancelledError):
                await engine._add_request_runtime_v2(request)
            assert "req-cancel" not in engine._out_queue
            assert "req-cancel" not in engine._runtime_v2_inflight
        finally:
            engine.close()

    asyncio.run(_run())


def test_runtime_v2_client_setup_failure_shuts_down_manager():
    """If from_addresses raises AFTER the manager spawned the scheduler proc,
    __init__ must shut the manager down -- otherwise the nested scheduler / GPU
    worker processes leak (no later close() can reach the un-assigned engine)."""
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    od_config = _od_config(enable_runtime_v2=True)
    fake_manager = MagicMock(name="RuntimeV2SchedulerProcManager")
    fake_manager.addresses.inputs = ["ipc:///tmp/fake-in"]
    fake_manager.addresses.outputs = ["ipc:///tmp/fake-out"]

    with (
        _stub_process_funcs(),
        patch(
            "vllm_omni.diffusion.runtime_v2.scheduler_proc.RuntimeV2SchedulerProcManager",
            return_value=fake_manager,
        ),
        patch(
            "vllm_omni.diffusion.runtime_v2.scheduler_client.RuntimeV2SchedulerClient.from_addresses",
            side_effect=RuntimeError("IPC bind failed"),
        ),
        patch.object(DiffusionEngine, "_dummy_run"),
        patch("vllm_omni.diffusion.executor.abstract.DiffusionExecutor.get_class") as get_class,
    ):
        get_class.return_value = MagicMock()
        with pytest.raises(RuntimeError, match="IPC bind failed"):
            DiffusionEngine(od_config)
        # The spawned scheduler proc (+ GPU workers) was torn down, not leaked.
        fake_manager.shutdown.assert_called_once()


def test_register_backend_dead_callback_routes_to_client():
    """The engine forwards the stage's backend-death callback to the scheduler
    client's proc monitor (so idle nested-proc death wakes the stage)."""
    fake_manager, fake_client = _fake_manager_and_client()
    engine = _build_flag_on_engine(fake_manager, fake_client)
    try:

        def _cb() -> None:
            pass

        engine.register_backend_dead_callback(_cb)
        fake_client.set_on_engine_dead.assert_called_once_with(_cb)
    finally:
        engine.close()


def test_runtime_v2_forwards_stage_init_timeout_to_scheduler_proc_manager():
    """The nested scheduler proc must inherit od_config.stage_init_timeout (the
    user's --stage-init-timeout), not a hardcoded 300s -- otherwise a slow
    checkpoint load in RuntimeV2SchedulerProc is killed at 300s regardless."""
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    od_config = _od_config(enable_runtime_v2=True, stage_init_timeout=1234)

    fake_manager = MagicMock(name="RuntimeV2SchedulerProcManager")
    fake_manager.addresses.inputs = ["ipc:///tmp/fake-in"]
    fake_manager.addresses.outputs = ["ipc:///tmp/fake-out"]
    fake_client = MagicMock(name="RuntimeV2SchedulerClient")

    with (
        _stub_process_funcs(),
        patch(
            "vllm_omni.diffusion.runtime_v2.scheduler_proc.RuntimeV2SchedulerProcManager",
            return_value=fake_manager,
        ) as manager_cls,
        patch(
            "vllm_omni.diffusion.runtime_v2.scheduler_client.RuntimeV2SchedulerClient.from_addresses",
            return_value=fake_client,
        ),
        patch.object(DiffusionEngine, "_dummy_run"),
        patch("vllm_omni.diffusion.executor.abstract.DiffusionExecutor.get_class") as get_class,
    ):
        get_class.return_value = MagicMock()
        engine = DiffusionEngine(od_config)
        try:
            manager_cls.assert_called_once()
            _, mkwargs = manager_cls.call_args
            assert mkwargs["stage_init_timeout"] == 1234
        finally:
            engine.close()


def test_runtime_v2_abort_tombstone_drains_late_result_and_frees_shm():
    """A terminal the scheduler proc had ALREADY sent before the abort must still
    be drained + discarded so its packed SHM handle is unlinked (not leaked).

    abort() resolves the future and drops the id from _runtime_v2_inflight, so the
    main drain never fetches that late result; the tombstone path
    (_drain_runtime_v2_aborted_tombstones) must pull it, materialize it (unlink
    the /dev/shm segment), and drop the tombstone -- WITHOUT re-resolving.
    """
    import os
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.ipc import pack_diffusion_output_shm

    fake_manager, fake_client = _fake_manager_and_client()

    async def _run() -> None:
        engine = _build_flag_on_engine(fake_manager, fake_client)
        try:
            engine.main_loop = asyncio.get_running_loop()
            rid = "req-late-abort"
            fut = engine.main_loop.create_future()
            engine._out_queue[rid] = fut
            engine._runtime_v2_inflight.add(rid)

            # Abort: future resolves aborted, inflight cleared, tombstone registered.
            engine.abort(rid)
            assert fut.done() and fut.result().aborted is True
            assert rid not in engine._runtime_v2_inflight
            assert rid in engine._runtime_v2_aborted

            # The late terminal: a raw DiffusionOutput whose big tensor is packed
            # to a /dev/shm handle (as the worker sends it, kept packed).
            packed = DiffusionOutput(output=torch.arange(300_000, dtype=torch.float32))
            pack_diffusion_output_shm(packed)
            assert isinstance(packed.output, dict) and packed.output.get("__tensor_shm__")
            shm_name = packed.output["name"]
            assert shm_name in os.listdir("/dev/shm")

            fake_client.get_result_nowait = MagicMock(side_effect=[packed])
            engine._drain_runtime_v2_aborted_tombstones(fake_client, dict(engine._runtime_v2_aborted))

            # SHM segment unlinked, tombstone dropped, future not re-resolved.
            assert shm_name not in os.listdir("/dev/shm")
            assert rid not in engine._runtime_v2_aborted
        finally:
            engine.close()

    asyncio.run(_run())


def test_runtime_v2_abort_tombstone_gc_when_no_late_result():
    """When no late terminal arrives, the tombstone is GC'd once its window
    elapses (the request was aborted before it finished -> nothing will come)."""
    fake_manager, fake_client = _fake_manager_and_client()
    engine = _build_flag_on_engine(fake_manager, fake_client)
    try:
        # Deadline already in the past; client has nothing buffered.
        engine._runtime_v2_aborted = {"req-gone": 0.0}
        fake_client.get_result_nowait = MagicMock(return_value=None)

        engine._drain_runtime_v2_aborted_tombstones(fake_client, dict(engine._runtime_v2_aborted))

        assert "req-gone" not in engine._runtime_v2_aborted
    finally:
        engine.close()


def test_runtime_v2_engine_dead_delivers_buffered_result_before_failing():
    """When the proc dies AFTER buffering a completed result, the engine_dead
    branch must DELIVER that result (freeing its SHM), failing only requests with
    nothing buffered -- not blanket-fail everything and discard the completed one."""
    import threading

    from vllm.v1.engine.exceptions import EngineDeadError

    from vllm_omni.diffusion.data import DiffusionOutput

    fake_manager, fake_client = _fake_manager_and_client()
    good = DiffusionOutput(output="delivered")

    def _get(rid):
        if rid == "rid-good":
            return good  # buffered before death
        raise EngineDeadError()  # nothing buffered for this one

    fake_client.get_result_nowait = MagicMock(side_effect=_get)
    fake_client.engine_dead = True

    async def _run() -> None:
        engine = _build_flag_on_engine(fake_manager, fake_client)
        try:
            engine.main_loop = asyncio.get_running_loop()
            engine.stop_event = threading.Event()
            fut_good = engine.main_loop.create_future()
            fut_dead = engine.main_loop.create_future()
            engine._out_queue["rid-good"] = fut_good
            engine._out_queue["rid-dead"] = fut_dead
            engine._runtime_v2_inflight.update({"rid-good", "rid-dead"})

            task = engine.main_loop.create_task(engine._runtime_v2_drain_loop())
            await asyncio.wait_for(
                asyncio.gather(fut_good, fut_dead, return_exceptions=True), timeout=3.0
            )
            engine.stop_event.set()
            await asyncio.wait_for(task, timeout=3.0)

            # The buffered result was delivered (not discarded); the other failed.
            assert fut_good.result().output == "delivered"
            assert fut_good.result().error is None
            assert fut_dead.result().error is not None
        finally:
            engine.close()

    asyncio.run(_run())


def test_runtime_v2_drain_materialize_failure_isolated_to_one_request():
    """A bad/orphaned SHM handle in ONE result must fail only that request, not
    tear down _runtime_v2_drain_loop and strand every other in-flight future."""
    import threading

    from vllm_omni.diffusion.data import DiffusionOutput

    fake_manager, fake_client = _fake_manager_and_client()
    good_result = DiffusionOutput(output="good")
    bad_result = DiffusionOutput(output="bad")
    pending = {"req-bad": bad_result, "req-good": good_result}
    fake_client.engine_dead = False
    fake_client.get_result_nowait = MagicMock(side_effect=lambda rid: pending.pop(rid, None))

    async def _run() -> None:
        engine = _build_flag_on_engine(fake_manager, fake_client)
        try:
            engine.main_loop = asyncio.get_running_loop()
            engine.stop_event = threading.Event()
            fut_bad = engine.main_loop.create_future()
            fut_good = engine.main_loop.create_future()
            engine._out_queue["req-bad"] = fut_bad
            engine._out_queue["req-good"] = fut_good
            engine._runtime_v2_inflight.update({"req-bad", "req-good"})

            def _materialize(payload):
                if payload is bad_result:
                    raise RuntimeError("shm segment gone")
                return payload

            with patch.object(engine, "_materialize_runtime_v2_output", side_effect=_materialize):
                task = engine.main_loop.create_task(engine._runtime_v2_drain_loop())
                await asyncio.wait_for(
                    asyncio.gather(fut_bad, fut_good, return_exceptions=True), timeout=3.0
                )
                engine.stop_event.set()
                await asyncio.wait_for(task, timeout=3.0)

            # Bad -> contained to an error output; good -> resolved normally; the
            # loop survived (task returned cleanly rather than dying on the raise).
            assert fut_bad.done() and fut_bad.result().error is not None
            assert fut_good.done() and fut_good.result().output == "good"
            assert task.done() and task.exception() is None
        finally:
            engine.close()

    asyncio.run(_run())


def test_runtime_v2_abort_resolves_future_and_clears_inflight():
    """abort() under runtime_v2 must not leak the future / in-flight entry.

    The scheduler proc's abort branch sends NOTHING back, so if abort() only
    forwarded to the client, the request's future would never resolve, its
    _out_queue entry + _runtime_v2_inflight membership would leak, and the drain
    loop would keep polling it forever. This test registers an in-flight
    request (as _add_request_runtime_v2 does), calls abort(), and asserts: the
    client was told to abort, the future resolves with an aborted
    DiffusionOutput, and the id is gone from both _out_queue and
    _runtime_v2_inflight.
    """
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    od_config = _od_config(enable_runtime_v2=True)

    fake_manager = MagicMock(name="RuntimeV2SchedulerProcManager")
    fake_manager.addresses.inputs = ["ipc:///tmp/fake-in"]
    fake_manager.addresses.outputs = ["ipc:///tmp/fake-out"]
    fake_client = MagicMock(name="RuntimeV2SchedulerClient")

    async def _run() -> None:
        with (
            _stub_process_funcs(),
            patch(
                "vllm_omni.diffusion.runtime_v2.scheduler_proc.RuntimeV2SchedulerProcManager",
                return_value=fake_manager,
            ),
            patch(
                "vllm_omni.diffusion.runtime_v2.scheduler_client.RuntimeV2SchedulerClient.from_addresses",
                return_value=fake_client,
            ),
            patch.object(DiffusionEngine, "_dummy_run"),
            patch("vllm_omni.diffusion.executor.abstract.DiffusionExecutor.get_class") as get_class,
        ):
            get_class.return_value = MagicMock()
            engine = DiffusionEngine(od_config)
            try:
                # Register an in-flight request the way _add_request_runtime_v2
                # does: a future in _out_queue + membership in the inflight set.
                request_id = "req-abort-1"
                engine.main_loop = asyncio.get_running_loop()
                fut = engine.main_loop.create_future()
                engine._out_queue[request_id] = fut
                engine._runtime_v2_inflight.add(request_id)

                engine.abort(request_id)

                # Client received the abort forward (kept behavior).
                fake_client.abort_nowait.assert_called_once()
                forwarded_ids = fake_client.abort_nowait.call_args[0][0]
                assert request_id in list(forwarded_ids)

                # The future is resolved with an aborted DiffusionOutput...
                assert fut.done()
                result = fut.result()
                assert isinstance(result, DiffusionOutput)
                assert result.aborted is True
                # ...and the id no longer leaks anywhere.
                assert request_id not in engine._out_queue
                assert request_id not in engine._runtime_v2_inflight

                # A double-abort (or a real terminal racing in) must not raise
                # or double-resolve; the second call is a no-op.
                engine.abort(request_id)
            finally:
                engine.close()

    asyncio.run(_run())
