# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU socket-level round-trip for the runtime_v2 scheduler client.

Stands up ``RuntimeV2SchedulerClient``'s real ZMQ sockets against an in-test
*fake* scheduler proc (a bare ZMQ socket pair, no GPU, no real proc spawn).
The fake mirrors ``RuntimeV2SchedulerProc``'s wire contract:

  * it PULLs ``add_request`` messages off the client's request address, and
  * PUSHes back ``{"type":"result","request_id","output": DiffusionOutput}`` on
    the client's response address,

and we assert the client surfaces that raw ``DiffusionOutput`` for the matching
``request_id`` via ``get_result_nowait``. A second test drives the
death-sentinel path: the fake sends ``RUNTIME_V2_SCHEDULER_PROC_DEAD`` and the
client raises ``EngineDeadError``.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import zmq
from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.runtime_v2.scheduler_client import RuntimeV2SchedulerClient
from vllm_omni.diffusion.runtime_v2.scheduler_proc import RuntimeV2SchedulerProc
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


class _FakeSchedulerProc:
    """A bare ZMQ peer that impersonates ``RuntimeV2SchedulerProc``.

    The client BINDS both sockets (PUSH request / PULL response), so this fake
    CONNECTs the opposite ends (PULL on the request address to receive, PUSH on
    the response address to reply) -- exactly as the real proc does in
    ``RuntimeV2SchedulerProc.run_loop``.
    """

    def __init__(self, request_address: str, response_address: str) -> None:
        self._ctx = zmq.Context()
        # Receive add_request/abort/shutdown (client PUSHes here).
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.connect(request_address)
        # Send results/errors/sentinel (client PULLs here).
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.connect(response_address)
        self._encoder = OmniMsgpackEncoder()
        self._decoder = OmniMsgpackDecoder()

    def recv_request(self, timeout_ms: int = 2000) -> dict:
        if self._pull.poll(timeout=timeout_ms) == 0:
            raise TimeoutError("fake proc did not receive an add_request in time")
        return self._decoder.decode(self._pull.recv())

    def send_result(self, request_id: str, output: DiffusionOutput) -> None:
        self._push.send(
            self._encoder.encode({"type": "result", "request_id": request_id, "output": output})
        )

    def send_error(self, request_id: str, error: str) -> None:
        self._push.send(
            self._encoder.encode(
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": error,
                    "status_code": 500,
                    "error_type": "internal_error",
                }
            )
        )

    def send_death_sentinel(self) -> None:
        self._push.send(RuntimeV2SchedulerProc.RUNTIME_V2_SCHEDULER_PROC_DEAD)

    def close(self) -> None:
        self._pull.close(linger=0)
        self._push.close(linger=0)
        self._ctx.term()


def _make_request(request_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt={"prompt": "a small test image"},
        sampling_params=OmniDiffusionSamplingParams(height=64, width=64, num_inference_steps=1),
        request_id=request_id,
    )


def _make_client() -> tuple[RuntimeV2SchedulerClient, _FakeSchedulerProc]:
    request_address = get_open_zmq_ipc_path()
    response_address = get_open_zmq_ipc_path()
    # No proc_manager: this is a pure socket round-trip, no real subprocess.
    client = RuntimeV2SchedulerClient.from_addresses(
        request_address=request_address,
        response_address=response_address,
        proc_manager=None,
    )
    fake = _FakeSchedulerProc(request_address, response_address)
    return client, fake


async def _drain_until(client: RuntimeV2SchedulerClient, request_id: str, timeout_s: float = 3.0):
    """Poll ``get_result_nowait`` until the terminal arrives or we time out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = client.get_result_nowait(request_id)
        if result is not None:
            return result
        await _async_sleep()
    return None


async def _async_sleep() -> None:
    import asyncio

    await asyncio.sleep(0.01)


def _bare_client(*, engine_dead: bool = False, shutting_down: bool = False) -> RuntimeV2SchedulerClient:
    """A RuntimeV2SchedulerClient with only the death-tracking attrs wired (no
    sockets / subprocess), for unit-testing the death-notification helpers."""
    from threading import Lock

    client = object.__new__(RuntimeV2SchedulerClient)
    client._engine_dead = engine_dead
    client._shutting_down = shutting_down
    client._on_engine_dead = None
    client._engine_dead_lock = Lock()
    return client


def test_scheduler_client_set_on_engine_dead_fires_immediately_if_already_dead():
    """Registering the callback after the proc already died (a detector won the
    race) must fire it immediately, so a late-registering stage still wakes."""
    client = _bare_client(engine_dead=True)

    fired: list = []
    client.set_on_engine_dead(lambda: fired.append(True))

    assert fired == [True]


def test_scheduler_client_set_on_engine_dead_stores_when_alive():
    """When the proc is still alive, the callback is stored (fired later by a
    detector) and NOT invoked eagerly."""
    client = _bare_client(engine_dead=False)

    fired: list = []
    client.set_on_engine_dead(lambda: fired.append(True))

    assert fired == []
    assert client._on_engine_dead is not None


def test_mark_engine_dead_fires_callback_exactly_once():
    """Every death-detection path routes through _mark_engine_dead; the callback
    must fire exactly once even if multiple paths race to mark death."""
    client = _bare_client(engine_dead=False)
    fired: list = []
    client._on_engine_dead = lambda: fired.append(True)

    client._mark_engine_dead()
    client._mark_engine_dead()  # second detector no-ops (already dead)

    assert fired == [True]
    assert client._engine_dead is True


def test_mark_engine_dead_skips_during_graceful_shutdown():
    """A clean teardown (shutting_down) must NOT be reported as a fatal death:
    the flag stays down and the callback does not fire."""
    client = _bare_client(shutting_down=True)
    fired: list = []
    client._on_engine_dead = lambda: fired.append(True)

    client._mark_engine_dead()

    assert fired == []
    assert client._engine_dead is False


@pytest.mark.asyncio
async def test_client_delivers_result_for_request_id():
    """add_request -> fake echoes a result -> client surfaces it by request_id."""
    client, fake = _make_client()
    try:
        request_id = "req-roundtrip-1"
        request = _make_request(request_id)

        # Send the add_request over ZMQ (the client encodes prompt +
        # sampling_params in the shape the proc reconstructs).
        await client.add_request(request_id, request)

        # The fake proc receives the add_request and echoes a raw DiffusionOutput
        # carrying a small tensor (no SHM handles at this layer).
        msg = fake.recv_request()
        assert msg["type"] == "add_request"
        assert msg["request_id"] == request_id
        assert msg["prompt"] == {"prompt": "a small test image"}
        assert msg["sampling_params"]["height"] == 64

        payload = DiffusionOutput(output=torch.zeros(2, 3))
        fake.send_result(request_id, payload)

        result = await _drain_until(client, request_id)
        assert result is not None, "client never surfaced the result"
        assert isinstance(result, DiffusionOutput)
        assert result.error is None
        assert torch.equal(result.output, torch.zeros(2, 3))

        # Once delivered, it is popped: a second fetch returns None.
        assert client.get_result_nowait(request_id) is None
    finally:
        fake.close()
        client.close()


@pytest.mark.asyncio
async def test_client_delivers_result_with_none_output():
    """A terminal DiffusionOutput with output=None still round-trips."""
    client, fake = _make_client()
    try:
        request_id = "req-none-output"
        request = _make_request(request_id)
        await client.add_request(request_id, request)

        assert fake.recv_request()["request_id"] == request_id
        fake.send_result(request_id, DiffusionOutput(output=None, finished=True))

        result = await _drain_until(client, request_id)
        assert result is not None
        assert isinstance(result, DiffusionOutput)
        assert result.output is None
        assert result.error is None
    finally:
        fake.close()
        client.close()


@pytest.mark.asyncio
async def test_client_surfaces_error_for_request_id():
    """An 'error' response resolves to a DiffusionOutput carrying the error."""
    client, fake = _make_client()
    try:
        request_id = "req-error"
        await client.add_request(request_id, _make_request(request_id))
        assert fake.recv_request()["request_id"] == request_id

        fake.send_error(request_id, "boom in the scheduler")

        result = await _drain_until(client, request_id)
        assert result is not None
        assert isinstance(result, DiffusionOutput)
        assert result.error == "boom in the scheduler"
        assert result.error_status_code == 500
        assert result.error_type == "internal_error"
    finally:
        fake.close()
        client.close()


@pytest.mark.asyncio
async def test_client_death_sentinel_path():
    """The death sentinel marks the client dead and raises EngineDeadError."""
    client, fake = _make_client()
    try:
        request_id = "req-before-death"
        await client.add_request(request_id, _make_request(request_id))
        assert fake.recv_request()["request_id"] == request_id

        fake.send_death_sentinel()

        # Drain until the sentinel is processed, then the next fetch raises.
        deadline = time.monotonic() + 3.0
        raised = False
        while time.monotonic() < deadline:
            try:
                client.get_result_nowait(request_id)
            except EngineDeadError:
                raised = True
                break
            await _async_sleep()

        assert raised, "client did not raise EngineDeadError after death sentinel"
        assert client.engine_dead is True
    finally:
        fake.close()
        client.close()


@pytest.mark.asyncio
async def test_client_drain_sentinel_fires_on_engine_dead_callback():
    """When _drain_responses observes the death sentinel (no proc-monitor here,
    so the drain is the sole detector), the on_engine_dead callback MUST fire --
    otherwise an idle StageDiffusionProc is never woken to demote the replica."""
    client, fake = _make_client()
    try:
        fired: list = []
        client.set_on_engine_dead(lambda: fired.append(True))

        fake.send_death_sentinel()

        # get_result_nowait drains responses (processing the sentinel) each call.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fired:
            try:
                client.get_result_nowait("nonexistent")
            except EngineDeadError:
                break
            await _async_sleep()

        assert fired == [True], "drain-detected death must fire the on_engine_dead callback"
        assert client.engine_dead is True
    finally:
        fake.close()
        client.close()


@pytest.mark.asyncio
async def test_add_request_after_death_raises():
    """Once dead, add_request itself refuses to send."""
    client, fake = _make_client()
    try:
        # Force the dead state directly via the sentinel round-trip.
        fake.send_death_sentinel()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not client.engine_dead:
            try:
                client.get_result_nowait("nonexistent")
            except EngineDeadError:
                break
            await _async_sleep()
        assert client.engine_dead is True

        with pytest.raises(EngineDeadError):
            await client.add_request("late-req", _make_request("late-req"))
    finally:
        fake.close()
        client.close()


# ----------------------------------------------------------------------------
# _runner_busy_loop dead-pool escalation (the submit-into-dead-pool path).
#
# When an idle scheduler (no inflight requests) submits into a worker pool that
# has already died, submit() raises. Delivering only a per-request error and
# continuing leaves the proc reporting healthy forever while every later request
# fails the same way -- the end-of-loop health check is never reached while
# inflight stays empty. The loop must escalate to signal_fatal so the proc dies
# and the replica is demoted.
# ----------------------------------------------------------------------------


def _make_proc() -> RuntimeV2SchedulerProc:
    # od_config is only touched by initialize()/run_loop(); the busy loop under
    # test uses only self._runner, so a bare Mock config is sufficient.
    return RuntimeV2SchedulerProc(model="fake-model", od_config=Mock())


class _DeadPoolRunner:
    """submit() raises AND check_health() raises -> pool is permanently dead."""

    def submit(self, request):
        raise RuntimeError("dispatch to dead worker pool failed")

    def check_health(self):
        raise RuntimeError("worker pool has a dead worker")


class _BenignFailRunner:
    """submit() raises but check_health() passes -> a per-request error only."""

    def submit(self, request):
        raise ValueError("bad request")

    def check_health(self):
        return None


class _IdleOnlyQueue(queue.Queue):
    """Delivers its one item ONLY via ``get(timeout=...)`` (the idle path).

    ``_runner_busy_loop`` has two submit sites: the drain loop (``get_nowait``)
    and the idle branch (``get(timeout=0.1)`` when nothing is inflight). By
    making ``get_nowait`` always report empty, the drain loop never sees the
    request, so it is delivered deterministically through the idle branch --
    with no thread and no timing race.
    """

    def __init__(self, item) -> None:
        super().__init__()
        self._item = item

    def get_nowait(self):
        raise queue.Empty

    def get(self, block=True, timeout=None):
        if self._item is None:
            raise queue.Empty
        item, self._item = self._item, None
        return item


def _run_busy_loop_once(proc, submit_queue, stop_event, errors, fatals) -> None:
    proc._runner_busy_loop(
        loop=None,  # unused by the busy loop itself (send callbacks are injected)
        submit_queue=submit_queue,
        abort_queue=queue.Queue(),
        stop_event=stop_event,
        send_result=lambda *a, **k: None,
        send_error=lambda rid, msg, code, etype: errors.append((rid, msg, code, etype)),
        signal_fatal=lambda reason: fatals.append(reason),
    )


def test_busy_loop_drain_submit_into_dead_pool_escalates_to_fatal():
    """DRAIN path: a queued submit whose pool is dead delivers the error AND
    signals fatal. The request is enqueued before the loop runs, so the inner
    ``while True`` drain (get_nowait) processes it."""
    proc = _make_proc()
    proc._runner = _DeadPoolRunner()

    submit_q: queue.Queue = queue.Queue()
    submit_q.put(SimpleNamespace(request_id="req-dead"))
    errors: list = []
    fatals: list = []

    # Returns promptly: drain -> submit raises -> send_error -> dead -> fatal -> return.
    _run_busy_loop_once(proc, submit_q, threading.Event(), errors, fatals)

    # The per-request error is still surfaced to the waiting caller...
    assert [e[0] for e in errors] == ["req-dead"]
    # ...AND the dead pool escalated to fatal so the proc exits (not stays UP).
    assert fatals == ["worker pool reported permanent failure"]


def test_busy_loop_idle_submit_into_dead_pool_escalates_to_fatal():
    """IDLE path: the SAME escalation must fire when the request arrives while no
    request is inflight (the idle ``get(timeout=0.1)`` branch, a SEPARATE handler
    from the drain path above). In production the first request always arrives
    idle, so this is the common path -- without its own guard a dead pool would
    go undetected. _IdleOnlyQueue forces this path deterministically."""
    proc = _make_proc()
    proc._runner = _DeadPoolRunner()

    submit_q = _IdleOnlyQueue(SimpleNamespace(request_id="req-dead-idle"))
    errors: list = []
    fatals: list = []

    # Drain finds nothing (get_nowait empty) -> idle get returns the item ->
    # submit raises -> send_error -> dead -> fatal -> return.
    _run_busy_loop_once(proc, submit_q, threading.Event(), errors, fatals)

    assert [e[0] for e in errors] == ["req-dead-idle"]
    assert fatals == ["worker pool reported permanent failure"]


def test_busy_loop_idle_worker_death_signals_fatal_without_any_request():
    """A worker that dies while the scheduler is IDLE (no inflight, EMPTY queue)
    must be detected on the idle tick and trip fatal -- otherwise the proc stays
    UP (check_health only sees the proc) until the next request happens to fail.
    No request is involved here: the empty queue's get(timeout) times out and the
    loop polls worker liveness."""
    proc = _make_proc()
    proc._runner = _DeadPoolRunner()  # check_health raises -> pool is dead

    submit_q: queue.Queue = queue.Queue()  # empty -> idle get(timeout) -> Empty
    errors: list = []
    fatals: list = []
    stop_event = threading.Event()

    # The idle get blocks ~0.1s then times out and the loop checks liveness;
    # run in a thread so a hang can't wedge the test.
    t = threading.Thread(
        target=_run_busy_loop_once,
        args=(proc, submit_q, stop_event, errors, fatals),
        daemon=True,
    )
    t.start()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fatals:
            time.sleep(0.01)
        assert fatals == ["worker pool reported permanent failure"]
        assert errors == [], "no request was involved -- no per-request error expected"
    finally:
        stop_event.set()
        t.join(timeout=3.0)
        assert not t.is_alive(), "busy loop did not exit after idle worker-death fatal"


def test_busy_loop_benign_submit_failure_does_not_signal_fatal():
    """A per-request submit error on a LIVE pool must NOT kill the whole proc."""
    proc = _make_proc()
    proc._runner = _BenignFailRunner()

    submit_q: queue.Queue = queue.Queue()
    submit_q.put(SimpleNamespace(request_id="req-bad"))
    errors: list = []
    fatals: list = []
    stop_event = threading.Event()

    # The pool stays healthy, so the loop keeps running (idle-waits on the queue);
    # run it in a thread and stop it once the error has been delivered.
    t = threading.Thread(
        target=_run_busy_loop_once,
        args=(proc, submit_q, stop_event, errors, fatals),
        daemon=True,
    )
    t.start()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not errors:
            time.sleep(0.01)
        assert [e[0] for e in errors] == ["req-bad"]
        # A live-pool per-request failure never escalates to fatal.
        assert fatals == []
    finally:
        stop_event.set()
        t.join(timeout=3.0)
        assert not t.is_alive(), "busy loop did not stop after stop_event"
