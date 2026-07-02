# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subprocess entry point for the runtime_v2 scheduler.

``RuntimeV2SchedulerProc`` runs a :class:`RuntimeV2Runner` (``GlobalScheduler``
+ ``MultiprocWorkerPool``) in a child process that nests **below**
``StageDiffusionProc``, communicating with a ``RuntimeV2SchedulerClient`` via
ZMQ (PUSH/PULL). It is a ~structural clone of ``stage_diffusion_proc.py`` with
two adaptations:

  * :meth:`initialize` builds a ``RuntimeV2Runner`` (which spawns the GPU
    worker pool — so THIS process becomes the parent of the GPU workers)
    instead of a ``DiffusionEngine``.
  * ``run_loop`` drives the runner (``submit`` + non-blocking ``poll_once`` +
    ``get_request_status``) and, on terminal, PUSHes the **raw**
    ``DiffusionOutput`` with SHM handles KEPT PACKED. It NEVER runs CPU
    postprocess (``post_process_func`` / ``format_diffusion_outputs`` /
    tensor→PIL) — that stays in ``StageDiffusionProc``, a different process.

Loop model: this proc is single-tenant (only the scheduler lives here — no postprocess, no
other requests), so it mirrors the engine's Thread-A/Thread-B split, but as a
clean two-thread split inside a *dedicated* process where GIL contention is
harmless:

  * **Thread A** (asyncio event loop): ZMQ recv/decode + send/encode. Hands
    off ``add_request`` / ``abort`` to Thread B via a thread-safe queue.
  * **Thread B** (``_runner_busy_loop``): the SOLE owner of the runner /
    ``GlobalScheduler``. Drains the hand-off queue (the only ``submit`` site),
    then drives the runner with a blocking ``poll_once`` (efficiently
    block-waits on worker events) and, when a tracked request reaches a
    terminal state, schedules the raw ``DiffusionOutput`` back onto the event
    loop for ZMQ send via ``loop.call_soon_threadsafe``.

Because Thread B is the single owner of the lock-free scheduler, no lock guards
scheduler access; the client (in ``StageDiffusionProc``) is the only ZMQ peer.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing.connection
import os
import queue
import signal
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
import zmq.asyncio
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngine, EngineZmqAddresses, wait_for_engine_startup
from vllm.v1.utils import shutdown

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.engine.stage_init_utils import set_death_signal
from vllm_omni.errors import client_error_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


class RuntimeV2SchedulerProc:
    """Subprocess entry point for the runtime_v2 scheduler.

    Manages ``RuntimeV2Runner`` lifecycle, drives its submit/poll loop on a
    dedicated thread, and relays raw (SHM-packed) ``DiffusionOutput`` results
    to a ``RuntimeV2SchedulerClient`` over ZMQ.
    """

    RUNTIME_V2_SCHEDULER_PROC_DEAD = b"RUNTIME_V2_SCHEDULER_PROC_DEAD"

    def __init__(self, model: str, od_config: OmniDiffusionConfig) -> None:
        self._model = model
        self._od_config = od_config
        self._runner: RuntimeV2Runner | None = None
        self._closed = False
        # Number of in-flight requests (registered by the event loop [Thread A],
        # retired by the runner thread [Thread B] via send_result/send_error).
        # Reported for the OmniCoordinator heartbeat hook. Mutated from BOTH
        # threads, so every add/discard and the length read is guarded by
        # ``_inflight_lock``.
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()
        # Set when the runner thread detects a fatal, non-recoverable failure
        # (e.g. the worker pool died); ``run_loop`` wakes, sends the death
        # sentinel and exits non-zero, mirroring StageDiffusionProc.
        self._fatal_event: asyncio.Event | None = None

    @property
    def queue_length(self) -> int:
        """Number of in-flight runtime_v2 requests.

        Returns 0 before :meth:`run_loop` starts and after it exits.
        """
        with self._inflight_lock:
            return len(self._inflight)

    def _is_runner_dead(self) -> bool:
        """True iff the runner's worker pool has a dead worker.

        Mirrors ``StageDiffusionProc._is_executor_dead``: once any GPU worker
        exits, every subsequent dispatch/fetch fails the same way, so the loop
        treats it as fatal. ``MultiprocWorkerPool.check_health()`` raises when a
        worker is no longer alive.
        """
        if self._runner is None:
            return False
        runner = self._runner
        try:
            runner.check_health()
        except Exception:
            return True
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Enrich config and create the runtime_v2 runner (spawns GPU workers)."""
        self._enrich_config()
        od_config = self._od_config
        self._runner = RuntimeV2Runner(
            pipeline=None,
            default_step_chunk_size=int(getattr(od_config, "runtime_v2_denoise_chunk_size", 1) or 1),
            scheduler_policy=str(getattr(od_config, "runtime_v2_scheduler_policy", "fcfs")),
            omni_diffusion_config=od_config,
        )
        logger.info("RuntimeV2SchedulerProc initialized with model: %s", self._model)

    def _enrich_config(self) -> None:
        """Load model metadata from HuggingFace and populate od_config fields."""
        self._od_config.enrich_config()

    # ------------------------------------------------------------------
    # Request reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_sampling_params(self, sampling_params_dict: dict) -> OmniDiffusionSamplingParams:
        """Reconstruct OmniDiffusionSamplingParams from a dict, handling LoRA."""
        lora_req = sampling_params_dict.get("lora_request")
        if lora_req is not None:
            from vllm.lora.request import LoRARequest

            if not isinstance(lora_req, LoRARequest):
                sampling_params_dict["lora_request"] = msgspec.convert(lora_req, LoRARequest)

        return OmniDiffusionSamplingParams(**sampling_params_dict)

    def _reconstruct_request(self, msg: dict[str, Any]) -> OmniDiffusionRequest:
        """Reconstruct an ``OmniDiffusionRequest`` from a decoded add_request msg.

        The client may send either a fully-formed ``request`` object (msgpack
        round-trips ``OmniDiffusionRequest`` via ``OmniMsgpack*``) or the
        stage-diffusion-style ``prompt`` + ``sampling_params`` dict fields; we
        support both so the same proc works against either serialization.
        """
        request = msg.get("request")
        if request is not None:
            if isinstance(request, OmniDiffusionRequest):
                if not request.request_id:
                    request.request_id = msg["request_id"]
                return request
            # A dict-shaped request (e.g. from a plain msgpack peer).
            return msgspec.convert(request, OmniDiffusionRequest)

        sampling_params = self._reconstruct_sampling_params(msg["sampling_params"])
        return OmniDiffusionRequest(
            prompt=msg["prompt"],
            sampling_params=sampling_params,
            request_id=msg["request_id"],
            kv_sender_info=msg.get("kv_sender_info"),
        )

    # ------------------------------------------------------------------
    # Runner busy loop (Thread B — the SOLE scheduler owner)
    # ------------------------------------------------------------------

    def _runner_busy_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        submit_queue: queue.Queue,
        abort_queue: queue.Queue,
        stop_event: threading.Event,
        send_result,
        send_error,
        signal_fatal,
    ) -> None:
        """Drive the runner: drain submits/aborts, poll, deliver terminals.

        This is the ONLY caller of ``runner.submit`` / ``poll_once`` /
        ``get_request_status`` / ``release_request``, so it is the single owner
        of the lock-free ``GlobalScheduler``. Terminal results (raw, SHM-packed
        ``DiffusionOutput``) and errors are handed back to the event loop
        thread (for ZMQ send) via the thread-safe ``send_result``/``send_error``
        callbacks (which use ``loop.call_soon_threadsafe`` internally).
        """
        inflight: set[str] = set()
        runner = self._runner
        assert runner is not None
        while not stop_event.is_set():
            # Drain the submit hand-off queue. This is the ONLY submit site, so
            # Thread B is the sole owner of the GlobalScheduler.
            while True:
                try:
                    request = submit_queue.get_nowait()
                except queue.Empty:
                    break
                request_id = request.request_id
                try:
                    runner.submit(request)
                    inflight.add(request_id)
                except Exception as exc:  # noqa: BLE001 - propagate to caller
                    logger.error("runtime_v2 submit failed for %s", request_id, exc_info=True)
                    status_code, error_type = client_error_metadata(exc)
                    send_error(request_id, str(exc), status_code, error_type)
                    # A submit can fail on a benign per-request error OR because
                    # the worker pool died mid-dispatch. In the latter case every
                    # future submit fails the same way, so escalate to fatal here
                    # instead of quietly erroring each request while the proc
                    # keeps reporting healthy (the poll path below is unreachable
                    # while inflight stays empty).
                    if self._is_runner_dead():
                        signal_fatal("worker pool reported permanent failure")
                        return

            # Drain aborts.
            while True:
                try:
                    abort_id = abort_queue.get_nowait()
                except queue.Empty:
                    break
                inflight.discard(abort_id)
                # Use abort_request (NOT release_request): release only frees
                # controller state, leaving the FCFS policy's active_request_by_group
                # slot pinned to the aborted id, so the next request for that group
                # is parked forever (single-group => whole-group deadlock).
                # abort_request emits a synthetic REQUEST_FAILED through the policy
                # so the slot is freed and any queued request is promoted +
                # dispatched. Guarded no-op for an unknown/already-finished id.
                with contextlib.suppress(Exception):
                    runner.abort_request(abort_id)

            if not inflight:
                # Nothing to poll for; briefly wait for the next submit/abort so
                # we don't spin. A short sleep keeps abort latency low.
                try:
                    request = submit_queue.get(timeout=0.1)
                except queue.Empty:
                    # Idle: with no inflight requests, poll_once + the end-of-loop
                    # health check never run, so a GPU worker that dies WHILE the
                    # scheduler is idle would go undetected (the proc stays alive
                    # and RuntimeV2SchedulerClient.check_health only sees the proc)
                    # until the next request fails. Poll worker liveness on each
                    # idle tick so idle worker death trips fatal + demotes the
                    # replica promptly. check_health is a cheap local is_alive()
                    # scan (no IPC), so a 0.1s idle cadence is fine.
                    if self._is_runner_dead():
                        signal_fatal("worker pool reported permanent failure")
                        return
                    continue
                request_id = request.request_id
                try:
                    runner.submit(request)
                    inflight.add(request_id)
                except Exception as exc:  # noqa: BLE001
                    logger.error("runtime_v2 submit failed for %s", request_id, exc_info=True)
                    status_code, error_type = client_error_metadata(exc)
                    send_error(request_id, str(exc), status_code, error_type)
                    # Idle path: no inflight requests means the end-of-loop
                    # health check is never reached, so a pool that died while
                    # this request was dispatched would otherwise go undetected
                    # and every later submit would fail the same way. Escalate.
                    if self._is_runner_dead():
                        signal_fatal("worker pool reported permanent failure")
                        return
                continue

            # Advance the runner one tick, block-waiting on worker events. A poll
            # failure is fatal to all in-flight requests.
            try:
                runner.poll_once(timeout_s=0.05)
            except Exception:  # noqa: BLE001
                logger.error("runtime_v2 poll_once failed", exc_info=True)
                signal_fatal("poll_once failed")
                return

            for request_id in list(inflight):
                if stop_event.is_set():
                    return
                try:
                    status, payload = runner.get_request_status(request_id)
                except Exception as exc:  # noqa: BLE001
                    logger.error("runtime_v2 get_request_status failed for %s", request_id, exc_info=True)
                    status_code, error_type = client_error_metadata(exc)
                    inflight.discard(request_id)
                    send_error(request_id, str(exc), status_code, error_type)
                    continue
                if status == "pending":
                    continue
                inflight.discard(request_id)
                if status == "finished":
                    # payload is the RAW DiffusionOutput fetched with
                    # unpack_shm=False (SHM handles KEPT PACKED). Do NOT unpack
                    # or postprocess here — StageDiffusionProc materializes it.
                    send_result(request_id, payload)
                else:  # "failed"
                    message = str(payload) if payload is not None else "runtime_v2 request failed"
                    send_error(request_id, message, 500, "internal_error")
                # Retire controller-side state now that the terminal is delivered.
                with contextlib.suppress(Exception):
                    runner.release_request(request_id)

            if self._is_runner_dead():
                signal_fatal("worker pool reported permanent failure")
                return

    # ------------------------------------------------------------------
    # ZMQ event loop (Thread A)
    # ------------------------------------------------------------------

    async def run_loop(
        self,
        request_address: str,
        response_address: str,
    ) -> None:
        """Async event loop handling ZMQ messages from RuntimeV2SchedulerClient."""
        ctx = zmq.asyncio.Context()

        request_socket = ctx.socket(zmq.PULL)
        request_socket.connect(request_address)

        response_socket = ctx.socket(zmq.PUSH)
        response_socket.connect(response_address)

        encoder = OmniMsgpackEncoder()
        decoder = OmniMsgpackDecoder()

        loop = asyncio.get_running_loop()
        submit_queue: queue.Queue = queue.Queue()
        abort_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        # Serializes ZMQ sends: the runner thread schedules sends onto this
        # queue via the event loop, and a single drain coroutine performs the
        # actual awaited ``response_socket.send``.
        send_queue: asyncio.Queue = asyncio.Queue()

        fatal_event = asyncio.Event()
        self._fatal_event = fatal_event

        def _enqueue_send(frame: bytes) -> None:
            send_queue.put_nowait(frame)

        def send_result(request_id: str, output: Any) -> None:
            frame = encoder.encode({"type": "result", "request_id": request_id, "output": output})
            with self._inflight_lock:
                self._inflight.discard(request_id)
            loop.call_soon_threadsafe(_enqueue_send, frame)

        def send_error(request_id: str, error: str, status_code: int, error_type: str) -> None:
            frame = encoder.encode(
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": error,
                    "status_code": status_code,
                    "error_type": error_type,
                }
            )
            with self._inflight_lock:
                self._inflight.discard(request_id)
            loop.call_soon_threadsafe(_enqueue_send, frame)

        def signal_fatal(reason: str) -> None:
            logger.error(
                "[RuntimeV2SchedulerProc] fatal runner failure detected (%s); "
                "signaling run_loop to send RUNTIME_V2_SCHEDULER_PROC_DEAD and exit.",
                reason,
            )
            loop.call_soon_threadsafe(fatal_event.set)

        runner_thread = threading.Thread(
            target=self._runner_busy_loop,
            name="RuntimeV2SchedulerRunner",
            kwargs={
                "loop": loop,
                "submit_queue": submit_queue,
                "abort_queue": abort_queue,
                "stop_event": stop_event,
                "send_result": send_result,
                "send_error": send_error,
                "signal_fatal": signal_fatal,
            },
            daemon=True,
        )
        runner_thread.start()

        async def _send_drain() -> None:
            while True:
                frame = await send_queue.get()
                await response_socket.send(frame)

        send_task = asyncio.ensure_future(_send_drain())

        try:
            while True:
                # Await recv and fatal_event concurrently so the loop wakes up
                # immediately when the runner thread signals a fatal failure —
                # even if no fresh ZMQ frame arrives.
                recv_task: asyncio.Task = asyncio.ensure_future(request_socket.recv())
                fatal_task: asyncio.Task = asyncio.ensure_future(fatal_event.wait())
                try:
                    await asyncio.wait(
                        [recv_task, fatal_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for waiter in (recv_task, fatal_task):
                        if not waiter.done():
                            waiter.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await waiter
                if fatal_event.is_set():
                    raise RuntimeError(
                        "RuntimeV2SchedulerProc runner reported permanent failure; "
                        "tearing down the scheduler subprocess."
                    )
                raw = recv_task.result()
                msg = decoder.decode(raw)
                msg_type = msg.get("type")

                if msg_type == "add_request":
                    request_id = msg["request_id"]
                    try:
                        request = self._reconstruct_request(msg)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Failed to reconstruct runtime_v2 request %s: %s", request_id, exc)
                        status_code, error_type = client_error_metadata(exc)
                        send_error(request_id, str(exc), status_code, error_type)
                        continue
                    with self._inflight_lock:
                        self._inflight.add(request_id)
                    submit_queue.put(request)

                elif msg_type == "abort":
                    for rid in msg.get("request_ids", []):
                        with self._inflight_lock:
                            self._inflight.discard(rid)
                        abort_queue.put(rid)

                elif msg_type == "shutdown":
                    break

        except Exception:
            # Send the death sentinel so the client can detect the fatal failure
            # promptly (mirrors StageDiffusionProc.DIFFUSION_PROC_DEAD).
            try:
                response_socket.setsockopt(zmq.LINGER, 4000)
                await response_socket.send(RuntimeV2SchedulerProc.RUNTIME_V2_SCHEDULER_PROC_DEAD)
            except Exception:
                logger.warning("Failed to send RUNTIME_V2_SCHEDULER_PROC_DEAD sentinel to client.")
            raise

        finally:
            stop_event.set()
            send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await send_task
            runner_thread.join(timeout=5.0)
            self._fatal_event = None
            request_socket.close()
            response_socket.close()
            ctx.term()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the runner (and its GPU worker pool)."""
        if self._closed:
            return
        self._closed = True

        if self._runner is not None:
            try:
                self._runner.shutdown()
            except Exception as e:
                logger.warning("Error shutting down runtime_v2 runner: %s", e)

    # ------------------------------------------------------------------
    # Subprocess entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _open_startup_handshake(
        handshake_address: str,
        *,
        local_client: bool,
        headless: bool,
    ) -> tuple[zmq.Context, zmq.Socket, EngineZmqAddresses]:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, (0).to_bytes(2, "little"))
        socket.connect(handshake_address)
        addresses = EngineCoreProc.startup_handshake(
            socket,
            local_client=local_client,
            headless=headless,
            parallel_config=None,
        )
        return ctx, socket, addresses

    @staticmethod
    def _send_startup_ready(
        handshake_socket: zmq.Socket,
        *,
        local_client: bool,
        headless: bool,
    ) -> None:
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "READY",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

    @classmethod
    def run_scheduler_proc(
        cls,
        model: str,
        od_config: OmniDiffusionConfig,
        handshake_address: str,
        *,
        local_client: bool,
        headless: bool,
    ) -> None:
        """Entry point for the runtime_v2 scheduler subprocess.

        Mirrors :meth:`StageDiffusionProc.run_diffusion_proc` but hosts a
        ``RuntimeV2Runner``. This proc nests below ``StageDiffusionProc`` and is
        the parent of the GPU workers spawned by the runner's
        ``MultiprocWorkerPool``. No OmniCoordinator wiring here — the nested
        scheduler proc is an internal, single-peer backend of
        ``StageDiffusionProc`` (coordinator wiring stays at the StageDiffusionProc
        layer; see design open-item 4).
        """
        shutdown_requested = False

        # Announce the process identity as early as possible. This is the
        # grep-proof that the scheduler runs in its OWN process, distinct from
        # the StageDiffusionProc that spawned it (whose pid == our os.getppid()).
        # The GPU smoke asserts pid != ppid on this line to prove the scheduler
        # is genuinely process-isolated from postprocess.
        logger.info(
            "RuntimeV2SchedulerProc starting: scheduler_proc_pid=%s stage_diffusion_proc_pid=%s (separate process)",
            os.getpid(),
            os.getppid(),
        )

        set_death_signal(signal.SIGTERM)

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit(_signal_exit_code(signum))

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        proc = cls(model, od_config)
        handshake_ctx: zmq.Context | None = None
        handshake_socket: zmq.Socket | None = None
        try:
            handshake_ctx, handshake_socket, addresses = cls._open_startup_handshake(
                handshake_address,
                local_client=local_client,
                headless=headless,
            )
            request_address = addresses.inputs[0]
            response_address = addresses.outputs[0]

            proc.initialize()

            cls._send_startup_ready(
                handshake_socket,
                local_client=local_client,
                headless=headless,
            )
            handshake_socket.close()
            handshake_ctx.term()
            handshake_socket = None
            handshake_ctx = None

            asyncio.run(proc.run_loop(request_address, response_address))

        except SystemExit:
            logger.debug("RuntimeV2SchedulerProc exiting.")
            raise
        except Exception:
            logger.exception("RuntimeV2SchedulerProc encountered a fatal error.")
            raise
        finally:
            if handshake_socket is not None:
                handshake_socket.close(linger=0)
            if handshake_ctx is not None:
                handshake_ctx.term()
            proc.close()


class RuntimeV2SchedulerProcManager:
    """Owns a ``RuntimeV2SchedulerProc`` subprocess.

    Structural clone of ``StageDiffusionProcManager``: spawns the scheduler
    proc, waits for its startup handshake, and exposes the small
    process-lifecycle surface (``sentinels`` / ``monitor_engine_liveness`` /
    ``finished_procs`` / ``shutdown``). The ZMQ addresses are freshly allocated
    and distinct from any ``StageDiffusionProc`` addresses — this proc nests
    BELOW ``StageDiffusionProc``.
    """

    def __init__(
        self,
        *,
        model: str,
        od_config: OmniDiffusionConfig,
        stage_init_timeout: int,
        handshake_address: str | None = None,
        addresses: EngineZmqAddresses | None = None,
    ) -> None:
        handshake_address = handshake_address or get_open_zmq_ipc_path()
        addresses = addresses or EngineZmqAddresses(
            inputs=[get_open_zmq_ipc_path()],
            outputs=[get_open_zmq_ipc_path()],
        )

        ctx = get_mp_context()
        proc = ctx.Process(
            target=RuntimeV2SchedulerProc.run_scheduler_proc,
            name="RuntimeV2SchedulerProc",
            kwargs={
                "model": model,
                "od_config": od_config,
                "handshake_address": handshake_address,
                "local_client": True,
                "headless": False,
            },
        )
        proc.start()
        self.proc = proc
        self.addresses = addresses
        self.manager_stopped = False
        self.failed_proc_name: str | None = None

        self._wait_until_started(handshake_address, stage_init_timeout)

    def _wait_until_started(self, handshake_address: str, stage_init_timeout: int) -> None:
        try:
            with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
                wait_for_engine_startup(
                    handshake_socket,
                    self.addresses,
                    [CoreEngine(index=0, local=True)],
                    SimpleNamespace(
                        data_parallel_size_local=1,
                        data_parallel_hybrid_lb=False,
                        data_parallel_external_lb=False,
                    ),
                    False,
                    None,
                    self,
                    None,
                )
        except Exception:
            shutdown([self.proc])
            raise

    def shutdown(self, timeout: float | None = None) -> None:
        self.manager_stopped = True
        shutdown([self.proc], timeout=timeout)

    def sentinels(self) -> list[int]:
        return [self.proc.sentinel]

    def finished_procs(self) -> dict[str, int]:
        if self.proc.exitcode is None:
            return {}
        return {self.proc.name: self.proc.exitcode}

    def monitor_engine_liveness(self) -> None:
        try:
            multiprocessing.connection.wait([self.proc.sentinel])
        except Exception:
            return
        if self.proc.exitcode not in (None, 0) and not self.manager_stopped:
            self.failed_proc_name = self.proc.name
        self.shutdown()
