# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend client for the runtime_v2 scheduler subprocess.

``RuntimeV2SchedulerClient`` owns the ``StageDiffusionProc``-side ZMQ sockets
that talk to a :class:`RuntimeV2SchedulerProc`. It is a structural clone of
``stage_diffusion_client.StageDiffusionClient`` with three adaptations that
follow the runtime_v2 message shapes (see the plan's "Message shapes"):

  * ``add_request`` sends ``{"type":"add_request","request_id","prompt",
    "sampling_params","kv_sender_info"}`` -- the SAME serialization
    ``StageDiffusionClient`` uses (``prompt`` + a plain ``sampling_params``
    dict), because ``RuntimeV2SchedulerProc._reconstruct_request`` reconstructs
    from exactly that shape. (A raw ``OmniDiffusionRequest`` object does NOT
    msgpack round-trip -- its ``prompt`` is a union of TypedDicts that
    ``msgspec.convert`` rejects -- so we mirror the proven stage-diffusion
    encoding.)
  * The death sentinel is ``RUNTIME_V2_SCHEDULER_PROC_DEAD`` (not
    ``DIFFUSION_PROC_DEAD``).
  * Delivered ``"result"`` payloads are RAW ``DiffusionOutput`` objects (with
    SHM handles still packed), NOT ``OmniRequestOutput``. Results are delivered
    per ``request_id`` (the engine correlates by id), not through a single
    FIFO output queue -- so callers fetch by ``get_result_nowait(request_id)``.

The engine (in ``StageDiffusionProc``) is the only ZMQ peer and touches this
client on exactly one thread (the event loop), so no lock guards it.
"""

from __future__ import annotations

import asyncio
import multiprocessing.connection
from dataclasses import fields, is_dataclass
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, Callable
import weakref

import zmq
import zmq.asyncio
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.runtime_v2.scheduler_proc import (
    RuntimeV2SchedulerProc,
    RuntimeV2SchedulerProcManager,
)
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

logger = init_logger(__name__)


class RuntimeV2SchedulerClient:
    """Communicates with a ``RuntimeV2SchedulerProc`` via ZMQ.

    Owns the request PUSH socket and response PULL socket that nest below
    ``StageDiffusionProc``; delivers raw (SHM-packed) ``DiffusionOutput``
    results keyed by ``request_id``.
    """

    def __init__(
        self,
        request_address: str,
        response_address: str,
        *,
        proc_manager: RuntimeV2SchedulerProcManager | None = None,
    ) -> None:
        self._proc_manager = proc_manager
        self._connect_transport(request_address, response_address)

        # Per-request delivery: the engine correlates by id, so results and
        # errors are stashed under their request_id rather than a shared FIFO.
        self._results: dict[str, DiffusionOutput] = {}
        self._shutting_down = False
        self._engine_dead: bool = False
        # Fired ONCE when the scheduler proc is detected dead, so the owning stage
        # can wake its idle run loop. Without it an idle StageDiffusionProc keeps
        # reporting healthy until a request happens to touch the dead client.
        # Death can be detected by EITHER the proc-monitor thread OR
        # _drain_responses seeing the death sentinel, so the flag flip + callback
        # go through _mark_engine_dead under this lock to fire the callback exactly
        # once regardless of which path (and registration timing) wins the race.
        self._on_engine_dead: Callable[[], None] | None = None
        self._engine_dead_lock = Lock()

        if self._proc_manager is not None:
            self._start_proc_monitor()

        logger.info(
            "[RuntimeV2SchedulerClient] initialized (owns_process=%s)",
            self._proc_manager is not None,
        )

    @classmethod
    def from_addresses(
        cls,
        request_address: str,
        response_address: str,
        *,
        proc_manager: RuntimeV2SchedulerProcManager | None = None,
    ) -> RuntimeV2SchedulerClient:
        """Create a client for an already-running scheduler subprocess."""
        return cls(
            request_address,
            response_address,
            proc_manager=proc_manager,
        )

    def _connect_transport(self, request_address: str, response_address: str) -> None:
        # Mirror StageDiffusionClient: the client BINDS both sockets and the
        # proc CONNECTs (see RuntimeV2SchedulerProc.run_loop).
        self.request_address = request_address
        self.response_address = response_address

        self._zmq_ctx = zmq.Context()
        self._request_socket = self._zmq_ctx.socket(zmq.PUSH)
        self._request_socket.bind(request_address)
        self._response_socket = self._zmq_ctx.socket(zmq.PULL)
        self._response_socket.bind(response_address)

        self._response_poller = zmq.asyncio.Poller()
        self._response_poller.register(self._response_socket, zmq.POLLIN)

        self._encoder = OmniMsgpackEncoder()
        self._decoder = OmniMsgpackDecoder()

    # ------------------------------------------------------------------
    # Process monitor (mirrors StageDiffusionClient._start_proc_monitor)
    # ------------------------------------------------------------------

    def _start_proc_monitor(self) -> None:
        """Watch the scheduler subprocess sentinel for silent death.

        When the subprocess dies without sending the ZMQ death sentinel
        (e.g. SIGKILL, segfault), this daemon thread sets ``_engine_dead`` so
        subsequent calls raise ``EngineDeadError``.
        """
        proc = self._proc_manager.proc
        self_ref = weakref.ref(self)

        def _monitor() -> None:
            try:
                multiprocessing.connection.wait([proc.sentinel])
            except Exception:
                return
            client = self_ref()
            if client is None or client._shutting_down or client._engine_dead:
                return
            logger.error(
                "[RuntimeV2SchedulerClient] RuntimeV2SchedulerProc died unexpectedly (exit code %s).",
                proc.exitcode,
            )
            client._mark_engine_dead()

        Thread(target=_monitor, daemon=True, name="RuntimeV2SchedulerProcMonitor").start()

    def _mark_engine_dead(self) -> None:
        """Flip ``_engine_dead`` and notify the owner, exactly once.

        Called from BOTH death-detection paths -- the proc-monitor thread AND
        ``_drain_responses`` seeing ``RUNTIME_V2_SCHEDULER_PROC_DEAD`` -- whichever
        wins the race. The ``_on_engine_dead`` callback (wired by the owning stage
        to wake its idle run loop) MUST fire regardless of which path detected
        death first, so route both here. The first caller flips the flag and
        fires the callback; later callers no-op. Skipped during graceful shutdown
        so a clean teardown is not reported as a fatal death.
        """
        with self._engine_dead_lock:
            if self._engine_dead or self._shutting_down:
                return
            self._engine_dead = True
            callback = self._on_engine_dead
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.warning("[RuntimeV2SchedulerClient] on_engine_dead callback failed", exc_info=True)

    def set_on_engine_dead(self, callback: Callable[[], None] | None) -> None:
        """Register a callback fired once when the scheduler proc is detected dead.

        The death-detection paths otherwise only flip ``_engine_dead``, so an idle
        StageDiffusionProc (blocked on recv) never notices nested-proc death. The
        stage registers a callback here that wakes its run loop. Invoked
        immediately if the proc already died before registration (race). The
        store + already-dead check happen under the same lock as
        ``_mark_engine_dead`` so the callback fires exactly once even if death
        transitions concurrently with registration.
        """
        with self._engine_dead_lock:
            self._on_engine_dead = callback
            already_dead = self._engine_dead
        if callback is not None and already_dead:
            try:
                callback()
            except Exception:
                logger.warning("[RuntimeV2SchedulerClient] on_engine_dead callback failed", exc_info=True)

    # ------------------------------------------------------------------
    # Serialization helpers (cloned from StageDiffusionClient)
    # ------------------------------------------------------------------

    # Fields that are subprocess-local and cannot cross process boundaries;
    # they are recreated in the subprocess with their default values.
    _NON_SERIALIZABLE_FIELDS = frozenset(
        {
            "generator",  # torch.Generator -- recreated from seed
            "modules",  # model components -- loaded in subprocess
        }
    )

    @staticmethod
    def _sampling_params_to_dict(sampling_params: Any) -> dict[str, Any]:
        """Convert sampling params to a plain dict for serialization.

        Uses ``dataclasses.fields`` + ``getattr`` (not ``asdict``) to avoid
        deep-copying large tensors, and skips fields that cannot cross process
        boundaries. Preserves a ``torch.Generator``'s seed so the subprocess can
        recreate deterministic random state. Identical to
        ``StageDiffusionClient._sampling_params_to_dict``.
        """
        if is_dataclass(sampling_params) and not isinstance(sampling_params, type):
            result = {
                f.name: getattr(sampling_params, f.name)
                for f in fields(sampling_params)
                if f.name not in RuntimeV2SchedulerClient._NON_SERIALIZABLE_FIELDS
            }
        elif not isinstance(sampling_params, dict):
            raise TypeError(f"sampling_params is not a dict but {sampling_params.__class__.__name__}")
        else:
            result = {
                k: v
                for k, v in sampling_params.items()
                if k not in RuntimeV2SchedulerClient._NON_SERIALIZABLE_FIELDS
            }

        if result.get("seed") is None:
            generator = (
                getattr(sampling_params, "generator", None)
                if not isinstance(sampling_params, dict)
                else sampling_params.get("generator")
            )
            if generator is not None:
                if isinstance(generator, list) and generator:
                    generator = generator[0]
                if hasattr(generator, "initial_seed"):
                    result["seed"] = generator.initial_seed()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_output(output: Any) -> DiffusionOutput:
        """Reconstruct a ``DiffusionOutput`` from a decoded ``"result"`` payload.

        ``OmniMsgpack*`` round-trips a ``DiffusionOutput`` as a plain dict of its
        fields (nested tensor / SHM-handle values are preserved by the encoder's
        hooks). Rebuild the dataclass from that dict; if a decoder ever hands
        back a real ``DiffusionOutput`` (or some other bare payload), handle both
        so the SHM handles stay intact for the engine to unpack.
        """
        if isinstance(output, DiffusionOutput):
            return output
        if isinstance(output, dict):
            try:
                return DiffusionOutput(**output)
            except TypeError:
                # Not a DiffusionOutput-shaped dict; treat it as the raw .output.
                return DiffusionOutput(output=output)
        return DiffusionOutput(output=output)

    def _drain_responses(self) -> None:
        """Non-blocking drain of all available responses from the subprocess.

        Routes ``"result"`` and ``"error"`` into ``self._results`` keyed by
        ``request_id`` (as raw ``DiffusionOutput``); the death sentinel sets
        ``_engine_dead``.
        """
        while True:
            try:
                raw = self._response_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            # Death sentinel: raw bytes, not msgpack-encoded. Route through
            # _mark_engine_dead (NOT a bare flag flip) so the owner's
            # on_engine_dead callback fires even when the drain observes death
            # before the proc-monitor thread -- otherwise the monitor returns
            # early (engine_dead already set) and the stage's fatal event is
            # never tripped, leaving an idle replica advertised healthy.
            if raw == RuntimeV2SchedulerProc.RUNTIME_V2_SCHEDULER_PROC_DEAD:
                logger.error(
                    "[RuntimeV2SchedulerClient] received RUNTIME_V2_SCHEDULER_PROC_DEAD sentinel from subprocess.",
                )
                self._mark_engine_dead()
                break

            msg = self._decoder.decode(raw)
            msg_type = msg.get("type")

            if msg_type == "result":
                request_id = msg["request_id"]
                output = msg["output"]
                # The proc sends a RAW DiffusionOutput (SHM handles kept packed).
                # OmniMsgpack does NOT register DiffusionOutput, so it arrives as
                # a plain dict of its fields (the nested tensor/SHM-handle values
                # are preserved). Reconstruct the DiffusionOutput here so the
                # engine's _materialize_runtime_v2_output (which unpacks SHM) and
                # postprocess see the same type the legacy path produced. A bare
                # (non-dict) payload is wrapped as the .output field.
                self._results[request_id] = self._reconstruct_output(output)
            elif msg_type == "error":
                request_id = msg.get("request_id")
                error_msg = msg.get("error") or "Unknown runtime_v2 scheduler subprocess error."
                status_code = msg.get("status_code")
                error_type = msg.get("error_type")
                logger.error(
                    "[RuntimeV2SchedulerClient] subprocess error for %s: %s",
                    request_id,
                    error_msg,
                )
                if request_id is not None:
                    # Surface as a DiffusionOutput carrying the error so the
                    # engine resolves the future instead of hanging forever.
                    self._results[request_id] = DiffusionOutput(
                        error=error_msg,
                        error_status_code=status_code,
                        error_type=error_type,
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_request(
        self,
        request_id: str,
        request: Any,
    ) -> None:
        """Send an ``add_request`` for ``request`` (an ``OmniDiffusionRequest``).

        Serializes ``request`` in the ``prompt`` + ``sampling_params`` dict shape
        that ``RuntimeV2SchedulerProc._reconstruct_request`` reconstructs from
        (the same shape ``StageDiffusionClient`` uses), because a raw
        ``OmniDiffusionRequest`` object does not msgpack round-trip.
        """
        if self._engine_dead:
            raise EngineDeadError()
        logger.info("[RuntimeV2SchedulerClient] add request: %s", request_id)
        frame = self._encoder.encode(
            {
                "type": "add_request",
                "request_id": request_id,
                "prompt": request.prompt,
                "sampling_params": self._sampling_params_to_dict(request.sampling_params),
                "kv_sender_info": getattr(request, "kv_sender_info", None),
            }
        )
        # This runs on the engine's event loop. A plain (blocking) send() would
        # stall the whole loop if the PUSH socket hit its high-water mark (peer
        # slow / not yet connected). Send NOBLOCK and, on zmq.Again, yield to the
        # loop and retry a bounded number of times so we never spin-block it.
        await self._send_request_nowait(frame)

    async def _send_request_nowait(self, frame: bytes) -> None:
        """Send ``frame`` on the request PUSH socket without blocking the loop.

        Uses ``zmq.NOBLOCK``; on ``zmq.Again`` (HWM reached) it ``await``s a
        short, backing-off sleep and retries. Bounded so a permanently-wedged
        peer surfaces as an error instead of hanging the caller forever.
        """
        backoff = 0.001
        max_backoff = 0.05
        deadline = asyncio.get_running_loop().time() + 30.0
        while True:
            try:
                self._request_socket.send(frame, flags=zmq.NOBLOCK)
                return
            except zmq.Again:
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(
                        "runtime_v2 add_request send timed out (request socket high-water mark); "
                        "the scheduler proc is not draining requests."
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, max_backoff)

    def get_result_nowait(self, request_id: str) -> DiffusionOutput | None:
        """Return this request's raw ``DiffusionOutput`` if it has arrived.

        Drains the ZMQ socket first (non-blocking) and pops the result for
        ``request_id``. Returns ``None`` if the terminal has not arrived yet.
        Mirrors ``StageDiffusionClient.get_diffusion_output_nowait`` but keyed
        by ``request_id`` (the engine correlates by id).
        """
        self._drain_responses()
        result = self._results.pop(request_id, None)
        if result is not None:
            return result
        if self._engine_dead:
            if self._shutting_down:
                return None
            raise EngineDeadError()
        if self._proc_manager is None:
            return None
        proc = self._proc_manager.proc
        if not self._shutting_down and not proc.is_alive():
            self._mark_engine_dead()
            exitcode = proc.exitcode
            # One final drain -- the last ZMQ frame may have arrived between the
            # first drain and the is_alive() check.
            self._drain_responses()
            result = self._results.pop(request_id, None)
            if result is not None:
                return result
            if exitcode is not None and exitcode > 128:
                sig = exitcode - 128
                logger.warning(
                    "RuntimeV2SchedulerProc was killed by signal %d; treating as external shutdown.", sig
                )
                self._shutting_down = True
                return None
            raise EngineDeadError(f"RuntimeV2SchedulerProc died unexpectedly (exit code {exitcode})")
        return None

    async def abort(self, request_ids: list[str]) -> None:
        """Abort the given request ids in the scheduler proc."""
        self.abort_nowait(request_ids)

    def abort_nowait(self, request_ids: list[str]) -> None:
        """Synchronously forward an abort to the scheduler proc.

        Uses ``NOBLOCK`` so a dead/disconnected proc peer cannot block the caller
        in ZMQ's PUSH "mute" state. Best-effort: a failed send is swallowed.
        """
        try:
            self._request_socket.send(
                self._encoder.encode({"type": "abort", "request_ids": list(request_ids)}),
                flags=zmq.NOBLOCK,
            )
        except Exception:
            logger.warning("runtime_v2 abort forward failed for %s", request_ids, exc_info=True)

    @property
    def engine_dead(self) -> bool:
        return self._engine_dead

    def check_health(self) -> None:
        """Raise ``EngineDeadError`` if the scheduler subprocess is dead."""
        if self._engine_dead:
            raise EngineDeadError("RuntimeV2SchedulerProc is dead")
        if self._proc_manager is None:
            return
        proc = self._proc_manager.proc
        if not proc.is_alive():
            self._mark_engine_dead()
            raise EngineDeadError(
                f"RuntimeV2SchedulerProc is not alive (exit code: {proc.exitcode})."
            )

    def close(self) -> None:
        self._shutting_down = True
        # Non-blocking send: a PUSH socket blocks forever in ZMQ's "mute" state
        # when it has no connected peer (e.g. the scheduler proc already exited,
        # or -- in tests -- there is no proc at all). NOBLOCK makes shutdown
        # best-effort: if the proc is still connected it receives the shutdown,
        # otherwise we fall through to closing the sockets.
        try:
            self._request_socket.send(self._encoder.encode({"type": "shutdown"}), flags=zmq.NOBLOCK)
        except Exception:
            pass

        if self._proc_manager is not None and self._proc_manager.proc.is_alive():
            self._proc_manager.shutdown(timeout=10)

        self._request_socket.close(linger=0)
        self._response_socket.close(linger=0)
        self._zmq_ctx.term()
