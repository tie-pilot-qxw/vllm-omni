# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PR1 runtime_v2 runner: a single-group bootstrap over the FCFS scheduler.

This is the PR1 slice of the upstream ``runtime_v2/runner.py``. The full runner
supports cost-model-driven EDF/SRTF/wave-stress policies, explicit
``groups_json``/``group_sizes`` partitions, GFC collectives and async migration.
PR1 ships only the minimal end-to-end path:

  * one execution group spanning every GPU
    (``RuntimeTopology.single_group(num_gpus, parallel_spec)``),
  * a :class:`MultiprocWorkerPool`,
  * a :class:`GlobalScheduler` driven by :class:`FCFSSchedulerPolicy`,
  * the per-model adapter resolved through :func:`get_runtime_v2_adapter`.

The ``ParallelSpec`` is derived from ``od_config.parallel_config`` (tp/sp/cfg).
Everything else (cost model, alternate policies, reshard) is intentionally
absent and will land in later PRs.
"""

from __future__ import annotations

import time
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2.interfaces import RuntimeV2Adapter, SchedulerPolicy
from vllm_omni.diffusion.runtime_v2.multiproc_worker import MultiprocWorkerPool
from vllm_omni.diffusion.runtime_v2.policies import FCFSSchedulerPolicy
from vllm_omni.diffusion.runtime_v2.protocol import ParallelSpec
from vllm_omni.diffusion.runtime_v2.registry import get_runtime_v2_adapter
from vllm_omni.diffusion.runtime_v2.scheduler import GlobalScheduler, InMemoryArtifactStore
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

logger = init_logger(__name__)


class RuntimeV2Runner:
    """Bootstrap runtime_v2 runner using an explicit model adapter (PR1).

    Owns the single-group topology, worker pool, global scheduler and the
    per-model adapter. Public API mirrors the engine's expectations:
    :meth:`submit`, :meth:`poll_once`, :meth:`get_request_status`,
    :meth:`wait`, :meth:`execute`, :meth:`shutdown`.
    """

    def __init__(
        self,
        pipeline: Any | None = None,
        default_step_chunk_size: int = 1,
        *,
        scheduler_policy: str | SchedulerPolicy = "fcfs",
        vllm_config: Any = None,
        omni_diffusion_config: Any = None,
    ) -> None:
        self.default_step_chunk_size = default_step_chunk_size
        if omni_diffusion_config is None:
            omni_diffusion_config = getattr(pipeline, "od_config", None)
        if omni_diffusion_config is None:
            raise ValueError("runtime_v2 runner requires omni_diffusion_config")
        self.od_config = omni_diffusion_config
        self.adapter = self._resolve_adapter(omni_diffusion_config)
        if pipeline is not None:
            self.adapter.validate_pipeline(pipeline, omni_diffusion_config)

        topology = self._build_topology(omni_diffusion_config)
        policy = self._build_scheduler_policy(
            topology=topology,
            scheduler_policy=scheduler_policy,
        )

        worker_pool = MultiprocWorkerPool(
            topology=topology,
            od_config=omni_diffusion_config,
        )
        self.worker_pool = worker_pool

        self.scheduler = GlobalScheduler(
            topology=topology,
            worker_pool=worker_pool,
            compiler=self.adapter.build_task_compiler(
                default_denoise_chunk_size=default_step_chunk_size,
                od_config=omni_diffusion_config,
                pipeline=pipeline,
            ),
            artifact_store=InMemoryArtifactStore(),
            policy=policy,
        )
        # Start the worker pool with the configured stage init timeout so a slow
        # checkpoint load is bounded by --stage-init-timeout (forwarded onto
        # od_config), not the worker pool's hardcoded default. Fall back to that
        # default when the field is absent.
        start_timeout = getattr(self.od_config, "stage_init_timeout", None)
        self.scheduler.start(timeout_s=float(start_timeout) if start_timeout is not None else None)
        self._poll_interval_s = 0.01
        logger.info(
            "runtime_v2 runner started: model_class_name=%s adapter=%s default_step_chunk_size=%s "
            "groups=%s workers=%s policy=%s",
            getattr(omni_diffusion_config, "model_class_name", None),
            type(self.adapter).__name__,
            default_step_chunk_size,
            len(topology.groups),
            len(topology.workers),
            type(self.scheduler.policy).__name__,
        )

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_adapter(omni_diffusion_config: Any) -> RuntimeV2Adapter:
        return get_runtime_v2_adapter(getattr(omni_diffusion_config, "model_class_name", None))

    @staticmethod
    def _build_topology(omni_diffusion_config: Any) -> RuntimeTopology:
        """Derive a single execution group spanning all GPUs.

        ``num_gpus`` is taken from ``od_config`` (falling back to
        ``parallel_config.world_size``); the per-group ``ParallelSpec`` mirrors
        the configured tensor/sequence/cfg parallel degrees. PR1 uses exactly
        one group, so ``num_gpus`` must equal the parallel world size.
        """
        parallel_config = omni_diffusion_config.parallel_config
        num_gpus = int(getattr(omni_diffusion_config, "num_gpus", None) or parallel_config.world_size)
        parallel_spec = ParallelSpec(
            tp=int(parallel_config.tensor_parallel_size),
            sp=int(parallel_config.sequence_parallel_size or 1),
            cfg=int(parallel_config.cfg_parallel_size),
        )
        tp, sp, cfg = parallel_spec.tp, parallel_spec.sp, parallel_spec.cfg
        if tp > 1 or sp > 1 or cfg > 1 or num_gpus != 1:
            raise NotImplementedError(
                "runtime_v2 PR1 supports single-rank execution groups only "
                "(SP1/TP1/CFG1, num_gpus=1); got tp=%d sp=%d cfg=%d num_gpus=%d. "
                "Multi-rank groups require the artifact codec/migration layer (later PR)."
                % (tp, sp, cfg, num_gpus)
            )
        return RuntimeTopology.single_group(num_gpus=num_gpus, parallel_spec=parallel_spec)

    @staticmethod
    def _build_scheduler_policy(
        *,
        topology: RuntimeTopology,
        scheduler_policy: str | SchedulerPolicy,
    ) -> SchedulerPolicy:
        if isinstance(scheduler_policy, SchedulerPolicy):
            return scheduler_policy
        policy_name = str(scheduler_policy).lower()
        if policy_name == "fcfs":
            return FCFSSchedulerPolicy(topology=topology)
        raise ValueError(
            f"runtime_v2 PR1 supports only scheduler_policy='fcfs', got {scheduler_policy!r}"
        )

    # ------------------------------------------------------------------ #
    # Request lifecycle                                                    #
    # ------------------------------------------------------------------ #
    def _to_runtime_request(self, request: Any, *, denoise_chunk_size: int | None = None) -> Any:
        return self.adapter.normalize_request(
            request,
            denoise_chunk_size=int(denoise_chunk_size or self.default_step_chunk_size),
        )

    def submit(self, request: Any, *, denoise_chunk_size: int | None = None) -> str:
        runtime_req = self._to_runtime_request(request, denoise_chunk_size=denoise_chunk_size)
        req = runtime_req.diffusion_request
        sp = req.sampling_params
        logger.info(
            "runtime_v2 submit: request_id=%s model_class_name=%s chunk=%s steps=%s frames=%s size=%sx%s",
            runtime_req.request_id,
            getattr(self.od_config, "model_class_name", None),
            runtime_req.denoise_chunk_size,
            getattr(sp, "num_inference_steps", None),
            getattr(sp, "num_frames", None),
            getattr(sp, "width", None),
            getattr(sp, "height", None),
        )
        return self.scheduler.submit_request(runtime_req)

    def poll_once(self, timeout_s: float = 0.0) -> None:
        self.scheduler.poll_once(timeout_s=timeout_s)

    def get_request_status(self, request_id: str) -> tuple[str, Any | None]:
        return self.scheduler.get_request_status(request_id)

    def _release_request(self, request_id: str) -> None:
        # Retire all controller-side state for a request once its terminal status
        # has been delivered, so plans / artifact store / completed-output cache do
        # not accumulate for the process lifetime.
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "release_request"):
            scheduler.release_request(request_id)

    def release_request(self, request_id: str) -> None:
        self._release_request(request_id)

    def abort_request(self, request_id: str) -> None:
        """Abort an in-flight request, freeing its scheduler policy slot.

        Unlike :meth:`release_request` (which only frees controller-side state),
        this drives the scheduler's ``abort_request`` so the FCFS active slot for
        the request's group is released and any queued request is promoted --
        otherwise aborting the active request deadlocks its group. No-op if the
        scheduler does not support it (older/alternate schedulers)."""
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "abort_request"):
            scheduler.abort_request(request_id)
        else:
            self._release_request(request_id)

    def wait(self, request_id: str, *, timeout_s: float | None = None):
        if timeout_s is not None and timeout_s <= 0:
            status, payload = self.get_request_status(request_id)
            if status == "failed":
                logger.error("runtime_v2 wait failed: request_id=%s elapsed=%.3fs", request_id, 0.0)
                self._release_request(request_id)
                raise RuntimeError(payload)
            if status == "finished":
                logger.info("runtime_v2 wait finished: request_id=%s elapsed=%.3fs", request_id, 0.0)
                self._release_request(request_id)
                return payload
            self.poll_once(timeout_s=0.0)
            status, payload = self.get_request_status(request_id)
            if status == "failed":
                logger.error("runtime_v2 wait failed: request_id=%s elapsed=%.3fs", request_id, 0.0)
                self._release_request(request_id)
                raise RuntimeError(payload)
            if status == "finished":
                logger.info("runtime_v2 wait finished: request_id=%s elapsed=%.3fs", request_id, 0.0)
                self._release_request(request_id)
                return payload
            raise TimeoutError(f"request {request_id} timed out")

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        start_t = time.monotonic()
        last_progress_log_t = 0.0
        while True:
            status, payload = self.get_request_status(request_id)
            if status == "failed":
                logger.error(
                    "runtime_v2 wait failed: request_id=%s elapsed=%.3fs",
                    request_id,
                    time.monotonic() - start_t,
                )
                self._release_request(request_id)
                raise RuntimeError(payload)
            if status == "finished":
                logger.info(
                    "runtime_v2 wait finished: request_id=%s elapsed=%.3fs",
                    request_id,
                    time.monotonic() - start_t,
                )
                self._release_request(request_id)
                return payload
            now = time.monotonic()
            if now - last_progress_log_t >= 2.0:
                scheduler = getattr(self, "scheduler", None)
                if scheduler is not None:
                    progress = scheduler.get_request_progress(request_id)
                    logger.info(
                        "runtime_v2 wait progress: request_id=%s finished=%s/%s running=%s queued=%s",
                        request_id,
                        progress["finished_tasks"],
                        progress["total_tasks"],
                        progress["running_tasks"],
                        progress["pending_tasks"],
                    )
                last_progress_log_t = now
            sleep_s = self._poll_interval_s
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "runtime_v2 wait timeout: request_id=%s elapsed=%.3fs",
                        request_id,
                        time.monotonic() - start_t,
                    )
                    raise TimeoutError(f"request {request_id} timed out")
                sleep_s = min(sleep_s, remaining)
            try:
                self.poll_once(timeout_s=sleep_s)
            except Exception as exc:
                raise RuntimeError(f"runtime_v2 poll failed while waiting for {request_id}: {exc}") from exc

    def execute(self, request: Any, *, denoise_chunk_size: int | None = None, timeout_s: float | None = None):
        request_id = self.submit(request, denoise_chunk_size=denoise_chunk_size)
        return self.wait(request_id, timeout_s=timeout_s)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        self.scheduler.shutdown()
        logger.info("runtime_v2 runner shut down")

    # Backwards-compatible alias used by the upstream runner's callers.
    def close(self) -> None:
        self.shutdown()

    def check_health(self) -> None:
        if hasattr(self.worker_pool, "check_health"):
            self.worker_pool.check_health()
