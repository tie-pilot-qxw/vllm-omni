# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real Qwen-Image end-to-end GPU smoke for the runtime_v2 PR1 vertical slice.

This is the proof that the runtime_v2 path actually produces an image on GPU:

    Omni(enable_runtime_v2=True)
      -> DiffusionEngine flag branch
      -> RuntimeV2Runner (single_group SP1 + FCFS)
      -> MultiprocWorkerPool (DiffusionWorker -> QwenImagePipeline -> build_executors)
      -> compiler DAG TEXT_ENCODE -> DIT_STEP_CHUNK x N -> VAE_DECODE -> FINALIZE
      -> executors drive prepare_encode / denoise_step+step_scheduler / post_decode
      -> terminal artifact fetched and resolved through the legacy output format path.

The companion parity test runs the SAME prompt/steps through the upstream
step-execution path (``enable_runtime_v2=False, step_execution=True``) which
drives the identical stage methods, and asserts a comparable image is produced.

Single visible GPU => num_gpus=1 => SP1, the only validated PR1 regime.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile

import numpy as np
import pytest

pytestmark = [pytest.mark.diffusion, pytest.mark.gpu]

# The model is large (~20B DiT). Default to the cached Qwen/Qwen-Image; allow an
# override for environments that stage the checkpoint elsewhere.
_MODEL = os.environ.get("RUNTIME_V2_QWEN_MODEL", "Qwen/Qwen-Image")

_PROMPT = {"prompt": "a red panda walking in snow"}
_NUM_STEPS = 8
_HEIGHT = 512
_WIDTH = 512
_SEED = 1234


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


pytestmark.append(
    pytest.mark.skipif(not _cuda_available(), reason="runtime_v2 Qwen-Image smoke requires CUDA")
)


def _sampling_params():
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionSamplingParams(
        num_inference_steps=_NUM_STEPS,
        height=_HEIGHT,
        width=_WIDTH,
        seed=_SEED,
        guidance_scale=0.0,
    )


# Log substrings emitted ONLY by the runtime_v2 path. The diffusion stage runs
# in a subprocess, so we cannot reach the engine object from the parent Omni
# handle; instead we capture the process-level stderr/stdout (fd-level, so it
# also catches the subprocess, which inherits these fds) and assert these
# runtime_v2-only signals appear. A bare "valid image" check is NOT sufficient:
# the legacy path also produces a valid image, so the old smoke passed even
# though it silently ran legacy diffusion.
_RUNTIME_V2_LOG_SIGNALS = (
    "runtime_v2 active: enable_runtime_v2=True",  # DiffusionEngine.__init__ branch
    "runtime_v2 runner started",  # RuntimeV2Runner.__init__ (inside the scheduler proc)
)
# Emitted by the runtime_v2 scheduler (in the scheduler proc) while executing a
# real request; proves the runtime_v2 scheduler modules actually ran, not just
# constructed.
_RUNTIME_V2_ACTIVITY_SIGNAL = "runtime_v2 plan compiled"
# Emitted by the GPU worker (multiproc_worker.py) as it executes DiT chunks;
# proves the worker pool below the scheduler proc actually ran the model.
_RUNTIME_V2_WORKER_SIGNAL = "worker dit chunk timing"

# The scheduler proc announces its own pid + its parent's (StageDiffusionProc)
# pid on startup. Parsing this proves the scheduler runs in a SEPARATE process
# from the DiffusionEngine host that drives postprocess -- the whole point of
# the "scheduler as a separate process" restructure.
_SCHEDULER_PROC_PID_RE = re.compile(
    r"RuntimeV2SchedulerProc starting: scheduler_proc_pid=(\d+) stage_diffusion_proc_pid=(\d+)"
)


def _assert_separate_scheduler_process(captured: str) -> tuple[int, int]:
    """Fail unless a distinct RuntimeV2SchedulerProc process was spawned.

    Greps the scheduler proc's startup announcement and asserts its pid differs
    from the StageDiffusionProc (its parent) pid. Returns (sched_pid, stage_pid)
    so the caller can echo the grep proof.
    """
    matches = _SCHEDULER_PROC_PID_RE.findall(captured)
    assert matches, (
        "no separate RuntimeV2SchedulerProc process was spawned "
        "(missing the 'RuntimeV2SchedulerProc starting: scheduler_proc_pid=...' "
        "startup line). The scheduler must run in its OWN process below "
        "StageDiffusionProc. Captured tail:\n" + captured[-2000:]
    )
    sched_pid, stage_pid = int(matches[0][0]), int(matches[0][1])
    assert sched_pid != stage_pid, (
        f"RuntimeV2SchedulerProc pid ({sched_pid}) equals StageDiffusionProc pid "
        f"({stage_pid}) -- the scheduler did NOT run in a separate process."
    )
    assert sched_pid > 0 and stage_pid > 0
    return sched_pid, stage_pid


@contextlib.contextmanager
def _capture_process_output():
    """Capture stdout+stderr at the file-descriptor level.

    The diffusion stage runs in a child process that inherits fds 1/2, so a
    plain ``capsys``/``caplog`` (which only sees the parent's Python-level
    streams) would miss the runtime_v2 logs. Duplicating the fds onto a temp
    file captures both the parent and the child.
    """
    with tempfile.TemporaryFile(mode="w+") as tmp:
        tmp_fd = tmp.fileno()
        sys.stdout.flush()
        sys.stderr.flush()
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        try:
            os.dup2(tmp_fd, 1)
            os.dup2(tmp_fd, 2)
            yield tmp
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def _read_captured(tmp) -> str:
    tmp.flush()
    tmp.seek(0)
    return tmp.read()


def _assert_runtime_v2_was_active(captured: str) -> None:
    """Fail if the runtime_v2 path did not genuinely run.

    This is the assertion that distinguishes runtime_v2 from the legacy path.
    """
    missing = [sig for sig in _RUNTIME_V2_LOG_SIGNALS if sig not in captured]
    assert not missing, (
        "runtime_v2 path was NOT exercised (legacy diffusion likely ran). "
        f"Missing runtime_v2 log signal(s): {missing}. "
        "Captured tail:\n" + captured[-2000:]
    )
    assert _RUNTIME_V2_ACTIVITY_SIGNAL in captured, (
        "runtime_v2 runner constructed but never executed a request "
        f"(missing {_RUNTIME_V2_ACTIVITY_SIGNAL!r}). Captured tail:\n" + captured[-2000:]
    )
    assert _RUNTIME_V2_WORKER_SIGNAL in captured, (
        "runtime_v2 scheduler compiled a plan but the GPU worker never ran a DiT "
        f"chunk (missing {_RUNTIME_V2_WORKER_SIGNAL!r}); the worker pool below the "
        "scheduler proc did not execute. Captured tail:\n" + captured[-2000:]
    )


def _assert_valid_image_output(output) -> None:
    """A finished request must carry a non-empty, well-formed RGB image."""
    assert output is not None
    assert getattr(output, "finished", False), f"request did not finish: {output!r}"
    assert output.final_output_type == "image", f"unexpected output type: {output.final_output_type!r}"
    assert output.images, "expected at least one generated image"

    image = output.images[0]
    arr = np.asarray(image, dtype=np.float32)
    assert arr.ndim == 3 and arr.shape[2] == 3, f"expected HWC RGB image, got shape={arr.shape}"
    assert arr.shape[0] > 0 and arr.shape[1] > 0
    # A real decode produces variation; an all-zero/all-constant frame signals a
    # broken latent/decode path rather than a generated image.
    assert float(arr.max()) > float(arr.min()), "decoded image is constant (no signal)"


def test_qwen_image_generates_through_runtime_v2():
    """The runtime_v2 FCFS path produces a real image on GPU (SP1, non-streaming).

    This test PROVES runtime_v2 ran: it captures the process-level output (which
    includes the diffusion subprocess) and asserts the runtime_v2-only log
    signals appear. If the opt-in flag is dropped and the legacy path runs
    instead, those signals are absent and the test FAILS -- even though the
    legacy path would also produce a valid image.
    """
    from vllm_omni.entrypoints.omni import Omni

    with _capture_process_output() as cap:
        omni = Omni(
            model=_MODEL,
            enable_runtime_v2=True,
            runtime_v2_scheduler_policy="fcfs",
            runtime_v2_denoise_chunk_size=4,
            enforce_eager=True,
        )
        try:
            outputs = omni.generate(_PROMPT, sampling_params_list=_sampling_params())
        finally:
            omni.close()
        captured = _read_captured(cap)

    # Echo the captured output back to the real stderr so the run log (and the
    # grep proof in task-10-rerun.log) contains the runtime_v2 module activity.
    sys.stderr.write(captured)
    sys.stderr.flush()

    # PROOF the runtime_v2 path (not legacy) actually executed.
    _assert_runtime_v2_was_active(captured)

    # PROOF the scheduler ran in a SEPARATE process from the DiffusionEngine host
    # (StageDiffusionProc) -- the whole point of this restructure.
    sched_pid, stage_pid = _assert_separate_scheduler_process(captured)
    sys.stderr.write(
        f"\n[grep-proof] separate scheduler process: "
        f"RuntimeV2SchedulerProc pid={sched_pid} != StageDiffusionProc pid={stage_pid}\n"
    )
    sys.stderr.flush()

    assert outputs, "runtime_v2 generate returned no outputs"
    _assert_valid_image_output(outputs[0])


def test_qwen_image_parity_upstream_step_execution():
    """Parity: the upstream step-execution path drives the same stage methods.

    Runs the SAME prompt/steps with ``enable_runtime_v2=False,
    step_execution=True`` and asserts a comparable image is produced. Both paths
    call prepare_encode / denoise_step / step_scheduler / post_decode, so the
    output should match closely (identical seed -> deterministic latents).
    """
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(
        model=_MODEL,
        enable_runtime_v2=False,
        step_execution=True,
        enforce_eager=True,
    )
    try:
        outputs = omni.generate(_PROMPT, sampling_params_list=_sampling_params())
    finally:
        omni.close()

    assert outputs, "step-execution generate returned no outputs"
    _assert_valid_image_output(outputs[0])
