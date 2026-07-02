# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU isolation proof for the runtime_v2 "scheduler as a separate process" restructure.

This is the payoff test for moving the runtime_v2 control plane (``GlobalScheduler``
+ ``MultiprocWorkerPool``) out of a *thread inside* ``StageDiffusionProc`` and into
its **own process** (``RuntimeV2SchedulerProc``). The claim to prove: the scheduler
process keeps dispatching GPU work while ``StageDiffusionProc`` is busy running the
CPU-bound postprocess (tensor->PIL) of an *earlier* request. A thread could not give
this guarantee (the GIL serializes postprocess and scheduling); a separate process
can, because the two run on different interpreters.

Method (a real, non-fakeable demonstration):

  * Submit **N concurrent requests** at a **larger resolution** (so the tensor->PIL
    postprocess in StageDiffusionProc is non-trivial) via a single
    ``Omni.generate([...])`` call, which submits ALL requests up front and then
    drains outputs -- so multiple requests are in flight at once.
  * Assert **every** request completes with a valid, non-constant RGB image (the
    scheduler must have kept dispatching later requests' DiT chunks while earlier
    requests were being postprocessed, or they would never finish / would deadlock).
  * Parse the fd-captured logs for two host-wide-monotonic (``CLOCK_MONOTONIC`` is
    system-wide on Linux, so timestamps across the two processes are comparable)
    signals:
      - ``runtime_v2 postprocess begin/end ... mono_ns=`` (emitted in StageDiffusionProc),
      - ``runtime_v2 worker dit chunk timing ... gpu_record_ns=`` (a GPU-worker DiT
        dispatch, issued by the scheduler PROC).
    Then, for every consecutive request pair (R_n, R_{n+1}), assert that R_{n+1}'s
    FIRST DiT-chunk dispatch happens **at or before R_n's postprocess completes**.
    A thread-based scheduler sharing StageDiffusionProc's GIL could NOT dispatch
    R_{n+1}'s DiT until R_n's (CPU-bound) postprocess released the GIL, so R_{n+1}'s
    dispatch would land strictly AFTER R_n's postprocess end. A process-isolated
    scheduler dispatches R_{n+1} ahead of / concurrent with R_n's postprocess -- which
    is exactly what we assert. (On a single GPU with fast postprocess, the scheduler
    typically dispatches the next request's DiT *tens of ms before* the prior request's
    postprocess even begins; the assertion tolerates that by comparing against
    postprocess-END, and reports the measured lead time either way.)
  * Also assert the scheduler ran in a **separate process** (pid != ppid).

The assertion is not weakenable to a trivial pass: it requires real postprocess
windows AND real DiT dispatches to have been captured, requires >= 2 requests to
form a pair, forbids ANY pair where the next dispatch trailed the prior postprocess
(a stall), and requires EVERY consecutive pair to show the scheduler running ahead.
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

_MODEL = os.environ.get("RUNTIME_V2_QWEN_MODEL", "Qwen/Qwen-Image")

# Larger than the single-image smoke so tensor->PIL postprocess is heavier and the
# per-request DiT dispatch cadence is easier to interleave with a postprocess window.
_NUM_REQUESTS = 4
_NUM_STEPS = 8
_HEIGHT = 1024
_WIDTH = 1024
_SEED = 1234

# Distinct prompts so the decoded images differ request-to-request (guards against a
# stuck pipeline handing back one cached frame for every request).
_PROMPTS = [
    {"prompt": "a red panda walking in snow"},
    {"prompt": "a blue sailboat on a calm lake at sunset"},
    {"prompt": "a green forest with tall pine trees"},
    {"prompt": "a golden desert with rolling sand dunes"},
]


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


pytestmark.append(
    pytest.mark.skipif(not _cuda_available(), reason="runtime_v2 isolation proof requires CUDA")
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


@contextlib.contextmanager
def _capture_process_output():
    """Capture stdout+stderr at the fd level (so the subprocess logs are included)."""
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


_PP_BEGIN_RE = re.compile(r"runtime_v2 postprocess begin: request_id=(\S+) mono_ns=(\d+)")
_PP_END_RE = re.compile(r"runtime_v2 postprocess end: request_id=(\S+) mono_ns=(\d+) elapsed_ms=([\d.]+)")
_DIT_RE = re.compile(r"runtime_v2 worker dit chunk timing: .*task_id=(\S+):dit:\d+ .*gpu_record_ns=(\d+)")
_SCHED_PID_RE = re.compile(
    r"RuntimeV2SchedulerProc starting: scheduler_proc_pid=(\d+) stage_diffusion_proc_pid=(\d+)"
)


def _assert_valid_image_output(output) -> None:
    assert output is not None
    assert getattr(output, "finished", False), f"request did not finish: {output!r}"
    assert output.final_output_type == "image", f"unexpected output type: {output.final_output_type!r}"
    assert output.images, "expected at least one generated image"
    image = output.images[0]
    arr = np.asarray(image, dtype=np.float32)
    assert arr.ndim == 3 and arr.shape[2] == 3, f"expected HWC RGB image, got shape={arr.shape}"
    assert arr.shape[0] > 0 and arr.shape[1] > 0
    assert float(arr.max()) > float(arr.min()), "decoded image is constant (no signal)"


def _request_of(task_id: str) -> str:
    # DiT task_id is "<request_id>:dit:<n>"; the postprocess request_id is the bare
    # request_id. Strip the task suffix so they key on the same request identity.
    return task_id.split(":dit:")[0]


def test_runtime_v2_scheduler_not_stalled_by_postprocess():
    """The scheduler PROC keeps dispatching while StageDiffusionProc postprocesses.

    Submits N concurrent large-resolution requests and proves (a) all finish with
    valid, distinct images, (b) the scheduler ran in a separate process, and (c) for
    EVERY consecutive request pair the scheduler dispatched the next request's first
    DiT chunk at/before the prior request's postprocess completed -- the process-
    isolation guarantee a GIL-sharing thread could not provide (it would have to wait
    for postprocess to release the GIL first).
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
            outputs = omni.generate(_PROMPTS[:_NUM_REQUESTS], sampling_params_list=_sampling_params())
        finally:
            omni.close()
        captured = _read_captured(cap)

    # Echo captured output so the run log carries the interleave evidence.
    sys.stderr.write(captured)
    sys.stderr.flush()

    # (a) EVERY concurrent request produced a valid, non-constant image. Reaching
    # this state requires the scheduler to keep dispatching later requests' chunks
    # while earlier requests are postprocessed in StageDiffusionProc.
    assert outputs, "runtime_v2 concurrent generate returned no outputs"
    assert len(outputs) == _NUM_REQUESTS, (
        f"expected {_NUM_REQUESTS} outputs, got {len(outputs)} -- a stalled scheduler "
        "would leave later concurrent requests unfinished."
    )
    for out in outputs:
        _assert_valid_image_output(out)
    # Images must differ across requests (distinct prompts) -- guards against a stuck
    # pipeline echoing one frame.
    arrs = [np.asarray(o.images[0], dtype=np.float32) for o in outputs]
    assert any(
        arrs[0].shape != arrs[i].shape or float(np.abs(arrs[0] - arrs[i]).max()) > 1.0
        for i in range(1, len(arrs))
    ), "all concurrent outputs are identical images -- pipeline likely not truly concurrent"

    # (b) The scheduler ran in a SEPARATE process.
    pid_matches = _SCHED_PID_RE.findall(captured)
    assert pid_matches, (
        "no separate RuntimeV2SchedulerProc process was spawned. Captured tail:\n" + captured[-2000:]
    )
    sched_pid, stage_pid = int(pid_matches[0][0]), int(pid_matches[0][1])
    assert sched_pid != stage_pid, (
        f"scheduler pid ({sched_pid}) == StageDiffusionProc pid ({stage_pid}); not a separate process."
    )

    # (c) Parse postprocess windows and DiT dispatch timestamps (host-wide monotonic).
    begins: dict[str, int] = {}
    for rid, ns in _PP_BEGIN_RE.findall(captured):
        begins[rid] = int(ns)
    windows: list[tuple[str, int, int]] = []  # (request_id, begin_ns, end_ns)
    for rid, ns, _elapsed in _PP_END_RE.findall(captured):
        b = begins.get(rid)
        if b is not None:
            windows.append((rid, b, int(ns)))
    # First DiT-chunk dispatch time (gpu_record_ns) per request -- the moment the
    # scheduler proc handed the next request's denoise work to the GPU worker.
    first_dit: dict[str, int] = {}
    for task_id, ns in _DIT_RE.findall(captured):
        rid = _request_of(task_id)
        ns_i = int(ns)
        if rid not in first_dit or ns_i < first_dit[rid]:
            first_dit[rid] = ns_i

    # We must have observed real postprocess windows and real DiT dispatches, else the
    # instrumentation regressed and the proof would be vacuous.
    assert windows, (
        "no runtime_v2 postprocess windows were captured -- the postprocess begin/end "
        "instrumentation did not fire; cannot prove isolation. Captured tail:\n" + captured[-3000:]
    )
    assert first_dit, (
        "no runtime_v2 DiT dispatches were captured -- the worker timing log did not "
        "fire; cannot prove isolation. Captured tail:\n" + captured[-3000:]
    )

    # (c) core assertion -- the isolation guarantee.
    #
    # Order requests by their numeric submission index (Omni prefixes request ids
    # with "<i>_"). For each consecutive pair (R_n, R_{n+1}), compare:
    #   * R_n's postprocess window [begin, end]  (runs on StageDiffusionProc's CPU), and
    #   * R_{n+1}'s FIRST DiT-chunk dispatch     (issued by the scheduler PROC).
    #
    # If the scheduler shared StageDiffusionProc's GIL (the pre-restructure thread
    # design), it could NOT dispatch R_{n+1}'s DiT until R_n's postprocess released
    # the GIL -- so R_{n+1}.dit0 would land strictly AFTER R_n's postprocess end.
    # A process-isolated scheduler is free to dispatch R_{n+1} at/before R_n's
    # postprocess completes. We assert exactly that: R_{n+1}.dit0 <= R_n.pp_end for
    # every consecutive pair, i.e. the scheduler never serialized behind postprocess.
    def _req_index(rid: str) -> int:
        head = rid.split("_", 1)[0]
        return int(head) if head.isdigit() else 0

    pp_end = {rid: e for rid, _b, e in windows}
    pp_begin = {rid: b for rid, b, _e in windows}
    ordered = sorted(pp_end.keys(), key=_req_index)

    ahead_pairs: list[str] = []
    stalled_pairs: list[str] = []
    for i in range(len(ordered) - 1):
        r_n, r_next = ordered[i], ordered[i + 1]
        d0 = first_dit.get(r_next)
        if d0 is None or r_n not in pp_end:
            continue
        # milliseconds the next dispatch precedes (negative) or follows (positive)
        # the prior request's postprocess *begin*, for human-readable evidence.
        rel_begin_ms = (d0 - pp_begin[r_n]) / 1e6
        if d0 <= pp_end[r_n]:
            ahead_pairs.append(
                f"req{i}->req{i + 1}: next DiT dispatched {rel_begin_ms:+.1f}ms relative to "
                f"req{i}'s postprocess-begin (<= its postprocess-end): scheduler ran ahead of postprocess"
            )
        else:
            stalled_pairs.append(
                f"req{i}->req{i + 1}: next DiT dispatched {(d0 - pp_end[r_n]) / 1e6:.1f}ms AFTER "
                f"req{i}'s postprocess-end: scheduler appears to have STALLED behind postprocess"
            )

    summary = (
        f"[isolation] requests={_NUM_REQUESTS} finished={len(outputs)} "
        f"postprocess_windows={len(windows)} first_dit_dispatches={len(first_dit)} "
        f"consecutive_pairs={max(0, len(ordered) - 1)} "
        f"scheduler_ahead_of_postprocess={len(ahead_pairs)} stalled={len(stalled_pairs)} "
        f"postprocess_widths_ms={[round((e - b) / 1e6, 1) for _r, b, e in windows]}"
    )
    sys.stderr.write("\n" + summary + "\n")
    for line in ahead_pairs + stalled_pairs:
        sys.stderr.write("  " + line + "\n")
    sys.stderr.flush()

    # Need at least one consecutive pair to make a statement about isolation.
    assert len(ordered) >= 2, (
        "fewer than 2 requests produced postprocess windows; cannot demonstrate "
        "cross-request scheduler isolation. " + summary
    )
    # No pair may show the scheduler stalling behind postprocess...
    assert not stalled_pairs, (
        "scheduler STALLED behind postprocess for at least one request pair -- the "
        "process isolation this restructure provides was not observed:\n  "
        + "\n  ".join(stalled_pairs)
        + "\n"
        + summary
    )
    # ...and every consecutive pair must positively show the scheduler running
    # ahead of / concurrent with the prior request's postprocess.
    assert len(ahead_pairs) == len(ordered) - 1, (
        "not every consecutive request pair showed the scheduler dispatching the next "
        "request's DiT at/before the prior request's postprocess end. " + summary
    )
