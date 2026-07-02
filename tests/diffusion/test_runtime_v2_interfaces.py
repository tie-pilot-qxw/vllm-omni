# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from abc import ABC
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle, ArtifactKind, ArtifactLayout, ArtifactValue,
    ArtifactValue as AV_via_protocol,
    InferenceTask, ParallelSpec, RequestExecutionPlan, TaskKind, WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.interfaces import (
    ArtifactLayoutCodec, ArtifactStore, RuntimeV2Adapter,
    SchedulerPolicy, TaskCompiler, WorkerExecutor,
    ArtifactValue as AV_via_interfaces,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def test_five_abcs_are_abstract():
    """All five runtime ABCs cannot be instantiated without implementing abstractmethods."""
    for cls in (TaskCompiler, ArtifactStore, SchedulerPolicy, WorkerExecutor, RuntimeV2Adapter):
        assert issubclass(cls, ABC), f"{cls.__name__} must be ABC"
        with pytest.raises(TypeError):
            cls()  # type: ignore[abstract]


def test_scheduler_policy_hold_methods_have_default_noop():
    """acquire/release_migration_hold are non-abstract; concrete subclass need not override."""
    class MinimalPolicy(SchedulerPolicy):
        def on_request_submitted(self, plan):
            return ()
        def on_tasks_runnable(self, tasks):
            return ()
        def on_worker_event(self, event):
            return ()

    policy = MinimalPolicy()
    # Must not raise — default is a no-op
    policy.acquire_migration_hold((0, 1))
    policy.release_migration_hold((0, 1))


def test_artifact_layout_codec_is_abstract():
    """ArtifactLayoutCodec is an ABC with abstract methods."""
    assert issubclass(ArtifactLayoutCodec, ABC)
    with pytest.raises(TypeError):
        ArtifactLayoutCodec()  # type: ignore[abstract]


def test_artifact_value_importable_from_protocol():
    """ArtifactValue is importable from protocol (not only from interfaces)."""
    h = ArtifactHandle(
        request_id="r", artifact_id="out", kind=ArtifactKind.OUTPUT,
        layout=ArtifactLayout.HOST,
    )
    av = ArtifactValue(handle=h, value=b"bytes")
    assert av.handle is h
    assert av.value == b"bytes"


def test_artifact_value_reexported_from_interfaces():
    """ArtifactValue re-exported from interfaces resolves to the same object as protocol.ArtifactValue."""
    assert AV_via_interfaces is AV_via_protocol
