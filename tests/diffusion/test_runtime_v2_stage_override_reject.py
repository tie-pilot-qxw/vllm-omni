# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""runtime_v2 opt-in is a top-level-only flag; a per-stage ``stage_<id>_*``
override for it would be silently dropped by the OrchestratorArgs blacklist,
so ``build_stage_runtime_overrides`` must reject it loudly instead."""
from __future__ import annotations

import pytest

from vllm_omni.config.stage_config import build_stage_runtime_overrides

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    "key",
    [
        "stage_0_enable_runtime_v2",
        "stage_0_runtime_v2_denoise_chunk_size",
        "stage_1_runtime_v2_scheduler_policy",
    ],
)
def test_stage_override_rejects_runtime_v2_flags(key: str) -> None:
    with pytest.raises(ValueError, match="top level"):
        build_stage_runtime_overrides(0, {key: True}, internal_keys=frozenset())


def test_stage_override_still_passes_normal_stage_keys() -> None:
    # A non-runtime_v2 per-stage knob still flows through for the matching stage.
    out = build_stage_runtime_overrides(
        0, {"stage_0_diffusion_batch_size": 4}, internal_keys=frozenset()
    )
    assert out == {"diffusion_batch_size": 4}
    # ...and is scoped to its stage: stage 1's overrides don't pick up stage 0's.
    assert build_stage_runtime_overrides(
        1, {"stage_0_diffusion_batch_size": 4}, internal_keys=frozenset()
    ) == {}
