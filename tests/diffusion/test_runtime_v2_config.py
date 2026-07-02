# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def test_enable_runtime_v2_defaults_false():
    cfg = OmniDiffusionConfig(model_class_name="QwenImagePipeline")
    assert cfg.enable_runtime_v2 is False


def test_runtime_v2_denoise_chunk_size_defaults_one():
    cfg = OmniDiffusionConfig(model_class_name="QwenImagePipeline")
    assert cfg.runtime_v2_denoise_chunk_size == 1


def test_runtime_v2_scheduler_policy_defaults_fcfs():
    cfg = OmniDiffusionConfig(model_class_name="QwenImagePipeline")
    assert cfg.runtime_v2_scheduler_policy == "fcfs"


def test_enable_runtime_v2_can_be_set_true():
    cfg = OmniDiffusionConfig(
        model_class_name="QwenImagePipeline",
        enable_runtime_v2=True,
        runtime_v2_denoise_chunk_size=8,
        runtime_v2_scheduler_policy="fifo",
    )
    assert cfg.enable_runtime_v2 is True
    assert cfg.runtime_v2_denoise_chunk_size == 8
    assert cfg.runtime_v2_scheduler_policy == "fifo"


def test_stage_init_timeout_defaults_300():
    cfg = OmniDiffusionConfig(model_class_name="QwenImagePipeline")
    assert cfg.stage_init_timeout == 300


def test_stage_init_timeout_survives_from_kwargs():
    # from_kwargs filters to declared fields; the field must exist so the nested
    # runtime_v2 scheduler proc reads the user's --stage-init-timeout (forwarded
    # via stage_engine_args) instead of the hardcoded 300s default.
    cfg = OmniDiffusionConfig.from_kwargs(
        model_class_name="QwenImagePipeline", stage_init_timeout=900
    )
    assert cfg.stage_init_timeout == 900
