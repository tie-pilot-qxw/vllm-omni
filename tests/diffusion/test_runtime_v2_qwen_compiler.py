# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from vllm_omni.diffusion.runtime_v2.protocol import TaskKind
from vllm_omni.diffusion.runtime_v2.registry import get_runtime_v2_adapter
pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]

def _fake_qwen_request(num_steps=8):
    # minimal OmniDiffusionRequest carrying sampling_params with num_inference_steps=num_steps.
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    sp = OmniDiffusionSamplingParams(num_inference_steps=num_steps, height=512, width=512)
    return OmniDiffusionRequest(request_id="r", prompt="a cat", sampling_params=sp)

def test_qwen_compiler_emits_linear_stage_dag():
    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    compiler = adapter.build_task_compiler(default_denoise_chunk_size=4)
    plan = compiler.compile_request(adapter.normalize_request(_fake_qwen_request(num_steps=8), 4))
    kinds = [t.kind for t in plan.tasks.values()]
    # PR1: prep is bundled into one TEXT_ENCODE task (prepare_encode does encode+latents+timesteps)
    assert kinds.count(TaskKind.TEXT_ENCODE) == 1
    assert kinds.count(TaskKind.DIT_STEP_CHUNK) == 2          # 8 steps / chunk 4
    assert TaskKind.VAE_DECODE in kinds and TaskKind.FINALIZE in kinds
    # linear dependency chain: TEXT_ENCODE → DIT_STEP_CHUNK[0] → DIT_STEP_CHUNK[1] → VAE_DECODE → FINALIZE
    fin = next(t for t in plan.tasks.values() if t.kind == TaskKind.FINALIZE)
    assert fin.task_id in plan.terminal_task_ids
    vae = next(t for t in plan.tasks.values() if t.kind == TaskKind.VAE_DECODE)
    assert any(plan.tasks[d].kind == TaskKind.DIT_STEP_CHUNK for d in vae.dependencies)


def test_qwen_compiler_ragged_chunk_and_step_ranges():
    # 9 steps / chunk 4 -> 3 chunks with the last chunk ragged (8..9).
    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    compiler = adapter.build_task_compiler(default_denoise_chunk_size=4)
    plan = compiler.compile_request(adapter.normalize_request(_fake_qwen_request(num_steps=9), 4))
    dits = sorted(
        (t for t in plan.tasks.values() if t.kind == TaskKind.DIT_STEP_CHUNK),
        key=lambda t: t.step_range.start,
    )
    assert [(d.step_range.start, d.step_range.end) for d in dits] == [(0, 4), (4, 8), (8, 9)]
    # Linear chain: each dit depends on the previous; chunk 0 depends on TEXT_ENCODE.
    text = next(t for t in plan.tasks.values() if t.kind == TaskKind.TEXT_ENCODE)
    assert dits[0].dependencies == (text.task_id,)
    assert dits[1].dependencies == (dits[0].task_id,)
    assert dits[2].dependencies == (dits[1].task_id,)


def test_qwen_compiler_terminal_output_handle_and_artifacts():
    from vllm_omni.diffusion.runtime_v2.protocol import (
        ArtifactKind,
        ArtifactLayout,
    )

    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    compiler = adapter.build_task_compiler(default_denoise_chunk_size=4)
    plan = compiler.compile_request(adapter.normalize_request(_fake_qwen_request(num_steps=8), 4))

    # The shared per-stage artifact is a REQUEST_STATE handle, WORKER_LOCAL.
    text = next(t for t in plan.tasks.values() if t.kind == TaskKind.TEXT_ENCODE)
    state_out = text.outputs[0]
    assert state_out.kind is ArtifactKind.REQUEST_STATE
    assert state_out.layout is ArtifactLayout.WORKER_LOCAL

    # The terminal FINALIZE output handle is an OUTPUT artifact.
    fin = next(t for t in plan.tasks.values() if t.kind == TaskKind.FINALIZE)
    assert fin.outputs[0].kind is ArtifactKind.OUTPUT
    assert plan.terminal_task_ids == (fin.task_id,)

    # The plan seeds the raw request as an initial artifact for TEXT_ENCODE.
    assert len(plan.initial_artifacts) == 1
    assert plan.initial_artifacts[0].handle.artifact_id == text.inputs[0].artifact_id


def _mock_qwen_pipeline():
    from unittest.mock import MagicMock

    pipeline = MagicMock()
    pipeline.supports_step_execution = True
    # The 4 step-exec stage methods must be present and callable.
    for name in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
        setattr(pipeline, name, MagicMock(name=name))
    return pipeline


def test_qwen_validate_pipeline_accepts_step_exec_mock():
    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    adapter.validate_pipeline(_mock_qwen_pipeline(), od_config=None)  # no raise


def test_qwen_validate_pipeline_rejects_missing_contract():
    from types import SimpleNamespace

    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    with pytest.raises(ValueError):
        adapter.validate_pipeline(None, od_config=None)
    with pytest.raises(ValueError):
        adapter.validate_pipeline(SimpleNamespace(supports_step_execution=False), od_config=None)
    # Has the flag but is missing the stage methods.
    with pytest.raises(ValueError):
        adapter.validate_pipeline(SimpleNamespace(supports_step_execution=True), od_config=None)


def test_qwen_validate_pipeline_rejects_cache_backend():
    """runtime_v2 executors bypass DiffusionModelRunner's cache refresh AND its
    'Step mode does not support cache_backend' guard, so a non-none cache backend
    would run with stale cache state. validate_pipeline must reject it at startup."""
    from types import SimpleNamespace

    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    pipeline = _mock_qwen_pipeline()

    with pytest.raises(ValueError, match="cache_backend"):
        adapter.validate_pipeline(pipeline, od_config=SimpleNamespace(cache_backend="cache_dit"))

    # 'none' / None / absent are all accepted (no raise).
    adapter.validate_pipeline(pipeline, od_config=SimpleNamespace(cache_backend="none"))
    adapter.validate_pipeline(pipeline, od_config=SimpleNamespace(cache_backend=None))
    adapter.validate_pipeline(pipeline, od_config=SimpleNamespace())


def test_qwen_build_executors_returns_four_entry_dict():
    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    executors = adapter.build_executors(_mock_qwen_pipeline())
    assert set(executors.keys()) == {
        TaskKind.TEXT_ENCODE,
        TaskKind.DIT_STEP_CHUNK,
        TaskKind.VAE_DECODE,
        TaskKind.FINALIZE,
    }
    # Each executor holds the pipeline and implements execute().
    for executor in executors.values():
        assert hasattr(executor, "pipeline")
        assert callable(executor.execute)


def test_qwen_adapter_supported_task_kinds_and_normalize():
    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    assert adapter.model_class_name == "QwenImagePipeline"
    assert TaskKind.TEXT_ENCODE in adapter.supported_task_kinds
    assert TaskKind.DIT_STEP_CHUNK in adapter.supported_task_kinds
    # normalize_request wraps an OmniDiffusionRequest and is idempotent.
    req = _fake_qwen_request(num_steps=4)
    normalized = adapter.normalize_request(req, 2)
    assert normalized.denoise_chunk_size == 2
    assert adapter.normalize_request(normalized, 8) is normalized


def test_registry_supports_only_qwen_image():
    from vllm_omni.diffusion.runtime_v2.registry import supports_runtime_v2_model

    assert supports_runtime_v2_model("QwenImagePipeline") is True
    assert supports_runtime_v2_model("WanPipeline") is False
    assert supports_runtime_v2_model(None) is False
    with pytest.raises(KeyError):
        get_runtime_v2_adapter("NopePipeline")
