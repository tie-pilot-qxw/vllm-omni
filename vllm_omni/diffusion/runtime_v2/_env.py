# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared environment-variable parsing for runtime_v2.

Centralizes boolean flag parsing so the same env var is interpreted identically
everywhere. Previously several modules each inlined
``os.environ.get(...).strip().lower() in {...}`` with *different* accepted-value
sets, so e.g. ``y`` was truthy in one module but not another.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def env_flag(name: str, default: bool = False) -> bool:
    """Return the boolean value of env var ``name``.

    Empty/unset -> ``default``. Recognized truthy/falsey strings are matched
    case-insensitively; anything else logs a warning and falls back to ``default``.
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    logger.warning("runtime_v2 invalid bool env %s=%r, using default=%s", name, raw, default)
    return default
