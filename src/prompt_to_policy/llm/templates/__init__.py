"""Per-env system prompt builders."""

from __future__ import annotations

from ._base import PROMPT_VERSION
from .ant import build_system_prompt as build_ant_system_prompt
from .halfcheetah import build_system_prompt as build_halfcheetah_system_prompt
from .hopper import build_system_prompt as build_hopper_system_prompt

__all__ = [
    "PROMPT_VERSION",
    "build_ant_system_prompt",
    "build_halfcheetah_system_prompt",
    "build_hopper_system_prompt",
    "build_system_prompt",
]


# Legacy name retained for callers that imported the HalfCheetah builder directly.
build_system_prompt = build_halfcheetah_system_prompt
