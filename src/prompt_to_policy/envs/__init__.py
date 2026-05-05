"""Environment registry: ``halfcheetah``, ``hopper``, ``ant``.

Importing this package registers all three envs. Use ``get(name)`` to
look up an ``EnvSpec`` (env id, make_env, features, prompt template).
"""

from __future__ import annotations

from . import ant, halfcheetah, hopper
from .registry import EnvSpec, _register, get, list_envs

_register(halfcheetah.SPEC)
_register(hopper.SPEC)
_register(ant.SPEC)

__all__ = [
    "EnvSpec",
    "ant",
    "get",
    "halfcheetah",
    "hopper",
    "list_envs",
]
