"""CLI parsing tests. The actual training and LLM paths are exercised
elsewhere; here we verify tyro wires up the subcommands and flags
correctly.
"""

from __future__ import annotations

import pytest
import tyro

from prompt_to_policy.cli import (
    Command,
    Generate,
    Run,
    TrainSpec,
    _resolve_timesteps,
)
from prompt_to_policy.train import HALFCHEETAH_PPO


def test_run_parses_positional_prompt():
    cmd = tyro.cli(Command, args=["run", "make the cheetah run forward"])
    assert isinstance(cmd, Run)
    assert cmd.prompt == "make the cheetah run forward"
    assert cmd.timesteps is None
    assert cmd.quick is False
    assert cmd.no_video is False


def test_run_quick_flag():
    cmd = tyro.cli(Command, args=["run", "stand still", "--quick"])
    assert isinstance(cmd, Run)
    assert cmd.quick is True


def test_run_explicit_timesteps():
    cmd = tyro.cli(Command, args=["run", "stand still", "--timesteps", "50000"])
    assert isinstance(cmd, Run)
    assert cmd.timesteps == 50000


def test_run_no_video_flag():
    cmd = tyro.cli(Command, args=["run", "stand still", "--no-video"])
    assert isinstance(cmd, Run)
    assert cmd.no_video is True


def test_run_env_defaults_to_halfcheetah():
    cmd = tyro.cli(Command, args=["run", "go forward"])
    assert isinstance(cmd, Run)
    assert cmd.env == "halfcheetah"


def test_run_env_hopper():
    cmd = tyro.cli(Command, args=["run", "hop forward", "--env", "hopper"])
    assert isinstance(cmd, Run)
    assert cmd.env == "hopper"


def test_run_env_ant():
    cmd = tyro.cli(Command, args=["run", "walk forward", "--env", "ant"])
    assert isinstance(cmd, Run)
    assert cmd.env == "ant"


def test_run_provider_gemini():
    cmd = tyro.cli(Command, args=["run", "go forward", "--provider", "gemini"])
    assert isinstance(cmd, Run)
    assert cmd.provider == "gemini"


def test_run_provider_local():
    cmd = tyro.cli(Command, args=["run", "go forward", "--provider", "local"])
    assert isinstance(cmd, Run)
    assert cmd.provider == "local"


def test_generate_parses():
    cmd = tyro.cli(Command, args=["generate", "stand still"])
    assert isinstance(cmd, Generate)
    assert cmd.prompt == "stand still"
    assert cmd.force_refresh is False


def test_generate_force_refresh():
    cmd = tyro.cli(Command, args=["generate", "x", "--force-refresh"])
    assert isinstance(cmd, Generate)
    assert cmd.force_refresh is True


def test_train_spec_parses_path(tmp_path):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text("{}")  # path is just parsed as a Path; not loaded yet
    cmd = tyro.cli(Command, args=["train-spec", str(spec_file)])
    assert isinstance(cmd, TrainSpec)
    assert cmd.spec == spec_file


def test_resolve_timesteps_explicit_wins():
    # Explicit --timesteps wins over --quick.
    assert _resolve_timesteps("halfcheetah", timesteps=42, quick=True) == 42


def test_resolve_timesteps_quick_when_no_explicit():
    assert (
        _resolve_timesteps("halfcheetah", timesteps=None, quick=True)
        == HALFCHEETAH_PPO.quick_timesteps
    )


def test_resolve_timesteps_default():
    assert (
        _resolve_timesteps("halfcheetah", timesteps=None, quick=False)
        == HALFCHEETAH_PPO.total_timesteps
    )


def test_resolve_timesteps_uses_per_env_config():
    from prompt_to_policy.train import HOPPER_PPO

    assert (
        _resolve_timesteps("hopper", timesteps=None, quick=True) == HOPPER_PPO.quick_timesteps
    )


def test_unknown_subcommand_errors():
    with pytest.raises(SystemExit):
        tyro.cli(Command, args=["whoknows", "foo"])
