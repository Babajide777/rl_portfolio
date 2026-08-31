"""Tests for validation-based checkpoint selection and evaluation paths."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config as cfg
from src.evaluate import _model_paths
from src.selection import (
    latest_checkpoint,
    list_checkpoints,
    select_checkpoint,
    selected_model_paths,
)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Isolate run output under a temporary directory."""
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(cfg, "RUNS_DIR", root)
    return root


@pytest.fixture
def run(runs_dir):
    return cfg.RunConfig(penalty_weight=0.0, seed=0)


def _write_checkpoint(run_dir: Path, steps: int, with_vec: bool = True) -> None:
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / f"ppo_{steps}_steps.zip").write_bytes(b"model")
    if with_vec:
        (ckpt / f"ppo_vecnormalize_{steps}_steps.pkl").write_bytes(b"vec")


def test_list_checkpoints_parses_step_counts(run, runs_dir):
    run_dir = runs_dir / run.label
    _write_checkpoint(run_dir, 50_000)
    _write_checkpoint(run_dir, 100_000)

    ckpts = list_checkpoints(run)
    assert [s for s, _, _ in ckpts] == [50_000, 100_000]
    assert ckpts[0][2] is not None
    assert ckpts[0][1].name == "ppo_50000_steps.zip"


def test_latest_checkpoint_returns_highest_steps(run, runs_dir):
    run_dir = runs_dir / run.label
    _write_checkpoint(run_dir, 50_000)
    _write_checkpoint(run_dir, 100_000)

    steps, path, _ = latest_checkpoint(run)
    assert steps == 100_000
    assert path.name == "ppo_100000_steps.zip"


def test_selected_model_paths_uses_checkpoint_when_json_present(run, runs_dir):
    run_dir = runs_dir / run.label
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoints" / "ppo_50000_steps.zip"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"model")
    vec = run_dir / "checkpoints" / "ppo_vecnormalize_50000_steps.pkl"
    vec.write_bytes(b"vec")
    (run_dir / "model.zip").write_bytes(b"final")

    record = {
        "fallback": False,
        "selected_path": str(ckpt),
        "selected_vecnormalize": str(vec),
    }
    (run_dir / "checkpoint_selection.json").write_text(json.dumps(record))

    model_path, vec_path = selected_model_paths(run)
    assert model_path == ckpt
    assert vec_path == vec


def test_selected_model_paths_falls_back_to_final_model(run, runs_dir):
    run_dir = runs_dir / run.label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.zip").write_bytes(b"final")
    (run_dir / "vecnormalize.pkl").write_bytes(b"vec")

    model_path, vec_path = selected_model_paths(run)
    assert model_path == run_dir / "model.zip"
    assert vec_path == run_dir / "vecnormalize.pkl"


def test_select_checkpoint_fallback_when_no_checkpoints_and_no_model(run, runs_dir):
    run_dir = runs_dir / run.label
    run_dir.mkdir(parents=True, exist_ok=True)

    rec = select_checkpoint(run, val_rel=_dummy_val())
    assert rec["fallback"] is True
    assert rec["error"] == "no checkpoints and no final model"
    assert rec["selected_path"] is None


def test_select_checkpoint_fallback_to_final_model(run, runs_dir):
    run_dir = runs_dir / run.label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.zip").write_bytes(b"final")

    rec = select_checkpoint(run, val_rel=_dummy_val())
    assert rec["fallback"] is True
    assert rec["selected_path"] == str(run_dir / "model.zip")
    assert "error" not in rec


def test_model_paths_use_final_model_when_requested(run, runs_dir):
    run_dir = runs_dir / run.label
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoints" / "ppo_50000_steps.zip"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"model")
    (run_dir / "model.zip").write_bytes(b"final")
    (run_dir / "vecnormalize.pkl").write_bytes(b"vec")
    (run_dir / "checkpoint_selection.json").write_text(json.dumps({
        "fallback": False,
        "selected_path": str(ckpt),
        "selected_vecnormalize": None,
    }))

    selected = _model_paths(run, use_final_model=False)
    final = _model_paths(run, use_final_model=True)

    assert selected[0] == ckpt
    assert final[0] == run_dir / "model.zip"
    assert final[1] == run_dir / "vecnormalize.pkl"


def _dummy_val():
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(1.0004, 0.011, (200, 9)),
        columns=list(cfg.ASSETS),
    )


# ---------------------------------------------------------------------------
# Batch skip logic
# ---------------------------------------------------------------------------
def test_smoke_run_gets_distinct_label():
    """A smoke run must not occupy the directory a full run would use.

    RunConfig.label encodes eta and seed only, so without this the 20k-step
    smoke model would be mistaken for a completed 2M-step run and silently
    skipped in the batch.
    """
    full = cfg.RunConfig(penalty_weight=0.005, seed=0)
    smoke = cfg.RunConfig(penalty_weight=0.005, seed=0, total_timesteps=20_000)
    assert full.label != smoke.label
    assert "smoke" in smoke.label


def test_completed_timesteps_detects_undertrained_run(tmp_path, monkeypatch):
    """An undertrained model must not be treated as complete."""
    import json
    from src.train import completed_timesteps

    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path)
    run = cfg.RunConfig(penalty_weight=0.005, seed=0)

    assert completed_timesteps(run) == 0            # nothing saved yet

    (run.dir / "model.zip").write_bytes(b"stub")
    (run.dir / "metadata.json").write_text(json.dumps({"total_timesteps": 20_000}))
    assert completed_timesteps(run) == 20_000
    assert completed_timesteps(run) < run.total_timesteps   # -> will retrain

    (run.dir / "metadata.json").write_text(
        json.dumps({"total_timesteps": cfg.TOTAL_TIMESTEPS}))
    assert completed_timesteps(run) >= run.total_timesteps  # -> will skip


def test_completed_timesteps_handles_corrupt_metadata(tmp_path, monkeypatch):
    """Unreadable metadata must be treated as untrained, not crash."""
    from src.train import completed_timesteps
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path)
    run = cfg.RunConfig(penalty_weight=0.005, seed=1)
    (run.dir / "model.zip").write_bytes(b"stub")
    (run.dir / "metadata.json").write_text("{not valid json")
    assert completed_timesteps(run) == 0
