"""
Validation-based checkpoint selection (Section 3.5).

Training to a fixed budget and evaluating the final model assumes that more
training is monotonically better. That is not true here: the agent replays
the same ~3,500 trading days several hundred times, so beyond some point it
memorises the training period rather than learning a transferable policy.

This module gives the validation partition its role. Each checkpoint written
during training is rolled out on validation, and the checkpoint achieving the
highest net-of-cost Sharpe ratio is carried forward to test evaluation. The
test partition is never consulted in this decision.

Selection is cheap: it requires rollouts, not retraining.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from . import config as cfg
from . import metrics as mt

log = logging.getLogger(__name__)

_STEP_RE = re.compile(r"_(\d+)_steps")


def list_checkpoints(run: cfg.RunConfig) -> list[tuple[int, Path, Path | None]]:
    """Return (timesteps, model_path, vecnormalize_path) sorted by timesteps.

    CheckpointCallback writes ``ppo_<n>_steps.zip`` and, with
    ``save_vecnormalize=True``, ``ppo_vecnormalize_<n>_steps.pkl``.
    """
    ckpt_dir = run.dir / "checkpoints"
    if not ckpt_dir.exists():
        return []

    found = []
    for model_path in sorted(ckpt_dir.glob("ppo_*_steps.zip")):
        m = _STEP_RE.search(model_path.stem)
        if not m:
            continue
        steps = int(m.group(1))
        vec = ckpt_dir / f"ppo_vecnormalize_{steps}_steps.pkl"
        found.append((steps, model_path, vec if vec.exists() else None))
    return sorted(found, key=lambda x: x[0])


def latest_checkpoint(run: cfg.RunConfig) -> tuple[int, Path, Path | None] | None:
    """Most recent checkpoint, used by the resume logic in train.py."""
    ckpts = list_checkpoints(run)
    return ckpts[-1] if ckpts else None


def _rollout_checkpoint(model_path: Path, vec_path: Path | None,
                        rel: pd.DataFrame, penalty_weight: float,
                        cost_rate: float) -> pd.DataFrame:
    """Deterministic rollout of a saved checkpoint over one partition."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from .portfolio_env import PortfolioEnv

    env = PortfolioEnv(rel, cost_rate=cost_rate, penalty_weight=penalty_weight,
                       record_history=True)
    vec = DummyVecEnv([lambda: env])
    if vec_path is not None:
        # Without the saved statistics the network receives unnormalised
        # observations and the rollout is silently meaningless.
        vec = VecNormalize.load(str(vec_path), vec)
        vec.training = False
        vec.norm_reward = False

    model = PPO.load(str(model_path), device=cfg.DEVICE)
    obs = vec.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec.step(action)
        done = bool(dones[0])
    vec.close()
    return env.history


def select_checkpoint(
    run: cfg.RunConfig,
    val_rel: pd.DataFrame,
    metric: str = "sharpe",
    cost_rate: float | None = None,
) -> dict:
    """Evaluate every checkpoint on validation and select the best.

    Returns a record of the selection, also written to
    ``runs/<label>/checkpoint_selection.json`` together with the full
    validation curve, which supports the convergence claim in Chapter 5.
    """
    c = cfg.COST_RATE if cost_rate is None else cost_rate
    ckpts = list_checkpoints(run)

    if not ckpts:
        final_model = run.dir / "model.zip"
        if not final_model.exists():
            log.warning("%s: no checkpoints and no final model", run.label)
            return {
                "label": run.label,
                "selected_steps": None,
                "selected_path": None,
                "fallback": True,
                "error": "no checkpoints and no final model",
                "curve": [],
            }
        log.warning("%s: no checkpoints; falling back to final model", run.label)
        return {"label": run.label, "selected_steps": run.total_timesteps,
                "selected_path": str(final_model),
                "fallback": True, "curve": []}

    curve = []
    for steps, model_path, vec_path in ckpts:
        hist = _rollout_checkpoint(model_path, vec_path, val_rel,
                                   run.penalty_weight, c)
        m = mt.compute_metrics(hist)
        curve.append({"steps": steps, "path": str(model_path),
                      "vecnormalize": str(vec_path) if vec_path else None,
                      **{k: m[k] for k in
                         ("sharpe", "sortino", "calmar", "max_drawdown",
                          "ann_turnover", "total_cost", "final_value")}})
        log.debug("%s @ %d steps: val %s = %.4f", run.label, steps, metric, m[metric])

    best = max(curve, key=lambda r: r[metric])
    record = {
        "label": run.label,
        "selection_metric": metric,
        "selection_partition": "validation",
        "n_checkpoints": len(curve),
        "selected_steps": best["steps"],
        "selected_path": best["path"],
        "selected_vecnormalize": best["vecnormalize"],
        "selected_val_metric": best[metric],
        "final_val_metric": curve[-1][metric],
        # If the best checkpoint is well short of the budget, the training
        # budget was sufficient and later training degraded generalisation.
        "selected_fraction_of_budget": best["steps"] / max(curve[-1]["steps"], 1),
        "fallback": False,
        "curve": curve,
    }
    (run.dir / "checkpoint_selection.json").write_text(
        json.dumps(record, indent=2, default=float))
    pd.DataFrame(curve).to_csv(run.dir / "validation_curve.csv", index=False)

    log.info("%s: selected %d steps (%.0f%% of budget), val %s %.4f "
             "(final checkpoint %.4f)",
             run.label, best["steps"], 100 * record["selected_fraction_of_budget"],
             metric, best[metric], curve[-1][metric])
    return record


def select_all(val_rel: pd.DataFrame, metric: str = "sharpe") -> pd.DataFrame:
    """Run checkpoint selection across the whole batch."""
    rows = []
    for run in cfg.all_runs():
        if not (run.dir / "model.zip").exists():
            continue
        rec = select_checkpoint(run, val_rel, metric=metric)
        if rec.get("error"):
            continue
        rows.append({k: v for k, v in rec.items() if k != "curve"})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(cfg.RESULTS_DIR / "checkpoint_selection.csv", index=False)
    return df


def selected_model_paths(run: cfg.RunConfig) -> tuple[Path, Path | None]:
    """Paths chosen by validation, or the final model if selection never ran."""
    sel = run.dir / "checkpoint_selection.json"
    if sel.exists():
        rec = json.loads(sel.read_text())
        if not rec.get("fallback"):
            vec = rec.get("selected_vecnormalize")
            return Path(rec["selected_path"]), (Path(vec) if vec else None)
    return run.dir / "model.zip", run.dir / "vecnormalize.pkl"
