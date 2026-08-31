"""
Training: the full experimental batch of Section 3.6.

Four penalty weights x ten seeds = forty runs. The cost-blind condition is
eta = 0, so the core ablation and the penalty sweep are a single experiment
rather than two.

Run:
    python -m src.train                 # full batch
    python -m src.train --eta 0 --seed 3
    python -m src.train --smoke         # 20k steps, one run, for wiring checks
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from . import config as cfg
from .data_pipeline import build
from .portfolio_env import make_env

log = logging.getLogger(__name__)


def _sb3():
    """Import Stable-Baselines3 lazily so the rest of the package works without it."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, SubprocVecEnv, VecNormalize,
    )
    return PPO, CheckpointCallback, DummyVecEnv, SubprocVecEnv, VecNormalize


def build_vec_env(train_rel, penalty_weight: float, seed: int,
                  n_envs: int = cfg.N_ENVS, subproc: bool = True):
    """Vectorised, observation-normalised training environment.

    Observation normalisation is enabled because the two components of the
    observation occupy different scales: price relatives cluster tightly
    around 1.0 while weights lie in [0, 1]. Without rescaling the
    larger-variance component dominates early gradients.

    Reward normalisation is DISABLED. Rescaling the reward would distort the
    very quantity under experimental manipulation, and would make rewards
    incomparable across penalty weights.
    """
    _, _, DummyVecEnv, SubprocVecEnv, VecNormalize = _sb3()
    fns = [make_env(train_rel, penalty_weight) for _ in range(n_envs)]
    # SubprocVecEnv gives true parallelism but pays inter-process overhead;
    # for a cheap env DummyVecEnv can be faster. Benchmark both.
    venv = SubprocVecEnv(fns) if (subproc and n_envs > 1) else DummyVecEnv(fns)
    venv.seed(seed)
    return VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)


def train_one(run: cfg.RunConfig, train_rel, n_envs: int = cfg.N_ENVS,
              subproc: bool = True) -> Path:
    """Train a single agent and persist model, statistics and metadata."""
    PPO, CheckpointCallback, *_ = _sb3()

    run.save()
    out = run.dir
    log.info("=== %s | eta=%g (%gx c) | seed=%d | %s steps ===",
             run.label, run.penalty_weight, run.eta_multiple,
             run.seed, f"{run.total_timesteps:,}")

    env = build_vec_env(train_rel, run.penalty_weight, run.seed, n_envs, subproc)

    # Resume from the most recent checkpoint if one exists. Without this an
    # interruption at 90% of a 2M-step run costs the entire run.
    from .selection import latest_checkpoint
    resume = latest_checkpoint(run)
    completed_steps = 0

    if resume is not None:
        completed_steps, ckpt_model, ckpt_vec = resume
        if completed_steps >= run.total_timesteps:
            log.info("%s already at %d steps; nothing to do", run.label, completed_steps)
            env.close()
            return out
        log.info("Resuming %s from checkpoint at %d steps", run.label, completed_steps)
        if ckpt_vec is not None:
            from stable_baselines3.common.vec_env import VecNormalize
            env = VecNormalize.load(str(ckpt_vec), env.venv)
            env.training = True
            env.norm_reward = False
        model = PPO.load(str(ckpt_model), env=env, device=cfg.DEVICE)
    else:
        model = PPO(
            "MlpPolicy", env,
            seed=run.seed,             # seeds policy init and action sampling
            device=cfg.DEVICE,
            verbose=0,
            tensorboard_log=str(out / "tb"),
            **cfg.PPO_KWARGS,
        )

    # save_freq counts CALLBACK invocations, not environment steps: each
    # step advances n_envs timesteps, so divide.
    ckpt = CheckpointCallback(
        save_freq=max(cfg.CHECKPOINT_EVERY // n_envs, 1),
        save_path=str(out / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,        # ESSENTIAL: see note below
    )

    remaining = run.total_timesteps - completed_steps
    t0 = time.time()
    # reset_num_timesteps=False continues the step counter and the learning
    # rate schedule rather than restarting them.
    model.learn(total_timesteps=remaining, callback=ckpt,
                reset_num_timesteps=(completed_steps == 0),
                progress_bar=False)
    elapsed = time.time() - t0

    model.save(out / "model")
    # VecNormalize statistics are NOT stored inside the model file. Restoring
    # a checkpoint without them presents unnormalised observations to a
    # network trained on normalised ones, which manifests as a severe and
    # initially inexplicable performance collapse. The failure is silent:
    # the code runs and produces plausible output.
    env.save(str(out / "vecnormalize.pkl"))

    meta = {
        "label": run.label,
        "penalty_weight": run.penalty_weight,
        "eta_multiple": run.eta_multiple,
        "cost_rate": run.cost_rate,
        "seed": run.seed,
        "total_timesteps": run.total_timesteps,
        "n_envs": n_envs,
        "device": cfg.DEVICE,
        "wall_clock_seconds": round(elapsed, 1),
        "resumed_from_steps": completed_steps,
        "fps": round(remaining / elapsed, 1),
        "ppo_kwargs": cfg.PPO_KWARGS,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    env.close()

    log.info("Finished %s in %.1f min (%.0f fps)", run.label, elapsed / 60, meta["fps"])
    return out


def completed_timesteps(run: cfg.RunConfig) -> int:
    """Timesteps a saved run actually completed, or 0 if none exists.

    ``RunConfig.label`` encodes only eta and seed, so a smoke run and a full
    run share a directory. Comparing the recorded budget rather than merely
    testing for ``model.zip`` prevents an undertrained model from being
    mistaken for a finished one.
    """
    meta = run.dir / "metadata.json"
    model = run.dir / "model.zip"
    if not model.exists():
        return 0
    if meta.exists():
        try:
            return int(json.loads(meta.read_text()).get("total_timesteps", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            log.warning("%s: unreadable metadata.json; treating as untrained",
                        run.label)
            return 0
    # A model with no metadata predates this check; assume it is complete.
    return run.total_timesteps


def main() -> None:
    ap = argparse.ArgumentParser(description="Train cost-aware PPO agents")
    ap.add_argument("--eta", type=float, default=None,
                    help="single penalty weight (absolute, not a multiple)")
    ap.add_argument("--seed", type=int, default=None, help="single seed")
    ap.add_argument("--n-envs", type=int, default=cfg.N_ENVS)
    ap.add_argument("--no-subproc", action="store_true",
                    help="use DummyVecEnv; often faster for cheap environments")
    ap.add_argument("--smoke", action="store_true",
                    help="one short run to verify wiring")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Limit PyTorch threading BEFORE any model is constructed. With a small
    # MLP and vectorised environments, multithreaded operations contend for
    # the same cores and slow training down.
    import torch
    torch.set_num_threads(cfg.TORCH_NUM_THREADS)
    log.info("torch threads set to %d", cfg.TORCH_NUM_THREADS)

    parts = build()
    log.info("Training partition: %d rows, %s to %s",
             len(parts.train), parts.train.index[0].date(), parts.train.index[-1].date())

    if args.smoke:
        runs = [cfg.RunConfig(penalty_weight=10 * cfg.COST_RATE, seed=0,
                              total_timesteps=20_000)]
    elif args.eta is not None and args.seed is not None:
        runs = [cfg.RunConfig(penalty_weight=args.eta, seed=args.seed)]
    else:
        runs = cfg.all_runs()

    log.info("Batch: %d run(s)", len(runs))
    for i, run in enumerate(runs, 1):
        done = completed_timesteps(run)
        if done >= run.total_timesteps:
            log.info("[%d/%d] %s already trained (%s steps), skipping",
                     i, len(runs), run.label, f"{done:,}")
            continue
        if done > 0:
            # A shorter model exists at this path -- typically a smoke run,
            # which uses the same label because RunConfig.label encodes only
            # eta and seed. Silently keeping it would mix an undertrained
            # agent into the batch.
            log.warning(
                "[%d/%d] %s has a model trained for only %s of %s steps; "
                "retraining from scratch. Move or delete %s to keep it.",
                i, len(runs), run.label, f"{done:,}",
                f"{run.total_timesteps:,}", run.dir,
            )
        log.info("[%d/%d] starting %s", i, len(runs), run.label)
        train_one(run, parts.train, n_envs=args.n_envs, subproc=not args.no_subproc)


if __name__ == "__main__":
    main()
