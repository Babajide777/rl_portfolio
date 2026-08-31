"""Benchmark throughput before committing to the 40-run batch.

PPO with a small MLP is often FASTER on CPU than GPU: the bottleneck is
environment stepping, not network computation, and host-device transfer
costs more than it saves. Run this first and adopt the fastest setting.

    python scripts/benchmark.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from src import config as cfg
from src.portfolio_env import make_env

STEPS = 100_000


def bench(device, n_envs, subproc, torch_threads=None):
    import torch
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, SubprocVecEnv, VecNormalize)
    rng = np.random.default_rng(0)
    rel = pd.DataFrame(rng.normal(1.0004, 0.011, (3500, 9)),
                       columns=list(cfg.ASSETS))
    fns = [make_env(rel, 10 * cfg.COST_RATE) for _ in range(n_envs)]
    venv = SubprocVecEnv(fns) if (subproc and n_envs > 1) else DummyVecEnv(fns)
    venv.seed(0)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, device=device, verbose=0, **cfg.PPO_KWARGS)
    t0 = time.time()
    model.learn(total_timesteps=STEPS, progress_bar=False)
    dt = time.time() - t0
    venv.close()
    return dt, STEPS / dt


def preflight():
    """Fail fast with a clear message if the env is incompatible with SB3."""
    from stable_baselines3.common.env_checker import check_env
    from src.portfolio_env import PortfolioEnv
    rng = np.random.default_rng(0)
    rel = pd.DataFrame(rng.normal(1.0004, 0.011, (200, 9)),
                       columns=list(cfg.ASSETS))
    check_env(PortfolioEnv(rel), warn=True, skip_render_check=True)
    print("Pre-flight: environment is SB3-compatible.\n")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        preflight()
    except Exception as e:
        print(f"Pre-flight FAILED -- {type(e).__name__}: {e}")
        print("Fix this before benchmarking; every configuration will fail.")
        raise SystemExit(1)
    cores = mp.cpu_count()
    print(f"{cores} cores detected. Benchmarking {STEPS:,} timesteps each.\n")
    results = []
    # torch_threads is swept as a fourth dimension: for a small MLP,
    # multithreaded PyTorch operations contend with the vectorised
    # environments for the same cores and can dominate throughput.
    configs = [
        ("cpu",  cores,     False, None),  # torch default threading
        ("cpu",  cores,     False, 1),     # single-threaded torch
        ("cpu",  cores,     True,  1),
        ("cpu",  cores // 2, False, 1),
        ("cpu",  1,         False, 1),
        ("cuda", cores,     False, 1),
    ]
    for dev, n, sub, thr in configs:
        try:
            dt, fps = bench(dev, n, sub, thr)
            kind = "Subproc" if sub else "Dummy"
            tl = "default" if thr is None else str(thr)
            print(f"{dev:<5} n_envs={n:<3} {kind:<8} threads={tl:<7} "
                  f"{dt:6.1f}s {fps:8.0f} fps")
            results.append((fps, dev, n, sub, thr))
        except Exception as e:
            # Print the MESSAGE, not just the type. A bare type name hides
            # the cause and turns a two-minute fix into a guessing game.
            kind = "Subproc" if sub else "Dummy"
            tl = "default" if thr is None else str(thr)
            print(f"{dev:<5} n_envs={n:<3} {kind:<8} threads={tl:<7} FAILED")
            print(f"      {type(e).__name__}: {e}")
            if dev == "cuda" and isinstance(e, (AssertionError, RuntimeError)):
                print("      (no CUDA device is expected on a CPU-only machine)")
    if results:
        fps, dev, n, sub, thr = max(results)
        print(f"\nFastest: device={dev} n_envs={n} subproc={sub} "
              f"torch_threads={thr}")
        print(f"Set N_ENVS={n} and TORCH_NUM_THREADS={thr or 'default'} "
              f"in src/config.py")
        print(f"Projected: 2M steps ~ {2_000_000/fps/60:.0f} min/run, "
              f"40 runs ~ {40*2_000_000/fps/3600:.1f} hours")
