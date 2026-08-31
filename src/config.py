"""
Central configuration for the cost-aware portfolio rebalancing experiment.

Every experimental constant lives here. No magic numbers appear elsewhere in
the codebase, so that the full specification of a run is recoverable from a
single file. This supports the reproducibility provisions of Section 3.8.

Reference: Oyafemi, B. (2026). Reinforcement Learning for Cost-Aware
Portfolio Rebalancing. MSc dissertation, University of Chester.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------
# Nine ETFs spanning four asset classes. Ordering is fixed and load-bearing:
# it determines the row order of the state matrix and the column order of
# every weight vector, so it must not be changed without regenerating all
# cached data and retraining.
ASSETS: tuple[str, ...] = (
    "SPY",  # US large-cap equity
    "QQQ",  # US technology equity
    "IWM",  # US small-cap equity
    "EFA",  # International developed equity
    "TLT",  # Long US Treasuries
    "LQD",  # Investment-grade corporate credit
    "GLD",  # Gold
    "VNQ",  # US real estate
    "USO",  # Crude oil
)
N_ASSETS: int = len(ASSETS)

# USO (United States Oil Fund) was launched on 10 April 2006 and is the
# latest inception in the universe. Retrieval therefore begins in 2006 so
# that a complete lookback window exists before the first modelled decision
# in 2007. See Section 3.3.
DATA_START: str = "2006-01-01"
DATA_END: str = "2024-12-31"

# ---------------------------------------------------------------------------
# Partitioning  (Section 3.3, Table 3.2)
# ---------------------------------------------------------------------------
TRAIN_END: str = "2020-12-31"
VAL_END: str = "2021-12-31"
# Test runs from VAL_END to DATA_END.

LOOKBACK: int = 30   # trading days in the state window
EMBARGO: int = 21    # trading days embargoed after each partition boundary

# ---------------------------------------------------------------------------
# Environment  (Section 3.4)
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 100_000.0

# COST_RATE (c) is the proportional charge levied by the market. It applies
# to EVERY agent in EVERY condition and is never swept.
COST_RATE: float = 0.0005          # 5 basis points

# PENALTY_WEIGHTS (eta) are training hyperparameters appearing only in the
# reward. eta = 0 is the cost-blind condition. Values above COST_RATE are
# deliberate reward shaping: see Section 3.6.
PENALTY_WEIGHTS: tuple[float, ...] = (
    0.0,                  # cost-blind
    1.0 * COST_RATE,      # penalty at true cost
    10.0 * COST_RATE,     # 10x
    50.0 * COST_RATE,     # 50x
)

# Floor applied to the argument of the logarithm. Necessary rather than
# precautionary: front-month oil settled negative in April 2020, inside the
# training partition. Every activation is counted and reported.
LOG_FLOOR: float = 1e-8

# Episode terminates if portfolio value falls below this fraction of initial
# capital, preventing accumulation of experience in unrecoverable states.
RUIN_THRESHOLD: float = 0.20

# Finite bound on policy logits. SB3 requires a finite Box action space.
# Softmax makes +/-10 more than sufficient for single-asset concentration.
LOGIT_BOUND: float = 10.0

# ---------------------------------------------------------------------------
# Training  (Section 3.5, Table 3.3)
# ---------------------------------------------------------------------------
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
TOTAL_TIMESTEPS: int = 2_000_000
# TOTAL_TIMESTEPS: int = 500_000
N_ENVS: int = 8                    # parallel environments; tune to core count
CHECKPOINT_EVERY: int = 50_000     # environment steps between checkpoints

# Stable-Baselines3 PPO defaults, stated explicitly rather than left implicit.
# Held identical across all conditions to preserve internal validity: the
# reward specification must be the only quantity that differs.
PPO_KWARGS: dict = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,                 # RL discount factor -- NOT the cost rate
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": dict(pi=[64, 64], vf=[64, 64])},
}

# PPO with a small MLP policy is typically faster on CPU than GPU: the
# bottleneck is environment stepping, not network computation, and the
# host-device transfer costs more than it saves. Benchmark both before
# committing (see scripts/benchmark.py).
# DEVICE: str = "cpu"
DEVICE: str = "cuda"

# PyTorch spawns multiple threads per operation by default. With a small MLP
# the per-operation work is tiny, so thread coordination overhead exceeds any
# parallel benefit -- and those threads contend with the vectorised
# environments for the same cores. Limiting PyTorch to a single thread is a
# documented Stable-Baselines3 performance tip for this configuration.
# scripts/benchmark.py sweeps this; adopt whichever value is fastest.
TORCH_NUM_THREADS: int = 1

# ---------------------------------------------------------------------------
# Evaluation  (Section 3.6, 3.7)
# ---------------------------------------------------------------------------
# Cost rates at which trained policies are re-scored without retraining.
EVAL_COST_RATES: tuple[float, ...] = (0.0001, 0.0005, 0.0010, 0.0020, 0.0050)

TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE: float = 0.0        # excess returns computed against zero
BOOTSTRAP_RESAMPLES: int = 10_000

# Baseline rebalancing rules
CALENDAR_FREQ: str = "MS"          # month start
THRESHOLD_BAND: float = 0.05       # +/- 5 percentage points
CLASSICAL_LOOKBACK: int = 252      # estimation window for MVO / risk parity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, RUNS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunConfig:
    """Identifies a single training run and derives its output paths."""

    penalty_weight: float
    seed: int
    cost_rate: float = COST_RATE
    total_timesteps: int = TOTAL_TIMESTEPS

    @property
    def eta_multiple(self) -> float:
        """Penalty weight expressed as a multiple of the cost rate."""
        return 0.0 if self.cost_rate == 0 else self.penalty_weight / self.cost_rate

    @property
    def label(self) -> str:
        """Directory name for this run.

        Smoke runs are labelled separately so a short diagnostic run cannot
        occupy the directory a full run would use.
        """
        base = f"eta{self.eta_multiple:g}c_seed{self.seed}"
        if self.total_timesteps < TOTAL_TIMESTEPS:
            return f"{base}_smoke{self.total_timesteps}"
        return base

    @property
    def is_cost_blind(self) -> bool:
        return self.penalty_weight == 0.0

    @property
    def dir(self) -> Path:
        d = RUNS_DIR / self.label
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self) -> None:
        (self.dir / "config.json").write_text(json.dumps(asdict(self), indent=2))


def all_runs() -> list[RunConfig]:
    """The full experimental batch: 4 penalty weights x 10 seeds = 40 runs."""
    return [
        RunConfig(penalty_weight=eta, seed=s)
        for eta in PENALTY_WEIGHTS
        for s in SEEDS
    ]
