"""
Custom Gymnasium environment for cost-aware portfolio rebalancing.

This is the principal software artefact of the project. It implements
Section 3.4 of the dissertation.

THE CENTRAL DESIGN DECISION
---------------------------
Two distinct parameters govern transaction costs, and conflating them would
invalidate the experiment:

    cost_rate (c)       the proportional charge levied by the MARKET.
                        Applies to every agent in every condition. Never
                        swept. Updates portfolio value.

    penalty_weight (eta) the coefficient in the REWARD. A training
                        hyperparameter, not a market property. Swept
                        experimentally. eta = 0 is the cost-blind agent.

A cost-blind agent therefore pays exactly the same real trading costs as a
cost-aware one; it simply receives no signal about them. Were a single
parameter to serve both roles, setting it to zero would remove costs
entirely and the comparison would be between two different markets rather
than two differently instructed agents.

See ``tests/test_env.py::test_eta_does_not_affect_portfolio_value`` for the
assertion that enforces this.
"""
from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from . import config as cfg

log = logging.getLogger(__name__)


class PortfolioEnv(gym.Env):
    """Daily long-only portfolio rebalancing over a fixed asset universe.

    Observation
        Flattened concatenation of a ``lookback x n_assets`` matrix of price
        relatives with the current portfolio weight vector. For the default
        configuration this is 30 x 9 + 9 = 279 values.

    Action
        Nine real logits in ``[-LOGIT_BOUND, LOGIT_BOUND]`` (see config);
        softmax maps them to the simplex. Weights are necessarily non-negative
        and sum to one, so the portfolio is long-only and fully invested by
        construction rather than by penalty.

    Reward
        ``log(w . y - eta * turnover)``, floored at ``log_floor``.
        With ``eta = 0`` this reduces to ``log(w . y)``.

    Termination
        Portfolio value below ``ruin_threshold`` of initial capital.
    Truncation
        End of the data partition.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        price_relatives: pd.DataFrame | np.ndarray,
        cost_rate: float = cfg.COST_RATE,
        penalty_weight: float = 0.0,
        lookback: int = cfg.LOOKBACK,
        initial_capital: float = cfg.INITIAL_CAPITAL,
        log_floor: float = cfg.LOG_FLOOR,
        ruin_threshold: float = cfg.RUIN_THRESHOLD,
        record_history: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(price_relatives, pd.DataFrame):
            self._dates = price_relatives.index
            self.rel = price_relatives.to_numpy(dtype=np.float64)
        else:
            self._dates = None
            self.rel = np.asarray(price_relatives, dtype=np.float64)

        if self.rel.ndim != 2:
            raise ValueError(f"expected 2-D array, got shape {self.rel.shape}")
        if cost_rate < 0 or penalty_weight < 0:
            raise ValueError("cost_rate and penalty_weight must be non-negative")

        self.T, self.n = self.rel.shape
        if self.T <= lookback + 1:
            raise ValueError(
                f"need more than lookback+1={lookback + 1} rows, got {self.T}"
            )

        self.c = float(cost_rate)
        self.eta = float(penalty_weight)
        self.L = int(lookback)
        self.V0 = float(initial_capital)
        self.eps = float(log_floor)
        self.ruin = float(ruin_threshold)
        # History is read only at evaluation. During training it is never
        # touched, yet accumulating a dict plus an array copy every step
        # costs roughly 1.3 GB across eight environments over two million
        # steps. Evaluation and checkpoint selection opt in explicitly.
        self.record_history = bool(record_history)
        # Vectorised wrappers auto-reset the moment an episode ends, which
        # would otherwise discard the completed episode before the caller can
        # read it. The finished record is carried over here.
        self._history: list[dict] = []
        self._last_history: list[dict] = []

        obs_dim = self.L * self.n + self.n
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Logits are softmaxed inside step(); SB3 nonetheless requires finite
        # Box bounds (see config.LOGIT_BOUND and the regression tests).
        self.action_space = spaces.Box(
            low=-cfg.LOGIT_BOUND, high=cfg.LOGIT_BOUND,
            shape=(self.n,), dtype=np.float32,
        )

        # Diagnostics, reset per episode
        self.floor_events = 0
        self.total_floor_events = 0
        self.episodes = 0
        self.ruin_episodes = 0

        self._t = self.L
        self._value = self.V0
        self._w = np.full(self.n, 1.0 / self.n)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._t = self.L
        self._value = self.V0
        # Equal-weight initialisation: no asset is privileged at the outset.
        self._w = np.full(self.n, 1.0 / self.n)
        self.floor_events = 0
        # Preserve the episode just finished before clearing. DummyVecEnv and
        # SubprocVecEnv call reset() automatically on termination, so without
        # this the caller reads an empty history.
        if self._history:
            self._last_history = self._history
        self._history = []
        return self._observe(), {"date": self._date()}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # 1. logits -> simplex. Shift by the max for numerical stability:
        #    exp of a large logit overflows, exp of (logit - max) cannot.
        a = np.asarray(action, dtype=np.float64).ravel()
        a = np.exp(a - a.max())
        a = a / a.sum()

        # 2. Turnover against the existing (post-drift) position. One-way
        #    measure per DeMiguel et al. (2009): captures deliberate
        #    reallocation, not drift.
        turnover = float(np.abs(a - self._w).sum())

        # 3. Cost charged by the MARKET at rate c. Independent of eta:
        #    every agent pays this.
        cost = self.c * turnover

        # 4. Realised market return over the following period.
        y = self.rel[self._t]
        gross = float(a @ y)

        # 5. Reward uses the PENALTY WEIGHT eta, not the cost rate c.
        #    When eta = 0 this reduces to log(gross): the agent is blind to
        #    a cost it has nonetheless incurred at step 3.
        shaped = gross - self.eta * turnover
        if shaped <= self.eps:
            self.floor_events += 1
            self.total_floor_events += 1
        reward = float(np.log(max(shaped, self.eps)))

        # 6. Portfolio update at the TRUE cost, then overnight drift.
        net = gross - cost
        self._value *= max(net, self.eps)
        # Drift: prices move, so weights change without any trading. These
        # drifted weights become w_(t-1) for the next step, which is why
        # Section 3.4 specifies post-drift weights.
        self._w = (a * y) / gross if gross > self.eps else a.copy()

        info = {
            "date": self._date(),
            "turnover": turnover,
            "cost": cost,
            "gross_return": gross,
            "net_return": net,
            "portfolio_value": self._value,
            # No copy: `a` is freshly allocated each step and never mutated
            # afterwards, and the info dict is discarded by the training
            # loop. The history list takes its own copy below.
            "weights": a,
        }
        # The transient info dict is discarded by the training loop each
        # step, so including weights there is free. The HISTORY LIST is what
        # grows unboundedly, which is why it is gated.
        if self.record_history:
            self._history.append({**info, "weights": a.copy()})

        self._t += 1
        terminated = bool(self._value < self.ruin * self.V0)
        truncated = bool(self._t >= self.T)
        if terminated or truncated:
            self.episodes += 1
            if terminated:
                self.ruin_episodes += 1
            info["episode_floor_events"] = self.floor_events

        return self._observe(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        """Concatenate the flattened lookback window with current weights.

        The window is transposed before flattening so the vector is
        asset-major: all ``lookback`` observations for asset 0, then asset 1,
        and so on. Arbitrary for a fully connected network, but fixed and
        documented so that learned weights can be related to specific assets.
        """
        t = min(self._t, self.T)
        window = self.rel[t - self.L : t]          # (L, n)
        return np.concatenate(
            [window.T.ravel(), self._w]            # (L*n,) + (n,)
        ).astype(np.float32)

    def _date(self):
        if self._dates is None or self._t >= self.T:
            return None
        return self._dates[self._t]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def history(self) -> pd.DataFrame:
        """Per-step record of the current episode."""
        rows = self._history or self._last_history
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([
            {k: v for k, v in h.items() if k != "weights"} for h in rows
        ])
        w = np.vstack([h["weights"] for h in rows])
        for i in range(self.n):
            df[f"w_{cfg.ASSETS[i]}" if self.n == len(cfg.ASSETS) else f"w_{i}"] = w[:, i]
        if "date" in df and df["date"].notna().all():
            df = df.set_index("date")
        return df

    def diagnostics(self) -> dict[str, float]:
        return {
            "episodes": self.episodes,
            "ruin_episodes": self.ruin_episodes,
            "ruin_rate": self.ruin_episodes / max(self.episodes, 1),
            "total_floor_events": self.total_floor_events,
        }

    def __repr__(self) -> str:
        return (
            f"PortfolioEnv(T={self.T}, n={self.n}, c={self.c:g}, "
            f"eta={self.eta:g}, lookback={self.L})"
        )


def make_env(price_relatives, penalty_weight: float,
             monitor_path: str | None = None, **kw):
    """Factory returning a zero-argument callable, as VecEnv requires.

    The environment is wrapped in Stable-Baselines3's ``Monitor``, which
    records episode return, length and duration and exposes them to the
    training loop. Without it, no ``rollout/*`` statistics are logged: the
    optimiser diagnostics under ``train/*`` would still appear, but there
    would be no record of whether the agent is actually improving at the
    task, and no training curve to report.
    """
    def _init():
        env = PortfolioEnv(price_relatives, penalty_weight=penalty_weight, **kw)
        try:
            from stable_baselines3.common.monitor import Monitor
        except ImportError:      # keep the package usable without SB3
            return env
        # info_keywords surfaces per-episode turnover and cost alongside the
        # standard return/length, so behaviour can be tracked during training
        # rather than only reconstructed at evaluation.
        return Monitor(env, filename=monitor_path,
                       info_keywords=("turnover", "cost"))
    return _init
