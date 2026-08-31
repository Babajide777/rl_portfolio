"""
Classical and rule-based baseline strategies.

Implements Section 4.6. Every baseline is evaluated on the identical test
partition and charged at the identical cost rate, so that all reported
figures are directly comparable with the agents'.

Classical baselines are computed WALK-FORWARD: at each rebalancing date the
covariance matrix is estimated using only observations available at that
date. Estimating once over the full test period would grant information no
live investor possesses and would flatter the baselines relative to the
agent, which sees no future data.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from . import config as cfg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared simulation engine
# ---------------------------------------------------------------------------
def simulate(
    rel: pd.DataFrame,
    target_fn: Callable[[int, np.ndarray], np.ndarray | None],
    cost_rate: float = cfg.COST_RATE,
    initial_capital: float = cfg.INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Run a weight-generating rule through the same accounting as the agent.

    ``target_fn(t, current_weights)`` returns the desired weight vector for
    step ``t``, or ``None`` to hold the current position (incurring no cost).

    Using one engine for agents and baselines alike guarantees that any
    performance difference reflects the strategies rather than divergent
    accounting.
    """
    R = rel.to_numpy(dtype=np.float64)
    T, n = R.shape
    w = np.full(n, 1.0 / n)
    value = float(initial_capital)
    rows = []

    for t in range(T):
        target = target_fn(t, w)
        if target is None:
            a = w                       # hold: zero turnover, zero cost
        else:
            a = np.asarray(target, dtype=np.float64)
            if a.min() < -1e-12:
                raise ValueError("baseline produced a negative weight")
            a = a / a.sum()

        turnover = float(np.abs(a - w).sum())
        cost = cost_rate * turnover
        y = R[t]
        gross = float(a @ y)
        net = gross - cost
        value *= max(net, cfg.LOG_FLOOR)
        w = (a * y) / gross if gross > 0 else a.copy()

        rows.append({
            "date": rel.index[t],
            "turnover": turnover,
            "cost": cost,
            "gross_return": gross,
            "net_return": net,
            "portfolio_value": value,
        })

    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Rule-based baselines
# ---------------------------------------------------------------------------
def equal_weight_buy_and_hold(rel: pd.DataFrame, **kw) -> pd.DataFrame:
    """Allocate equally once, then let weights drift.

    Turnover is near zero after inception, which is precisely why this is a
    demanding benchmark: DeMiguel et al. (2009) found no optimising model
    consistently beat 1/N out of sample once costs were counted.
    """
    n = rel.shape[1]
    eq = np.full(n, 1.0 / n)
    return simulate(rel, lambda t, w: eq if t == 0 else None, **kw)


def calendar_rebalance(rel: pd.DataFrame, freq: str = cfg.CALENDAR_FREQ, **kw) -> pd.DataFrame:
    """Restore equal weights on the first trading day of each period."""
    n = rel.shape[1]
    eq = np.full(n, 1.0 / n)
    marks = set(rel.groupby(pd.Grouper(freq=freq)).head(1).index)
    return simulate(rel, lambda t, w: eq if rel.index[t] in marks else None, **kw)


def threshold_rebalance(rel: pd.DataFrame, band: float = cfg.THRESHOLD_BAND, **kw) -> pd.DataFrame:
    """Rebalance to equal weights only when any weight breaches the band.

    The empirical analogue of the no-trade region of Constantinides (1979)
    and Davis and Norman (1990): tolerate drift, trade only at the boundary.
    """
    n = rel.shape[1]
    eq = np.full(n, 1.0 / n)
    return simulate(
        rel,
        lambda t, w: eq if (t == 0 or np.abs(w - eq).max() > band) else None,
        **kw,
    )


# ---------------------------------------------------------------------------
# Classical baselines (walk-forward)
# ---------------------------------------------------------------------------
def _shrunk_covariance(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a scaled identity target.

    The sample covariance matrix is a poor estimator in high dimensions and
    its errors propagate into extreme allocations -- Michaud's (1989) error
    maximiser. Shrinkage trades a little bias for a large variance reduction.
    """
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(returns).covariance_


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance weights by projected gradient descent.

    Solved numerically rather than in closed form because the closed-form
    solution admits negative weights, which the long-only constraint of
    Section 1.6 forbids.
    """
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    step = 1.0 / (np.trace(cov) + 1e-12)
    for _ in range(500):
        grad = 2.0 * cov @ w
        w = _project_simplex(w - step * grad)
    return w


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (Wang & Carreira-Perpinan, 2013)."""
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _erc_weights(cov: np.ndarray, iters: int = 500) -> np.ndarray:
    """Equal risk contribution weights by fixed-point iteration.

    Each asset contributes an identical share of total portfolio variance
    (Maillard, Roncalli & Teiletche, 2010). Depends only on the covariance
    matrix, so it avoids the return-estimation problem entirely.
    """
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        mrc = cov @ w                       # marginal risk contributions
        w = w * (1.0 / np.maximum(mrc, 1e-12))
        w /= w.sum()
    return w


def _walk_forward(
    rel: pd.DataFrame,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    lookback: int = cfg.CLASSICAL_LOOKBACK,
    freq: str = cfg.CALENDAR_FREQ,
    **kw,
) -> pd.DataFrame:
    """Re-estimate and reallocate at each rebalancing date using past data only."""
    logret = np.log(rel.to_numpy(dtype=np.float64))
    marks = set(rel.groupby(pd.Grouper(freq=freq)).head(1).index)
    n = rel.shape[1]
    cache: dict[int, np.ndarray] = {}

    def target(t: int, w: np.ndarray):
        if rel.index[t] not in marks and t != 0:
            return None
        if t < lookback:
            return np.full(n, 1.0 / n)      # insufficient history yet
        if t not in cache:
            window = logret[t - lookback : t]   # strictly past data
            cache[t] = weight_fn(_shrunk_covariance(window))
        return cache[t]

    return simulate(rel, target, **kw)


def min_variance(rel: pd.DataFrame, **kw) -> pd.DataFrame:
    """Walk-forward long-only minimum-variance portfolio."""
    return _walk_forward(rel, _min_variance_weights, **kw)


def risk_parity(rel: pd.DataFrame, **kw) -> pd.DataFrame:
    """Walk-forward equal-risk-contribution portfolio."""
    return _walk_forward(rel, _erc_weights, **kw)


# ---------------------------------------------------------------------------
BASELINES: dict[str, Callable[..., pd.DataFrame]] = {
    "equal_weight_bh": equal_weight_buy_and_hold,
    "calendar_monthly": calendar_rebalance,
    "threshold_5pct": threshold_rebalance,
    "min_variance": min_variance,
    "risk_parity": risk_parity,
}


def run_all_baselines(rel: pd.DataFrame, cost_rate: float = cfg.COST_RATE) -> dict[str, pd.DataFrame]:
    out = {}
    for name, fn in BASELINES.items():
        log.info("Running baseline: %s", name)
        out[name] = fn(rel, cost_rate=cost_rate)
    return out
