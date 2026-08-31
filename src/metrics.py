"""
Evaluation metrics and the two-level statistical protocol of Section 3.7.

Level 1 -- training stochasticity across seeds.
    Welch's t-test on per-seed summary metrics, with Cohen's d and bootstrap
    confidence intervals. Welch rather than Student because RL runs routinely
    violate equal variances (Colas et al., 2019).

Level 2 -- the realised return series.
    Studentised stationary bootstrap for the difference in Sharpe ratios
    (Ledoit & Wolf, 2008). The closed-form test of Jobson and Korkie (1981),
    as corrected by Memmel (2003), assumes IID normal returns and is
    inappropriate for daily financial data, which are autocorrelated and
    heavy-tailed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def _net_returns(hist: pd.DataFrame) -> np.ndarray:
    """Daily net-of-cost simple returns from a simulation history."""
    v = hist["portfolio_value"].to_numpy(dtype=np.float64)
    v0 = np.concatenate([[cfg.INITIAL_CAPITAL], v[:-1]])
    return v / v0 - 1.0


def sharpe_ratio(r: np.ndarray, rf: float = cfg.RISK_FREE_RATE,
                 periods: int = cfg.TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe ratio (Sharpe, 1966, 1994)."""
    ex = r - rf / periods
    sd = ex.std(ddof=1)
    return float(np.sqrt(periods) * ex.mean() / sd) if sd > 0 else 0.0


def sortino_ratio(r: np.ndarray, rf: float = cfg.RISK_FREE_RATE,
                  periods: int = cfg.TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sortino ratio: downside deviation only (Sortino & van der Meer, 1991)."""
    ex = r - rf / periods
    downside = ex[ex < 0]
    dd = downside.std(ddof=1) if downside.size > 1 else 0.0
    return float(np.sqrt(periods) * ex.mean() / dd) if dd > 0 else 0.0


def max_drawdown(hist: pd.DataFrame) -> float:
    """Largest peak-to-trough decline as a positive fraction."""
    v = hist["portfolio_value"].to_numpy(dtype=np.float64)
    return float((1.0 - v / np.maximum.accumulate(v)).max())


def calmar_ratio(hist: pd.DataFrame, periods: int = cfg.TRADING_DAYS_PER_YEAR) -> float:
    """Annualised return divided by maximum drawdown (Young, 1991)."""
    v = hist["portfolio_value"].to_numpy(dtype=np.float64)
    years = len(v) / periods
    cagr = (v[-1] / cfg.INITIAL_CAPITAL) ** (1.0 / years) - 1.0
    mdd = max_drawdown(hist)
    return float(cagr / mdd) if mdd > 0 else 0.0


def annualised_turnover(hist: pd.DataFrame, periods: int = cfg.TRADING_DAYS_PER_YEAR) -> float:
    """Mean daily turnover annualised. The behavioural variable of interest."""
    return float(hist["turnover"].mean() * periods)


def compute_metrics(hist: pd.DataFrame) -> dict[str, float]:
    """All reported metrics for one simulation history."""
    r = _net_returns(hist)
    v = hist["portfolio_value"].to_numpy(dtype=np.float64)
    years = len(v) / cfg.TRADING_DAYS_PER_YEAR
    return {
        "final_value": float(v[-1]),
        "total_return": float(v[-1] / cfg.INITIAL_CAPITAL - 1.0),
        "cagr": float((v[-1] / cfg.INITIAL_CAPITAL) ** (1.0 / years) - 1.0),
        "volatility": float(r.std(ddof=1) * np.sqrt(cfg.TRADING_DAYS_PER_YEAR)),
        "sharpe": sharpe_ratio(r),              # PRIMARY ENDPOINT
        "sortino": sortino_ratio(r),
        "calmar": calmar_ratio(hist),
        "max_drawdown": max_drawdown(hist),
        "ann_turnover": annualised_turnover(hist),   # secondary
        "total_cost": float(hist["cost"].sum() * cfg.INITIAL_CAPITAL),
        "total_cost_pct": float(hist["cost"].sum()),
        "n_days": int(len(v)),
    }


# ---------------------------------------------------------------------------
# Level 1: seed-level comparison
# ---------------------------------------------------------------------------
@dataclass
class ComparisonResult:
    metric: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    difference: float
    welch_t: float
    welch_p: float
    cohens_d: float
    ci_low: float
    ci_high: float
    mannwhitney_u: float
    mannwhitney_p: float

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        sig = "significant" if self.welch_p < 0.05 else "not significant"
        return (
            f"{self.metric}: {self.mean_a:.4f} vs {self.mean_b:.4f} "
            f"(diff {self.difference:+.4f}), Welch t={self.welch_t:.3f}, "
            f"p={self.welch_p:.4f} [{sig}], d={self.cohens_d:.3f}, "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
        )


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference with pooled standard deviation."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def bootstrap_ci(
    a: np.ndarray, b: np.ndarray,
    resamples: int = cfg.BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the difference in means."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for i in range(resamples):
        diffs[i] = (rng.choice(a, len(a), replace=True).mean()
                    - rng.choice(b, len(b), replace=True).mean())
    return float(np.percentile(diffs, 100 * alpha / 2)), \
           float(np.percentile(diffs, 100 * (1 - alpha / 2)))


def compare_seeds(
    values_a: np.ndarray, values_b: np.ndarray, metric: str = "sharpe", seed: int = 0
) -> ComparisonResult:
    """Level-1 comparison of two conditions across seeds."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)

    # equal_var=False -> Welch. RL runs routinely violate equal variances.
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    lo, hi = bootstrap_ci(a, b, seed=seed)

    return ComparisonResult(
        metric=metric, n_a=len(a), n_b=len(b),
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        std_a=float(a.std(ddof=1)), std_b=float(b.std(ddof=1)),
        difference=float(a.mean() - b.mean()),
        welch_t=float(t_stat), welch_p=float(p_val),
        cohens_d=cohens_d(a, b), ci_low=lo, ci_high=hi,
        mannwhitney_u=float(u_stat), mannwhitney_p=float(u_p),
    )


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Holm-Bonferroni step-down correction for the family of hypotheses.

    Controls the family-wise error rate while being uniformly more powerful
    than Bonferroni.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    rows, rejected_so_far = [], True
    for i, (name, p) in enumerate(items):
        threshold = alpha / (m - i)
        rejected = rejected_so_far and (p <= threshold)
        rejected_so_far = rejected
        rows.append({"hypothesis": name, "p_value": p,
                     "threshold": threshold, "rejected": rejected})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Level 2: Ledoit-Wolf Sharpe difference test
# ---------------------------------------------------------------------------
def _stationary_bootstrap_indices(
    n: int, expected_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap index sequence.

    Blocks of geometrically distributed length preserve the serial
    dependence structure that an IID bootstrap would destroy.
    """
    p = 1.0 / max(expected_block, 1.0)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    for i in range(1, n):
        idx[i] = rng.integers(0, n) if rng.random() < p else (idx[i - 1] + 1) % n
    return idx


def sharpe_difference_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    resamples: int = cfg.BOOTSTRAP_RESAMPLES,
    expected_block: float = 5.0,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Ledoit & Wolf (2008) test for the difference of two Sharpe ratios.

    Both series must cover the same period, so the pair is resampled with a
    common index to preserve their contemporaneous correlation. Since two
    agents on the same universe are strongly positively correlated, this
    materially raises power relative to independent resampling.
    """
    a = np.asarray(returns_a, dtype=np.float64)
    b = np.asarray(returns_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("return series must be the same length")

    observed = sharpe_ratio(a) - sharpe_ratio(b)
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for i in range(resamples):
        idx = _stationary_bootstrap_indices(len(a), expected_block, rng)
        boot[i] = sharpe_ratio(a[idx]) - sharpe_ratio(b[idx])

    centred = boot - boot.mean()
    p = float((np.abs(centred) >= abs(observed)).mean())
    return {
        "sharpe_a": sharpe_ratio(a),
        "sharpe_b": sharpe_ratio(b),
        "difference": float(observed),
        "ci_low": float(np.percentile(boot, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot, 100 * (1 - alpha / 2))),
        "p_value": p,
        "significant": bool(p < alpha),
        "resamples": resamples,
    }

# ---------------------------------------------------------------------------
# Ensemble construction for Level-2 testing
# ---------------------------------------------------------------------------
def ensemble_returns(histories: list[pd.DataFrame]) -> np.ndarray:
    """Equal-weighted mean of daily net returns across seeds.

    Level 2 requires two contemporaneous series. Selecting a single
    representative seed per condition discards nine tenths of the runs; this
    instead averages all seeds within a condition, giving the expected daily
    return of a randomly initialised agent trained under that condition.

    All histories must cover the same dates, which holds by construction
    since every agent is evaluated on the same test partition.
    """
    if not histories:
        raise ValueError("no histories supplied")
    series = [_net_returns(h) for h in histories]
    lengths = {len(s) for s in series}
    if len(lengths) != 1:
        raise ValueError(f"histories differ in length: {sorted(lengths)}")
    return np.vstack(series).mean(axis=0)


def pairwise_sharpe_differences(
    hists_a: list[pd.DataFrame], hists_b: list[pd.DataFrame]
) -> pd.DataFrame:
    """All seed-by-seed Sharpe differences, reported as a distribution.

    Reported for robustness, NOT as significance tests: the comparisons are
    not independent, so p-values would require correction so severe as to be
    uninformative. The value is in showing how consistently one condition
    exceeds the other across the full seed grid.
    """
    rows = []
    for i, ha in enumerate(hists_a):
        sa = sharpe_ratio(_net_returns(ha))
        for j, hb in enumerate(hists_b):
            sb = sharpe_ratio(_net_returns(hb))
            rows.append({"seed_a": i, "seed_b": j, "sharpe_a": sa,
                         "sharpe_b": sb, "difference": sa - sb})
    df = pd.DataFrame(rows)
    return df
