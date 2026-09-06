"""
Allocation and trade-off analysis (Objectives 2 and 5).

Produces the results Chapter 5 reports on what the agents held and on the
relationship between turnover and risk-adjusted return. Both were previously
computed ad hoc; this script makes them reproducible from the repository, as
Section 3.8 requires.

Reads   runs/<label>/test_history.csv   (written by src.evaluate)
        results/agent_metrics.csv
        results/baseline_metrics.csv

Writes  results/weight_summary.csv          per-run allocation statistics
        results/allocation_by_condition.csv mean and terminal allocation
        results/tradeoff_regression.json    turnover vs Sharpe regression
        results/figures/fig5_allocation_heatmap.png
        results/figures/fig6_turnover_tradeoff.png

Run after src.evaluate:
    python scripts/allocation_analysis.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EQUAL = 100.0 / len(cfg.ASSETS)


# ---------------------------------------------------------------------------
# Per-run allocation statistics
# ---------------------------------------------------------------------------
def collect_weights() -> pd.DataFrame:
    """Summarise the weight path of every completed run.

    The environment records the post-softmax action at each step, so these
    describe the allocations the agent chose rather than those the market
    left it holding after drift.
    """
    rows = []
    for hist_path in sorted(cfg.RUNS_DIR.glob("*/test_history.csv")):
        label = hist_path.parent.name
        if "smoke" in label:
            continue
        d = pd.read_csv(hist_path)
        wcols = [c for c in d.columns if c.startswith("w_")]
        if not wcols:
            log.warning("%s: no weight columns; skipping", label)
            continue
        row = {"label": label}
        for c in wcols:
            asset = c[2:]
            row[f"mean_{asset}"] = d[c].mean()
            row[f"final_{asset}"] = d[c].iloc[-1]
            row[f"std_{asset}"] = d[c].std()
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            "No test_history.csv files found. Run `python -m src.evaluate` first."
        )
    df = pd.DataFrame(rows)
    df["eta_multiple"] = df.label.str.extract(r"eta(\d+)c")[0].astype(float)
    df["seed"] = df.label.str.extract(r"seed(\d+)")[0].astype(int)

    # Departure from equal weighting: sum of absolute deviations from 1/N,
    # in percentage points. Zero means the portfolio was held exactly at
    # equal weight on average. This measures how distinctive the position
    # is, which is NOT the same as how much trading it required.
    df["departure"] = df.apply(
        lambda r: sum(abs(100 * r[f"mean_{a}"] - EQUAL) for a in cfg.ASSETS), axis=1
    )
    log.info("Collected weights for %d runs", len(df))
    return df


def allocation_by_condition(w: pd.DataFrame) -> pd.DataFrame:
    """Mean and terminal allocation per condition, in per cent."""
    rows = []
    for e in sorted(w.eta_multiple.unique()):
        g = w[w.eta_multiple == e]
        for kind in ("mean", "final"):
            r = {"eta_multiple": e, "statistic": kind}
            for a in cfg.ASSETS:
                r[a] = round(100 * g[f"{kind}_{a}"].mean(), 3)
            rows.append(r)
    eq = {"eta_multiple": np.nan, "statistic": "equal_weight"}
    eq.update({a: round(EQUAL, 3) for a in cfg.ASSETS})
    return pd.DataFrame(rows + [eq])


def departure_tests(w: pd.DataFrame) -> dict:
    """Does the penalty produce a more distinctive allocation?"""
    rho, p = stats.spearmanr(w.eta_multiple, w.departure)
    hi = w[w.eta_multiple == w.eta_multiple.max()].departure
    lo = w[w.eta_multiple == 0].departure
    tt = stats.ttest_ind(hi, lo, equal_var=False)
    pooled = np.sqrt((hi.var(ddof=1) + lo.var(ddof=1)) / 2)
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "welch_t": float(tt.statistic),
        "welch_p": float(tt.pvalue),
        "cohens_d": float((hi.mean() - lo.mean()) / pooled) if pooled else 0.0,
        "departure_by_condition": {
            f"{e:g}c": {
                "mean": float(w[w.eta_multiple == e].departure.mean()),
                "sd": float(w[w.eta_multiple == e].departure.std()),
            }
            for e in sorted(w.eta_multiple.unique())
        },
    }


# ---------------------------------------------------------------------------
# Turnover / return trade-off  (Objective 5)
# ---------------------------------------------------------------------------
def tradeoff_regression(agents: pd.DataFrame, baselines: pd.DataFrame) -> dict:
    """Regress net Sharpe on turnover across all runs.

    The intercept estimates the Sharpe ratio the agents' allocation
    behaviour would attain at zero turnover, and is directly comparable to
    the buy-and-hold baseline whose turnover is zero by construction. If the
    two coincide, the agents hold no allocation advantage over naive
    diversification and their whole shortfall is attributable to trading.

    Baselines are excluded from the fit: their turnover is zero or near
    zero, so including them would let a handful of clustered points
    determine the intercept the regression exists to estimate independently.
    """
    x = agents.ann_turnover.to_numpy(dtype=float)
    y = agents.sharpe.to_numpy(dtype=float)
    slope, intercept, r, p, se = stats.linregress(x, y)
    rho, prho = stats.spearmanr(x, y)

    bh = baselines.loc[baselines.strategy == "equal_weight_bh", "sharpe"]
    bh_sharpe = float(bh.iloc[0]) if len(bh) else float("nan")

    return {
        "n_runs": int(len(agents)),
        "slope": float(slope),
        "slope_se": float(se),
        "intercept": float(intercept),
        "r_squared": float(r ** 2),
        "p_value": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(prho),
        "sharpe_cost_per_100_turnover": float(abs(slope) * 100),
        "equal_weight_baseline_sharpe": bh_sharpe,
        "intercept_minus_baseline": float(intercept - bh_sharpe),
        "turnover_range": [float(x.min()), float(x.max())],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_allocation(w: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    etas = sorted(w.eta_multiple.unique())
    labels = ["\u03b7 = 0\n(cost-blind)" if e == 0 else f"\u03b7 = {e:g}c" for e in etas]
    mean = np.array([[100 * w[w.eta_multiple == e][f"mean_{a}"].mean()
                      for a in cfg.ASSETS] for e in etas])
    fin = np.array([[100 * w[w.eta_multiple == e][f"final_{a}"].mean()
                     for a in cfg.ASSETS] for e in etas])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    vmax = max(mean.max(), fin.max())
    for ax, M, ttl in [(axes[0], mean, "Mean allocation over test period"),
                       (axes[1], fin, "Terminal allocation")]:
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cfg.ASSETS)))
        ax.set_xticklabels(cfg.ASSETS, fontsize=9)
        ax.set_yticks(range(len(etas)))
        ax.set_yticklabels(labels, fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.65 * vmax else "black")
        ax.set_title(ttl, fontsize=11)
        plt.colorbar(im, ax=ax, label="weight (%)")
    fig.suptitle(f"Portfolio allocation by penalty weight "
                 f"(equal weight = {EQUAL:.1f}% each)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_allocation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_tradeoff(agents: pd.DataFrame, baselines: pd.DataFrame,
                    reg: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.6))
    palette = ["#C0392B", "#E67E22", "#2E86C1", "#1E8449", "#7D3C98"]
    for i, e in enumerate(sorted(agents.eta_multiple.unique())):
        g = agents[agents.eta_multiple == e]
        lbl = "\u03b7 = 0 (cost-blind)" if e == 0 else f"\u03b7 = {e:g}c"
        ax.scatter(g.ann_turnover, g.sharpe, s=64, alpha=0.8,
                   color=palette[i % len(palette)], edgecolor="white",
                   linewidth=0.8, label=lbl, zorder=3)
    x = np.linspace(0, agents.ann_turnover.max() * 1.05, 100)
    ax.plot(x, reg["intercept"] + reg["slope"] * x, "--", color="#555555", lw=1.4,
            zorder=2, label=f"OLS fit (R\u00b2 = {reg['r_squared']:.2f})")
    ax.scatter(baselines.ann_turnover, baselines.sharpe, marker="*", s=280,
               color="black", zorder=4, label="baselines")
    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.set_xlabel("Annualised turnover")
    ax.set_ylabel("Net-of-cost Sharpe ratio")
    ax.set_title("Turnover against risk-adjusted return")
    ax.grid(alpha=0.3, zorder=0)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_turnover_tradeoff.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    agents = pd.read_csv(cfg.RESULTS_DIR / "agent_metrics.csv")
    baselines = pd.read_csv(cfg.RESULTS_DIR / "baseline_metrics.csv")

    w = collect_weights()
    w.to_csv(cfg.RESULTS_DIR / "weight_summary.csv", index=False)

    alloc = allocation_by_condition(w)
    alloc.to_csv(cfg.RESULTS_DIR / "allocation_by_condition.csv", index=False)
    log.info("Allocation by condition (mean, %%):\n%s",
             alloc[alloc.statistic == "mean"].to_string(index=False))

    dep = departure_tests(w)
    reg = tradeoff_regression(agents, baselines)
    (cfg.RESULTS_DIR / "tradeoff_regression.json").write_text(
        json.dumps({"departure": dep, "tradeoff": reg}, indent=2))

    log.info("Departure from equal weight: Spearman rho=%+.3f p=%.4f; "
             "extremes Welch p=%.4f d=%+.3f",
             dep["spearman_rho"], dep["spearman_p"], dep["welch_p"], dep["cohens_d"])
    log.info("Trade-off: Sharpe = %+.4f %+.6f x turnover, R2=%.3f, p=%.4g",
             reg["intercept"], reg["slope"], reg["r_squared"], reg["p_value"])
    log.info("Intercept %+.4f vs equal-weight baseline %+.4f (difference %+.4f)",
             reg["intercept"], reg["equal_weight_baseline_sharpe"],
             reg["intercept_minus_baseline"])

    fig_dir = cfg.RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_allocation(w, fig_dir)
    figure_tradeoff(agents, baselines, reg, fig_dir)
    log.info("Figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
