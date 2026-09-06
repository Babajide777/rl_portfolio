"""
Evaluation: validation-based checkpoint selection, deterministic test
rollout, cost sensitivity, baseline comparison and the two-level statistical
protocol.

Implements Sections 3.5, 3.7, 4.6 and 4.8.

Pipeline order matters. Checkpoint selection consults the VALIDATION
partition only; the test partition is touched exactly once, after selection
is complete.

    python -m src.evaluate
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import metrics as mt
from .baselines import run_all_baselines
from .data_pipeline import build
from .portfolio_env import PortfolioEnv
from .selection import select_all, selected_model_paths

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy rollout
# ---------------------------------------------------------------------------
def _model_paths(run: cfg.RunConfig, use_final_model: bool) -> tuple[Path, Path | None]:
    """Return (model_path, vecnormalize_path) for rollout."""
    if use_final_model:
        return run.dir / "model.zip", run.dir / "vecnormalize.pkl"
    return selected_model_paths(run)


def rollout(run: cfg.RunConfig, rel: pd.DataFrame,
            cost_rate: float | None = None,
            use_final_model: bool = False) -> pd.DataFrame:
    """Step the validation-selected policy once through a partition.

    Actions are taken deterministically (the mode of the policy
    distribution) rather than sampled, so evaluation reflects the learned
    policy rather than exploration noise.

    When ``use_final_model`` is True, load the final ``model.zip`` rather
    than the validation-selected checkpoint (see ``--skip-selection``).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    c = cfg.COST_RATE if cost_rate is None else cost_rate
    model_path, vec_path = _model_paths(run, use_final_model)

    env = PortfolioEnv(rel, cost_rate=c, penalty_weight=run.penalty_weight,
                       record_history=True)
    vec = DummyVecEnv([lambda: env])
    if vec_path is not None and vec_path.exists():
        # Reload the normalisation statistics saved with the model. Omitting
        # this presents unnormalised observations to a network trained on
        # normalised ones; the run completes without error and is wrong.
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

    hist = env.history
    # Attach environment diagnostics promised in Sections 3.4 and 4.4. The
    # epsilon floor should never activate on price movement alone -- the
    # worst price relative in the sample is 0.7468 -- so a non-zero count
    # would indicate the penalty, not the data, driving the argument of the
    # logarithm non-positive.
    hist.attrs["floor_events"] = int(env.total_floor_events)
    hist.attrs["ruin_episodes"] = int(env.ruin_episodes)
    hist.attrs["episodes"] = int(env.episodes)
    return hist


def evaluate_run(run: cfg.RunConfig, test_rel: pd.DataFrame,
                 use_final_model: bool = False) -> dict:
    """Metrics for one agent at the true cost rate, on the test partition."""
    hist = rollout(run, test_rel, use_final_model=use_final_model)
    m = mt.compute_metrics(hist)
    m.update({"label": run.label, "eta": run.penalty_weight,
              "eta_multiple": run.eta_multiple, "seed": run.seed,
              "floor_events": hist.attrs.get("floor_events", 0),
              "ruin_episodes": hist.attrs.get("ruin_episodes", 0)})
    hist.to_csv(run.dir / "test_history.csv")
    return m


def load_test_histories(df: pd.DataFrame, eta_multiple: float) -> list[pd.DataFrame]:
    """Saved test histories for every seed of one condition."""
    labels = df.loc[df["eta_multiple"] == eta_multiple, "label"]
    return [pd.read_csv(cfg.RUNS_DIR / lbl / "test_history.csv") for lbl in labels]


# ---------------------------------------------------------------------------
# Cost sensitivity (no retraining)
# ---------------------------------------------------------------------------
def cost_sensitivity(run: cfg.RunConfig, test_rel: pd.DataFrame,
                     rates: tuple[float, ...] = cfg.EVAL_COST_RATES,
                     use_final_model: bool = False) -> pd.DataFrame:
    """Re-score a fixed policy under alternative cost assumptions.

    The policy does not change, so only the arithmetic of the portfolio
    update is repeated. Addresses Assumption A4 by establishing how far
    conclusions depend on the constant-cost simplification.
    """
    rows = []
    for c in rates:
        hist = rollout(run, test_rel, cost_rate=c,
                       use_final_model=use_final_model)
        m = mt.compute_metrics(hist)
        m.update({"cost_rate_bps": c * 10_000, "label": run.label,
                  "eta_multiple": run.eta_multiple, "seed": run.seed})
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------
def evaluate_all(test_rel: pd.DataFrame,
                 use_final_model: bool = False) -> pd.DataFrame:
    rows = []
    for run in cfg.all_runs():
        if not (run.dir / "model.zip").exists():
            log.warning("%s not trained, skipping", run.label)
            continue
        log.info("Evaluating %s", run.label)
        rows.append(evaluate_run(run, test_rel,
                                 use_final_model=use_final_model))
    df = pd.DataFrame(rows)
    df.to_csv(cfg.RESULTS_DIR / "agent_metrics.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Two-level statistical protocol (Section 3.7)
# ---------------------------------------------------------------------------
def statistical_tests(df: pd.DataFrame) -> dict:
    """Level 1 across seeds; Level 2 on ensemble daily return series.

    Level 1 asks whether the reward specification changes the distribution
    of training outcomes. Level 2 asks whether the realised equity curves
    are statistically distinguishable in risk-adjusted terms, using the full
    daily series rather than ten seed summaries.
    """
    results: dict = {"level_1": {}, "level_2": {}, "pairwise": {}}
    blind = df[df["eta_multiple"] == 0]
    if blind.empty:
        raise ValueError("no cost-blind (eta = 0) runs found")

    blind_hists = load_test_histories(df, 0.0)
    blind_ens = mt.ensemble_returns(blind_hists)
    p_values: dict[str, float] = {}

    for mult in sorted(m for m in df["eta_multiple"].unique() if m > 0):
        aware = df[df["eta_multiple"] == mult]
        key = f"eta_{mult:g}c_vs_blind"

        # ---- Level 1: seed-level summaries --------------------------------
        results["level_1"][key] = {}
        for metric in ("sharpe", "ann_turnover", "total_cost"):
            cmp = mt.compare_seeds(aware[metric].to_numpy(),
                                   blind[metric].to_numpy(), metric=metric)
            results["level_1"][key][metric] = cmp.as_dict()
            log.info("%s | %s", key, cmp)
            if metric == "sharpe":               # primary endpoint only
                p_values[key] = cmp.welch_p

        # ---- Level 2: ensemble return series ------------------------------
        aware_hists = load_test_histories(df, mult)
        aware_ens = mt.ensemble_returns(aware_hists)
        lw = mt.sharpe_difference_test(aware_ens, blind_ens)
        results["level_2"][key] = lw
        log.info("%s | Ledoit-Wolf (ensemble of %d seeds): diff=%+.4f p=%.4f",
                 key, len(aware_hists), lw["difference"], lw["p_value"])

        # ---- Robustness: full seed grid, reported as a distribution -------
        pw = mt.pairwise_sharpe_differences(aware_hists, blind_hists)
        pw.to_csv(cfg.RESULTS_DIR / f"pairwise_{key}.csv", index=False)
        results["pairwise"][key] = {
            "n_comparisons": int(len(pw)),
            "mean_difference": float(pw["difference"].mean()),
            "median_difference": float(pw["difference"].median()),
            "fraction_favouring_cost_aware": float((pw["difference"] > 0).mean()),
            "min_difference": float(pw["difference"].min()),
            "max_difference": float(pw["difference"].max()),
        }

    # Family-wise correction across primary-endpoint comparisons
    results["holm_bonferroni"] = mt.holm_bonferroni(p_values).to_dict("records")
    (cfg.RESULTS_DIR / "statistical_tests.json").write_text(
        json.dumps(results, indent=2, default=float))
    return results


# ---------------------------------------------------------------------------
# Figures and reporting
# ---------------------------------------------------------------------------
def make_figures(df: pd.DataFrame, baselines: dict[str, pd.DataFrame]) -> None:
    """Chapter 5 figures: equity curves, the eta trade-off, validation curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = cfg.RESULTS_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Figure 1: cumulative value, median-seed agents vs baselines
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for mult in sorted(df["eta_multiple"].unique()):
        sub = df[df["eta_multiple"] == mult]
        med = sub.iloc[(sub["sharpe"] - sub["sharpe"].median()).abs().argsort()].iloc[0]
        h = pd.read_csv(cfg.RUNS_DIR / med["label"] / "test_history.csv")
        lbl = "cost-blind (eta = 0)" if mult == 0 else f"eta = {mult:g}c"
        ax.plot(h["portfolio_value"].to_numpy(), label=lbl, lw=1.6)
    for name, h in baselines.items():
        ax.plot(h["portfolio_value"].to_numpy(), lw=1.0, ls="--",
                alpha=0.65, label=name)
    ax.set_xlabel("Trading day (test partition)")
    ax.set_ylabel("Portfolio value (GBP)")
    ax.set_title("Net-of-cost portfolio value, test period")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_equity_curves.png", dpi=200)
    plt.close(fig)

    # Figure 2: the eta trade-off -- primary endpoint and mechanism
    agg = df.groupby("eta_multiple").agg(
        sharpe_mean=("sharpe", "mean"), sharpe_std=("sharpe", "std"),
        turnover_mean=("ann_turnover", "mean"), turnover_std=("ann_turnover", "std"),
    ).reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.errorbar(agg["eta_multiple"], agg["sharpe_mean"], yerr=agg["sharpe_std"],
                marker="o", capsize=4)
    a1.set_xlabel("Penalty weight (multiples of c)")
    a1.set_ylabel("Net-of-cost Sharpe ratio")
    a1.set_title("Primary endpoint")
    a2.errorbar(agg["eta_multiple"], agg["turnover_mean"], yerr=agg["turnover_std"],
                marker="s", color="darkred", capsize=4)
    a2.set_xlabel("Penalty weight (multiples of c)")
    a2.set_ylabel("Annualised turnover")
    a2.set_title("Mechanism")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_eta_tradeoff.png", dpi=200)
    plt.close(fig)

    # Figure 4: training curves from the Monitor logs
    curves = {}
    for run in cfg.all_runs():
        mdir = run.dir / "monitor"
        if not mdir.exists():
            continue
        frames = []
        for f in sorted(mdir.glob("*.csv")):
            try:
                frames.append(pd.read_csv(f, skiprows=1))
            except Exception:
                continue
        if frames:
            d = pd.concat(frames).sort_values("t")
            curves.setdefault(run.eta_multiple, []).append(d)

    if curves:
        fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for mult in sorted(curves):
            d = pd.concat(curves[mult]).sort_values("t").reset_index(drop=True)
            w = max(len(d) // 50, 1)
            lbl = "cost-blind (eta = 0)" if mult == 0 else f"eta = {mult:g}c"
            b1.plot(d["r"].rolling(w, min_periods=1).mean().to_numpy(), label=lbl, lw=1.4)
            b2.plot(d["turnover"].rolling(w, min_periods=1).mean().to_numpy(), label=lbl, lw=1.4)
        b1.set_xlabel("Episode"); b1.set_ylabel("Episode return")
        b1.set_title("Training reward")
        b2.set_xlabel("Episode"); b2.set_ylabel("Mean turnover per step")
        b2.set_title("Turnover during training")
        for a in (b1, b2):
            a.grid(alpha=0.3); a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig4_training_curves.png", dpi=200)
        plt.close(fig)

    # Figure 3: validation curves, evidencing the training budget
    if (cfg.RESULTS_DIR / "checkpoint_selection.csv").exists():
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for run in cfg.all_runs():
            f = run.dir / "validation_curve.csv"
            if f.exists():
                cur = pd.read_csv(f)
                ax.plot(cur["steps"], cur["sharpe"], alpha=0.35, lw=0.9)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel("Validation Sharpe ratio")
        ax.set_title("Validation performance during training (all runs)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig3_validation_curves.png", dpi=200)
        plt.close(fig)

    log.info("Figures written to %s", fig_dir)


def quantstats_report(run_label: str) -> None:
    """Full tear sheet for the best agent (Appendix C)."""
    try:
        import quantstats as qs
    except ImportError:
        log.warning("quantstats not installed; skipping tear sheet")
        return
    h = pd.read_csv(cfg.RUNS_DIR / run_label / "test_history.csv",
                    parse_dates=["date"], index_col="date")
    returns = pd.Series(mt._net_returns(h), index=h.index)
    out = cfg.RESULTS_DIR / f"tearsheet_{run_label}.html"
    qs.reports.html(returns, output=str(out),
                    title=f"Cost-aware PPO ({run_label})")
    log.info("Tear sheet written to %s", out)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained agents")
    ap.add_argument("--skip-selection", action="store_true",
                    help="use final models rather than validation-selected ones")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parts = build()

    # ---- 1. Checkpoint selection on VALIDATION (test untouched) -----------
    if not args.skip_selection:
        log.info("--- checkpoint selection on validation (%d rows, %s to %s) ---",
                 len(parts.val), parts.val.index[0].date(),
                 parts.val.index[-1].date())
        sel = select_all(parts.val)
        if not sel.empty:
            log.info("Median selected budget fraction: %.0f%%",
                     100 * sel["selected_fraction_of_budget"].median())

    # ---- 2. Baselines -----------------------------------------------------
    log.info("--- baselines ---")
    base = run_all_baselines(parts.test)
    brows = []
    for name, hist in base.items():
        m = mt.compute_metrics(hist)
        m["strategy"] = name
        brows.append(m)
        hist.to_csv(cfg.RESULTS_DIR / f"baseline_{name}.csv")
    pd.DataFrame(brows).to_csv(cfg.RESULTS_DIR / "baseline_metrics.csv", index=False)

    # ---- 3. Agents on TEST ------------------------------------------------
    log.info("--- agents on test partition ---")
    agents = evaluate_all(parts.test, use_final_model=args.skip_selection)
    if agents.empty:
        log.error("No trained agents found. Run src.train first.")
        return

    # ---- 4. Statistics ----------------------------------------------------
    log.info("--- statistical tests ---")
    statistical_tests(agents)

    # ---- 5. Cost sensitivity ---------------------------------------------
    if not args.skip_sensitivity:
        log.info("--- cost sensitivity ---")
        frames = [cost_sensitivity(r, parts.test,
                                   use_final_model=args.skip_selection)
                  for r in cfg.all_runs()
                  if (r.dir / "model.zip").exists()]
        if frames:
            pd.concat(frames).to_csv(cfg.RESULTS_DIR / "cost_sensitivity.csv",
                                     index=False)

    # ---- 6. Figures and tear sheet ---------------------------------------
    if not args.skip_figures:
        make_figures(agents, base)
        best = agents.loc[agents["sharpe"].idxmax(), "label"]
        quantstats_report(best)

    if "floor_events" in agents.columns:
        fe, re_ = int(agents["floor_events"].sum()), int(agents["ruin_episodes"].sum())
        log.info("Environment diagnostics across %d runs: %d log-floor "
                 "activation(s), %d ruin termination(s)", len(agents), fe, re_)
        if fe:
            log.warning("The epsilon floor activated %d time(s); inspect whether "
                        "the penalty drove the log argument non-positive", fe)

    log.info("Results written to %s", cfg.RESULTS_DIR)


if __name__ == "__main__":
    main()
