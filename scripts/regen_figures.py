"""Rebuild Chapter 5 figures from saved evaluation outputs.

Does not re-run checkpoint selection or test rollouts. Requires
``results/agent_metrics.csv`` and ``results/baseline_*.csv`` from a prior
``python -m src.evaluate``.

    python scripts/regen_figures.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config as cfg
from src.evaluate import make_figures, quantstats_report

log = logging.getLogger(__name__)

# Names written by evaluate.main via ``baseline_{name}.csv``.
_BASELINE_NAMES = (
    "equal_weight_bh",
    "calendar_monthly",
    "threshold_5pct",
    "min_variance",
    "risk_parity",
)


def _load_baselines() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in _BASELINE_NAMES:
        path = cfg.RESULTS_DIR / f"baseline_{name}.csv"
        if not path.exists():
            log.warning("Missing %s; skipping that baseline in figures", path.name)
            continue
        out[name] = pd.read_csv(path)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    metrics_path = cfg.RESULTS_DIR / "agent_metrics.csv"
    if not metrics_path.exists():
        log.error("No %s — run python -m src.evaluate first", metrics_path)
        sys.exit(1)

    agents = pd.read_csv(metrics_path)
    if agents.empty:
        log.error("agent_metrics.csv is empty")
        sys.exit(1)

    baselines = _load_baselines()
    if not baselines:
        log.warning("No baseline CSVs found; equity figure will show agents only")

    make_figures(agents, baselines)

    best = agents.loc[agents["sharpe"].idxmax(), "label"]
    quantstats_report(best)

    fig_dir = cfg.RESULTS_DIR / "figures"
    expected = (
        "fig1_equity_curves.png",
        "fig2_eta_tradeoff.png",
        "fig3_validation_curves.png",
        "fig4_training_curves.png",
    )
    for name in expected:
        path = fig_dir / name
        if path.exists():
            log.info("OK %s", path)
        else:
            log.warning("Missing %s (may be skipped if source logs absent)", path)


if __name__ == "__main__":
    main()
