# Reinforcement Learning for Cost-Aware Portfolio Rebalancing

Implementation accompanying the MSc dissertation (CO7047, University of Chester).

Trains PPO agents to rebalance a nine-ETF portfolio under a reward function in
which transaction costs are embedded directly, and compares them against
otherwise-identical cost-blind agents and against classical and rule-based
baselines.

---

## The central design decision

Two parameters govern transaction costs, and conflating them invalidates the
experiment:

| Parameter | Role | Behaviour |
|---|---|---|
| `cost_rate` (**c**) | What the **market** charges | Fixed at 5 bps. Applies to *every* agent in *every* condition. Updates portfolio value. |
| `penalty_weight` (**η**) | What the **reward** penalises | A training hyperparameter. Swept experimentally. `η = 0` is the cost-blind agent. |

A cost-blind agent pays exactly the same real trading costs as a cost-aware
one — it simply receives no signal about them. Were one parameter to serve
both roles, setting it to zero would remove costs altogether, and the
comparison would be between two different *markets* rather than two
differently *instructed* agents.

This is enforced by `tests/test_env.py::test_eta_does_not_affect_portfolio_value`,
which asserts that four environments differing only in η, driven by identical
actions, produce identical portfolio values but different rewards. That test
fails under the conflated formulation.

---

## Installation

```bash
python -m venv .venv

# Unix / macOS
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Usage

```bash
# 1. Verify the environment before spending compute
python -m pytest tests/ -v

# 2. Establish throughput and pick the fastest configuration
python scripts/benchmark.py
# then set N_ENVS and TORCH_NUM_THREADS in src/config.py to whatever it reports

# 3. Check the wiring end to end (20k steps, ~1 minute)
python -m src.train --smoke

# 4. Full batch: 4 penalty weights x 10 seeds = 40 runs
python -m src.train

# 5. Evaluate, compare against baselines, run statistical tests
python -m src.evaluate

# 6. (Optional) Rebuild Chapter 5 figures from existing CSVs without re-rolling out
python scripts/regen_figures.py
```

Training is genuinely resumable. Completed runs are skipped; an interrupted
run reloads its most recent checkpoint (model *and* `VecNormalize`
statistics) and continues with `reset_num_timesteps=False`, so the step
counter and learning-rate schedule carry on rather than restarting.

## Layout

```
src/config.py          Every experimental constant. No magic numbers elsewhere.
src/data_pipeline.py   Acquisition, integrity checks, price relatives,
                       partitioning with purge and embargo.
src/portfolio_env.py   The Gymnasium environment. Principal artefact.
src/baselines.py       MVO, risk parity, equal-weight B&H, calendar, threshold.
                       Classical baselines are walk-forward.
src/metrics.py         Performance metrics and the two-level statistical protocol.
src/train.py           The 40-run batch, resumable from checkpoints.
src/selection.py       Validation-based checkpoint selection.
src/evaluate.py        Rollout, cost sensitivity, statistics, figures.
tests/test_env.py      Environment verification tests.
tests/test_selection.py Checkpoint selection and evaluation path tests.
scripts/benchmark.py   CPU/GPU and vectorisation throughput comparison.
scripts/regen_figures.py Rebuild figures from saved evaluation CSVs.
```

A local `pics/` folder (if present) is for personal notes only and is
gitignored; it is not part of the reproducible codebase.

## Experimental design

| | |
|---|---|
| Universe | SPY, QQQ, IWM, EFA, TLT, LQD, GLD, VNQ, USO |
| Period | 2006–2024 (2006 reserved for the initial lookback) |
| Splits | Train 2007–2020 · Validate 2021 · Test 2022–2024 |
| Boundaries | 30-day purge + 21-day embargo |
| State | 9 × 30 price relatives + 9 current weights = 279 values |
| Action | Softmax over 9 logits → long-only, fully invested |
| Reward | `log(w·y − η·turnover)`, floored at 1e-8 |
| Conditions | η ∈ {0, c, 10c, 50c} |
| Seeds | 10 per condition |
| Timesteps | 2,000,000 per run |

USO launched 10 April 2006, which is the latest inception in the universe and
determines the study start date.

## Statistical protocol

Two sources of randomness, tested separately:

**Level 1 — training stochasticity.** Per-seed Sharpe and turnover compared
with Welch's *t*-test (not Student's: RL runs violate equal variances),
Cohen's *d*, bootstrap confidence intervals, Mann–Whitney U as corroboration.

**Level 2 — realised returns.** An equal-weighted *ensemble* daily return
series is formed per condition by averaging across all ten seeds, and the two
ensembles are compared with the Ledoit–Wolf (2008) studentised stationary
bootstrap. Averaging across seeds uses every run rather than a single
representative, and the ensemble has a clean interpretation: the expected
daily return of a randomly initialised agent under that condition. The
closed-form Jobson–Korkie–Memmel test assumes IID normal returns and is
inappropriate for autocorrelated, heavy-tailed daily data.

**Robustness.** All 100 seed-by-seed Sharpe differences are reported as a
distribution — mean, median, range, and the fraction favouring the cost-aware
condition. These are *not* significance tests: the comparisons are not
independent, so correction would be severe enough to be uninformative. Their
value is in showing how consistently one condition exceeds the other.

Holm–Bonferroni correction is applied across the family of primary-endpoint
comparisons.

## Performance

`scripts/benchmark.py` sweeps device, environment count, vectorisation class
and PyTorch thread count, then reports the fastest configuration and a
projected total runtime for the batch. Findings worth knowing before
committing compute:

- **Benchmark before choosing device.** The policy is a `[64, 64]` MLP, so
  environment stepping often dominates network compute; either CPU or GPU can
  win depending on the machine. This project uses `DEVICE = "auto"` in
  `src/config.py` (and records the resolved device in each run’s
  `metadata.json`).
- **PyTorch threading matters.** Multithreaded operations on a tiny network
  contend with the vectorised environments for the same cores.
  `TORCH_NUM_THREADS = 1` is a documented Stable-Baselines3 tip for exactly
  this configuration.

The environment itself runs at roughly 50,000 steps/second in isolation, so
it accounts for about 1% of training time; the remainder is PyTorch and SB3.
Episode history is therefore recorded only when `record_history=True`
(evaluation and checkpoint selection), which avoids about 1.3 GB of
allocation across eight environments over two million steps.

## Model selection

The validation partition (2021) selects checkpoints. Each checkpoint written
during training is rolled out on validation, and the one with the highest
net-of-cost Sharpe is carried forward to test. The test partition is touched
exactly once, after selection is complete.

This is why the training budget is a ceiling rather than a target: if the
selected checkpoint sits well short of 2M steps, the budget was ample and
further training degraded generalisation. `results/figures/fig3` plots the
validation curves that evidence this.

## What to expect after evaluate

After a successful `python -m src.evaluate` (or after regenerating figures):

| Artefact | Role |
|---|---|
| `results/agent_metrics.csv` | Per-seed test metrics for all agents |
| `results/baseline_metrics.csv` | Classical / rule-based baselines |
| `results/checkpoint_selection.csv` | Validation-selected checkpoints |
| `results/statistical_tests.json` | Level 1 / Level 2 / pairwise protocol |
| `results/cost_sensitivity.csv` | Fixed policies re-scored at alternative cost rates |
| `results/partition_boundaries.csv` | Train / val / test date ranges |
| `results/figures/fig1_equity_curves.png` | Test equity curves |
| `results/figures/fig2_eta_tradeoff.png` | Sharpe vs turnover by η |
| `results/figures/fig3_validation_curves.png` | Validation Sharpe during training |
| `results/figures/fig4_training_curves.png` | Monitor reward / turnover |
| `results/tearsheet_*.html` | QuantStats report for the best seed |

Per-run detail lives under `runs/<label>/` (histories, selection JSON, models).

## Sending this to a supervisor

- **GitHub** carries the **code** only (`src/`, `tests/`, `scripts/`, READMEs,
  `requirements.txt`). Experiment outputs under `runs/`, `results/`, and
  `data/` are gitignored because they are large and regenerable.
- **Attach a results zip** separately when sharing findings. Include
  `results/**` (tables, figures, tearsheet) and, optionally, light provenance
  from each run (`config.json`, `metadata.json`, `checkpoint_selection.json`).
  Do **not** send `.venv/`, full `checkpoints/`, `model.zip` trees, or
  TensorBoard logs unless specifically requested.

Example (PowerShell, from this directory):

```powershell
Compress-Archive -Path results -DestinationPath ..\rl_portfolio_results.zip -Force
```

To include light run provenance as well:

```powershell
$staging = Join-Path $env:TEMP "rl_portfolio_submission"
Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
New-Item $staging -ItemType Directory | Out-Null
Copy-Item results $staging -Recurse
Get-ChildItem runs -Directory | ForEach-Object {
  $dest = Join-Path $staging "runs\$($_.Name)"
  New-Item $dest -ItemType Directory -Force | Out-Null
  foreach ($f in "config.json","metadata.json","checkpoint_selection.json") {
    $src = Join-Path $_.FullName $f
    if (Test-Path $src) { Copy-Item $src $dest }
  }
}
Compress-Archive -Path "$staging\*" -DestinationPath ..\rl_portfolio_results.zip -Force
```

## Reproducibility

- All seeds fixed and recorded in `config.py`
- Data cached with its retrieval date; yfinance output is subject to revision
- Partition boundaries written to `results/partition_boundaries.csv`
- Per-run metadata in `runs/<label>/metadata.json`
- `VecNormalize` statistics saved with every checkpoint — omitting them
  produces a silent, severe performance collapse on reload

Determinism caveat: seeding fixes initialisation and sampling, but bitwise
reproduction additionally requires identical library versions and, on GPU,
deterministic kernel selection. Results are reproducible in distribution
rather than bitwise across differing hardware.

## Notes

Backtested performance is not predictive of live performance. This code
implements a methodological study of reward design and does not constitute
investment advice.
