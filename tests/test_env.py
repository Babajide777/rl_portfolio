"""
Verification suite for the portfolio environment (Section 4.8).

An environment that runs without error may nonetheless be incorrect. Because
the entire experiment rests on the environment computing turnover and cost
correctly, these tests are run before any training run is launched.

    python -m pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.portfolio_env import PortfolioEnv


@pytest.fixture
def rel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(1.0004, 0.011, (300, 9)),
        index=pd.bdate_range("2020-01-01", periods=300),
        columns=list(cfg.ASSETS),
    )


@pytest.fixture
def actions() -> np.ndarray:
    return np.random.default_rng(7).normal(size=(60, 9))


# ---------------------------------------------------------------------------
# 1. Interface conformance
# ---------------------------------------------------------------------------
def test_gymnasium_env_checker(rel):
    """The environment satisfies the Gymnasium API contract."""
    from gymnasium.utils.env_checker import check_env
    check_env(PortfolioEnv(rel), skip_render_check=True)


def test_observation_shape(rel):
    env = PortfolioEnv(rel)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (cfg.LOOKBACK * 9 + 9,) == (279,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# 2. Invariants
# ---------------------------------------------------------------------------
def test_actions_lie_on_simplex(rel, actions):
    """Softmax must yield non-negative weights summing to one."""
    env = PortfolioEnv(rel)
    env.reset(seed=0)
    for a in actions:
        _, _, _, _, info = env.step(a)
        w = info["weights"]
        assert np.all(w >= 0.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_extreme_logits_do_not_overflow(rel):
    """Softmax shifts by the max, so large logits must not produce NaN."""
    env = PortfolioEnv(rel)
    env.reset(seed=0)
    _, r, _, _, info = env.step(np.array([1e4, -1e4, 0, 0, 0, 0, 0, 0, 0]))
    assert np.isfinite(r)
    assert info["weights"].sum() == pytest.approx(1.0)


def test_portfolio_value_stays_positive(rel, actions):
    env = PortfolioEnv(rel)
    env.reset(seed=0)
    for a in actions:
        _, _, _, _, info = env.step(a)
        assert info["portfolio_value"] > 0


def test_weights_drift_but_remain_normalised(rel, actions):
    """Overnight drift changes weights without trading; they stay on the simplex."""
    env = PortfolioEnv(rel)
    env.reset(seed=0)
    env.step(actions[0])
    assert env._w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(env._w >= 0.0)


# ---------------------------------------------------------------------------
# 3. Turnover and cost reconciliation
# ---------------------------------------------------------------------------
def test_turnover_matches_independent_calculation(rel, actions):
    """Environment turnover must equal an independent pandas computation.

    Confirms the cost charged corresponds to the reallocation actually
    performed, which is the quantity the entire study measures.
    """
    env = PortfolioEnv(rel)
    env.reset(seed=0)
    prev = np.full(9, 1 / 9)
    for a in actions[:20]:
        _, _, _, _, info = env.step(a)
        expected = np.abs(info["weights"] - prev).sum()
        assert info["turnover"] == pytest.approx(expected, abs=1e-12)
        prev = env._w.copy()          # post-drift weights


def test_cost_equals_rate_times_turnover(rel, actions):
    c = 0.0005
    env = PortfolioEnv(rel, cost_rate=c)
    env.reset(seed=0)
    for a in actions[:20]:
        _, _, _, _, info = env.step(a)
        assert info["cost"] == pytest.approx(c * info["turnover"], abs=1e-15)


def test_holding_position_incurs_no_cost(rel):
    """Repeating the current weights exactly must cost nothing."""
    env = PortfolioEnv(rel, cost_rate=0.0005)
    env.reset(seed=0)
    logits = np.log(env._w)           # softmax(log w) == w
    _, _, _, _, info = env.step(logits)
    assert info["turnover"] == pytest.approx(0.0, abs=1e-12)
    assert info["cost"] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 4. Degenerate cases -- the separation of c from eta
# ---------------------------------------------------------------------------
def test_cost_blind_reward_is_log_gross(rel, actions):
    """With eta = 0 the reward must be exactly log of the gross return."""
    env = PortfolioEnv(rel, cost_rate=0.0005, penalty_weight=0.0)
    env.reset(seed=0)
    for a in actions[:20]:
        _, r, _, _, info = env.step(a)
        assert r == pytest.approx(np.log(info["gross_return"]), abs=1e-12)


def test_eta_does_not_affect_portfolio_value(rel, actions):
    """THE CRITICAL TEST.

    Two environments differing only in eta, driven by identical actions,
    must produce identical portfolio values while producing different
    rewards. This is the direct confirmation that the cost rate and the
    penalty weight are properly separated: every agent pays the same real
    cost, and only the training signal differs.

    Under the conflated formulation -- where a single parameter served as
    both the charged rate and the reward penalty -- this test fails, because
    eta = 0 would mean trading was free.
    """
    values, rewards = {}, {}
    for eta in (0.0, cfg.COST_RATE, 10 * cfg.COST_RATE, 50 * cfg.COST_RATE):
        env = PortfolioEnv(rel, cost_rate=cfg.COST_RATE, penalty_weight=eta)
        env.reset(seed=0)
        total = 0.0
        for a in actions:
            _, r, _, _, info = env.step(a)
            total += r
        values[eta] = info["portfolio_value"]
        rewards[eta] = total

    vs = list(values.values())
    assert all(v == pytest.approx(vs[0], rel=1e-12) for v in vs), \
        "portfolio value must not depend on eta"

    rs = list(rewards.values())
    assert len(set(np.round(rs, 9))) == len(rs), \
        "reward must differ across eta"

    # Larger penalty must yield a smaller reward for identical behaviour
    ordered = [rewards[e] for e in sorted(rewards)]
    assert all(a > b for a, b in zip(ordered, ordered[1:])), \
        "reward must decrease monotonically in eta"


def test_cost_rate_does_affect_portfolio_value(rel, actions):
    """Conversely, changing the CHARGED rate must change the outcome."""
    finals = []
    for c in (0.0, 0.0005, 0.005):
        env = PortfolioEnv(rel, cost_rate=c, penalty_weight=0.0)
        env.reset(seed=0)
        for a in actions:
            _, _, _, _, info = env.step(a)
        finals.append(info["portfolio_value"])
    assert finals[0] > finals[1] > finals[2], "higher charged cost must reduce value"


# ---------------------------------------------------------------------------
# 5. Episode mechanics
# ---------------------------------------------------------------------------
def test_equal_weight_initialisation(rel):
    env = PortfolioEnv(rel)
    env.reset(seed=0)
    assert env._w == pytest.approx(np.full(9, 1 / 9))


def test_reset_restores_capital(rel, actions):
    env = PortfolioEnv(rel)
    env.reset(seed=0)
    for a in actions[:10]:
        env.step(a)
    env.reset(seed=0)
    assert env._value == cfg.INITIAL_CAPITAL
    assert env._t == cfg.LOOKBACK


def test_truncates_at_end_of_data(rel):
    env = PortfolioEnv(rel)
    env.reset(seed=0)
    trunc = False
    for _ in range(len(rel)):
        _, _, term, trunc, _ = env.step(np.zeros(9))
        if term or trunc:
            break
    assert trunc


def test_log_floor_counts_activations(rel):
    """A catastrophic penalty forces the floor; activations must be counted."""
    env = PortfolioEnv(rel, cost_rate=0.0005, penalty_weight=1e6)
    env.reset(seed=0)
    env.step(np.array([10.0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert env.floor_events >= 1
    assert env.total_floor_events >= 1


def test_rejects_negative_parameters(rel):
    with pytest.raises(ValueError):
        PortfolioEnv(rel, cost_rate=-0.001)
    with pytest.raises(ValueError):
        PortfolioEnv(rel, penalty_weight=-0.001)


def test_determinism_under_fixed_seed(rel, actions):
    """Identical seed and actions must give bit-identical trajectories."""
    out = []
    for _ in range(2):
        env = PortfolioEnv(rel)
        env.reset(seed=123)
        vals = [env.step(a)[4]["portfolio_value"] for a in actions]
        out.append(vals)
    assert out[0] == out[1]

def test_action_space_bounds_are_finite(rel):
    """REGRESSION TEST.

    Stable-Baselines3 asserts that a Box action space has finite bounds
    (base_class.py: "Continuous action space must have a finite lower and
    upper bound"). An unbounded space raises AssertionError at model
    construction, which surfaces as every benchmark configuration failing
    identically -- including CPU with a single environment, ruling out any
    device or vectorisation cause.
    """
    env = PortfolioEnv(rel)
    assert np.all(np.isfinite(env.action_space.low)), "lower bound must be finite"
    assert np.all(np.isfinite(env.action_space.high)), "upper bound must be finite"


def test_action_bounds_permit_concentration(rel):
    """The finite bound must not meaningfully restrict the policy.

    An action at the extreme of the permitted range should still allow
    almost all capital in one asset, confirming the bound guards against
    numerical pathology rather than constraining strategy.
    """
    env = PortfolioEnv(rel)
    env.reset(seed=0)
    extreme = np.full(9, -cfg.LOGIT_BOUND)
    extreme[0] = cfg.LOGIT_BOUND
    _, _, _, _, info = env.step(extreme)
    assert info["weights"][0] > 0.999, "bound is too tight to concentrate"


def test_env_passes_sb3_action_space_assertion(rel):
    """Replicate the exact SB3 check that failed at model construction."""
    env = PortfolioEnv(rel)
    from gymnasium import spaces
    assert isinstance(env.action_space, spaces.Box)
    assert np.all(np.isfinite(
        np.array([env.action_space.low, env.action_space.high])
    )), "Continuous action space must have a finite lower and upper bound"


def test_history_disabled_by_default(rel, actions):
    """History must NOT accumulate during training.

    Recording a dict plus an array copy every step costs roughly 1.3 GB
    across eight environments over two million steps, for data that the
    training loop never reads. Evaluation and checkpoint selection opt in
    explicitly via ``record_history=True``.

    The transient ``info`` dict still carries weights in both modes, since
    it is discarded each step and costs nothing.
    """
    env = PortfolioEnv(rel)
    env.reset(seed=0)
    for a in actions[:30]:
        _, _, _, _, info = env.step(a)
    assert env._history == [], "history accumulated with record_history=False"
    assert "weights" in info, "info must still expose weights"

    env2 = PortfolioEnv(rel, record_history=True)
    env2.reset(seed=0)
    for a in actions[:30]:
        env2.step(a)
    assert len(env2._history) == 30
    assert "weights" in env2._history[-1]


def test_make_env_wraps_in_monitor(rel, tmp_path):
    """REGRESSION TEST.

    Without a Monitor wrapper, Stable-Baselines3 logs no ``rollout/*``
    statistics: the optimiser diagnostics under ``train/*`` still appear, so
    training looks healthy, but there is no record of episode return and
    therefore no training curve. The absence is silent, which is why it is
    asserted here.
    """
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.monitor import Monitor
    from src.portfolio_env import make_env

    env = make_env(rel, penalty_weight=0.005,
                   monitor_path=str(tmp_path / "env0"))()
    assert isinstance(env, Monitor)
    # training must not accumulate history through the wrapper
    assert env.unwrapped.record_history is False


def test_monitor_records_episode_statistics(rel, tmp_path):
    """Monitor must emit episode return, length, turnover and cost."""
    pytest.importorskip("stable_baselines3")
    from src.portfolio_env import make_env

    env = make_env(rel, penalty_weight=0.005,
                   monitor_path=str(tmp_path / "env0"))()
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    info = {}
    for _ in range(len(rel)):
        _, _, term, trunc, info = env.step(rng.normal(size=9))
        if term or trunc:
            break

    assert "episode" in info, "Monitor did not emit episode statistics"
    for key in ("r", "l", "t"):
        assert key in info["episode"]

    csvs = list(tmp_path.glob("*.csv"))
    assert csvs, "Monitor wrote no CSV"
    header = csvs[0].read_text().splitlines()[1]
    for field in ("r", "l", "t", "turnover", "cost"):
        assert field in header, f"'{field}' absent from monitor CSV"
