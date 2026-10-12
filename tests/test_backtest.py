"""Backtest accounting and the out-of-sample lock.

Small worked examples with numbers you can check by hand, plus the invented
market from fake_market.py. None of it is market data.
"""

import numpy as np
import pandas as pd
import pytest
from fake_market import random_market

from divergence import backtest, db
from divergence.backtest import BacktestConfig, OutOfSampleRefused
from divergence.signals import SignalConfig

COST = 0.0005  # 5 basis points

SIGNAL = SignalConfig(sentiment_model="test", return_days=5, zscore_window=60,
                      min_history=20, threshold=1.5, holding_days=5)


def hand_panel(positions: dict, returns: dict) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=len(next(iter(positions.values()))))
    return pd.concat([pd.DataFrame({"ticker": t, "date": dates, "position": positions[t],
                                    "return_open_to_open": returns[t]}) for t in positions],
                     ignore_index=True)


def test_daily_returns_split_money_equally_and_charge_every_weight_change():
    panel = hand_panel(
        positions={"A": [1, 1, 0, 0], "B": [0, -1, 0, 0]},
        returns={"A": [0.01, 0.02, 0.03, np.nan], "B": [0.05, -0.04, 0.01, np.nan]},
    )
    result = backtest.daily_returns(panel, "position", COST)

    # Day 1: all money long A.            0.01
    # Day 2: half long A, half short B.   0.5*0.02 + (-0.5)*(-0.04) = 0.03
    # Day 3: flat.                        0
    assert result["gross"].tolist() == pytest.approx([0.01, 0.03, 0.0])
    # Weight changes each day: buy A (1.0); sell half A + short half B (1.0); close both (1.0).
    assert result["cost"].tolist() == pytest.approx([COST, COST, COST])
    assert len(result) == 3  # last day dropped: its next open is unknown


def test_buy_and_hold_lets_weights_drift_and_pays_to_enter_and_exit():
    dates = pd.bdate_range("2023-01-02", periods=3)
    panel = pd.DataFrame({"ticker": ["A"] * 3 + ["B"] * 3, "date": list(dates) * 2,
                          "adj_open": [100, 110, 121, 100, 100, 100]})
    returns = backtest.buy_and_hold_returns(panel, dates[0], dates[-1], COST)
    # Portfolio value 1.0 -> 1.05 -> 1.105 (A grows, B flat, equal money at the start)
    assert returns.tolist() == pytest.approx([0.05 - COST, 1.105 / 1.05 - 1 - COST])


def test_trades_are_unbroken_runs_and_a_flip_is_two_trades():
    panel = hand_panel(positions={"A": [0, 1, 1, -1, 0]},
                       returns={"A": [0.0, 0.1, 0.1, -0.1, 0.0]})
    trades = backtest.trade_list(panel, "position", COST)
    assert trades["direction"].tolist() == [1, -1]
    assert trades["days"].tolist() == [2, 1]
    assert trades["net_return"].tolist() == pytest.approx([1.1 * 1.1 - 1 - 2 * COST, 0.1 - 2 * COST])


def test_summary_statistics():
    stats = backtest.summarize(pd.Series([0.1, -0.5, 0.2]))
    assert stats["max_drawdown"] == pytest.approx(-0.5)         # 1.1 -> 0.55
    assert stats["annual_return"] == pytest.approx(0.66 ** (252 / 3) - 1)
    # A loss on the very first day is a drawdown from the starting money.
    assert backtest.summarize(pd.Series([-0.1, 0.05]))["max_drawdown"] == pytest.approx(-0.1)


def test_placebo_keeps_the_scores_but_scrambles_their_days():
    _, sentiment = random_market(seed=3)
    shuffled = backtest.shuffled_sentiment(sentiment, np.random.default_rng(0))
    for ticker in ["AAA", "BBB"]:
        real = sentiment[sentiment["ticker"] == ticker]
        fake = shuffled[shuffled["ticker"] == ticker]
        assert sorted(real["sentiment"]) == sorted(fake["sentiment"])        # same scores
        assert real["trading_day"].tolist() == fake["trading_day"].tolist()  # same news days
        assert real["sentiment"].tolist() != fake["sentiment"].tolist()      # different order


def bt_config(**overrides) -> BacktestConfig:
    return BacktestConfig(**{"in_sample_end": "2020-09-30", "cost_bps_per_side": 5,
                             "placebo_runs": 3, "locked": False, **overrides})


def test_unlocked_backtest_never_reports_the_test_period():
    prices, sentiment = random_market(seed=4)
    results = backtest.evaluate(prices, {"test": sentiment}, SIGNAL, bt_config(), include_out_of_sample=False)
    assert list(results["periods"]) == ["in_sample"]
    assert {row["period"] for row in results["table"]} == {"in_sample"}
    for curve in results["curves"].values():
        assert curve.index.max() <= pd.Timestamp("2020-09-30")


def test_locked_backtest_reports_both_periods_and_every_comparison():
    prices, sentiment = random_market(seed=4)
    results = backtest.evaluate(prices, {"test": sentiment}, SIGNAL, bt_config(locked=True),
                                include_out_of_sample=True)
    assert list(results["periods"]) == ["in_sample", "out_of_sample"]
    strategies = {row["strategy"] for row in results["table"]}
    assert strategies == {"news divergence (test)", "price only (no news)", "buy and hold (equal weight)"}
    for period in results["periods"]:
        assert 0 < results["placebo"][period]["p_value"] <= 1


def test_changing_parameters_after_an_out_of_sample_run_needs_a_reason():
    conn = db.connect(":memory:")
    config = bt_config(locked=True)
    backtest.check_out_of_sample_allowed(conn, SIGNAL, config, reason=None)  # first look: fine
    db.log_out_of_sample_run(conn, backtest.parameters_fingerprint(SIGNAL, config), None)

    backtest.check_out_of_sample_allowed(conn, SIGNAL, config, reason=None)  # same settings: fine

    tuned = SignalConfig(**{**SIGNAL.__dict__, "threshold": 2.0})
    with pytest.raises(OutOfSampleRefused):
        backtest.check_out_of_sample_allowed(conn, tuned, config, reason=None)
    backtest.check_out_of_sample_allowed(conn, tuned, config, reason="fixed a bug in X")  # allowed, and logged by caller
