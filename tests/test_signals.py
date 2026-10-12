"""Signal logic.

Inputs are small hand-made series, plus a seeded random walk used only to
exercise the code. None of it is market data.
"""

import numpy as np
import pandas as pd
import pytest
from fake_market import random_market

from divergence.signals import (SignalConfig, build_panel, divergence_flags, past_only_zscore,
                                positions_from_flags, price_only_flags)

CONFIG = SignalConfig(sentiment_model="test", return_days=5, zscore_window=60,
                      min_history=20, threshold=1.5, holding_days=5)


def test_zscore_compares_today_with_earlier_days_only():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    z = past_only_zscore(values, window=4, min_history=4)
    earlier = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert z.iloc[4] == pytest.approx((100.0 - earlier.mean()) / earlier.std())
    assert z.iloc[:4].isna().all()  # not enough history yet


def test_divergence_needs_both_sides_past_the_threshold_in_opposite_directions():
    sentiment_z = pd.Series([2.0, -2.0, 2.0, 0.0, np.nan, 2.0])
    return_z = pd.Series([-2.0, 2.0, 2.0, -2.0, -2.0, -1.0])
    # bullish, bearish, same direction, no news signal, missing news, price not far enough
    assert divergence_flags(sentiment_z, return_z, 1.5).tolist() == [1, -1, 0, 0, 0, 0]


def test_price_only_rule_buys_losers_and_shorts_winners():
    assert price_only_flags(pd.Series([-2.0, 2.0, 0.0, np.nan]), 1.5).tolist() == [1, -1, 0, 0]


def test_position_starts_the_day_after_a_flag_and_lasts_n_days():
    flags = pd.Series([0, 1, 0, 0, 0, 0, 0])
    assert positions_from_flags(flags, 3).tolist() == [0, 0, 1, 1, 1, 0, 0]


def test_newest_flag_wins_and_restarts_the_clock():
    flags = pd.Series([1, 0, -1, 0, 0, 0])
    assert positions_from_flags(flags, 2).tolist() == [0, 1, 1, -1, -1, 0]


def test_nothing_after_day_t_can_change_any_signal_on_or_before_day_t():
    prices, sentiment = random_market(seed=1)
    cutoff = pd.Timestamp("2020-09-01")
    next_day = pd.Timestamp("2020-09-02")
    before = build_panel(prices, sentiment, CONFIG)

    # Rewrite the future completely: different prices, flipped news.
    future_prices = prices["date"] > cutoff.strftime("%Y-%m-%d")
    prices.loc[future_prices, ["open", "close", "adj_close"]] *= 3.0
    sentiment.loc[sentiment["trading_day"] > cutoff, "sentiment"] *= -1
    after = build_panel(prices, sentiment, CONFIG)

    # Sanity check that the test has teeth: signals really were produced.
    assert (before["flag"] != 0).sum() > 0

    known_at_cutoff = ["return_z", "sentiment_z", "flag", "flag_price_only"]
    mask = before["date"] <= cutoff
    pd.testing.assert_frame_equal(before.loc[mask, known_at_cutoff], after.loc[mask, known_at_cutoff])

    # Tomorrow's position is fully decided by today's close.
    mask = before["date"] <= next_day
    pd.testing.assert_frame_equal(before.loc[mask, ["position", "position_price_only"]],
                                  after.loc[mask, ["position", "position_price_only"]])


def test_open_price_gets_the_same_adjustment_as_the_close():
    prices, sentiment = random_market(seed=2, days=30)
    panel = build_panel(prices, sentiment, CONFIG)
    assert np.allclose(panel["adj_open"], panel["open"] * 0.9)
