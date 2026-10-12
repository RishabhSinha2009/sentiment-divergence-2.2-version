"""The daily scan reuses the backtest's rules exactly. Inputs are hand-made, not market data."""

import numpy as np
import pandas as pd

from divergence import scan
from divergence.config import Config
from divergence.signals import SignalConfig, build_panel

SIGNAL = SignalConfig(sentiment_model="fake", return_days=5, zscore_window=60,
                      min_history=20, threshold=1.5, holding_days=3)
CONFIG = Config(tickers=["AAA"], news_aliases={}, start=None, end=None, database=None,
                news_page_size=50, signal=SIGNAL)


def test_next_session_positions_follow_the_holding_rule():
    dates = pd.bdate_range("2023-06-01", periods=4)
    panel = pd.DataFrame({
        "ticker": ["AAA"] * 4 + ["BBB"] * 4 + ["CCC"] * 4,
        "date": list(dates) * 3,
        "flag": [0, 1, 0, 0,     # 2 days into a 3-day hold: still held tomorrow
                 0, 0, 0, 0,     # never flagged
                 -1, 0, 0, 0],   # 3-day hold used up: flat tomorrow
    })
    positions = scan.next_session_positions(panel, holding_days=3)
    assert positions["ticker"].tolist() == ["AAA"]
    assert positions["position"].tolist() == ["LONG"]
    assert positions["signal_date"].tolist() == [dates[1].date()]


def history_panel(last_return_z: float) -> pd.DataFrame:
    """60 days of mild, alternating tone for one stock, ending on a chosen return z-score."""
    days = 60
    return pd.DataFrame({
        "ticker": "AAA",
        "date": pd.bdate_range("2023-01-02", periods=days),
        "sentiment": np.tile([0.1, -0.1], days // 2),
        "return_z": [0.0] * (days - 1) + [last_return_z],
    })


def test_good_news_after_a_price_drop_is_provisionally_bullish():
    view = scan.developing_view(pd.DataFrame({"ticker": ["AAA"], "score": [0.9]}),
                                history_panel(last_return_z=-2.0), CONFIG)
    row = view.iloc[0]
    assert row["provisional"] == "BULLISH"
    assert row["tone_z"] > 1.5
    assert row["headlines"] == 1


def test_bad_news_after_a_price_drop_is_not_a_divergence():
    view = scan.developing_view(pd.DataFrame({"ticker": ["AAA"], "score": [-0.9]}),
                                history_panel(last_return_z=-2.0), CONFIG)
    assert view.iloc[0]["provisional"] == "-"


def test_news_for_the_next_session_never_reaches_a_final_flag():
    dates = pd.bdate_range("2023-01-02", periods=3)
    prices = pd.DataFrame({"ticker": "AAA", "date": dates.strftime("%Y-%m-%d"),
                           "open": 100.0, "close": 100.0, "adj_close": 100.0})
    tomorrow = dates[-1] + pd.offsets.BDay(1)
    sentiment = pd.DataFrame({"ticker": ["AAA"], "trading_day": [tomorrow],
                              "sentiment": [0.99], "n_headlines": [1]})
    panel = build_panel(prices, sentiment, SIGNAL)
    assert panel["date"].max() == dates[-1]
    assert panel["sentiment"].isna().all()
