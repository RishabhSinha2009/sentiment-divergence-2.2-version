"""Live mode message handling.

Prices and headlines below are invented test inputs, not market data.
"""

import numpy as np
import pandas as pd
import pytest

from divergence import db
from divergence.config import Config
from divergence.errors import DataSourceError
from divergence.live import LiveMonitor
from divergence.signals import SignalConfig
from divergence.trading_calendar import TradingCalendar

SIGNAL = SignalConfig(sentiment_model="fake", return_days=5, zscore_window=60,
                      min_history=20, threshold=1.5, holding_days=5)
CALENDAR = TradingCalendar("2022-12-01", "2023-12-31")
AFTER_LAST_CLOSE = pd.Timestamp("2023-07-01 12:00", tz="UTC")  # Saturday; last close Fri 06-30


class FakeScorer:
    name = "fake"

    def score(self, texts):
        return [{"score": float(t.split()[-1]), "p_positive": None, "p_negative": None, "p_neutral": None}
                for t in texts]


@pytest.fixture
def monitor(tmp_path):
    config = Config(tickers=["AAA"], news_aliases={}, start=None, end=None,
                    database=tmp_path / "t.db", news_page_size=50, signal=SIGNAL)
    conn = db.connect(config.database)
    sessions = CALENDAR.sessions[(CALENDAR.sessions >= "2023-01-03") & (CALENDAR.sessions <= "2023-06-30")]
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(sessions))))
    close[-5:] = close[-6] * np.linspace(0.97, 0.85, 5)  # a sharp drop in the final week
    db.upsert_prices(conn, "AAA", pd.DataFrame({
        "date": sessions.strftime("%Y-%m-%d"), "open": close, "high": close, "low": close,
        "close": close, "adj_close": close, "volume": 1}))
    for i, day in enumerate(sessions):  # one mild headline per day as history
        db.insert_headlines(conn, [{"id": i + 1, "headline": "history",
                                    "published_at_utc": (day + pd.Timedelta(hours=15)).tz_localize("UTC").isoformat(),
                                    "source": "test", "url": None, "n_symbols": 1, "tickers": ["AAA"]}])
        db.insert_scores(conn, "fake", [{"headline_id": i + 1, "score": 0.1 if i % 2 else -0.1,
                                         "p_positive": None, "p_negative": None, "p_neutral": None}])
    return LiveMonitor(conn, CALENDAR, config, [FakeScorer()], now=AFTER_LAST_CLOSE)


def news_message(headline_id: int, text: str, symbols=("AAA",)) -> dict:
    return {"T": "n", "id": headline_id, "headline": text, "created_at": "2023-07-01T11:00:00Z",
            "updated_at": "2023-07-01T11:00:00Z", "symbols": list(symbols), "source": "test", "url": None}


def test_a_divergence_alert_fires_once_and_duplicates_are_ignored(monitor):
    lines = monitor.handle(news_message(9001, "Record quarter 0.9"), now=AFTER_LAST_CLOSE)
    assert any("AAA is now BULLISH" in line for line in lines)

    assert monitor.handle(news_message(9001, "Record quarter 0.9"), now=AFTER_LAST_CLOSE) == []

    lines = monitor.handle(news_message(9002, "Another good one 0.8"), now=AFTER_LAST_CLOSE)
    assert lines and not any(">>>" in line for line in lines)  # status unchanged, no repeat alert


def test_headlines_about_other_companies_are_skipped(monitor):
    assert monitor.handle(news_message(9003, "Unrelated 0.9", symbols=("ZZZ",)), now=AFTER_LAST_CLOSE) == []


def test_stream_errors_stop_the_monitor(monitor):
    with pytest.raises(DataSourceError):
        monitor.handle({"T": "error", "code": 402, "msg": "auth failed"})
