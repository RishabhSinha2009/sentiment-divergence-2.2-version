"""Price caching: unfinished days are never stored and nothing is downloaded twice.

The price tables below are small hand-written test inputs, not market data.
"""

from datetime import date

import pandas as pd
import pytest

from divergence import db, prices
from divergence.trading_calendar import TradingCalendar


def yfinance_frame(days: list[str], adj_close: list[float]) -> pd.DataFrame:
    """A frame shaped like yfinance's history() output."""
    index = pd.DatetimeIndex(pd.to_datetime(days)).tz_localize("America/New_York")
    return pd.DataFrame(
        {"Open": adj_close, "High": adj_close, "Low": adj_close, "Close": adj_close,
         "Adj Close": adj_close, "Volume": [1000] * len(days)},
        index=index,
    )


def price_table(days: list[str], adj_close: list[float]) -> pd.DataFrame:
    return prices.to_price_table(yfinance_frame(days, adj_close), date(2100, 1, 1))


def test_bar_for_a_session_still_trading_is_dropped():
    raw = yfinance_frame(["2023-06-12", "2023-06-13"], [100.0, 101.0])
    table = prices.to_price_table(raw, last_session=date(2023, 6, 12))
    assert table["date"].tolist() == ["2023-06-12"]


@pytest.mark.parametrize(
    "now, expected",
    [
        ("2023-06-13 14:00-04:00", "2023-06-12"),  # mid-session: today is not finished
        ("2023-06-13 16:05-04:00", "2023-06-13"),  # after the bell
        ("2023-06-17 12:00-04:00", "2023-06-16"),  # Saturday -> Friday
        ("2023-11-24 13:05-05:00", "2023-11-24"),  # after a 1 pm early close
    ],
)
def test_last_completed_session(now, expected):
    calendar = TradingCalendar("2023-01-01", "2023-12-31")
    assert calendar.last_completed_session(pd.Timestamp(now)).strftime("%Y-%m-%d") == expected


def test_adjustment_check():
    stored = price_table(["2023-06-12", "2023-06-13"], [100.0, 101.0])
    same = price_table(["2023-06-12", "2023-06-13", "2023-06-14"], [100.0, 101.0, 102.0])
    rounding_noise = price_table(["2023-06-12", "2023-06-13"], [100.0000001, 101.0])
    after_dividend = price_table(["2023-06-12", "2023-06-13"], [99.5, 100.495])  # old prices scaled down
    no_overlap = price_table(["2023-07-03"], [105.0])

    assert not prices.adjustments_changed(stored, same)
    assert not prices.adjustments_changed(stored, rounding_noise)
    assert prices.adjustments_changed(stored, after_dividend)
    assert prices.adjustments_changed(stored, no_overlap)


class FakeYahoo:
    """Stands in for download_history and records every request made."""

    def __init__(self, history: pd.DataFrame):
        self.history = history
        self.calls = []

    def __call__(self, ticker, start, last_session):
        self.calls.append((ticker, start, last_session))
        rows = self.history[(self.history["date"] >= start.isoformat())
                            & (self.history["date"] <= last_session.isoformat())]
        return rows.reset_index(drop=True)


def test_update_downloads_only_what_is_missing(monkeypatch):
    conn = db.connect(":memory:")
    yahoo = FakeYahoo(price_table(["2023-06-12", "2023-06-13", "2023-06-14"], [100.0, 101.0, 102.0]))
    monkeypatch.setattr(prices, "download_history", yahoo)

    # First run: full download.
    prices.update_prices(conn, ["TEST"], date(2023, 6, 1), date(2023, 6, 13))
    assert len(yahoo.calls) == 1
    assert db.last_price_date(conn, "TEST") == "2023-06-13"

    # Same day again: nothing to fetch.
    prices.update_prices(conn, ["TEST"], date(2023, 6, 1), date(2023, 6, 13))
    assert len(yahoo.calls) == 1

    # One new day: one short overlap request, and only the new row is added.
    prices.update_prices(conn, ["TEST"], date(2023, 6, 1), date(2023, 6, 14))
    assert len(yahoo.calls) == 2
    assert yahoo.calls[1][1] > date(2023, 6, 1)  # started near the end, not from the beginning
    assert db.load_prices(conn, ["TEST"])["date"].tolist() == ["2023-06-12", "2023-06-13", "2023-06-14"]


def test_update_redownloads_in_full_after_a_readjustment(monkeypatch):
    conn = db.connect(":memory:")
    monkeypatch.setattr(prices, "download_history",
                        FakeYahoo(price_table(["2023-06-12", "2023-06-13"], [100.0, 101.0])))
    prices.update_prices(conn, ["TEST"], date(2023, 6, 1), date(2023, 6, 13))

    # A dividend on 06-14 makes the provider scale every earlier price down.
    readjusted = FakeYahoo(price_table(["2023-06-12", "2023-06-13", "2023-06-14"], [99.5, 100.495, 102.0]))
    monkeypatch.setattr(prices, "download_history", readjusted)
    report = prices.update_prices(conn, ["TEST"], date(2023, 6, 1), date(2023, 6, 14))

    assert "re-downloaded" in report[0]
    assert readjusted.calls[-1][1] == date(2023, 6, 1)  # the full history was fetched again
    assert db.load_prices(conn, ["TEST"])["adj_close"].tolist() == [99.5, 100.495, 102.0]
