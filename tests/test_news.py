"""Parsing Alpaca articles and caching news by month.

The articles below follow the shape in Alpaca's API documentation; their
contents are invented test inputs, not real news.
"""

from datetime import date

import pandas as pd
import pytest

from divergence import db, news
from divergence.config import Config
from divergence.errors import DataSourceError

SYMBOL_TO_TICKER = {"AAPL": "AAPL", "GOOGL": "GOOGL", "GOOG": "GOOGL"}


def article(**overrides) -> dict:
    base = {
        "id": 101,
        "headline": "Test headline",
        "created_at": "2023-06-13T19:59:00Z",
        "updated_at": "2023-06-14T09:00:00Z",
        "symbols": ["AAPL"],
        "source": "benzinga",
        "url": "https://example.com/a",
    }
    return {**base, **overrides}


def test_publication_time_is_created_at_not_updated_at():
    parsed = news.parse_article(article(), SYMBOL_TO_TICKER)
    assert pd.Timestamp(parsed["published_at_utc"]) == pd.Timestamp("2023-06-13T19:59:00Z")


def test_aliases_map_to_our_ticker_and_outside_symbols_are_dropped():
    parsed = news.parse_article(article(symbols=["GOOG", "AAPL", "XYZ"]), SYMBOL_TO_TICKER)
    assert parsed["tickers"] == ["AAPL", "GOOGL"]
    assert parsed["n_symbols"] == 3  # still records how broad the story was


@pytest.mark.parametrize("bad", [article(headline="  "), article(symbols=["XYZ"])])
def test_unusable_articles_are_skipped(bad):
    assert news.parse_article(bad, SYMBOL_TO_TICKER) is None


def test_timestamp_without_timezone_stops_the_pipeline():
    with pytest.raises(DataSourceError):
        news.parse_article(article(created_at="2023-06-13T19:59:00"), SYMBOL_TO_TICKER)


def test_months_between():
    assert news.months_between(date(2015, 1, 1), date(2015, 3, 15)) == ["2015-01", "2015-02", "2015-03"]


def test_only_finished_months_count_as_complete():
    now = pd.Timestamp("2026-10-11 12:00", tz="UTC")
    assert news.month_is_complete("2026-09", now)
    assert not news.month_is_complete("2026-10", now)


def test_finished_months_are_never_downloaded_twice(monkeypatch, tmp_path):
    config = Config(tickers=["AAPL"], news_aliases={}, start=date(2026, 8, 1), end=None,
                    database=tmp_path / "t.db", news_page_size=50)
    conn = db.connect(config.database)
    requested = []

    def fake_fetch(session, symbols, month, page_size, now_utc):
        requested.append(month)
        return [article(id=int(month.replace("-", "")), created_at=f"{month}-15T15:00:00Z")]

    monkeypatch.setattr(news, "fetch_month", fake_fetch)
    monkeypatch.setattr(news, "load_credentials", lambda: {})
    now = pd.Timestamp("2026-10-11 12:00", tz="UTC")

    news.update_news(conn, config, now)
    assert requested == ["2026-08", "2026-09", "2026-10"]

    requested.clear()
    news.update_news(conn, config, now)
    assert requested == ["2026-10"]  # only the month still in progress is checked again
    assert conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0] == 3
