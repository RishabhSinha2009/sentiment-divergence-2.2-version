"""Database round-trips. Rows here are hand-written test inputs, not market data."""

import pandas as pd

from divergence import db


def prices(days, closes):
    return pd.DataFrame({"date": days, "open": closes, "high": closes, "low": closes,
                         "close": closes, "adj_close": closes, "volume": [1] * len(days)})


def test_writing_the_same_prices_twice_does_not_duplicate():
    conn = db.connect(":memory:")
    p = prices(["2023-06-12", "2023-06-13"], [1.0, 2.0])
    db.upsert_prices(conn, "TEST", p)
    db.upsert_prices(conn, "TEST", p)
    assert len(db.load_prices(conn, ["TEST"])) == 2


def test_replace_prices_drops_the_old_history():
    conn = db.connect(":memory:")
    db.upsert_prices(conn, "TEST", prices(["2023-06-12", "2023-06-13"], [1.0, 2.0]))
    db.replace_prices(conn, "TEST", prices(["2023-06-13"], [3.0]))
    assert db.load_prices(conn, ["TEST"])["adj_close"].tolist() == [3.0]


def test_headlines_are_stored_once_and_linked_to_each_ticker():
    conn = db.connect(":memory:")
    story = {"id": 7, "published_at_utc": "2023-06-13T19:59:00+00:00", "headline": "h",
             "source": "s", "url": None, "n_symbols": 2, "tickers": ["AAPL", "MSFT"]}
    assert db.insert_headlines(conn, [story]) == 1
    assert db.insert_headlines(conn, [story]) == 0

    loaded = db.load_headlines(conn)
    assert sorted(loaded["ticker"]) == ["AAPL", "MSFT"]
    assert str(loaded["published_at_utc"].dt.tz) == "UTC"
