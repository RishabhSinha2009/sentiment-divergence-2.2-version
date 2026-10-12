"""Sentiment scoring and the daily roll-up.

The headlines here are invented test sentences, not real news.
"""

import pandas as pd
import pytest

from divergence import db, sentiment
from divergence.trading_calendar import TradingCalendar


class FakeScorer:
    """Reads the score from the last word of the text, and records what it was asked to score."""

    name = "fake"

    def __init__(self):
        self.seen = []

    def score(self, texts):
        self.seen.extend(texts)
        return [{"score": float(t.split()[-1]), "p_positive": None, "p_negative": None, "p_neutral": None}
                for t in texts]


def add_headline(conn, headline_id, published, text, tickers=("AAPL",)):
    db.insert_headlines(conn, [{
        "id": headline_id,
        "published_at_utc": pd.Timestamp(published).tz_convert("UTC").isoformat(),
        "headline": text, "source": "test", "url": None,
        "n_symbols": len(tickers), "tickers": list(tickers),
    }])


def test_each_headline_is_scored_only_once():
    conn = db.connect(":memory:")
    scorer = FakeScorer()
    add_headline(conn, 1, "2023-06-13 10:00-04:00", "first 0.5")
    assert sentiment.score_new_headlines(conn, scorer) == 1
    add_headline(conn, 2, "2023-06-13 11:00-04:00", "second -0.5")
    assert sentiment.score_new_headlines(conn, scorer) == 1
    assert scorer.seen == ["first 0.5", "second -0.5"]


def test_daily_sentiment_follows_the_timing_rule():
    conn = db.connect(":memory:")
    add_headline(conn, 1, "2023-06-13 15:59-04:00", "just before the bell 1.0")
    add_headline(conn, 2, "2023-06-13 16:00-04:00", "at the bell -1.0")
    add_headline(conn, 3, "2023-06-14 09:00-04:00", "next morning 0.0")
    sentiment.score_new_headlines(conn, FakeScorer())

    daily = sentiment.daily_sentiment(conn, TradingCalendar("2023-05-01", "2023-07-31"), "fake")
    daily = daily.set_index("trading_day")

    assert daily.loc["2023-06-13", "sentiment"] == 1.0      # only the 3:59 pm story
    assert daily.loc["2023-06-13", "n_headlines"] == 1
    assert daily.loc["2023-06-14", "sentiment"] == -0.5     # the 4:00 pm story moved here
    assert daily.loc["2023-06-14", "n_headlines"] == 2
    assert pd.Timestamp("2023-06-15") not in daily.index    # no news is missing, not zero


def test_a_story_about_two_companies_counts_for_both():
    conn = db.connect(":memory:")
    add_headline(conn, 1, "2023-06-13 10:00-04:00", "shared story 0.8", tickers=("AAPL", "MSFT"))
    sentiment.score_new_headlines(conn, FakeScorer())
    daily = sentiment.daily_sentiment(conn, TradingCalendar("2023-05-01", "2023-07-31"), "fake")
    assert sorted(daily["ticker"]) == ["AAPL", "MSFT"]


def test_vader_knows_good_from_bad():
    good, bad = sentiment.VaderScorer().score([
        "Great quarter with strong growth and happy investors",
        "Terrible quarter with huge losses and angry investors",
    ])
    assert good["score"] > 0 > bad["score"]


def finbert_downloaded() -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(sentiment.FINBERT_MODEL, "config.json"), str)


@pytest.mark.skipif(not finbert_downloaded(), reason="FinBERT not downloaded yet; run `python -m divergence score`.")
def test_finbert_points_the_right_way():
    good, bad, flat = sentiment.FinBertScorer().score([
        "Company beats earnings estimates and raises full-year guidance",
        "Company misses revenue estimates and cuts outlook",
        "Company to present at investor conference on Tuesday",
    ])
    assert good["score"] > 0.5
    assert bad["score"] < -0.5
    assert abs(flat["score"]) < 0.3
    for s in (good, bad, flat):
        assert s["p_positive"] + s["p_negative"] + s["p_neutral"] == pytest.approx(1.0, abs=1e-5)
