"""Scores the tone of each headline, then rolls the scores up to one number
per stock per trading day.

Two scorers, both giving a score from -1 (very negative) to +1 (very positive):

  FinBERT  A language model trained on financial news. It reads the whole
           sentence, so it knows "beats estimates" is good and "cuts outlook"
           is bad. Score = P(positive) - P(negative).

  VADER    A word list built for social media posts. It knows nothing about
           finance. It is here as a simple baseline, to show whether the
           heavier model is actually worth it.
"""

import pandas as pd

from . import db
from .trading_calendar import TradingCalendar

FINBERT_MODEL = "ProsusAI/finbert"

# Headlines are short. 64 tokens covers nearly all of them and keeps
# scoring fast.
MAX_TOKENS = 64


class FinBertScorer:
    name = "finbert"

    def __init__(self, batch_size: int = 64):
        # torch and transformers take several seconds to import, so they load
        # only when FinBERT is actually used.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).eval()
        # Read the label order from the model instead of assuming it. A swapped
        # positive/negative would silently flip every signal downstream.
        label_index = {label.lower(): i for i, label in self.model.config.id2label.items()}
        self.pos = label_index["positive"]
        self.neg = label_index["negative"]
        self.neu = label_index["neutral"]

    def score(self, texts: list[str]) -> list[dict]:
        # Batching similar-length headlines together means less padding to
        # compute. The padding is masked out either way, so scores are unchanged.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        results: list[dict] = [{}] * len(texts)
        with self._torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                batch = self.tokenizer([texts[i] for i in indices], padding=True,
                                       truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
                probabilities = self._torch.softmax(self.model(**batch).logits, dim=-1).tolist()
                for i, p in zip(indices, probabilities):
                    results[i] = {"score": p[self.pos] - p[self.neg], "p_positive": p[self.pos],
                                  "p_negative": p[self.neg], "p_neutral": p[self.neu]}
        return results


class VaderScorer:
    name = "vader"

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()

    def score(self, texts: list[str]) -> list[dict]:
        return [{"score": self.analyzer.polarity_scores(t)["compound"],
                 "p_positive": None, "p_negative": None, "p_neutral": None} for t in texts]


def score_new_headlines(conn, scorer, chunk_size: int = 2000) -> int:
    """Score every headline that has no score from this scorer yet.

    Saves after each chunk, so an interrupted run keeps its progress.
    """
    todo = db.unscored_headlines(conn, scorer.name)
    for start in range(0, len(todo), chunk_size):
        chunk = todo.iloc[start:start + chunk_size]
        scores = scorer.score(chunk["headline"].tolist())
        db.insert_scores(conn, scorer.name,
                         [{"headline_id": int(i), **s} for i, s in zip(chunk["id"], scores)])
        print(f"  {scorer.name}: {min(start + chunk_size, len(todo))}/{len(todo)} scored", flush=True)
    return len(todo)


def daily_sentiment(conn, calendar: TradingCalendar, model: str,
                    tickers: list[str] | None = None, since=None) -> pd.DataFrame:
    """One row per stock per trading day that had news: mean score and headline count.

    Days without any headline are left out, not filled with zero. "No news"
    is not the same as "neutral news". Headlines published before `since`
    (the start of the study window) are ignored.
    """
    scored = db.load_scored_headlines(conn, model, tickers, published_after=study_start(since))
    if scored.empty:
        return pd.DataFrame(columns=["ticker", "trading_day", "sentiment", "n_headlines"])
    # The timing rule from Stage 1 decides which day each headline counts toward.
    scored["trading_day"] = calendar.assign_trading_day(scored["published_at_utc"])
    return (scored.groupby(["ticker", "trading_day"])["score"]
            .agg(sentiment="mean", n_headlines="count")
            .reset_index())


def study_start(since) -> pd.Timestamp | None:
    """Midnight UTC on the first day of the study window."""
    return pd.Timestamp(since, tz="UTC") if since is not None else None


def random_sample(conn, n: int = 20, seed: int | None = None) -> pd.DataFrame:
    """Random headlines with both models' scores side by side, for eyeballing."""
    both = pd.read_sql_query(
        "SELECT h.id, h.published_at_utc, h.headline, f.score AS finbert, v.score AS vader "
        "FROM headlines h "
        "JOIN headline_scores f ON f.headline_id = h.id AND f.model = 'finbert' "
        "JOIN headline_scores v ON v.headline_id = h.id AND v.model = 'vader'",
        conn,
    )
    tickers = pd.read_sql_query(
        "SELECT headline_id AS id, GROUP_CONCAT(ticker, ',') AS tickers FROM headline_tickers GROUP BY headline_id",
        conn,
    )
    sample = both.sample(n=min(n, len(both)), random_state=seed)
    return sample.merge(tickers, on="id")[["tickers", "published_at_utc", "finbert", "vader", "headline"]]
