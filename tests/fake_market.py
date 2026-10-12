"""A made-up market used only to exercise code in tests. It is never presented as data."""

import numpy as np
import pandas as pd


def random_market(seed: int, days: int = 300):
    """Two invented tickers: random-walk prices and random daily news tone."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=days)
    prices, sentiment = [], []
    for ticker in ["AAA", "BBB"]:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
        prices.append(pd.DataFrame({"ticker": ticker, "date": dates.strftime("%Y-%m-%d"),
                                    "open": close * (1 + rng.normal(0, 0.005, days)),
                                    "close": close, "adj_close": close * 0.9}))
        has_news = rng.random(days) < 0.7
        sentiment.append(pd.DataFrame({"ticker": ticker, "trading_day": dates[has_news],
                                       "sentiment": rng.uniform(-1, 1, has_news.sum()),
                                       "n_headlines": rng.integers(1, 5, has_news.sum())}))
    return pd.concat(prices, ignore_index=True), pd.concat(sentiment, ignore_index=True)
