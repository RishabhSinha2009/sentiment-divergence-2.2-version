"""Turns daily sentiment and recent returns into divergence flags and positions.

Each stock is measured only against its own past:

  sentiment z-score  how unusual today's news tone is, compared with this
                     stock's previous `zscore_window` trading days
  return z-score     how unusual the last `return_days` days' return is,
                     compared with the same stretch of history

A z-score of +2 means "two standard deviations above normal for this stock".
Comparing each stock with itself matters because some companies always get
cheerful coverage and others always get gloomy coverage.

  bullish divergence  news strongly positive, price strongly lagging  -> long
  bearish divergence  news strongly negative, price strongly rising   -> short
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    sentiment_model: str
    return_days: int
    zscore_window: int
    min_history: int
    threshold: float
    holding_days: int


def past_only_zscore(values: pd.Series, window: int, min_history: int) -> pd.Series:
    """How far today's value is from normal, where "normal" uses only earlier days.

    shift(1) is what keeps this honest: the average and spread come from
    yesterday and before, so today's value never shapes the yardstick it is
    measured against, and nothing later than today is ever touched.
    """
    history = values.shift(1).rolling(window, min_periods=min_history)
    z = (values - history.mean()) / history.std()
    # A perfectly flat history has zero spread; treat that as "no reading".
    return z.replace([np.inf, -np.inf], np.nan)


def divergence_flags(sentiment_z: pd.Series, return_z: pd.Series, threshold: float) -> pd.Series:
    """+1 bullish divergence, -1 bearish divergence, 0 otherwise (including missing data)."""
    bullish = (sentiment_z >= threshold) & (return_z <= -threshold)
    bearish = (sentiment_z <= -threshold) & (return_z >= threshold)
    return bullish.astype(int) - bearish.astype(int)


def price_only_flags(return_z: pd.Series, threshold: float) -> pd.Series:
    """The same rule with the news removed: buy big recent losers, short big recent winners.

    If this does as well as the news version, the news is adding nothing.
    """
    return (return_z <= -threshold).astype(int) - (return_z >= threshold).astype(int)


def positions_from_flags(flags: pd.Series, holding_days: int) -> pd.Series:
    """The position held each day, from that day's open to the next day's open.

    A flag raised at day t's close is acted on at day t+1's open and held for
    `holding_days` days. A day's position is recorded before that day's own
    flag is read, so a flag can never affect the day it appears on. If a new
    flag arrives while a position is open, the newest flag wins and the
    holding clock restarts.
    """
    positions = []
    held, days_left = 0, 0
    for flag in flags.to_numpy():
        positions.append(held if days_left > 0 else 0)
        days_left = max(days_left - 1, 0)
        if flag != 0:
            held, days_left = int(flag), holding_days
    return pd.Series(positions, index=flags.index, dtype=int)


def build_panel(prices: pd.DataFrame, sentiment: pd.DataFrame, config: SignalConfig) -> pd.DataFrame:
    """One row per stock per trading day with z-scores, flags and positions.

    prices:    columns ticker, date, open, close, adj_close (one row per trading day)
    sentiment: columns ticker, trading_day, sentiment, n_headlines (days with news only)
    """
    panel = prices[["ticker", "date", "open", "close", "adj_close"]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    # Yahoo adjusts only the close for dividends and splits. Scaling the open by
    # the same factor puts both prices in the same units.
    panel["adj_open"] = panel["open"] * panel["adj_close"] / panel["close"]

    daily = sentiment.rename(columns={"trading_day": "date"})[["ticker", "date", "sentiment", "n_headlines"]]
    daily = daily.assign(date=pd.to_datetime(daily["date"]))
    panel = panel.merge(daily, on=["ticker", "date"], how="left")
    panel["n_headlines"] = panel["n_headlines"].fillna(0).astype(int)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    by_ticker = panel.groupby("ticker", group_keys=False)
    panel["return_recent"] = by_ticker["adj_close"].pct_change(config.return_days, fill_method=None)
    panel["return_z"] = by_ticker["return_recent"].transform(
        lambda s: past_only_zscore(s, config.zscore_window, config.min_history))
    panel["sentiment_z"] = by_ticker["sentiment"].transform(
        lambda s: past_only_zscore(s, config.zscore_window, config.min_history))

    panel["flag"] = divergence_flags(panel["sentiment_z"], panel["return_z"], config.threshold)
    panel["flag_price_only"] = price_only_flags(panel["return_z"], config.threshold)
    panel["position"] = by_ticker["flag"].transform(lambda f: positions_from_flags(f, config.holding_days))
    panel["position_price_only"] = by_ticker["flag_price_only"].transform(
        lambda f: positions_from_flags(f, config.holding_days))
    return panel
