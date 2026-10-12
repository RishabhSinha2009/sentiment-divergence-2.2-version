"""Daily scan: which stocks are diverging right now.

It calls exactly the same functions as the backtest, so what the scan flags
is what the backtest tested. It shows three things:

  Final flags at the last close   Computed only from data up to that close.
                                  A position from one of these is opened at
                                  the next open, just like in the backtest.
  Positions held from next open   What the rule says to hold at the next
                                  open, including trades still inside their
                                  holding period.
  News since the last close       Headlines published after the close count
                                  toward the next session. Their tone so far
                                  is shown with a provisional z-score, which
                                  can change until the next close.
"""

from dataclasses import dataclass

import pandas as pd

from . import db, news, prices, sentiment
from .config import Config
from .signals import build_panel, divergence_flags, past_only_zscore, positions_from_flags
from .trading_calendar import TradingCalendar


@dataclass
class ScanResult:
    last_session: pd.Timestamp
    next_session: pd.Timestamp
    panel: pd.DataFrame          # full history, completed sessions only
    flags: pd.DataFrame          # final flags at the last close
    positions: pd.DataFrame      # what the rule holds from the next open
    developing: pd.DataFrame     # tone of news since the last close


def refresh(conn, calendar: TradingCalendar, config: Config, now: pd.Timestamp, scorers) -> None:
    """Bring prices, headlines and scores up to date. Only what is missing is fetched."""
    last_session = calendar.last_completed_session(now).date()
    prices.update_prices(conn, config.tickers, config.start, last_session)
    news.update_news(conn, config, now, verbose=False)
    for scorer in scorers:
        sentiment.score_new_headlines(conn, scorer)


def next_session_positions(panel: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    """The position each stock holds from the next open, by the backtest's own rule."""
    rows = []
    for ticker, history in panel.groupby("ticker"):
        # Adding one empty day and asking positions_from_flags for it reuses
        # the backtest's exact holding logic instead of re-implementing it.
        flags = pd.concat([history["flag"], pd.Series([0])], ignore_index=True)
        position = int(positions_from_flags(flags, holding_days).iloc[-1])
        if position != 0:
            flagged = history[history["flag"] != 0]
            rows.append({"ticker": ticker, "position": "LONG" if position > 0 else "SHORT",
                         "signal_date": flagged["date"].iloc[-1].date()})
    return pd.DataFrame(rows, columns=["ticker", "position", "signal_date"])


def developing_view(fresh: pd.DataFrame, panel: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Provisional divergence status from headlines that arrived after the last close.

    fresh: columns ticker, score (one row per headline since the last close)
    """
    signal = config.signal
    rows = []
    for ticker, scores in fresh.groupby("ticker"):
        history = panel[panel["ticker"] == ticker]
        if history.empty:
            continue
        tone_so_far = scores["score"].mean()
        # Put today's tone after the stock's history and z-score it with the
        # same function the backtest uses, so the yardstick is identical.
        tone = pd.concat([history["sentiment"], pd.Series([tone_so_far])], ignore_index=True)
        tone_z = past_only_zscore(tone, signal.zscore_window, signal.min_history).iloc[-1]
        return_z = history["return_z"].iloc[-1]  # price has not moved since the last close we know
        flag = divergence_flags(pd.Series([tone_z]), pd.Series([return_z]), signal.threshold).iloc[0]
        rows.append({"ticker": ticker, "headlines": len(scores), "tone_so_far": tone_so_far,
                     "tone_z": tone_z, "return_z": return_z,
                     "provisional": {1: "BULLISH", -1: "BEARISH"}.get(int(flag), "-")})
    columns = ["ticker", "headlines", "tone_so_far", "tone_z", "return_z", "provisional"]
    return pd.DataFrame(rows, columns=columns)


def fresh_scores(conn, calendar: TradingCalendar, config: Config, last_session: pd.Timestamp) -> pd.DataFrame:
    """Scored headlines that count toward the session after `last_session`."""
    last_close = calendar.close_of([last_session]).iloc[0]
    scored = db.load_scored_headlines(conn, config.signal.sentiment_model, config.tickers,
                                      published_after=last_close)
    if scored.empty:
        return scored
    scored["trading_day"] = calendar.assign_trading_day(scored["published_at_utc"])
    return scored[scored["trading_day"] > last_session]


def recent_headlines(conn, calendar: TradingCalendar, config: Config, last_session: pd.Timestamp,
                     sessions: int = 3) -> pd.DataFrame:
    """Scored headlines from roughly the last few trading days, newest included."""
    earlier = calendar.sessions[calendar.sessions <= last_session][-sessions - 1]
    since = calendar.close_of([earlier]).iloc[0]
    return db.load_scored_headlines(conn, config.signal.sentiment_model, config.tickers, published_after=since)


def build_panel_from_database(conn, calendar: TradingCalendar, config: Config) -> pd.DataFrame:
    daily = sentiment.daily_sentiment(conn, calendar, config.signal.sentiment_model, config.tickers, config.start)
    # Prices exist only for completed sessions, so news that counts toward the
    # next session drops out here and can never leak into a final flag.
    return build_panel(db.load_prices(conn, config.tickers), daily, config.signal)


def scan(conn, calendar: TradingCalendar, config: Config) -> ScanResult:
    panel = build_panel_from_database(conn, calendar, config)
    last_session = panel["date"].max()
    next_session = calendar.sessions[calendar.sessions > last_session][0]
    latest = panel[panel["date"] == last_session]
    flags = latest.loc[latest["flag"] != 0,
                       ["ticker", "flag", "sentiment", "sentiment_z", "return_recent", "return_z", "n_headlines"]]
    flags = flags.assign(flag=flags["flag"].map({1: "BULLISH", -1: "BEARISH"}))
    developing = developing_view(fresh_scores(conn, calendar, config, last_session), panel, config)
    return ScanResult(last_session, next_session, panel, flags,
                      next_session_positions(panel, config.signal.holding_days), developing)


def print_scan(result: ScanResult) -> None:
    def show(frame: pd.DataFrame, empty: str) -> None:
        print(frame.round(3).to_string(index=False) if not frame.empty else f"  {empty}")

    print(f"Last close: {result.last_session:%a %Y-%m-%d}    Next session: {result.next_session:%a %Y-%m-%d}\n")
    print("DIVERGENCE FLAGS AT THE LAST CLOSE (final)")
    show(result.flags, "none today")
    print("\nPOSITIONS THE RULE HOLDS FROM THE NEXT OPEN")
    show(result.positions, "none")
    print("\nNEWS SINCE THE LAST CLOSE (provisional until the next close)")
    show(result.developing, "no new headlines yet")
    print("\nResearch output, not financial advice.")
