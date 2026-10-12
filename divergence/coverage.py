"""Reports what is in the database and flags data problems.

Gaps and glitches have to be seen before any result depends on them.
"""

import pandas as pd

from . import db
from .sentiment import study_start
from .trading_calendar import TradingCalendar

# A one-day move this large in a mega-cap stock is rare enough to deserve a
# human look: it could be real news or a data glitch such as a spin-off the
# provider did not adjust for.
BIG_MOVE = 0.20


def price_coverage(conn, calendar: TradingCalendar, tickers: list[str]) -> pd.DataFrame:
    prices = db.load_prices(conn, tickers)
    rows = []
    for ticker in tickers:
        p = prices[prices["ticker"] == ticker]
        if p.empty:
            rows.append({"ticker": ticker, "days": 0})
            continue
        dates = pd.DatetimeIndex(pd.to_datetime(p["date"]))
        expected = calendar.sessions[(calendar.sessions >= dates.min()) & (calendar.sessions <= dates.max())]
        returns = p["adj_close"].pct_change()
        rows.append(
            {
                "ticker": ticker,
                "first": p["date"].iloc[0],
                "last": p["date"].iloc[-1],
                "days": len(p),
                "missing_sessions": len(expected.difference(dates)),
                "non_session_days": len(dates.difference(calendar.sessions)),
                "null_prices": int(p["adj_close"].isna().sum()),
                "moves_over_20pct": int((returns.abs() > BIG_MOVE).sum()),
            }
        )
    return pd.DataFrame(rows)


def big_moves(conn, tickers: list[str]) -> pd.DataFrame:
    prices = db.load_prices(conn, tickers)
    prices["return"] = prices.groupby("ticker")["adj_close"].pct_change()
    return prices.loc[prices["return"].abs() > BIG_MOVE, ["ticker", "date", "adj_close", "return"]]


def headline_coverage(conn, calendar: TradingCalendar, tickers: list[str], since=None) -> pd.DataFrame:
    """Headline counts per ticker per year, by the trading day each headline counts toward."""
    headlines = db.load_headlines(conn, tickers, published_after=study_start(since))
    if headlines.empty:
        return pd.DataFrame()
    headlines["year"] = calendar.assign_trading_day(headlines["published_at_utc"]).dt.year
    return headlines.pivot_table(index="ticker", columns="year", values="id", aggfunc="count", fill_value=0)


def print_report(conn, calendar: TradingCalendar, tickers: list[str], since=None) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 200):
        print("PRICES")
        print(price_coverage(conn, calendar, tickers).to_string(index=False))
        moves = big_moves(conn, tickers)
        print(f"\nOne-day moves over {BIG_MOVE:.0%} (check each one is real):")
        print(moves.to_string(index=False) if not moves.empty else "  none")

        print("\nHEADLINES per ticker per year")
        counts = headline_coverage(conn, calendar, tickers, since)
        print(counts.to_string() if not counts.empty else "  none downloaded yet")
