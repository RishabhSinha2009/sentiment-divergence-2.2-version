"""Daily prices from Yahoo Finance (via yfinance), cached in SQLite.

Yahoo's adjusted close is rewritten backwards in time whenever a company pays
a dividend or splits its stock. If we simply appended new days to an old
download, the series would mix two adjustment bases and show fake jumps. So
each update re-downloads a short overlap with what is already stored. If the
overlap still matches, only the new days are added. If not, that one ticker
is re-downloaded in full.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from . import db
from .errors import DataSourceError

OVERLAP_DAYS = 10

# Dividend adjustments move old prices by roughly 0.1% or more, so this catches
# every real re-adjustment while ignoring rounding noise between downloads.
ADJUSTMENT_TOLERANCE = 1e-5


def download_history(ticker: str, start: date, last_session: date) -> pd.DataFrame:
    """Daily bars from start through last_session, inclusive."""
    try:
        raw = yf.Ticker(ticker).history(
            start=start,
            end=last_session + timedelta(days=1),  # yfinance's end date is exclusive
            auto_adjust=False,
            actions=False,
            raise_errors=True,
        )
    except Exception as exc:
        raise DataSourceError(f"Yahoo Finance download failed for {ticker}: {exc}") from exc
    if raw.empty:
        raise DataSourceError(f"Yahoo Finance returned no rows for {ticker} between {start} and {last_session}.")
    return to_price_table(raw, last_session)


def to_price_table(raw: pd.DataFrame, last_session: date) -> pd.DataFrame:
    """Convert yfinance's output into the database's column layout."""
    prices = pd.DataFrame(
        {
            "date": raw.index.strftime("%Y-%m-%d"),
            "open": raw["Open"].to_numpy(),
            "high": raw["High"].to_numpy(),
            "low": raw["Low"].to_numpy(),
            "close": raw["Close"].to_numpy(),
            "adj_close": raw["Adj Close"].to_numpy(),
            "volume": raw["Volume"].to_numpy(),
        }
    )
    # A bar for a session that has not closed yet is not a closing price, just
    # the latest trade. Storing it would plant a fake "close" in history.
    return prices[prices["date"] <= last_session.isoformat()].reset_index(drop=True)


def adjustments_changed(stored: pd.DataFrame, fresh: pd.DataFrame) -> bool:
    """True if the provider has re-adjusted prices we already stored."""
    overlap = stored.merge(fresh, on="date", suffixes=("_stored", "_fresh"))
    if overlap.empty:
        # Nothing to compare means we can't confirm the two downloads line up.
        return True
    return not np.allclose(
        overlap["adj_close_stored"], overlap["adj_close_fresh"], rtol=ADJUSTMENT_TOLERANCE, atol=0
    )


def update_prices(conn, tickers: list[str], start: date, last_session: date) -> list[str]:
    """Bring every ticker up to last_session. Returns one status line per ticker."""
    report = []
    for ticker in tickers:
        stored_through = db.last_price_date(conn, ticker)

        if stored_through is None:
            prices = download_history(ticker, start, last_session)
            db.replace_prices(conn, ticker, prices)
            report.append(f"{ticker}: downloaded {len(prices)} days")
            continue

        if stored_through >= last_session.isoformat():
            report.append(f"{ticker}: already up to date through {stored_through}")
            continue

        overlap_start = date.fromisoformat(stored_through) - timedelta(days=OVERLAP_DAYS)
        fresh = download_history(ticker, overlap_start, last_session)
        if adjustments_changed(db.load_prices(conn, [ticker]), fresh):
            prices = download_history(ticker, start, last_session)
            db.replace_prices(conn, ticker, prices)
            report.append(f"{ticker}: provider re-adjusted old prices; re-downloaded {len(prices)} days")
        else:
            new_rows = fresh[fresh["date"] > stored_through]
            db.upsert_prices(conn, ticker, new_rows)
            report.append(f"{ticker}: added {len(new_rows)} new days")
    return report
