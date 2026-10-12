"""SQLite storage.

One file holds prices, headlines, and a log of which months of news have
already been downloaded, so re-running the pipeline only fetches what is
missing.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker     TEXT NOT NULL,
    date       TEXT NOT NULL,     -- trading session, YYYY-MM-DD (New York)
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,              -- adjusted for splits only
    adj_close  REAL,              -- adjusted for splits and dividends; use for returns
    volume     INTEGER,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS headlines (
    id               INTEGER PRIMARY KEY,   -- the provider's article id
    published_at_utc TEXT NOT NULL,         -- first publication time, ISO-8601 UTC
    headline         TEXT NOT NULL,
    source           TEXT,
    url              TEXT,
    n_symbols        INTEGER                -- tickers the provider tagged, ours or not
);

-- An article can be about several of our stocks, so the link is its own table.
CREATE TABLE IF NOT EXISTS headline_tickers (
    headline_id INTEGER NOT NULL REFERENCES headlines(id),
    ticker      TEXT NOT NULL,
    PRIMARY KEY (headline_id, ticker)
);

CREATE TABLE IF NOT EXISTS news_fetch_log (
    ticker      TEXT NOT NULL,
    month       TEXT NOT NULL,        -- YYYY-MM
    n_articles  INTEGER NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (ticker, month)
);

-- Scoring a quarter-million headlines takes about an hour, so each score is
-- saved once and never recomputed.
CREATE TABLE IF NOT EXISTS headline_scores (
    headline_id INTEGER NOT NULL REFERENCES headlines(id),
    model       TEXT NOT NULL,        -- 'finbert' or 'vader'
    score       REAL NOT NULL,        -- -1 (negative) to +1 (positive)
    p_positive  REAL,                 -- FinBERT's probabilities; empty for VADER
    p_negative  REAL,
    p_neutral   REAL,
    PRIMARY KEY (headline_id, model)
);

-- Every look at the out-of-sample period is recorded, so the report can
-- show honestly how many times the test data was used.
CREATE TABLE IF NOT EXISTS out_of_sample_runs (
    id          INTEGER PRIMARY KEY,
    ran_at      TEXT NOT NULL,
    parameters  TEXT NOT NULL,        -- JSON of every setting that shapes the result
    reason      TEXT                  -- required when parameters changed since the last run
);

-- Paper trading (fake money) records, for the forward-test track record.
CREATE TABLE IF NOT EXISTS paper_orders (
    id               INTEGER PRIMARY KEY,
    submitted_at     TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    side             TEXT NOT NULL,
    qty              INTEGER NOT NULL,
    reason           TEXT,
    alpaca_order_id  TEXT,
    status           TEXT
);

CREATE TABLE IF NOT EXISTS paper_equity (
    date    TEXT PRIMARY KEY,   -- one snapshot per day; later ones replace earlier
    equity  REAL NOT NULL
);
"""

PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def connect(path: Path | str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # A generous wait lets two commands share the database (e.g. scoring
    # while news is still downloading) instead of failing on a brief lock.
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


# --- prices -----------------------------------------------------------------

def upsert_prices(conn: sqlite3.Connection, ticker: str, prices: pd.DataFrame) -> None:
    rows = [(ticker, *row) for row in prices[PRICE_COLUMNS].itertuples(index=False)]
    conn.executemany(
        "INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, adj_close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def replace_prices(conn: sqlite3.Connection, ticker: str, prices: pd.DataFrame) -> None:
    """Swap out a ticker's whole history in one transaction.

    Used when the provider has re-adjusted old prices: mixing old and new
    adjustments in one series would create fake jumps in returns.
    """
    with conn:
        conn.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
    upsert_prices(conn, ticker, prices)


def last_price_date(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM prices WHERE ticker = ?", (ticker,)).fetchone()
    return row[0]


def load_prices(conn: sqlite3.Connection, tickers: list[str] | None = None) -> pd.DataFrame:
    query = "SELECT ticker, " + ", ".join(PRICE_COLUMNS) + " FROM prices"
    params: list[str] = []
    if tickers:
        query += " WHERE ticker IN (" + ",".join("?" * len(tickers)) + ")"
        params = list(tickers)
    query += " ORDER BY ticker, date"
    return pd.read_sql_query(query, conn, params=params)


# --- headlines ----------------------------------------------------------------

def insert_headlines(conn: sqlite3.Connection, articles: list[dict]) -> int:
    """Store parsed articles; ones already stored are left untouched.

    Returns how many were new.
    """
    before = conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
    conn.executemany(
        "INSERT OR IGNORE INTO headlines (id, published_at_utc, headline, source, url, n_symbols) "
        "VALUES (:id, :published_at_utc, :headline, :source, :url, :n_symbols)",
        articles,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO headline_tickers (headline_id, ticker) VALUES (?, ?)",
        [(a["id"], t) for a in articles for t in a["tickers"]],
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0] - before


def month_already_fetched(conn: sqlite3.Connection, ticker: str, month: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM news_fetch_log WHERE ticker = ? AND month = ?", (ticker, month)
    ).fetchone()
    return row is not None


def log_month_fetched(conn: sqlite3.Connection, ticker: str, month: str, n_articles: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO news_fetch_log (ticker, month, n_articles, fetched_at) VALUES (?, ?, ?, ?)",
        (ticker, month, n_articles, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def load_headlines(conn: sqlite3.Connection, tickers: list[str] | None = None,
                   published_after: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per (headline, ticker) pair, with publication time as UTC datetimes."""
    query = (
        "SELECT ht.ticker, h.id, h.published_at_utc, h.headline, h.source, h.url, h.n_symbols "
        "FROM headlines h JOIN headline_tickers ht ON ht.headline_id = h.id WHERE 1 = 1"
    )
    params: list[str] = []
    if tickers:
        query += " AND ht.ticker IN (" + ",".join("?" * len(tickers)) + ")"
        params = list(tickers)
    if published_after is not None:
        query += " AND h.published_at_utc >= ?"
        params.append(published_after.tz_convert("UTC").isoformat())
    query += " ORDER BY h.published_at_utc"
    return _parse_times(pd.read_sql_query(query, conn, params=params))


def _parse_times(frame: pd.DataFrame) -> pd.DataFrame:
    # Stored strings always carry an explicit UTC offset, so utc=True is a
    # conversion here, not a guess.
    frame["published_at_utc"] = pd.to_datetime(frame["published_at_utc"], utc=True, format="ISO8601")
    return frame


# --- out-of-sample audit trail --------------------------------------------------

def log_out_of_sample_run(conn: sqlite3.Connection, parameters: str, reason: str | None) -> None:
    conn.execute(
        "INSERT INTO out_of_sample_runs (ran_at, parameters, reason) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), parameters, reason),
    )
    conn.commit()


def load_out_of_sample_runs(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT ran_at, parameters, reason FROM out_of_sample_runs ORDER BY id", conn)


# --- paper trading ----------------------------------------------------------------

def log_paper_order(conn: sqlite3.Connection, order: dict, alpaca_order_id: str | None, status: str | None) -> None:
    conn.execute(
        "INSERT INTO paper_orders (submitted_at, ticker, side, qty, reason, alpaca_order_id, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), order["symbol"], order["side"],
         order["qty"], order["reason"], alpaca_order_id, status),
    )
    conn.commit()


def log_paper_equity(conn: sqlite3.Connection, equity: float) -> None:
    conn.execute("INSERT OR REPLACE INTO paper_equity (date, equity) VALUES (?, ?)",
                 (datetime.now(timezone.utc).date().isoformat(), equity))
    conn.commit()


def load_paper_equity(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT date, equity FROM paper_equity ORDER BY date", conn)


# --- sentiment scores ---------------------------------------------------------

def unscored_headlines(conn: sqlite3.Connection, model: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT h.id, h.headline FROM headlines h "
        "WHERE NOT EXISTS (SELECT 1 FROM headline_scores s WHERE s.headline_id = h.id AND s.model = ?) "
        "ORDER BY h.id",
        conn,
        params=[model],
    )


def insert_scores(conn: sqlite3.Connection, model: str, scores: list[dict]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO headline_scores (headline_id, model, score, p_positive, p_negative, p_neutral) "
        "VALUES (:headline_id, :model, :score, :p_positive, :p_negative, :p_neutral)",
        [{**s, "model": model} for s in scores],
    )
    conn.commit()


def load_scored_headlines(conn: sqlite3.Connection, model: str, tickers: list[str] | None = None,
                          published_after: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per (headline, ticker) pair that has a score from `model`."""
    query = (
        "SELECT ht.ticker, h.id, h.published_at_utc, h.headline, h.url, s.score "
        "FROM headlines h "
        "JOIN headline_tickers ht ON ht.headline_id = h.id "
        "JOIN headline_scores s ON s.headline_id = h.id AND s.model = ? "
        "WHERE 1 = 1"
    )
    params: list[str] = [model]
    if tickers:
        query += " AND ht.ticker IN (" + ",".join("?" * len(tickers)) + ")"
        params += list(tickers)
    if published_after is not None:
        # Stored times are all UTC in one ISO format, so text order is time order.
        query += " AND h.published_at_utc >= ?"
        params.append(published_after.tz_convert("UTC").isoformat())
    query += " ORDER BY h.published_at_utc"
    return _parse_times(pd.read_sql_query(query, conn, params=params))
