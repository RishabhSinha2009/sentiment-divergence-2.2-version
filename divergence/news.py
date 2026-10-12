"""Company news headlines from the Alpaca News API (Benzinga's feed), cached in SQLite.

Alpaca needs a free account. The keys are read from a .env file that stays on
this machine (it is git-ignored) and are never written into code.

News is fetched one ticker and one calendar month at a time. Each finished
month is logged in the database, so an interrupted download picks up where
it stopped and nothing is downloaded twice.
"""

import os
import time
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

from . import db
from .config import REPO_ROOT, Config
from .errors import DataSourceError

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
MAX_RETRIES = 5

# Free Alpaca accounts only see data at least 15 minutes old; asking for
# anything newer is refused, so the current month's request stops short.
FREE_PLAN_DELAY = pd.Timedelta(minutes=20)


def load_credentials() -> dict[str, str]:
    load_dotenv(REPO_ROOT / ".env")
    key_id = os.getenv("ALPACA_API_KEY_ID", "").strip()
    secret = os.getenv("ALPACA_API_SECRET_KEY", "").strip()
    if not key_id or not secret:
        raise DataSourceError(
            "Alpaca keys not found. Copy .env.example to .env and fill in "
            "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY."
        )
    return {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}


def months_between(start: date, end: date) -> list[str]:
    return [period.strftime("%Y-%m") for period in pd.period_range(start, end, freq="M")]


def month_is_complete(month: str, now_utc: pd.Timestamp) -> bool:
    """Only a month that is over gets logged as done; the current one is re-checked every run."""
    next_month_starts = (pd.Period(month, freq="M") + 1).start_time.tz_localize("UTC")
    return now_utc >= next_month_starts + FREE_PLAN_DELAY


def parse_article(raw: dict, symbol_to_ticker: dict[str, str]) -> dict | None:
    """Turn one API article into a database row, or None if it is unusable."""
    headline = (raw.get("headline") or "").strip()
    symbols = raw.get("symbols") or []
    tickers = sorted({symbol_to_ticker[s] for s in symbols if s in symbol_to_ticker})
    if not headline or not tickers:
        return None

    # created_at is when the story first appeared, which is when traders could
    # first react to it. updated_at only records the last time an editor
    # touched it, so it says nothing about when the news was known.
    published = pd.Timestamp(raw["created_at"])
    if published.tzinfo is None:
        raise DataSourceError(f"Article {raw.get('id')} has a timestamp with no timezone: {raw['created_at']}")

    return {
        "id": int(raw["id"]),
        "published_at_utc": published.tz_convert("UTC").isoformat(),
        "headline": headline,
        "source": raw.get("source"),
        "url": raw.get("url"),
        "n_symbols": len(symbols),
        "tickers": tickers,
    }


def fetch_month(session: requests.Session, symbols: list[str], month: str, page_size: int,
                now_utc: pd.Timestamp) -> list[dict]:
    period = pd.Period(month, freq="M")
    end = min(period.end_time.tz_localize("UTC"), now_utc - FREE_PLAN_DELAY)
    params = {
        "symbols": ",".join(symbols),
        "start": period.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": page_size,
        "sort": "asc",
    }
    articles = []
    while True:
        payload = _get_with_retries(session, params)
        articles.extend(payload.get("news") or [])
        next_page = payload.get("next_page_token")
        if not next_page:
            return articles
        params["page_token"] = next_page


def _get_with_retries(session: requests.Session, params: dict) -> dict:
    for attempt in range(MAX_RETRIES):
        response = session.get(NEWS_URL, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:
            # Over the rate limit: wait until Alpaca says the window resets.
            reset_at = float(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            time.sleep(max(1.0, reset_at - time.time()))
            continue
        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        raise DataSourceError(f"Alpaca news request failed ({response.status_code}): {response.text[:300]}")
    raise DataSourceError(f"Alpaca news request still failing after {MAX_RETRIES} attempts.")


def update_news(conn, config: Config, now_utc: pd.Timestamp, verbose: bool = True) -> None:
    headers = load_credentials()
    symbol_to_ticker = {symbol: t for t in config.tickers for symbol in config.news_symbols_for(t)}
    months = months_between(config.start, config.end or now_utc.date())
    # The API filters by last-edited date, so an old story edited recently can
    # come back too. Anything first published before the study window is dropped.
    window_start = pd.Timestamp(config.start, tz="UTC").isoformat()

    with requests.Session() as session:
        session.headers.update(headers)
        for ticker in config.tickers:
            months_checked = 0
            new_headlines = 0
            for month in months:
                if db.month_already_fetched(conn, ticker, month):
                    continue
                raw = fetch_month(session, config.news_symbols_for(ticker), month,
                                  config.news_page_size, now_utc)
                articles = [a for a in (parse_article(r, symbol_to_ticker) for r in raw)
                            if a and a["published_at_utc"] >= window_start]
                new_headlines += db.insert_headlines(conn, articles)
                if month_is_complete(month, now_utc):
                    db.log_month_fetched(conn, ticker, month, len(articles))
                months_checked += 1
            if verbose:
                print(f"{ticker}: checked {months_checked} month(s), {new_headlines} new headlines", flush=True)
