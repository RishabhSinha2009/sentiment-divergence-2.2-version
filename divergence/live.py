"""Live mode: scores each headline the moment it arrives.

Connects to Alpaca's real-time news stream. Each new headline about one of
our stocks is saved, scored by FinBERT (well under a second on a laptop CPU),
and that stock's provisional divergence status is recomputed. An alert prints
whenever a stock starts or stops diverging.

Final signals are still set only at the close, exactly as in the backtest.
Live mode shows which way things are heading before then.

If the connection drops, it reconnects and first fetches anything it missed
through the regular news API, so no headline is skipped.
"""

import asyncio
import json

import pandas as pd

from . import db, news, scan, sentiment
from .config import Config
from .errors import DataSourceError
from .trading_calendar import TradingCalendar

STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
RECONNECT_SECONDS = 5


class LiveMonitor:
    def __init__(self, conn, calendar: TradingCalendar, config: Config, scorers, now=None):
        self.conn = conn
        self.calendar = calendar
        self.config = config
        self.scorers = scorers
        self.symbol_to_ticker = {s: t for t in config.tickers for s in config.news_symbols_for(t)}
        self.status: dict[str, str] = {}  # last provisional status shown for each ticker
        self.last_session = None
        self.refresh_history(now or pd.Timestamp.now(tz="UTC"))

    def refresh_history(self, now: pd.Timestamp) -> None:
        """Rebuild the completed-session history once a new session has closed."""
        last_completed = self.calendar.last_completed_session(now)
        if last_completed != self.last_session:
            self.panel = scan.build_panel_from_database(self.conn, self.calendar, self.config)
            self.last_session = self.panel["date"].max()
            self.status = {}

    def handle(self, message: dict, now: pd.Timestamp | None = None) -> list[str]:
        """Process one stream message. Returns the lines to print."""
        if message.get("T") == "error":
            raise DataSourceError(f"Alpaca news stream error {message.get('code')}: {message.get('msg')}")
        if message.get("T") != "n":
            return []
        article = news.parse_article(message, self.symbol_to_ticker)
        if article is None or db.insert_headlines(self.conn, [article]) == 0:
            return []  # not about our stocks, or already seen

        scores = {}
        for scorer in self.scorers:
            scores[scorer.name] = scorer.score([article["headline"]])[0]
            db.insert_scores(self.conn, scorer.name, [{"headline_id": article["id"], **scores[scorer.name]}])
        tone = scores[self.config.signal.sentiment_model]["score"]

        self.refresh_history(now or pd.Timestamp.now(tz="UTC"))
        developing = scan.developing_view(
            scan.fresh_scores(self.conn, self.calendar, self.config, self.last_session),
            self.panel, self.config).set_index("ticker")

        lines = []
        for ticker in article["tickers"]:
            row = developing.loc[ticker] if ticker in developing.index else None
            tone_z = f"{row['tone_z']:+.2f}" if row is not None and pd.notna(row["tone_z"]) else "n/a"
            lines.append(f"{article['published_at_utc'][11:19]} UTC  {ticker:<5} tone {tone:+.2f}  "
                         f"(day so far z {tone_z})  {article['headline'][:90]}")
            status = row["provisional"] if row is not None else "-"
            if status != self.status.get(ticker, "-"):
                lines.append(f"  >>> {ticker} is now {status if status != '-' else 'no longer diverging'} "
                             f"(provisional until the close)")
                self.status[ticker] = status
        return lines

    async def run(self, headers: dict) -> None:
        import websockets

        symbols = sorted(self.symbol_to_ticker)
        while True:
            # Fill any gap left while disconnected before listening again.
            news.update_news(self.conn, self.config, pd.Timestamp.now(tz="UTC"), verbose=False)
            for scorer in self.scorers:
                sentiment.score_new_headlines(self.conn, scorer)
            try:
                async with websockets.connect(STREAM_URL, additional_headers=headers) as socket:
                    await socket.send(json.dumps({"action": "subscribe", "news": symbols}))
                    print(f"Listening for news on {len(self.config.tickers)} stocks. Ctrl+C to stop.", flush=True)
                    async for raw in socket:
                        for message in json.loads(raw):
                            for line in self.handle(message):
                                print(line, flush=True)
            except (websockets.ConnectionClosed, OSError) as exc:
                print(f"Connection lost ({exc}); reconnecting in {RECONNECT_SECONDS} s.", flush=True)
                await asyncio.sleep(RECONNECT_SECONDS)
