"""The public website. All inputs here are hand-made test values, not market data."""

import re

import pandas as pd

from divergence import website
from divergence.config import Config
from divergence.scan import ScanResult
from divergence.signals import SignalConfig

SIGNAL = SignalConfig(sentiment_model="finbert", return_days=5, zscore_window=60,
                      min_history=20, threshold=1.5, holding_days=5)
CONFIG = Config(tickers=["AAA", "BBB", "CCC"], news_aliases={}, start=None, end=None, database=None,
                news_page_size=50, signal=SIGNAL, repo_url="https://github.com/example/repo")
LAST = pd.Timestamp("2026-10-09")


def scan_result(flags=(1, 0, 0)) -> ScanResult:
    panel = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC"], "date": LAST,
        "sentiment": [0.6, 0.1, None], "sentiment_z": [2.1, 0.4, None],
        "return_recent": [-0.06, 0.01, 0.02], "return_z": [-1.9, 0.2, 0.5],
        "n_headlines": [7, 3, 0], "flag": list(flags), "adj_close": [100.0, 50.0, 20.0],
    })
    positions = pd.DataFrame({"ticker": ["AAA"], "position": ["LONG"], "signal_date": [LAST.date()]}) \
        if flags[0] == 1 else pd.DataFrame(columns=["ticker", "position", "signal_date"])
    return ScanResult(LAST, pd.Timestamp("2026-10-12"), panel, pd.DataFrame(), positions,
                      pd.DataFrame(columns=["ticker", "headlines", "tone_so_far", "tone_z", "return_z", "provisional"]))


RECENT = pd.DataFrame({
    "ticker": ["AAA", "BBB", "AAA"], "id": [1, 1, 2],
    "published_at_utc": pd.to_datetime(["2026-10-09 14:00", "2026-10-09 14:00", "2026-10-09 18:30"], utc=True),
    "headline": ["Shared story about two firms", "Shared story about two firms", "AAA beats estimates <b>"],
    "url": ["https://example.com/1", "https://example.com/1", None], "score": [0.2, 0.2, 0.9],
})


def test_example_sentence_explains_todays_trade_or_closest_call():
    latest = website.latest_rows(scan_result())
    assert "buy zone" in website.todays_example(latest, 1.5)
    latest = website.latest_rows(scan_result(flags=(0, 0, 0)))
    assert "Closest call" in website.todays_example(latest, 1.5)


def test_closeness_measures_progress_into_either_zone():
    assert website.closeness(2.0, -1.6) == 1.6      # buy zone
    assert website.closeness(-2.0, 1.7) == 1.7      # short zone
    assert website.closeness(2.0, 1.0) < 0          # same direction: nowhere near a trade


def test_page_builds_with_every_section(tmp_path):
    page = website.build(tmp_path, scan_result(), RECENT, CONFIG, results=None, paper=None,
                         now=pd.Timestamp("2026-10-11 15:00", tz="UTC")).read_text(encoding="utf-8")
    for expected in ["How the bot decides", "Today's signals", "Latest headlines", "paper-trading account",
                     "Did the idea work historically?", "Pending", "Built by Rishabh Sinha"]:
        assert expected in page
    assert ">AAA</text>" in page                   # the flagged stock is labeled on the chart
    assert "had no headlines" in page              # CCC has no tone, and the page says so
    assert "&lt;b&gt;" in page                     # headline text is escaped, never injected as HTML
    assert page.count("Shared story about two firms") == 1  # one row per article, tickers combined
    assert not re.search(r"\bnan\b", page, re.IGNORECASE)  # no unformatted missing values


def test_paper_section_shows_the_track_record():
    snapshot = website.PaperSnapshot(
        equity=pd.DataFrame({"date": pd.to_datetime(["2026-10-01", "2026-10-02"]), "equity": [100000.0, 100500.0]}),
        positions=[{"symbol": "AAA", "side": "long", "qty": "10", "market_value": "1005", "unrealized_plpc": "0.005"},
                   {"symbol": "ZZZ", "side": "long", "qty": "1", "market_value": "5", "unrealized_plpc": "0"}],
        orders=[])
    html = website.paper_section(snapshot, ["AAA"])
    assert "$100,500" in html and "+0.50%" in html
    assert "AAA" in html and "ZZZ" not in html     # only this strategy's stocks are shown
