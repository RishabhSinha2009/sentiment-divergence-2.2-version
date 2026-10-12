"""Paper trading: order planning and the safety rails. Prices here are made-up test inputs."""

from pathlib import Path

import pandas as pd
import pytest

from divergence import db, paper
from divergence.config import Config
from divergence.paper import PaperConfig
from divergence.scan import ScanResult


@pytest.mark.parametrize("new_york_time, accepted", [
    ("2023-06-13 08:00", True),    # before the 9:28 am cutoff
    ("2023-06-13 09:30", False),   # market open
    ("2023-06-13 18:59", False),
    ("2023-06-13 19:00", True),    # queued for the next opening auction
])
def test_opening_auction_window(new_york_time, accepted):
    now = pd.Timestamp(new_york_time, tz="America/New_York").tz_convert("UTC")
    assert paper.auction_orders_accepted(now) == accepted


def test_target_shares_are_whole_and_signed():
    positions = pd.DataFrame({"ticker": ["AAA", "BBB"], "position": ["LONG", "SHORT"]})
    targets = paper.target_shares(positions, pd.Series({"AAA": 190.0, "BBB": 60.0}), 5000)
    assert targets == {"AAA": 26, "BBB": -83}


def test_order_planning():
    universe = ["AAA", "BBB", "CCC", "DDD"]
    current = {"AAA": 10, "BBB": 5, "DDD": -7, "OTHER": 100}
    targets = {"AAA": 12, "BBB": -6, "CCC": 4}
    orders = paper.plan_orders(targets, current, universe)
    simple = [(o["symbol"], o["side"], o["qty"]) for o in orders]
    assert simple == [
        ("BBB", "sell", 5), ("BBB", "sell", 6),   # flip: close the long, then open the short
        ("CCC", "buy", 4),                        # new long
        ("DDD", "buy", 7),                        # holding period over: cover the short
    ]                                             # AAA already long: left alone; OTHER never touched


def test_no_real_money_address_exists_anywhere_in_the_code():
    assert paper.PAPER_API.startswith("https://paper-api.")
    package = Path(paper.__file__).parent
    for source in package.glob("*.py"):
        assert "https://api.alpaca.markets" not in source.read_text(), source.name


class FakeBroker:
    def __init__(self, held, pending=()):
        self.held, self.pending, self.sent = held, set(pending), []

    def positions(self):
        return self.held

    def pending_symbols(self):
        return self.pending

    def submit_on_open(self, order):
        self.sent.append(order)
        return {"id": f"order-{len(self.sent)}", "status": "accepted"}


def setup(tmp_path):
    config = Config(tickers=["AAA"], news_aliases={}, start=None, end=None, database=tmp_path / "t.db",
                    news_page_size=50, paper=PaperConfig(dollars_per_position=1000))
    last = pd.Timestamp("2023-06-12")
    result = ScanResult(last_session=last, next_session=pd.Timestamp("2023-06-13"),
                        panel=pd.DataFrame({"ticker": ["AAA"], "date": [last], "adj_close": [100.0]}),
                        flags=pd.DataFrame(), developing=pd.DataFrame(),
                        positions=pd.DataFrame({"ticker": ["AAA"], "position": ["LONG"],
                                                "signal_date": [last.date()]}))
    return db.connect(config.database), result, config


IN_WINDOW = pd.Timestamp("2023-06-13 08:00", tz="America/New_York").tz_convert("UTC")
OUT_OF_WINDOW = pd.Timestamp("2023-06-13 11:00", tz="America/New_York").tz_convert("UTC")


def test_nothing_is_sent_outside_the_auction_window(tmp_path):
    conn, result, config = setup(tmp_path)
    broker = FakeBroker({})
    paper.run(conn, broker, result, config, OUT_OF_WINDOW, ask=lambda _: "yes")
    assert broker.sent == []


def test_nothing_is_sent_without_typing_yes(tmp_path):
    conn, result, config = setup(tmp_path)
    broker = FakeBroker({})
    paper.run(conn, broker, result, config, IN_WINDOW, ask=lambda _: "y")
    assert broker.sent == []


def test_a_stock_with_an_unfilled_order_is_not_ordered_again(tmp_path):
    conn, result, config = setup(tmp_path)
    broker = FakeBroker({}, pending={"AAA"})
    paper.run(conn, broker, result, config, IN_WINDOW, ask=lambda _: "yes")
    assert broker.sent == []


def test_confirmed_orders_are_sent_and_logged(tmp_path):
    conn, result, config = setup(tmp_path)
    broker = FakeBroker({})
    paper.run(conn, broker, result, config, IN_WINDOW, ask=lambda _: "yes")
    assert [(o["symbol"], o["side"], o["qty"]) for o in broker.sent] == [("AAA", "buy", 10)]
    logged = conn.execute("SELECT ticker, side, qty, alpaca_order_id FROM paper_orders").fetchall()
    assert logged == [("AAA", "buy", 10, "order-1")]
