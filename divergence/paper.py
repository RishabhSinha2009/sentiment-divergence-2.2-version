"""Paper trading: places the strategy's trades in an Alpaca PAPER account (fake money).

Paper trading is a forward test. The rule runs on days that did not exist when
it was designed, against real prices and real order fills, so nothing about
the result can have been fitted to the past.

Safety rails, deliberately not configurable:
  - Only Alpaca's paper-trading address appears in this code. There is no
    setting that points it at a real-money account.
  - Nothing is sent until you have seen every order and typed "yes". The
    scheduled website job passes --yes instead; it can still only reach the
    paper account.
  - Orders are market-on-open (filled in the opening auction), matching the
    backtest's assumption that trades happen at the open. Alpaca accepts
    these only between 7:00 pm and 9:28 am New York time.
  - Only the stocks in config.yaml are touched. Any other positions in the
    paper account are left alone.
"""

import math
from dataclasses import dataclass
from datetime import time

import pandas as pd
import requests

from . import db
from .errors import DataSourceError
from .trading_calendar import MARKET_TZ

PAPER_API = "https://paper-api.alpaca.markets"

# Alpaca rejects market-on-open orders sent between these times (New York).
AUCTION_CUTOFF = time(9, 28)
AUCTION_REOPENS = time(19, 0)


@dataclass(frozen=True)
class PaperConfig:
    dollars_per_position: float


def auction_orders_accepted(now_utc: pd.Timestamp) -> bool:
    local = now_utc.tz_convert(MARKET_TZ).time()
    return not (AUCTION_CUTOFF <= local < AUCTION_REOPENS)


def target_shares(positions: pd.DataFrame, last_close: pd.Series, dollars_per_position: float) -> dict[str, int]:
    """Whole shares to hold per stock. Fractional shares cannot be sold short, so none are used."""
    targets = {}
    for row in positions.itertuples(index=False):
        shares = math.floor(dollars_per_position / last_close[row.ticker])
        targets[row.ticker] = shares if row.position == "LONG" else -shares
    return targets


def plan_orders(targets: dict[str, int], current: dict[str, int], universe: list[str]) -> list[dict]:
    """Orders that move the account from `current` to `targets`.

    A position already held in the right direction is left as it is, even if
    the price has moved and the share count would now differ slightly. Topping
    it up every day would pay trading costs for nothing.
    """
    orders = []
    for ticker in universe:
        have, want = current.get(ticker, 0), targets.get(ticker, 0)
        same_direction = have != 0 and want != 0 and (have > 0) == (want > 0)
        if have == want or same_direction:
            continue
        if have != 0:
            orders.append({"symbol": ticker, "side": "sell" if have > 0 else "buy",
                           "qty": abs(have), "reason": "close: holding period over or signal flipped"})
        if want != 0:
            orders.append({"symbol": ticker, "side": "buy" if want > 0 else "sell",
                           "qty": abs(want), "reason": "open: " + ("long" if want > 0 else "short")})
    return orders


class PaperBroker:
    """The few Alpaca paper-trading calls this project needs."""

    def __init__(self, headers: dict[str, str]):
        self.session = requests.Session()
        self.session.headers.update(headers)

    def _call(self, method: str, path: str, **kwargs) -> dict | list:
        response = self.session.request(method, PAPER_API + path, timeout=30, **kwargs)
        if response.status_code >= 400:
            raise DataSourceError(f"Alpaca paper API {method} {path} failed ({response.status_code}): "
                                  f"{response.text[:300]}")
        return response.json()

    def account(self) -> dict:
        return self._call("GET", "/v2/account")

    def positions(self) -> dict[str, int]:
        held = {}
        for p in self.open_positions():
            shares = abs(int(float(p["qty"])))
            held[p["symbol"]] = -shares if p["side"] == "short" else shares
        return held

    def equity_history(self) -> pd.DataFrame:
        """Daily account value as Alpaca records it, so no history needs storing here."""
        raw = self._call("GET", "/v2/account/portfolio/history", params={"period": "1A", "timeframe": "1D"})
        history = pd.DataFrame({"date": pd.to_datetime(raw.get("timestamp") or [], unit="s"),
                                "equity": raw.get("equity") or []}).dropna()
        # Alpaca reports 0 for days before the account existed.
        return history[history["equity"] > 0].reset_index(drop=True)

    def open_positions(self) -> list[dict]:
        return self._call("GET", "/v2/positions")

    def pending_symbols(self) -> set[str]:
        """Stocks with an order sent but not yet filled (for example, queued over a holiday)."""
        return {o["symbol"] for o in self._call("GET", "/v2/orders", params={"status": "open"})}

    def recent_orders(self, limit: int = 20) -> list[dict]:
        return self._call("GET", "/v2/orders", params={"status": "all", "limit": limit, "direction": "desc"})

    def submit_on_open(self, order: dict) -> dict:
        return self._call("POST", "/v2/orders", json={
            "symbol": order["symbol"], "qty": str(order["qty"]), "side": order["side"],
            "type": "market", "time_in_force": "opg",
        })


def run(conn, broker: PaperBroker, scan_result, config, now_utc: pd.Timestamp, ask=input) -> None:
    last_close = scan_result.panel[scan_result.panel["date"] == scan_result.last_session] \
        .set_index("ticker")["adj_close"]
    targets = target_shares(scan_result.positions, last_close, config.paper.dollars_per_position)
    orders = plan_orders(targets, broker.positions(), config.tickers)
    # An order that is queued but not filled is not a position yet. Skipping
    # those stocks stops a second run from sending the same trade twice.
    pending = broker.pending_symbols()
    for symbol in sorted(pending & {o["symbol"] for o in orders}):
        print(f"  {symbol}: an earlier order is still waiting to fill, so nothing new is sent for it.")
    orders = [o for o in orders if o["symbol"] not in pending]

    print(f"Paper orders for the {scan_result.next_session:%a %Y-%m-%d} open:")
    if not orders:
        print("  none: the account already matches what the rule holds.")
        return
    for o in orders:
        print(f"  {o['side'].upper():<4} {o['qty']:>5} {o['symbol']:<5}  {o['reason']}")

    if not auction_orders_accepted(now_utc):
        print("\nNot sent. Alpaca only accepts opening-auction orders between 7:00 pm and "
              "9:28 am New York time. Run this again in that window.")
        return
    if ask(f"\nType yes to send these {len(orders)} PAPER orders (fake money): ").strip().lower() != "yes":
        print("Nothing sent.")
        return
    for o in orders:
        result = broker.submit_on_open(o)
        db.log_paper_order(conn, o, result.get("id"), result.get("status"))
        print(f"  sent {o['side']} {o['qty']} {o['symbol']}: {result.get('status')}")


def status(conn, broker: PaperBroker, tickers: list[str]) -> None:
    """Print the paper account and record today's value, building a forward-test track record."""
    account = broker.account()
    equity = float(account["equity"])
    db.log_paper_equity(conn, equity)
    print(f"Paper account value: ${equity:,.2f}   cash: ${float(account['cash']):,.2f}")
    held = {t: q for t, q in broker.positions().items() if t in tickers}
    print("Positions from this strategy:", ", ".join(f"{t} {q:+d}" for t, q in held.items()) or "none")
    history = db.load_paper_equity(conn)
    if len(history) > 1:
        change = history["equity"].iloc[-1] / history["equity"].iloc[0] - 1
        print(f"Since first recorded {history['date'].iloc[0]}: {change:+.2%} over {len(history)} snapshots")
