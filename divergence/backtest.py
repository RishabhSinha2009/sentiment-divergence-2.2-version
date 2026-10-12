"""Backtest and honest evaluation.

How money is counted
  - Every trade happens at the opening price on the morning after the signal.
  - A position held on day d earns the open-to-open return from d to d+1.
  - Money is split equally across all open positions, longs and shorts, so the
    portfolio is fully invested whenever anything is open and sits in cash
    (earning nothing) otherwise.
  - Every change in a weight costs `cost_bps_per_side` of the amount traded,
    so opening and later closing a position costs twice that.

Periods
  - in-sample:     start of data through `in_sample_end`, used for development
  - out-of-sample: the day after `in_sample_end` to today, run only once the
                   parameters are locked; every run is logged in the database

What it is compared against
  - price-only:   the same rule with the news removed
  - placebo:      the same rule fed shuffled sentiment, many times over
  - buy-and-hold: equal money in all 30 stocks, bought once and left alone
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import db, sentiment
from .errors import DataSourceError
from .signals import SignalConfig, build_panel

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestConfig:
    in_sample_end: str
    cost_bps_per_side: float
    placebo_runs: int
    locked: bool


class OutOfSampleRefused(RuntimeError):
    """Raised instead of re-running the test period with changed parameters."""


# --- returns ------------------------------------------------------------------

def add_forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Returns that happen AFTER each day. Used only to score positions, never to choose them."""
    by_ticker = panel.groupby("ticker")
    panel["return_open_to_open"] = by_ticker["adj_open"].shift(-1) / panel["adj_open"] - 1
    panel["return_next_close"] = by_ticker["adj_close"].shift(-1) / panel["adj_close"] - 1
    return panel


def daily_returns(panel: pd.DataFrame, position_column: str, cost_per_side: float) -> pd.DataFrame:
    """Portfolio return for each day, before and after trading costs."""
    positions = panel.pivot(index="date", columns="ticker", values=position_column).fillna(0)
    returns = panel.pivot(index="date", columns="ticker", values="return_open_to_open")
    n_open = (positions != 0).sum(axis=1)
    weights = positions.div(n_open.where(n_open > 0), axis=0).fillna(0.0)

    gross = (weights * returns.fillna(0.0)).sum(axis=1)
    # Every weight change is a trade. The first day's weights are bought from cash.
    turnover = weights.diff().fillna(weights).abs().sum(axis=1)
    cost = turnover * cost_per_side
    result = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "n_positions": n_open})
    # The last day has no next open, so its return is unknown.
    return result[returns.notna().any(axis=1)]


def buy_and_hold_returns(panel: pd.DataFrame, start, end, cost_per_side: float) -> pd.Series:
    """Equal money in every stock at the first open of the period, then left alone."""
    opens = panel.pivot(index="date", columns="ticker", values="adj_open").loc[start:end]
    value = (opens / opens.iloc[0]).mean(axis=1)
    returns = (value.shift(-1) / value - 1).dropna()
    returns.iloc[0] -= cost_per_side   # buying everything on day one
    returns.iloc[-1] -= cost_per_side  # selling everything at the end
    return returns


def trade_list(panel: pd.DataFrame, position_column: str, cost_per_side: float) -> pd.DataFrame:
    """One row per trade: an unbroken run of the same position in one stock."""
    trades = []
    for ticker, rows in panel.groupby("ticker"):
        position = rows[position_column].to_numpy()
        returns = rows["return_open_to_open"].fillna(0.0).to_numpy()
        dates = rows["date"].to_numpy()
        start = None
        for i in range(len(position) + 1):
            current = position[i] if i < len(position) else 0
            if start is not None and (current != position[start]):
                direction = position[start]
                gross = np.prod(1 + direction * returns[start:i]) - 1
                trades.append({"ticker": ticker, "entry_date": dates[start], "direction": int(direction),
                               "days": i - start, "net_return": gross - 2 * cost_per_side})
                start = None
            if start is None and current != 0:
                start = i
    return pd.DataFrame(trades, columns=["ticker", "entry_date", "direction", "days", "net_return"])


# --- statistics ---------------------------------------------------------------

def summarize(net: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    n = len(net)
    equity = (1 + net).cumprod()
    std = net.std()
    stats = {
        "annual_return": equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / n) - 1,
        # Risk-free rate taken as zero for simplicity; see README.
        "sharpe": net.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else np.nan,
        # Peak never below 1.0, the starting value, so an early loss counts as a drawdown.
        "max_drawdown": (equity / equity.cummax().clip(lower=1.0) - 1).min(),
        # How many standard errors the average day is from zero. Below about 2,
        # the result is hard to tell apart from luck.
        "t_stat": net.mean() / (std / np.sqrt(n)) if std > 0 else np.nan,
        "days": n,
    }
    if trades is not None:
        stats["trades"] = len(trades)
        stats["hit_rate"] = (trades["net_return"] > 0).mean() if len(trades) else np.nan
    return stats


def sentiment_next_day_correlation(panel: pd.DataFrame, start, end) -> dict:
    """Correlation between a day's news tone and the next day's close-to-close return.

    Pooled across all stocks. Same-day returns of different stocks move
    together, so the observations are not fully independent and n overstates
    how much evidence there is.
    """
    rows = panel[(panel["date"] >= start) & (panel["date"] <= end)
                 & panel["sentiment"].notna() & panel["return_next_close"].notna()]
    return {"correlation": rows["sentiment"].corr(rows["return_next_close"]), "n": len(rows)}


def shuffled_sentiment(daily: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Same stocks, same news days, same scores, dealt out in random order.

    This keeps everything about the sentiment data except its timing, so any
    real link between a day's tone and what the price did next is destroyed.
    """
    shuffled = daily.copy()
    shuffled["sentiment"] = shuffled.groupby("ticker")["sentiment"].transform(
        lambda s: rng.permutation(s.to_numpy()))
    return shuffled


# --- the full evaluation ---------------------------------------------------------

def evaluate(prices: pd.DataFrame, daily_by_model: dict[str, pd.DataFrame], signal: SignalConfig,
             backtest: BacktestConfig, include_out_of_sample: bool, seed: int = 0) -> dict:
    cost = backtest.cost_bps_per_side / 10_000
    first_day = pd.Timestamp(prices["date"].min())
    in_sample_end = pd.Timestamp(backtest.in_sample_end)
    periods = {"in_sample": (first_day, in_sample_end)}
    if include_out_of_sample:
        periods["out_of_sample"] = (in_sample_end + pd.Timedelta(days=1), pd.Timestamp(prices["date"].max()))
    # Nothing past this day is reported, so while parameters are unlocked the
    # test period stays unseen, curves included.
    last_reported_day = periods[list(periods)[-1]][1]

    panels = {model: add_forward_returns(build_panel(prices, daily, signal))
              for model, daily in daily_by_model.items()}
    main = panels[signal.sentiment_model]
    strategies = {f"news divergence ({signal.sentiment_model})": (main, "position"),
                  "price only (no news)": (main, "position_price_only")}
    for model, panel in panels.items():
        strategies.setdefault(f"news divergence ({model})", (panel, "position"))

    table, curves = [], {}
    for name, (panel, column) in strategies.items():
        returns = daily_returns(panel, column, cost)
        trades = trade_list(panel, column, cost)
        curves[name] = returns["net"].loc[:last_reported_day]
        for period, (start, end) in periods.items():
            stats = summarize(returns["net"].loc[start:end],
                              trades[(trades["entry_date"] >= start) & (trades["entry_date"] <= end)])
            stats["exposure"] = (returns["n_positions"].loc[start:end] > 0).mean()
            table.append({"strategy": name, "period": period, **stats})

    for period, (start, end) in periods.items():
        stats = summarize(buy_and_hold_returns(main, start, end, cost))
        table.append({"strategy": "buy and hold (equal weight)", "period": period, **stats})
    curves["buy and hold (equal weight)"] = buy_and_hold_returns(main, first_day, last_reported_day, cost)

    placebo = run_placebo(prices, daily_by_model[signal.sentiment_model], signal, backtest, periods,
                          table, cost, seed)
    correlation = {period: {model: sentiment_next_day_correlation(panel, start, end)
                            for model, panel in panels.items()}
                   for period, (start, end) in periods.items()}

    return {
        "periods": {p: [s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")] for p, (s, e) in periods.items()},
        "signal_parameters": asdict(signal),
        "backtest_parameters": asdict(backtest),
        "table": table,
        "placebo": placebo,
        "correlation": correlation,
        "curves": curves,
    }


def run_placebo(prices, daily, signal, backtest, periods, table, cost, seed) -> dict:
    """Run the news strategy on shuffled sentiment many times.

    p_value = share of shuffled runs that did at least as well as the real
    one. If shuffled news does as well as real news, timing of the news is
    not what made money.
    """
    rng = np.random.default_rng(seed)
    sharpes = {period: [] for period in periods}
    for _ in range(backtest.placebo_runs):
        panel = add_forward_returns(build_panel(prices, shuffled_sentiment(daily, rng), signal))
        net = daily_returns(panel, "position", cost)["net"]
        for period, (start, end) in periods.items():
            sharpes[period].append(summarize(net.loc[start:end])["sharpe"])

    real_name = f"news divergence ({signal.sentiment_model})"
    result = {}
    for period, values in sharpes.items():
        values = np.array(values, dtype=float)
        real = next(r["sharpe"] for r in table if r["strategy"] == real_name and r["period"] == period)
        result[period] = {
            "runs": len(values),
            "sharpe_mean": float(np.nanmean(values)),
            "sharpe_95th_percentile": float(np.nanpercentile(values, 95)),
            # The +1s count the real run as one of the possibilities, the
            # standard way to keep a permutation p-value from ever being 0.
            "p_value": float((1 + np.sum(values >= real)) / (1 + len(values))),
            "sharpes": [float(v) for v in values],
        }
    return result


def run_from_database(conn, calendar, config, reason: str | None = None) -> dict:
    prices = db.load_prices(conn, config.tickers)
    daily_by_model = {model: sentiment.daily_sentiment(conn, calendar, model, config.tickers, config.start)
                      for model in ("finbert", "vader")}
    if any(d.empty for d in daily_by_model.values()):
        raise DataSourceError("No scored headlines yet. Run `news` and then `score` first.")

    signal, backtest = config.signal, config.backtest
    if backtest.locked:
        check_out_of_sample_allowed(conn, signal, backtest, reason)
    results = evaluate(prices, daily_by_model, signal, backtest, include_out_of_sample=backtest.locked)
    if backtest.locked:
        db.log_out_of_sample_run(conn, parameters_fingerprint(signal, backtest), reason)
    results["out_of_sample_runs"] = db.load_out_of_sample_runs(conn).to_dict("records")
    return results


def parameters_fingerprint(signal: SignalConfig, backtest: BacktestConfig) -> str:
    fixed = {**asdict(signal), "in_sample_end": backtest.in_sample_end,
             "cost_bps_per_side": backtest.cost_bps_per_side}
    return json.dumps(fixed, sort_keys=True)


def check_out_of_sample_allowed(conn, signal, backtest, reason: str | None) -> None:
    """Refuse to re-test the out-of-sample period with different parameters.

    Changing parameters after seeing out-of-sample results turns the test
    period into another training period. A genuine bug fix can still go
    ahead, but only with a written reason, which appears in every report.
    """
    previous = db.load_out_of_sample_runs(conn)
    current = parameters_fingerprint(signal, backtest)
    if not previous.empty and (previous["parameters"] != current).any() and not reason:
        raise OutOfSampleRefused(
            "The out-of-sample period was already tested with different parameters. "
            "Re-testing with new ones would be tuning on the test period. If this is a "
            "genuine bug fix, re-run with --reason \"...\" and it will be shown in the report."
        )


def save_results(results: dict, folder) -> None:
    """Write the numbers to results/ so the README and HTML report use exactly what was printed."""
    folder.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in results.items() if k != "curves"}
    (folder / "backtest.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    pd.DataFrame(results["curves"]).rename_axis("date").to_csv(folder / "daily_returns.csv")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    raise TypeError(f"Cannot save {type(value)} as JSON")


def format_table(results: dict) -> str:
    table = pd.DataFrame(results["table"])
    columns = ["annual_return", "sharpe", "max_drawdown", "hit_rate", "trades", "exposure", "t_stat"]
    wide = table.pivot(index="strategy", columns="period", values=columns)
    return wide.round(3).to_string()


def print_results(results: dict) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("Periods:", results["periods"])
        print("\n" + format_table(results))
        print("\nPlacebo (shuffled sentiment):")
        for period, p in results["placebo"].items():
            print(f"  {period}: mean Sharpe {p['sharpe_mean']:.3f}, 95th pct {p['sharpe_95th_percentile']:.3f}, "
                  f"p-value of real strategy {p['p_value']:.3f} ({p['runs']} runs)")
        print("\nCorrelation of today's sentiment with next-day return:")
        for period, by_model in results["correlation"].items():
            for model, c in by_model.items():
                print(f"  {period} {model}: {c['correlation']:.4f} (n={c['n']})")
        if "out_of_sample" not in results["periods"]:
            print("\nOut-of-sample results are locked away until `backtest.locked: true` in config.yaml.")
        print(f"\nOut-of-sample runs so far: {len(results['out_of_sample_runs'])}")
        for run in results["out_of_sample_runs"]:
            print(f"  {run['ran_at']}  reason: {run['reason'] or '-'}")
