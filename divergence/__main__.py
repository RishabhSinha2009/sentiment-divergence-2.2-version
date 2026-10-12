"""Command line:  divergence <command>   (or: python -m divergence <command>)

Build the dataset
    prices        download or refresh daily prices
    news          download or refresh headlines (needs Alpaca keys in .env)
    coverage      show what is stored and flag data problems
    score         score new headlines with VADER and FinBERT (saved, never redone)
    sample        show random headlines with both scores, for a sanity check

Test the idea
    backtest      run the strategy and every comparison; results go to results/
    report        build results/report.html from the latest backtest

Use it today
    scan          refresh everything and print stocks diverging right now
    live          watch the news stream and score each headline as it arrives
    paper-trade   place the rule's trades in an Alpaca PAPER account (fake money)
    paper-status  show the paper account and record its value
    site          build the public website in site/

Option --recent-days N (before the command) works on only the last N days of
data. The scheduled website job uses it to skip the 10-year history.
"""

import argparse
import asyncio
import dataclasses
import json
import sys
from datetime import timedelta

import pandas as pd

from . import backtest, coverage, db, live, news, paper, prices, report, scan, sentiment, website
from .config import REPO_ROOT, load_config
from .errors import DataSourceError
from .trading_calendar import TradingCalendar

RESULTS = REPO_ROOT / "results"


def scorers() -> list:
    # VADER first: it takes seconds, so its results arrive even if the much
    # slower FinBERT run is interrupted.
    return [sentiment.VaderScorer(), sentiment.FinBertScorer()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="divergence", description="News sentiment vs. price divergence.")
    parser.add_argument("--recent-days", type=int, help="use only the last N days of data (for the website job)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prices", help="download or refresh daily prices from Yahoo Finance")
    commands.add_parser("news", help="download or refresh headlines from Alpaca (needs keys in .env)")
    commands.add_parser("coverage", help="show what is stored and flag data problems")
    commands.add_parser("score", help="score new headlines with VADER and FinBERT")
    sample = commands.add_parser("sample", help="show random headlines with both scores")
    sample.add_argument("-n", type=int, default=20, help="how many headlines (default 20)")
    sample.add_argument("--seed", type=int, help="fix the random pick so it can be repeated")
    test = commands.add_parser("backtest", help="test the strategy against its comparisons")
    test.add_argument("--reason", help="required to re-run the out-of-sample test after changing parameters")
    commands.add_parser("report", help="build results/report.html from the latest backtest")
    commands.add_parser("scan", help="print stocks diverging right now")
    commands.add_parser("live", help="score headlines as they arrive and alert on divergences")
    trade = commands.add_parser("paper-trade", help="place the rule's trades in an Alpaca paper account")
    trade.add_argument("--yes", action="store_true",
                       help="send without asking (used by the scheduled job; paper account only)")
    commands.add_parser("paper-status", help="show the paper account and record its value")
    commands.add_parser("site", help="build the public website in site/")
    args = parser.parse_args(argv)

    config = load_config()
    conn = db.connect(config.database)
    now = pd.Timestamp.now(tz="UTC")
    if args.recent_days:
        config = dataclasses.replace(config, start=now.date() - timedelta(days=args.recent_days))
    calendar = TradingCalendar.covering(config.start, now.date())
    last_session = calendar.last_completed_session(now).date()
    if config.end:
        last_session = min(last_session, config.end)

    try:
        if args.command == "prices":
            for line in prices.update_prices(conn, config.tickers, config.start, last_session):
                print(line, flush=True)
        elif args.command == "news":
            news.update_news(conn, config, now)
        elif args.command == "coverage":
            coverage.print_report(conn, calendar, config.tickers, config.start)
        elif args.command == "score":
            for scorer in scorers():
                print(f"{scorer.name}: scored {sentiment.score_new_headlines(conn, scorer)} new headlines")
        elif args.command == "sample":
            picked = sentiment.random_sample(conn, args.n, args.seed)
            with pd.option_context("display.max_colwidth", 110, "display.width", 250):
                print(picked.to_string(index=False) if not picked.empty
                      else "No scored headlines yet. Run `news` and then `score` first.")
        elif args.command == "backtest":
            results = backtest.run_from_database(conn, calendar, config, args.reason)
            backtest.print_results(results)
            backtest.save_results(results, RESULTS)
        elif args.command == "report":
            if not (RESULTS / "backtest.json").exists():
                raise DataSourceError("No backtest results yet. Run `backtest` first.")
            path = report.build(RESULTS, coverage.headline_coverage(conn, calendar, config.tickers, config.start))
            print(f"Report written to {path}")
        elif args.command == "scan":
            scan.refresh(conn, calendar, config, now, scorers())
            scan.print_scan(scan.scan(conn, calendar, config))
        elif args.command == "live":
            monitor = live.LiveMonitor(conn, calendar, config, scorers())
            try:
                asyncio.run(monitor.run(news.load_credentials()))
            except KeyboardInterrupt:
                print("Stopped.")
        elif args.command == "paper-trade":
            broker = paper.PaperBroker(news.load_credentials())
            scan.refresh(conn, calendar, config, now, scorers())
            ask = (lambda _: "yes") if args.yes else input
            paper.run(conn, broker, scan.scan(conn, calendar, config), config, now, ask=ask)
        elif args.command == "paper-status":
            paper.status(conn, paper.PaperBroker(news.load_credentials()), config.tickers)
        elif args.command == "site":
            scan.refresh(conn, calendar, config, now, scorers())
            result = scan.scan(conn, calendar, config)
            recent = scan.recent_headlines(conn, calendar, config, result.last_session)
            try:
                snapshot = website.collect_paper(paper.PaperBroker(news.load_credentials()))
            except DataSourceError as exc:
                print(f"Paper account left out of the site: {exc}")
                snapshot = None
            has_results = (RESULTS / "backtest.json").exists()
            results = json.loads((RESULTS / "backtest.json").read_text()) if has_results else None
            report_file = RESULTS / "report.html" if (RESULTS / "report.html").exists() else None
            path = website.build(REPO_ROOT / "site", result, recent, config, results, snapshot, now, report_file)
            print(f"Website written to {path}")
    except backtest.OutOfSampleRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except DataSourceError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
