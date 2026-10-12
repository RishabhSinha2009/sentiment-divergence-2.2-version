"""Proof that no headline can reach a trading day before it was published.

The rule (see divergence/trading_calendar.py): a headline counts toward the
first trading session whose close comes strictly after it was published.
"""

import numpy as np
import pandas as pd
import pytest

from divergence import db
from divergence.config import load_config
from divergence.sentiment import study_start
from divergence.trading_calendar import TradingCalendar


@pytest.fixture(scope="module")
def calendar():
    return TradingCalendar("2014-12-01", "2027-01-31")


def assigned_day(calendar, published: str) -> str:
    return calendar.assign_trading_day([pd.Timestamp(published)]).iloc[0].strftime("%Y-%m-%d")


@pytest.mark.parametrize(
    "published, expected",
    [
        # Normal day (Tue 2023-06-13), close at 4:00 pm New York time.
        ("2023-06-13 07:00:00-04:00", "2023-06-13"),  # pre-market news counts today
        ("2023-06-13 15:59:59-04:00", "2023-06-13"),  # one second before the bell
        ("2023-06-13 16:00:00-04:00", "2023-06-14"),  # exactly at the bell: too late for today
        ("2023-06-13 16:00:01-04:00", "2023-06-14"),
        ("2023-06-13 23:59:59-04:00", "2023-06-14"),
        # Weekends and holidays roll forward to the next session.
        ("2023-06-16 18:00:00-04:00", "2023-06-20"),  # Fri evening; Mon 06-19 is Juneteenth
        ("2023-06-17 12:00:00-04:00", "2023-06-20"),  # Saturday
        ("2023-07-04 10:00:00-04:00", "2023-07-05"),  # Independence Day
        # Early close: the day after Thanksgiving the bell rings at 1:00 pm.
        ("2023-11-24 12:59:59-05:00", "2023-11-24"),
        ("2023-11-24 13:30:00-05:00", "2023-11-27"),  # after the early close, before 4 pm
        # Inputs in UTC. The close is 20:00 UTC in summer and 21:00 UTC in winter.
        ("2023-07-12 19:59:59+00:00", "2023-07-12"),
        ("2023-07-12 20:00:00+00:00", "2023-07-13"),
        ("2023-01-12 20:30:00+00:00", "2023-01-12"),  # 3:30 pm EST, still before the close
        ("2023-01-12 21:00:00+00:00", "2023-01-13"),
        # First trading day after clocks spring forward (Sun 2023-03-12).
        ("2023-03-13 19:30:00+00:00", "2023-03-13"),
        ("2023-03-13 20:00:00+00:00", "2023-03-14"),
        # First trading day after clocks fall back (Sun 2023-11-05).
        ("2023-11-06 20:30:00+00:00", "2023-11-06"),
        ("2023-11-06 21:00:00+00:00", "2023-11-07"),
    ],
)
def test_specific_cases(calendar, published, expected):
    assert assigned_day(calendar, published) == expected


def test_naive_times_are_rejected(calendar):
    # Without a timezone, "15:30" could be New York or UTC, a 4-5 hour
    # difference that could quietly leak tomorrow's news into today.
    with pytest.raises(TypeError):
        calendar.assign_trading_day([pd.Timestamp("2023-06-13 15:30:00")])


def check_rule(calendar, published_utc: pd.Series) -> None:
    days = calendar.assign_trading_day(published_utc)
    closes = calendar.close_of(days).to_numpy()
    previous_closes = calendar.previous_close_of(days).to_numpy()
    published = published_utc.to_numpy()

    # 1. No look-ahead: every headline existed before the close of the day it counts toward.
    assert (published < closes).all()
    # 2. No needless delay: it arrived at or after the previous close, so it
    #    could not have counted toward any earlier day.
    assert (published >= previous_closes).all()


def test_rule_holds_for_200k_random_times(calendar):
    # Random instants spread over 2015-2026, including nights, weekends,
    # holidays, early closes and daylight-saving switches.
    rng = np.random.default_rng(seed=0)
    low = pd.Timestamp("2015-01-01", tz="UTC").value
    high = pd.Timestamp("2026-10-01", tz="UTC").value
    published = pd.Series(pd.to_datetime(rng.integers(low, high, size=200_000), utc=True))
    check_rule(calendar, published)


def test_rule_holds_for_every_stored_headline():
    """The same check against the real downloaded headlines."""
    config = load_config()
    if not config.database.exists():
        pytest.skip("No database yet; run `python -m divergence news` first.")
    headlines = db.load_headlines(db.connect(config.database), published_after=study_start(config.start))
    if headlines.empty:
        pytest.skip("No headlines downloaded yet.")
    calendar = TradingCalendar.covering(config.start)
    check_rule(calendar, headlines["published_at_utc"])
