"""The NYSE trading calendar and the rule that assigns headlines to trading days.

The rule: a headline belongs to the first trading session whose closing bell
rings strictly after the headline was published.

    - 3:59 pm on a trading day           -> that day
    - 4:00 pm or later                   -> the next trading day
    - weekends and market holidays       -> the next trading day
    - early-close days (1:00 pm close)   -> the cutoff is 1:00 pm, not 4:00 pm

This is what makes the backtest honest. A signal for day t is computed at
day t's close, so it may only use headlines that existed before that close.
Anything later would be information from the future.
"""

from datetime import date, datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal

MARKET_TZ = "America/New_York"


class TradingCalendar:
    def __init__(self, start: date | str, end: date | str):
        schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
        # Sessions are plain dates; closes are exact UTC instants. Using the
        # real close each day handles daylight saving and early closes
        # without any special cases.
        self.sessions = pd.DatetimeIndex(schedule.index).as_unit("ns")
        self.closes_utc = pd.DatetimeIndex(schedule["market_close"]).as_unit("ns")

    @classmethod
    def covering(cls, start: date, today: date | None = None) -> "TradingCalendar":
        """A calendar with slack on both sides of the data.

        The extra month before `start` lets every headline have a previous
        close to compare against. The months after today let tonight's
        headlines be assigned to tomorrow's session.
        """
        today = today or datetime.now().date()
        return cls(start - timedelta(days=31), today + timedelta(days=90))

    def assign_trading_day(self, published_at) -> pd.Series:
        """Map each publication time to the trading session it counts toward."""
        times = _as_utc(published_at)
        # side="right" finds the first close strictly after each time, so a
        # headline stamped exactly 4:00:00 pm goes to the next session.
        positions = self.closes_utc.searchsorted(times.to_numpy(), side="right")
        if (positions == 0).any():
            raise ValueError("A headline is earlier than the calendar's first close; start the calendar earlier.")
        if (positions >= len(self.sessions)).any():
            raise ValueError("A headline is later than the calendar's last close; extend the calendar.")
        return pd.Series(self.sessions[positions], index=times.index, name="trading_day")

    def close_of(self, sessions) -> pd.Series:
        """Exact UTC closing time of each given session."""
        positions = self._positions(sessions)
        return pd.Series(self.closes_utc[positions], index=pd.Series(sessions).index)

    def previous_close_of(self, sessions) -> pd.Series:
        """Closing time of the session immediately before each given session."""
        positions = self._positions(sessions)
        if (positions == 0).any():
            raise ValueError("The first session in the calendar has no previous close.")
        return pd.Series(self.closes_utc[positions - 1], index=pd.Series(sessions).index)

    def last_completed_session(self, now_utc: pd.Timestamp) -> pd.Timestamp:
        """The most recent session whose closing bell has already rung.

        Prices for a session still in progress are not final, so they never
        get stored.
        """
        position = self.closes_utc.searchsorted(_as_utc([now_utc]).to_numpy(), side="right")[0] - 1
        return self.sessions[position]

    def is_session(self, day) -> bool:
        return pd.Timestamp(day) in self.sessions

    def _positions(self, sessions):
        sessions = pd.DatetimeIndex(pd.Series(sessions)).as_unit("ns")
        positions = self.sessions.get_indexer(sessions)
        if (positions == -1).any():
            raise ValueError("Not an NYSE trading session: " + str(sessions[positions == -1][0].date()))
        return positions


def _as_utc(published_at) -> pd.Series:
    times = pd.Series(published_at)
    if not isinstance(times.dtype, pd.DatetimeTZDtype):
        # Naive times are rejected outright. Guessing whether they meant
        # New York or UTC is exactly how a 4-5 hour look-ahead bug sneaks in.
        raise TypeError("Publication times must be timezone-aware datetimes.")
    return times.dt.tz_convert("UTC").dt.as_unit("ns")
