"""Builders of value histories with known log returns, for the simulator tests.

Intraday histories are hourly on XNYS, labelled as `simulate_buy_and_hold`
labels them: each session's open mark at 9:30 (the return ending there is
the gap), then bar closes at 10:00, 11:00, ..., the session close. So the
first bar of a session, 9:30 to 10:00, is short: ρ = τ/Δ = 0.5.
"""
import exchange_calendars as xcals
import numpy as np
import pandas as pd

NY = "America/New_York"
_CAL = xcals.get_calendar("XNYS", start="2024-01-01", end="2026-12-31")


def sessions(first, n):
    """Return `n` XNYS session dates from `first` on."""
    after = _CAL.sessions[_CAL.sessions >= pd.Timestamp(first)]
    return list(after[:n])


def hourly_marks(days):
    """Return the hourly marks of the sessions `days`: the open, then each bar close on the hour."""
    marks = []
    for day in days:
        open_ = _CAL.session_open(day).tz_convert(NY)
        close = _CAL.session_close(day).tz_convert(NY)
        closes = pd.date_range(open_.floor("h") + pd.Timedelta(hours=1), close, freq="h")
        marks += [open_] + list(closes)
    return pd.DatetimeIndex(marks)


def mark_kinds(index):
    """Return, for each return of hourly marks, ("gap" | "regular", ρ).

    The return ending at a 9:30 open mark (after the first mark) is the gap,
    ρ = 1; a bar ending at 10:00 runs from 9:30 and has ρ = 0.5; others 1.
    """
    kinds, rho = [], []
    for t in index[1:]:
        if (t.hour, t.minute) == (9, 30):
            kinds.append("gap")
            rho.append(1.0)
        else:
            kinds.append("regular")
            rho.append(0.5 if (t.hour, t.minute) == (10, 0) else 1.0)
    return np.array(kinds, dtype=object), np.array(rho)


def from_log_returns(index, ell, start=100.0):
    """Return the value series on `index` whose log returns are exactly `ell`."""
    return pd.Series(start * np.exp(np.cumsum(np.r_[0.0, ell])), index=index)


def intraday_history(n_sessions, gap_sd=0.01, bar_sd=0.003, first="2025-03-03", seed=0):
    """Return (values, kinds, rho): an hourly history with normal gaps and bars, short first bars at √ρ scale."""
    index = hourly_marks(sessions(first, n_sessions))
    kinds, rho = mark_kinds(index)
    rng = np.random.default_rng(seed)
    ell = np.where(kinds == "gap", rng.normal(0, gap_sd, len(kinds)),
                   rng.normal(0, bar_sd * np.sqrt(rho)))
    return from_log_returns(index, ell), kinds, rho


def daily_history(n, sd=0.01, mean=3e-4, seed=0, first="2024-01-02"):
    """Return a daily series on XNYS session dates with normal log returns."""
    days = pd.DatetimeIndex(sessions(first, n + 1))
    ell = np.random.default_rng(seed).normal(mean, sd, n)
    return from_log_returns(days, ell)


def log_steps(paths):
    """Return each path's per-mark log returns from a paths array."""
    return np.diff(np.log(paths), axis=1)
