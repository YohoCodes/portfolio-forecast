import math

# US equity session: 252 trading days, 6.5 hours (390 minutes) per day
TRADING_DAYS = 252
MINUTES_PER_DAY = 6.5 * 60
SECONDS_PER_DAY = MINUTES_PER_DAY * 60


def _bars_per_year(bar_seconds):
    """Bars in a year of regular-hours sessions for a bar of `bar_seconds`.

    Bars are counted with ceil, not division: a bar size that does not divide
    the 390-minute session still opens a bar for the leftover stub. On hourly
    bars 9:30-16:00 yields 7 bars (a 30-minute stub at the open plus six full
    hours), not 6.5.
    """
    return TRADING_DAYS * math.ceil(SECONDS_PER_DAY / bar_seconds)


# Bars per year for each supported bar size, used to annualize. Keys are the
# canonical bar-size strings; the reports also accept other spellings of the
# same sizes ('1D', '5min', a Timedelta) by parsing them.
PERIODS_PER_YEAR = {
    '1 secs':   _bars_per_year(1),
    '5 secs':   _bars_per_year(5),
    '10 secs':  _bars_per_year(10),
    '15 secs':  _bars_per_year(15),
    '30 secs':  _bars_per_year(30),
    '1 min':    _bars_per_year(60),
    '2 mins':   _bars_per_year(2 * 60),
    '3 mins':   _bars_per_year(3 * 60),
    '5 mins':   _bars_per_year(5 * 60),
    '10 mins':  _bars_per_year(10 * 60),
    '15 mins':  _bars_per_year(15 * 60),
    '20 mins':  _bars_per_year(20 * 60),
    '30 mins':  _bars_per_year(30 * 60),
    '1 hour':   _bars_per_year(60 * 60),
    '2 hours':  _bars_per_year(2 * 60 * 60),
    '3 hours':  _bars_per_year(3 * 60 * 60),
    '4 hours':  _bars_per_year(4 * 60 * 60),
    '8 hours':  _bars_per_year(8 * 60 * 60),
    '1 day':    TRADING_DAYS,
    '1 week':   52,
    '1 month':  12,
}
