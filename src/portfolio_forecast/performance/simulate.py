import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

# Close fields in order of preference, compared case-insensitively. An adjusted
# close includes dividends, so it is taken over a plain close when both exist.
_CLOSE_FIELDS = ('adj close', 'adj_close', 'adjclose', 'close')


def _find_close(labels):
    """The preferred close field among `labels`, or None."""
    by_name = {str(label).lower(): label for label in labels}
    return next((by_name[f] for f in _CLOSE_FIELDS if f in by_name), None)


def get_closing_prices(data):
    """Select closing prices from a price frame, as a (date x symbol) frame.

    Parameters
    ----------
    data : pandas.DataFrame
        Prices in one of these layouts:

        - MultiIndex columns with a close field on either level, e.g.
          ``(Price, Ticker)`` from ``yfinance.download`` or
          ``(symbol, field)``.
        - Flat OHLCV columns for one symbol.
        - Flat columns that are already closes, one per symbol.

    Returns
    -------
    pandas.DataFrame
        One column per symbol. For flat OHLCV input, the single close column
        keeps its name. A frame with no close column is returned unchanged.

    Raises
    ------
    KeyError
        If `data` has MultiIndex columns with no close field on any level.

    Notes
    -----
    Field names are matched case-insensitively. An adjusted close (``Adj
    Close``, ``adj_close`` or ``adjclose``) is preferred over ``Close``, since
    it includes dividends; ``yfinance.download(..., auto_adjust=False)``
    returns both.
    """
    if not isinstance(data.columns, pd.MultiIndex):
        # Single-symbol fetch() frames carry OHLCV columns; select close.
        close_col = _find_close(data.columns)
        if close_col is not None:
            return data[[close_col]]
        return data

    # Whichever level carries the close field, take it and let the other
    # level become the columns.
    for level in range(data.columns.nlevels):
        close = _find_close(data.columns.get_level_values(level).unique())
        if close is not None:
            return data.xs(close, axis=1, level=level)

    raise KeyError(f"no close field in columns named {data.columns.names}")


def simulate_buy_and_hold(data, w, br0=1):
    """Backtest a fixed-weight portfolio bought once and never rebalanced.

    The book is bought at the first close and held, so its weights drift
    with prices.

    Parameters
    ----------
    data : pandas.DataFrame
        Historical prices, date-indexed, in any layout `get_closing_prices`
        accepts.
    w : pandas.Series, dict or array-like
        Target weights. A Series or dict is aligned to the symbols by label,
        with missing symbols given zero weight; an array is matched to the
        columns by position. Weights are normalized to sum to 1.
    br0 : float, default 1
        Starting portfolio value (bankroll).

    Returns
    -------
    values : pandas.Series
        Portfolio value at each bar, starting at `br0` on the purchase date.
    dates : pandas.DatetimeIndex
        ``values.index``, returned for use with `performance_report`.

    Raises
    ------
    ValueError
        If the weights sum to zero or less after alignment, or an array of
        weights does not have one entry per symbol.

    Warns
    -----
    UserWarning
        If any symbol is dropped for missing prices, naming each one, how many
        returns it is missing and the date of its first price.

    Notes
    -----
    A gap inside a symbol's history is filled with its last known price, so
    the return across the gap is measured from that price. A symbol with no
    price at the start (e.g. one that listed partway through) is missing
    returns that cannot be filled, and is dropped entirely; the remaining
    weights are renormalized. A Series or dict of weights is realigned after
    the drop; an array must match the symbols that remain.

    See Also
    --------
    performance_report : Metrics for the resulting value series.
    """

    # Select the closing price across all tickers
    closing = get_closing_prices(data)
    # Calculating one-period returns. A return across a gap is measured from the
    # last known price. pandas < 3 did this by default; pandas 3 leaves the gap
    # NaN, so the fill is explicit to keep results the same on either version.
    returns = closing.ffill().pct_change(fill_method=None).iloc[1:]

    # A symbol missing any return (e.g. listed partway through the history) is
    # dropped entirely, which silently changes the book, so say which and why
    n_missing = returns.isna().sum()
    dropped = n_missing[n_missing > 0]
    if len(dropped):
        details = []
        for symbol, n in dropped.items():
            first = closing[symbol].first_valid_index()
            if first is None:
                since = "no prices"
            else:
                since = f"first price {first:%Y-%m-%d}" if hasattr(first, "strftime") else f"first price at {first}"
            details.append(f"{symbol} ({n} of {len(returns)} returns missing, {since})")
        warnings.warn(
            f"Dropped {len(dropped)} symbol(s) with missing prices: {'; '.join(details)}. "
            "Shorten the history to where every symbol has prices to include them.",
            UserWarning, stacklevel=2,
        )
    returns = returns.drop(columns=dropped.index).dropna(axis=0)

    # A dict of weights by symbol means the same as a Series
    if isinstance(w, Mapping):
        w = pd.Series(w, dtype=float)

    # Align weights to match available columns in returns.
    if isinstance(w, pd.Series):
        w = w.reindex(returns.columns).fillna(0.0)
        if w.sum() <= 0:
            raise ValueError("aligned weights sum to zero; cannot normalize")
        w = w / w.sum()
    else:
        w = np.asarray(w, dtype=float)
        if len(w) != returns.shape[1]:
            raise ValueError(
                f"received {len(w)} positional weights for {returns.shape[1]} assets"
            )
        if w.sum() <= 0:
            raise ValueError("weights sum to zero; cannot normalize")
        w = w / w.sum()

    # Cumulative returns
    cum_returns = (returns + 1).cumprod()

    # Use .dot() to sum across assets into a single portfolio series
    values = cum_returns.dot(w * br0)

    # The book is bought at the close before the first return, where it is
    # worth exactly br0. Starting the series there keeps the first period's
    # return, which rebasing on the first return bar would throw away.
    start = closing.index[closing.index.get_loc(returns.index[0]) - 1]
    values = pd.concat([pd.Series([float(br0)], index=[start]), values])

    return values, values.index
