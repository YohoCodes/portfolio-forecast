import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..utils.bar_times import _daily_sessions, close_times

# Close fields in order of preference, compared case-insensitively. An adjusted
# close includes dividends, so it is taken over a plain close when both exist.
_CLOSE_FIELDS = ('adj close', 'adj_close', 'adjclose', 'close')


def _find_field(labels, fields):
    """The first of `fields` among `labels`, compared case-insensitively, or
    None."""
    by_name = {str(label).lower(): label for label in labels}
    return next((by_name[f] for f in fields if f in by_name), None)


def _get_field(data, fields):
    """A (date x symbol) frame of the first of `fields` that `data` carries,
    or None. A flat frame (single-symbol OHLCV) gives its one matching column;
    a MultiIndex gives the field from whichever level carries it, with the
    other level as the columns."""
    if not isinstance(data.columns, pd.MultiIndex):
        col = _find_field(data.columns, fields)
        return None if col is None else data[[col]]
    for level in range(data.columns.nlevels):
        match = _find_field(data.columns.get_level_values(level).unique(), fields)
        if match is not None:
            return data.xs(match, axis=1, level=level)
    return None


def _get_opening_prices(data, closing):
    """Opens laid out like `closing`, on the same price basis, or None if
    `data` has no opens.

    When `closing` is an adjusted close, each bar's open is scaled by that
    bar's adjusted-to-traded close ratio; otherwise the first return would mix
    an adjusted close with a raw open. A bar missing either close carries the
    last ratio forward.
    """
    def align(frame):
        if not isinstance(data.columns, pd.MultiIndex):
            # One symbol: name each field's column after the close column
            return frame.set_axis(closing.columns, axis=1)
        return frame.reindex(columns=closing.columns)

    opening = _get_field(data, ('open',))
    if opening is None:
        return None
    opening = align(opening)

    # get_closing_prices takes an adjusted close over a plain one, so both
    # being present means `closing` is adjusted
    adjusted = _get_field(data, _CLOSE_FIELDS[:-1])
    plain = _get_field(data, ('close',))
    if adjusted is not None and plain is not None:
        opening = opening * (align(adjusted) / align(plain)).ffill()
    return opening


def _normalize_weights(w, columns):
    """Weights over `columns`, summing to 1. A Series or dict is aligned by
    symbol, with missing symbols at zero; an array is matched by position."""
    if isinstance(w, Mapping):
        w = pd.Series(w, dtype=float)
    if isinstance(w, pd.Series):
        w = w.reindex(columns).fillna(0.0)
        if w.sum() <= 0:
            raise ValueError("aligned weights sum to zero; cannot normalize")
        return (w / w.sum()).to_numpy()
    w = np.asarray(w, dtype=float)
    if len(w) != len(columns):
        raise ValueError(
            f"received {len(w)} positional weights for {len(columns)} assets"
        )
    if w.sum() <= 0:
        raise ValueError("weights sum to zero; cannot normalize")
    return w / w.sum()


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
    closes = _get_field(data, _CLOSE_FIELDS)
    if closes is not None:
        return closes
    if not isinstance(data.columns, pd.MultiIndex):
        # Flat columns without a close field are taken to be closes already
        return data
    raise KeyError(f"no close field in columns named {data.columns.names}")


def simulate_buy_and_hold(data, w, br0=1, fill="close", calendar="XNYS", tz=None):
    """Backtest a fixed-weight portfolio bought once and never rebalanced.

    The book is bought at the first bar's close (or open, with
    ``fill="open"``) and held, so its weights drift with prices. Every value
    is labelled with the moment it is true.

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
    fill : {"close", "open"}, default "close"
        Where the book is bought. ``"close"``: at the first bar's close, from
        closes alone. ``"open"``: at the first bar's open, which also counts
        the first bar's own move; `data` must carry opens, and bars must be
        intraday or daily.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code for labelling values with bar
        close times (see `close_times`). None keeps the input's labels, for
        data off any exchange calendar; ``fill="open"`` needs a calendar.
    tz : str or tzinfo, optional
        Timezone that naive intraday times are in. Defaults to this machine's
        local timezone.

    Returns
    -------
    values : pandas.Series
        Portfolio value, starting at `br0` at the purchase. With
        ``fill="close"`` there is one value per bar: intraday values are
        labelled with each bar's close time, daily and longer ones keep their
        dates. With ``fill="open"`` the first value is labelled with the first
        bar's start (daily: its session's open), then each bar's close
        follows, one more value than bars; daily labels are then
        timezone-aware session times.
    dates : pandas.DatetimeIndex
        ``values.index``, returned for use with `performance_report`.

    Raises
    ------
    ValueError
        If the weights sum to zero or less after alignment, an array of
        weights does not have one entry per symbol, `fill` is unknown, or
        ``fill="open"`` is given data without opens, without a calendar, or
        with weekly or longer bars. Also if a bar falls outside the calendar's
        sessions.

    Warns
    -----
    UserWarning
        If any symbol is dropped for missing prices, naming each one: with
        ``fill="close"``, how many returns it is missing and the date of its
        first price; with ``fill="open"``, that it has no first open.

    Notes
    -----
    A gap inside a symbol's history is filled with its last known price, so
    the return across the gap is measured from that price. A symbol with no
    price at the start (e.g. one that listed partway through) can't be bought,
    and is dropped entirely; the remaining weights are renormalized. A Series
    or dict of weights is realigned after the drop; an array must match the
    symbols that remain. Whether a symbol is dropped never depends on prices
    after the purchase.

    yfinance and Interactive Brokers label bars with their start time. A bar's
    close is only known at its end, which is why intraday values move to
    ``close_times``. A daily date names the whole session, so it is kept with
    ``fill="close"``.

    With an adjusted close, ``fill="open"`` scales each open by its bar's
    adjusted-to-traded close ratio, so both are on the same basis.

    See Also
    --------
    performance_report : Metrics for the resulting value series.
    close_times : The close label of each bar.
    """
    if fill not in ("close", "open"):
        raise ValueError(f"fill must be 'close' or 'open'; got {fill!r}")
    if fill == "open":
        return _buy_at_open(data, w, br0, calendar, tz)

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

    w = _normalize_weights(w, returns.columns)

    # Cumulative returns
    cum_returns = (returns + 1).cumprod()

    # Use .dot() to sum across assets into a single portfolio series
    values = cum_returns.dot(w * br0)

    # The book is bought at the close before the first return, where it is
    # worth exactly br0. Starting the series there keeps the first period's
    # return, which rebasing on the first return bar would throw away.
    start = closing.index[closing.index.get_loc(returns.index[0]) - 1]
    values = pd.concat([pd.Series([float(br0)], index=[start]), values])

    # Intraday values are true at their bar's close, not its start. The grid is
    # learned from every bar, so the closes are found before any row drop.
    index = closing.index
    if calendar is not None and not (index == index.normalize()).all():
        closes = pd.Series(close_times(index, calendar, tz), index=index)
        values.index = pd.DatetimeIndex(closes.loc[values.index])

    return values, values.index


def _buy_at_open(data, w, br0, calendar, tz):
    """simulate_buy_and_hold with fill="open": br0 at the first bar's open,
    then the book's value at every bar's close."""
    if calendar is None:
        raise ValueError(
            "fill='open' needs a calendar to label the fill and the closes"
        )
    closing = get_closing_prices(data)
    opening = _get_opening_prices(data, closing)
    if opening is None:
        raise ValueError(
            "fill='open' needs opening prices; pass data with an open field, "
            "or use fill='close'"
        )

    # The fill and each close need their own label. A daily date has only one,
    # so it is read as its session's open and close.
    index = closing.index
    if (index == index.normalize()).all():
        sessions = _daily_sessions(index, calendar)
        if sessions is None:
            raise ValueError(
                "fill='open' supports intraday and daily bars; weekly and "
                "longer bars aren't supported"
            )
        fill_label, close_labels = sessions[0][:1], sessions[1]
    else:
        fill_label, close_labels = index[:1], close_times(index, calendar, tz)

    # A symbol without a first open can't be bought. That is known at the
    # fill; a gap later on is not, so it doesn't count here.
    unquoted = opening.columns[opening.iloc[0].isna()]
    if len(unquoted):
        warnings.warn(
            f"Dropped {len(unquoted)} symbol(s) with no open on the first bar: "
            f"{', '.join(map(str, unquoted))}",
            UserWarning, stacklevel=3,
        )
    closing = closing.drop(columns=unquoted)
    opening = opening.drop(columns=unquoted)
    w = _normalize_weights(w, closing.columns)

    # The prices the book passes through: the fill open, then every close.
    # A gap holds the last quote (the fill price before the first close).
    px = pd.concat([opening.iloc[[0]], closing]).ffill()
    at_closes = (px / px.iloc[0]).to_numpy()[1:] @ w * br0

    values = pd.Series(np.concatenate([[float(br0)], at_closes]),
                       index=fill_label.append(close_labels))
    return values, values.index
