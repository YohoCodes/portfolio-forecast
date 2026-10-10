"""Buy-and-hold backtest: prices in, a `BacktestResult` of marks out.

Flow:

1. `get_closing_prices` and `_get_opening_prices` pick the close and open
   fields of any accepted price layout; opens go onto an adjusted close's
   basis with their own bar's factor.
2. `simulate_buy_and_hold` drops bars outside every session and bars
   still in progress, drops held symbols without a price at the fill
   and renormalizes the weights, then has the marks engine
   (`utils.marks._build_marks`) lay out the marks and class the returns.
3. It values the book at every mark at the last price known there, buying
   ``n_i = w_i br0 / P_i(fill)`` shares, so a symbol whose prices stop is
   held at its last price, as cash at 0%, and warns about every
   case that changes the book or the classes.
"""
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..types import BacktestResult
from ..utils.calendar import completed_mask
from ..utils.marks import IRREGULAR, _build_marks, _drop_off_session

# Close fields in order of preference, compared case-insensitively. An adjusted
# close includes dividends, so it is taken over a plain close when both exist.
_CLOSE_FIELDS = ('adj close', 'adj_close', 'adjclose', 'close')


def _find_field(labels, fields):
    """Return the first of `fields` among `labels`, compared
    case-insensitively, or None."""
    # Lower-cased label to label, searched in the order of `fields`
    by_name = {str(label).lower(): label for label in labels}
    return next((by_name[f] for f in fields if f in by_name), None)


def _get_field(data, fields):
    """Return a (date x symbol) frame of the first of `fields` that `data`
    carries, or None. A flat frame (single-symbol OHLCV) gives its one matching column;
    a MultiIndex gives the field from whichever level carries it, with the
    other level as the columns."""
    # Flat columns: one symbol's fields; MultiIndex: search each level for the field
    if not isinstance(data.columns, pd.MultiIndex):
        col = _find_field(data.columns, fields)
        return None if col is None else data[[col]]
    for level in range(data.columns.nlevels):
        match = _find_field(data.columns.get_level_values(level).unique(), fields)
        if match is not None:
            return data.xs(match, axis=1, level=level)
    return None


def _get_opening_prices(data, closing):
    """Return opens laid out like `closing`, on the same price basis, or None
    if `data` has no opens.

    When `closing` is an adjusted close, each bar's open is scaled by that
    bar's adjusted-to-traded close ratio; otherwise the first return would mix
    an adjusted close with a raw open. A bar missing either close carries the
    last ratio forward.
    """
    def align(frame):
        """Return `frame` with the columns of `closing`."""
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
    # Each open goes onto the adjusted basis by its own bar's factor
    # f_d = AdjC_d / C_d, so a dividend lands in the gap
    if adjusted is not None and plain is not None:
        opening = opening * (align(adjusted) / align(plain)).ffill()
    return opening


def _normalize_weights(w, columns):
    """Return weights over `columns`, summing to 1. A Series or dict is
    aligned by symbol, with missing symbols at zero; an array is matched by
    position."""
    # By symbol: align to the columns, missing symbols at zero
    if isinstance(w, Mapping):
        w = pd.Series(w, dtype=float)
    if isinstance(w, pd.Series):
        w = w.reindex(columns).fillna(0.0)
        if w.sum() <= 0:
            raise ValueError("aligned weights sum to zero; cannot normalize")
        return (w / w.sum()).to_numpy()
    # By position: one weight per column
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


def simulate_buy_and_hold(data, w, br0=1, fill="open", calendar="XNYS", tz=None, now=None) -> BacktestResult:
    """Backtest a fixed-weight portfolio bought once and never rebalanced.

    The book is bought at the first bar's open (or close, with
    ``fill="close"``) and held, so its weights drift with prices. Every value
    is the book's worth at prices known at its label, labelled with
    the moment it is true, on the exchange's clock.

    Parameters
    ----------
    data : pandas.DataFrame
        Historical prices, indexed by bar label (start times for intraday
        bars, dates for daily and longer bars), in any layout
        `get_closing_prices` accepts. Opens are read from an ``Open`` field.
    w : pandas.Series, dict or array-like
        Target weights. A Series or dict is aligned to the symbols by label,
        with missing symbols given zero weight; an array is matched to the
        columns by position. Weights are normalized to sum to 1.
    br0 : float, default 1
        Starting portfolio value (bankroll), the value of the first mark.
    fill : {"open", "close"}, default "open"
        Where the book is bought. ``"open"``: at the first bar's open,
        which is the first mark, so the first bar's own move counts; `data`
        must carry opens. ``"close"``: at the first bar's close; session 1
        has no open mark.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code for sessions, opens and closes.
        None is continuous mode for 24/7 data: a constant grid of bar
        size Δ (the most common spacing), no sessions or gaps. Weekday-only
        data off an exchange (NAVs, indices, FX) takes ``"24/5"``.
    tz : str or tzinfo, optional
        Timezone of naive intraday times (in continuous mode, of naive dates
        too) and of a naive `now`. Defaults to this machine's local timezone.
    now : datetime-like, optional
        The current time, for dropping bars still in progress. None uses
        the clock; tests inject it.

    Returns
    -------
    BacktestResult
        ``values``: the book's value at each mark, starting at `br0`.
        ``dates``: the marks, ``values.index``, timezone-aware. Intraday:
        each session's open mark (at its first bar's start), then each bar's
        close; daily and longer: the fill mark (``fill="open"``: the first
        session's open), then each bar's last session close; continuous
        mode: the fill at the first bar's start (``fill="open"``), then each
        bar's start + Δ. ``classes``: the class of each return, indexed by
        the mark where it ends: ``"gap"``, ``"regular"`` or ``"irregular"``.

    Raises
    ------
    ValueError
        If `fill` is unknown; intraday data has no opens;
        ``fill="open"`` is given data without opens (the message names
        ``fill="close"``); the weights sum to zero or less after
        alignment, an array of weights does not have one entry per symbol,
        or no held symbol has a price at the fill; a daily bar is dated on a
        non-session day; no bar falls inside the calendar's sessions;
        no bar has closed by `now`.

    Warns
    -----
    UserWarning
        One per case, each with its count and the symbols or dates involved:
        intraday bars dropped outside every session; held symbols
        dropped for having no price at the fill, with their first price
        date; held symbols whose prices stop early, with their last price
        date; sessions whose first bar is missing; returns made
        irregular by a stale held symbol or a missing scheduled bar;
        in continuous mode, more than 10% irregular returns.

    Notes
    -----
    - Intraday values: each session's open mark V(O_d), then each bar's
      close V(C_{d,k}). The gap g_d = V(O_d)/V(C_{d-1}) - 1 and the bars
      multiply to the session return: (1 + g_d) Π_k (1 + r_{d,k}) =
      V(C_d)/V(C_{d-1}), C_d the session's last close. A session
      opening the instant the previous one closed has no open mark and no
      gap.
    - Missing prices: a gap inside a symbol's history holds its last
      price. A return keeps its class only if every held symbol is priced
      fresh at both ends; otherwise it is irregular: kept in the values,
      left out of Monte Carlo fits.
    - No price at the fill: the symbol is dropped and the weights
      renormalized. The drop looks at the fill bar only, never later prices.
    - Prices stop early: the position is held at its last price, as
      cash at 0%. Survivorship bias: yfinance returns no data for dead
      tickers, so a portfolio built from today's tickers never holds one.
    - Adjusted prices: with an adjusted close, each open is put on the
      adjusted basis with its own bar's factor AdjC/C, so a dividend lands
      in the gap. yfinance intraday data has no adjusted close, so
      intraday backtests omit dividends.
    - Bars outside every session: extended-hours bars, or a 24/7
      asset's weekend bars in a mixed portfolio, are dropped; their moves
      fall into the next gap, so an off-session drawdown is not in the
      values.
    - Bars still in progress at `now` are dropped without a warning.

    See Also
    --------
    performance_report : Metrics for the resulting value series.
    portfolio_forecast.utils.classify_returns : Classes of an external series.
    """
    if fill not in ("open", "close"):
        raise ValueError(f"fill must be 'open' or 'close'; got {fill!r}")
    # Closes and opens (on the close's price basis) per symbol, and weights summing to 1
    closes = get_closing_prices(data)
    opens = _get_opening_prices(data, closes)
    weights = pd.Series(_normalize_weights(w, closes.columns), index=closes.columns)

    # A row with no price for any symbol is not a bar
    priced = closes.notna().any(axis=1)
    if opens is not None:
        priced |= opens.notna().any(axis=1)
        opens = opens[priced]
    closes = closes[priced]
    if closes.empty:
        raise ValueError("data has no prices")

    # Bars outside every session go first, before anything looks at
    # "the first bar"; then bars still in progress at `now`, silently
    closes, opens = _drop_off_session(closes, opens, calendar, tz, stacklevel=3)
    done = np.asarray(completed_mask(closes.index, calendar, tz, now))
    if not done.any():
        raise ValueError(f"No bar has closed by now ({now if now is not None else 'the current time'})")
    closes = closes[done]
    opens = opens[done] if opens is not None else None
    if fill == "open" and opens is None:
        raise ValueError("fill='open' needs opening prices; pass data with an open field, or use fill='close'")

    # A held symbol with no price at the fill is dropped and the other
    # weights renormalized
    weights = _drop_unpriced_at_fill(weights, opens if fill == "open" else closes, closes, fill)
    held = weights.index[weights > 0]

    # The marks, each mark's bar and whether it is an open, and every return's class
    marks = _build_marks(opens, closes, held, calendar, tz, fill, stacklevel=3)

    # Each value uses the last price known at its mark: the symbol's own (open at an open
    # mark, else close), else its last one, so after its last price it is cash at 0%
    own = closes[held].to_numpy()[marks.bar]
    if opens is not None and marks.is_open.any():
        own[marks.is_open] = opens[held].to_numpy()[marks.bar[marks.is_open]]
    _warn_stopped_early(own, marks.labels, held)
    px = pd.DataFrame(own).ffill().to_numpy()
    # Bought once at the fill, n_i = w_i br0 / P_i(fill), never rebalanced
    shares = weights[held].to_numpy() * br0 / px[0]
    # The book's value at every mark; the first is br0 exactly, free of floating-point round trip
    values = pd.Series(px @ shares, index=marks.labels)
    values.iloc[0] = float(br0)

    _warn_irregular(marks)
    return BacktestResult(values, values.index, marks.classes)


def _drop_unpriced_at_fill(weights, at_fill, closes, fill):
    """Return `weights` without the held symbols that have no price at the
    fill bar, renormalized, warning about each.

    `at_fill` is the frame the fill reads (opens or closes); only its first
    bar is looked at, so later prices never decide a drop.
    """
    # Held symbols with no price on the fill bar; if none is priced, nothing can be bought
    held = weights.index[weights > 0]
    missing = held[at_fill[held].iloc[0].isna().to_numpy()]
    if not len(missing):
        return weights
    if len(missing) == len(held):
        raise ValueError(
            f"No held symbol has a price at the fill ({fill} of the first bar, {closes.index[0]}); "
            "start the history where the book can be bought"
        )
    details = []
    for symbol in missing:
        first = at_fill[symbol].first_valid_index()
        details.append(f"{symbol} (first price {first})" if first is not None else f"{symbol} (no prices)")
    warnings.warn(
        f"Dropped {len(missing)} symbol(s) with no {fill} price at the fill, the first bar "
        f"({closes.index[0]}): {'; '.join(details)}. The remaining weights are renormalized; "
        "start the history where every symbol has prices to include them",
        UserWarning, stacklevel=3,
    )
    weights = weights.drop(missing)
    return weights / weights.sum()


def _warn_stopped_early(own, labels, held):
    """Warn about held symbols whose own prices at the marks stop before the
    last mark: each is held at its last price as cash at 0% from there.
    `own` is the (mark x held symbol) array of own prices, NaN where
    a symbol has none; "stopped" is judged on the whole history, as the
    marks engine judges freshness."""
    # Each symbol's last mark with its own price; stopped if that is before the end
    priced = ~np.isnan(own)
    last = len(labels) - 1 - np.argmax(priced[::-1], axis=0)
    stopped = [f"{symbol} (last price {labels[i]})" for symbol, i in zip(held, last, strict=True)
               if i < len(labels) - 1]
    if stopped:
        warnings.warn(
            f"{len(stopped)} held symbol(s) stop before the end of the history and are held as cash at 0% "
            f"from their last price: {'; '.join(stopped)}. Delisted tickers are often missing from "
            "data sources altogether, so backtests of today's tickers carry survivorship bias",
            UserWarning, stacklevel=3,
        )


def _warn_irregular(marks):
    """Warn with the count of sessions missing their first bar and of
    other irregular returns; both stay in the values and are
    left out of Monte Carlo fits."""
    # Late open marks first, then every other irregular return
    n_late = int(marks.late_open.sum())
    if n_late:
        first = marks.labels[1:][marks.late_open][0]
        warnings.warn(
            f"{n_late} session(s) have no bar at the open, first {first}: each open mark is the first "
            "available bar's open, labelled at its start, and the return into it is irregular",
            UserWarning, stacklevel=3,
        )
    other = (marks.classes == IRREGULAR).to_numpy() & ~marks.late_open
    if other.any():
        warnings.warn(
            f"{other.sum()} return(s) are irregular, first ending {marks.labels[1:][other][0]}: a held symbol "
            "is stale (no own price) at either end, or a scheduled bar is missing for every symbol. "
            "They stay in the values and metrics and are left out of Monte Carlo fits",
            UserWarning, stacklevel=3,
        )
