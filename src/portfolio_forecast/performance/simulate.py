import numpy as np
import pandas as pd


def get_closing_prices(data):
    """Select closing prices from a price frame, as a (date x symbol) frame.

    Parameters
    ----------
    data : pandas.DataFrame
        Prices in one of these layouts:

        - MultiIndex columns with a ``close`` field on either level, matched
          case-insensitively, e.g. ``(Price, Ticker)`` from
          ``yfinance.download`` or ``(symbol, field)``.
        - Flat OHLCV columns for one symbol.
        - Flat columns that are already closes, one per symbol.

    Returns
    -------
    pandas.DataFrame
        One column per symbol. For flat OHLCV input, the single ``close``
        column keeps its name. A frame with no ``close`` column is returned
        unchanged.

    Raises
    ------
    KeyError
        If `data` has MultiIndex columns with no ``close`` field on any level.
    """
    if not isinstance(data.columns, pd.MultiIndex):
        # Single-symbol fetch() frames carry OHLCV columns; select close.
        close_cols = [col for col in data.columns if str(col).lower() == 'close']
        if close_cols:
            close_col = close_cols[0]
            return data[[close_col]]
        return data

    # Whichever level carries the close field, take it and let the other
    # level become the columns.
    for level in range(data.columns.nlevels):
        labels = data.columns.get_level_values(level).unique()
        close = [label for label in labels if str(label).lower() == 'close']
        if close:
            return data.xs(close[0], axis=1, level=level)

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
    w : pandas.Series or array-like
        Target weights. A Series is aligned to the symbols by label, with
        missing symbols given zero weight; an array is matched to the columns
        by position. Weights are normalized to sum to 1.
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

    Notes
    -----
    Symbols with any missing return are dropped entirely, then any bar with
    a missing return is dropped. A Series of weights is realigned after the
    drop; an array must match the symbols that remain.

    See Also
    --------
    simulate_rebalancing_MVO : Re-weights the book on a schedule.
    performance_report : Metrics for the resulting value series.
    """

    # Select the closing price across all tickers
    closing = get_closing_prices(data)
    # Calculating cumulative returns
    returns = closing.pct_change().iloc[1:]
    returns = returns.dropna(axis=1).dropna(axis=0)

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


def _hold(closing, start_i, end_i, held, w_held):
    """Hold a book bought at close[start_i] through close[end_i].

    Returns the book's growth multiple at every bar of the span, and its
    weights at close[end_i] after prices have moved them, laid out over every
    column of `closing` (zero for names not held). An empty book is cash: the
    multiple stays flat and it carries no weights.
    """
    multiple = np.ones(end_i - start_i + 1)
    drifted = np.zeros(closing.shape[1])
    if not held.any():
        return multiple, drifted

    # Gaps after the fill hold the last quote, the usual reading of a missing bar
    px = closing.iloc[start_i : end_i + 1, np.flatnonzero(held)].ffill()

    # Dividing by the fill row rebases every price to 1, so each column is the
    # value of that position per unit of capital put into the book
    positions = (px / px.iloc[0]).to_numpy() * w_held
    multiple = positions.sum(axis=1)
    drifted[held] = positions[-1] / multiple[-1]
    return multiple, drifted


def _buyable(w_new, quotes):
    """The book a suggestion actually buys, over every column of `closing`.

    A name with no quote on the fill bar cannot be bought, so it is left out
    and the rest is renormalized to be fully invested, absorbing the dropped
    names and any drift in the stored weights. An empty suggestion is cash:
    all zeros.
    """
    book = np.where((w_new != 0) & quotes.notna().to_numpy(), w_new, 0.0)
    total = book.sum()
    return book / total if total > 0 else np.zeros(len(book))


def _turnover(a, b):
    """One-way turnover between two books: the fraction of the portfolio that
    changes hands moving from one to the other, in [0, 1].

    Cash is counted as the weight a book leaves uninvested, so selling
    everything into cash, or buying out of it, is a turnover of 1.
    """
    cash_gap = abs(a.sum() - b.sum())
    return 0.5 * (np.abs(a - b).sum() + cash_gap)


def simulate_rebalancing_MVO(data, w, beta=0, br0=1):
    """Backtest a book re-weighted on a schedule, with a no-trade band.

    At each rebalance timestamp the suggested book is compared with the
    weights actually held, and traded only if enough of the portfolio would
    change hands. Between trades the book is left alone, so its weights
    drift with prices.

    Parameters
    ----------
    data : pandas.DataFrame
        Historical prices, date-indexed, in any layout `get_closing_prices`
        accepts.
    w : pandas.DataFrame
        Suggested weights, one row per symbol and one column per rebalance
        timestamp, e.g. from a mean-variance optimizer (not included in this
        package). Each column is the book to buy at that bar's close and must
        be either all weights or all NaN; an all-NaN column means no
        suggestion, so nothing trades. Symbols missing from a column are
        unheld. Every column label must be a timestamp in the price index.
    beta : float, default 0
        No-trade band on one-way turnover, in [0, 1]. The book trades only if
        turnover exceeds `beta`: 0.10 reads "trade only if more than 10% of
        the portfolio would move". 0 trades at every suggestion; 1 never
        trades after the first.
    br0 : float, default 1
        Starting portfolio value (bankroll).

    Returns
    -------
    values : pandas.Series
        Portfolio value at every bar of the price index. Flat at `br0` until
        the first trade.
    dates : pandas.DatetimeIndex
        ``values.index``.
    decisions : pandas.Series
        One entry per rebalance timestamp: ``'traded'``, ``'held'`` (inside
        the no-trade band) or ``'skipped'`` (no suggestion).

    Raises
    ------
    TypeError
        If `w` is not a DataFrame.
    ValueError
        If `beta` is outside [0, 1], a column of `w` is partly NaN, a
        rebalance timestamp is not in the price index, or the timestamps are
        not strictly increasing.

    Notes
    -----
    One-way turnover between the suggested book and the drifted book held is
    ``0.5 * (sum(|w_new - w_drifted|) + |cash_new - cash_drifted|)``, the
    fraction of the portfolio that changes hands; cash is the weight a book
    leaves uninvested.

    A symbol with no quote on the fill bar cannot be bought: it is dropped
    and the rest of the book renormalized to be fully invested. Missing
    quotes while a book is held carry the last price forward. There are no
    transaction costs.

    See Also
    --------
    simulate_buy_and_hold : Buys once and never rebalances.
    performance_report : Metrics for the resulting value series.
    """
    if not isinstance(w, pd.DataFrame):
        raise TypeError(
            f"expected a (symbol x rebalance) weight frame, got {type(w).__name__}"
        )

    # beta is a threshold on one-way turnover, which lies in [0, 1]
    if beta < 0 or beta > 1:
        raise ValueError(
            f"Unsupported beta -> {beta}. "
            f"beta is a fraction of the portfolio traded, so it must be "
            f"between 0 and 1"
        )

    # Select the closing price across all tickers
    closing = get_closing_prices(data)

    # A column is either a full suggestion or no suggestion at all. A partly
    # NaN column has no single reading, so it is refused rather than guessed.
    solved = w.notna().all(axis=0)
    skipped = w.isna().all(axis=0)
    if not (solved | skipped).all():
        partial = w.columns[~(solved | skipped)]
        raise ValueError(
            f"{len(partial)} rebalance column(s) are partly NaN, starting at "
            f"{partial[0]}; a column must be all weights or all NaN"
        )

    # Align weights to the columns in closing, by ticker rather than by
    # position. A name the optimizer never saw is unheld on a solved date;
    # skipped columns stay NaN, so they are not mistaken for a move to cash.
    w = w.reindex(index=closing.columns)
    w.loc[:, solved] = w.loc[:, solved].fillna(0.0)

    # Each column is labelled with the bar it fills at, so the rebalance points
    # are looked up in the price index instead of being assumed.
    rb_idx = closing.index.get_indexer(w.columns)
    if (rb_idx < 0).any():
        missing = w.columns[rb_idx < 0]
        raise ValueError(
            f"{len(missing)} rebalance timestamp(s) absent from the price index, "
            f"starting at {missing[0]}"
        )
    if not (np.diff(rb_idx) > 0).all():
        raise ValueError("rebalance timestamps must be strictly increasing")

    # Nothing is held before the first trade, so the book sits flat at br0
    values = pd.Series(float(br0), index=closing.index)
    decisions = pd.Series("skipped", index=w.columns, dtype=object)

    # Running portfolio value at the last fill, carried so the periods compound
    V = float(br0)
    # The book currently held: its fill bar, which names it bought, and their
    # weights at the fill. None until the first trade.
    fill_i = held = w_held = None

    for t, i in enumerate(rb_idx):
        if skipped.iloc[t]:
            continue

        target = _buyable(w.iloc[:, t].to_numpy(dtype=float), closing.iloc[i])

        if fill_i is not None:
            multiple, drifted = _hold(closing, fill_i, i, held, w_held)

            # No-trade band, measured against the weights actually held after
            # drift: skip the trade if too little of the portfolio would move
            if _turnover(target, drifted) <= beta:
                decisions.iloc[t] = "held"
                continue

            # Trading: close out the book held since the last fill. The fill
            # bar belongs to both books; the new one rebases off it, so no
            # return is counted twice.
            values.iloc[fill_i : i + 1] = V * multiple
            V = float(V * multiple[-1])

        decisions.iloc[t] = "traded"

        # Buy the new book at this bar's close
        held = target != 0
        w_held = target[held]
        fill_i = i

    # The last book is held to the end of the sample
    if fill_i is not None:
        multiple, _ = _hold(closing, fill_i, len(closing) - 1, held, w_held)
        values.iloc[fill_i:] = V * multiple

    return values, values.index, decisions
