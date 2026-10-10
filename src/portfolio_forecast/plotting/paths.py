"""Fan charts of simulated paths, for one method or several side by side.

Flow
----
1. `_resolve_sims` reads each input: an array (one path per row, starting
   value in column 0) or a `SimulationResult`, whose future marks label the
   x-axis.
2. `_draw_paths` draws one fan on an axes: every path colored by its
   terminal value, the median path and, when every path shares it, the
   starting value.
3. `_label_dates` turns the x-axis ticks into the marks' dates.
4. `_add_colorbar` explains the colors. `plot_simulated_paths` draws one
   panel; `plot_path_comparison` draws one per method on a shared y-axis and
   color scale.

Conventions
-----------
Marks are evenly spaced on the x-axis, one step per mark, so an open mark
(the overnight gap) is its own step and nights, weekends and holidays
don't stretch the paths; only the tick labels are dates. Values are growth
multiples when the paths come from the simulators, which start every path
at 1.0; the y-axis label names the start only when every path shares it.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter, MaxNLocator, StrMethodFormatter

from ..types import SimulationResult

# Tick label of column 0 when only the future marks are known: a
# SimulationResult leaves out the last historical mark the paths start at
START_LABEL = 'Start'


def _resolve_sims(sims, dates):
    """Return (paths, dates, start_label) for one fan chart.

    A SimulationResult gives its paths and, when `dates` is None, its future
    marks: one per column after column 0, which is then labelled
    `START_LABEL`. Otherwise `dates` (if any) labels every column, column 0
    included, as in 0.3.0.
    """
    if isinstance(sims, SimulationResult):
        if dates is None and sims.dates is not None:
            return np.asarray(sims.paths), pd.DatetimeIndex(sims.dates), START_LABEL
        sims = sims.paths
    return np.asarray(sims), dates, None


def _label_dates(ax, dates, n_points, start_label=None):
    """Label the x-axis ticks with the marks' dates.

    `dates` labels every column (`n_points` entries), or with `start_label`
    every column after column 0 (`n_points - 1` entries, the future marks
    of a SimulationResult), column 0 then showing `start_label`. Marks
    all at midnight read as dates (day and year); otherwise each tick shows
    day and time, so an open mark reads as the session's opening time.

    Raises
    ------
    ValueError
        If `dates` has the wrong number of entries for `n_points`.
    """
    # One label per column, or per column after 0 when column 0 shows `start_label`
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    expected = n_points - 1 if start_label is not None else n_points
    if len(dates) != expected:
        if start_label is not None:
            raise ValueError(f'sims.dates has {len(dates)} future marks; expected {expected} '
                             '(one per column of sims.paths after column 0)')
        raise ValueError(f'dates has {len(dates)} entries; expected {n_points} '
                         '(one per column of sims, starting value included)')
    offset = 1 if start_label is not None else 0

    # The x-axis stays one step per mark; only the tick text becomes a date.
    # Midnight labels are dates, so their time of day carries no information
    daily = (dates == dates.normalize()).all()
    fmt = '%b %d\n%Y' if daily else '%b %d\n%H:%M'

    def label(x, pos):
        """Return the tick label at x-position `x`: its mark's date."""
        i = int(round(x))
        if i == 0 and start_label is not None:
            return start_label
        i -= offset
        return dates[i].strftime(fmt) if 0 <= i < len(dates) else ''

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(label))
    ax.set_xlabel('Date', fontsize=11)


def _draw_paths(ax, sims, method, cmap, norm, ylim, dates=None, start_label=None):
    """Draw one fan of simulated paths, the median path and the starting value on `ax`.

    `sims` is an array with one path per row; `cmap` and `norm` map terminal
    values to colors, so several panels can share one scale; `ylim` is the
    y-range. `dates` and `start_label` are passed to `_label_dates`; None
    leaves the ticks as mark numbers. When every path starts at the same
    value s, the y-axis reads "Portfolio Value (start = s)" and a dashed
    line marks s; otherwise it reads "Portfolio Value", with no line.
    """
    # One x step per mark, column 0 (the starting value) at x = 0
    steps = np.arange(sims.shape[1])

    # Draw in order of terminal value, so the highest paths end up on top
    order = np.argsort(sims[:, -1])
    colors = cmap(norm(sims[order, -1]))

    # One LineCollection instead of an ax.plot call per path: much faster at
    # 1,000 paths. Thin, semi-transparent lines keep overlapping paths readable
    paths = LineCollection([np.column_stack((steps, sims[i])) for i in order],
                           colors=colors, linewidths=0.6, alpha=0.35)
    ax.add_collection(paths)

    # A shared start (1.0 for the simulators' paths) names the y-axis scale and gets a
    # reference line; mixed starts get neither, as no one line would be true for every path
    starts = sims[:, 0]
    common_start = bool(np.all(starts == starts[0]))
    if common_start:
        ax.axhline(starts[0], color='#444444', linestyle='--', linewidth=1,
                   label=f'Starting value ({starts[0]:,.2f})')
    ax.plot(steps, np.median(sims, axis=0), color='#d62728', linewidth=2, label='Median path')

    # LineCollection doesn't autoscale the axes, so set the limits manually
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(*ylim)

    ax.set_title(f'{method}: {sims.shape[0]:,} Simulated Portfolio Paths',
                 fontsize=14, fontweight='bold', loc='left')
    ax.set_xlabel('Period', fontsize=11)
    ylabel = f'Portfolio Value (start = {starts[0]:g})' if common_start else 'Portfolio Value'
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    if dates is not None:
        _label_dates(ax, dates, sims.shape[1], start_label)

    ax.grid(True, color='#e0e0e0', linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    ax.legend(loc='upper left', frameon=False)


def _add_colorbar(fig, ax, cmap, norm):
    """Add a colorbar of terminal value beside `ax` (one axes or a list), on the paths' `cmap` and `norm`."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label('Terminal Value', fontsize=11)
    cbar.outline.set_visible(False)


def plot_simulated_paths(sims, method='Monte Carlo', dates=None, figsize=(12, 6.5), dpi=120):
    """Plot a fan chart of simulated portfolio paths.

    Draws every path, colored by its terminal value (viridis: low is purple,
    high is yellow), with the median path and the starting value marked.

    Parameters
    ----------
    sims : SimulationResult or array-like of shape (n_sims, n_marks + 1)
        One path per row, starting value in column 0, as returned by the
        `portfolio_forecast.forecast` simulators. A SimulationResult's
        future marks label the x-axis (column 0 reads ``'Start'``).
        When every path starts at the same value s (1.0 for the
        simulators' paths, which are growth multiples), the y-axis reads
        "Portfolio Value (start = s)" and a dashed line marks s; with
        mixed starts it reads "Portfolio Value", with no line.
    method : str, default 'Monte Carlo'
        Name of the simulation method, shown in the title.
    dates : array-like of datetimes, optional
        X-axis labels, one per column of `sims`: the last historical mark
        (the starting value) followed by the future marks, e.g.
        ``values.index[-1:].append(next_trading_dates(values, n).dates)``.
        Given with a SimulationResult, they replace its labels. None (and
        no SimulationResult dates) labels the marks 0 to n_marks.
    figsize : tuple of float, default (12, 6.5)
        Figure size in inches.
    dpi : int, default 120
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        Returned so the caller can adjust, show or save the figure.

    Raises
    ------
    ValueError
        If `dates` does not have one entry per column of `sims`, or a
        SimulationResult's dates one per column after column 0.

    Notes
    -----
    Marks stay evenly spaced on the x-axis even with dates, so weekends,
    holidays and overnight gaps do not stretch the paths; only the tick
    labels become dates. On intraday paths each session's open mark is its
    own step, labelled at the open.

    See Also
    --------
    plot_path_comparison : Several methods side by side.
    """
    # The paths, their column labels, and column 0's label when only future marks are known
    sims, dates, start_label = _resolve_sims(sims, dates)

    # The color scale spans the terminal values: low is purple, high is yellow
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(sims[:, -1].min(), sims[:, -1].max())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # A 2% margin above and below the paths
    _draw_paths(ax, sims, method, cmap, norm, ylim=(sims.min() * 0.98, sims.max() * 1.02), dates=dates,
                start_label=start_label)
    _add_colorbar(fig, ax, cmap, norm)

    fig.tight_layout()
    return fig, ax


def plot_path_comparison(sims_by_method, dates=None, figsize=None, dpi=120):
    """Plot the fan charts of several simulation methods side by side.

    Each method gets its own panel, drawn as in `plot_simulated_paths`. All
    panels share the y-axis and one color scale, so a height or a color means
    the same portfolio value in every panel and the spreads compare directly.

    Parameters
    ----------
    sims_by_method : dict of str to SimulationResult or array-like
        Method name to its paths (as passed to `plot_simulated_paths`), in
        the order the panels should appear.
    dates : array-like of datetimes, optional
        X-axis labels shared by every panel; see `plot_simulated_paths`. All
        paths must then have ``len(dates)`` columns. None labels each panel
        from its SimulationResult's dates, if it has them.
    figsize : tuple of float, optional
        Figure size in inches. Defaults to ``(6.5 * n_methods, 6)``.
    dpi : int, default 120
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes
        One axes per method, in the order of `sims_by_method`.

    Raises
    ------
    ValueError
        If `dates` does not have one entry per column of every method's
        paths, or a SimulationResult's dates one per column after column 0.

    See Also
    --------
    plot_simulated_paths : One method's fan chart.
    """
    # Each method's paths, column labels and column-0 label
    resolved = {method: _resolve_sims(sims, dates) for method, sims in sims_by_method.items()}
    sims_by_method = {method: paths for method, (paths, _, _) in resolved.items()}
    all_sims = list(sims_by_method.values())
    n = len(all_sims)

    # One color scale and one y-range over every method, so a color or a
    # height means the same value in every panel
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(min(s[:, -1].min() for s in all_sims), max(s[:, -1].max() for s in all_sims))
    ylim = (min(s.min() for s in all_sims) * 0.98, max(s.max() for s in all_sims) * 1.02)

    # One row of panels; constrained layout leaves room for the shared colorbar
    fig, axes = plt.subplots(1, n, figsize=figsize or (6.5 * n, 6), dpi=dpi, sharey=True,
                             layout='constrained', squeeze=False)
    axes = axes[0]

    for ax, (method, (sims, labels, start_label)) in zip(axes, resolved.items(), strict=True):
        _draw_paths(ax, sims, method, cmap, norm, ylim, dates=labels, start_label=start_label)
        # Narrower panels: put the path count on its own line
        ax.set_title(f'{method}\n{sims.shape[0]:,} simulated paths', fontsize=13,
                     fontweight='bold', loc='left')

    # The y-axis is shared, so only the first panel needs its label
    for ax in axes[1:]:
        ax.set_ylabel('')

    _add_colorbar(fig, list(axes), cmap, norm)

    return fig, axes
