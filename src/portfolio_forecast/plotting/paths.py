import matplotlib.pyplot as plt
import numpy as np

# Formats tick labels with thousands separators, and places/labels date ticks
import pandas as pd

# LineCollection draws many lines in one call (much faster than 1,000 ax.plot calls)
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter, MaxNLocator, StrMethodFormatter


# Labels the x-axis (one step per simulated period) with each period's date
def _label_dates(ax, dates, n_points):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(dates) != n_points:
        raise ValueError(f'dates has {len(dates)} entries; expected {n_points} '
                         '(one per column of sims, starting value included)')

    # Periods stay evenly spaced (one per trading period), so weekends, holidays and
    # overnight gaps don't stretch the paths; only the tick labels become dates
    daily = (dates == dates.normalize()).all()
    fmt = '%b %d\n%Y' if daily else '%b %d\n%H:%M'

    def label(x, pos):
        i = int(round(x))
        return dates[i].strftime(fmt) if 0 <= i < len(dates) else ''

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(label))
    ax.set_xlabel('Date', fontsize=11)


# Draws one fan of simulated paths onto an existing axes
def _draw_paths(ax, sims, method, cmap, norm, ylim, dates=None):
    # x-axis values: period 0 (starting value) through the last simulated period
    steps = np.arange(sims.shape[1])

    # Sort paths by terminal value so the best-performing paths are drawn last (on top)
    order = np.argsort(sims[:, -1])

    # Map each path's terminal value to a color: low = dark purple, high = yellow
    colors = cmap(norm(sims[order, -1]))

    # Build one (x, y) line segment per simulated path and add them all at once;
    # thin, semi-transparent lines keep 1,000 overlapping paths readable
    paths = LineCollection([np.column_stack((steps, sims[i])) for i in order],
                           colors=colors, linewidths=0.6, alpha=0.35)
    ax.add_collection(paths)

    # Reference line at the starting value (every path starts at 1, so values are growth multiples)
    start = sims[0, 0]
    ax.axhline(start, color='#444444', linestyle='--', linewidth=1,
               label=f'Starting value ({start:,.2f})')
    # Median portfolio value across all paths at each period
    ax.plot(steps, np.median(sims, axis=0), color='#d62728', linewidth=2, label='Median path')

    # LineCollection doesn't autoscale the axes, so set the limits manually
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(*ylim)

    # Title and axis labels
    ax.set_title(f'{method}: {sims.shape[0]:,} Simulated Portfolio Paths',
                 fontsize=14, fontweight='bold', loc='left')
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Portfolio Value (start = 1)', fontsize=11)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    if dates is not None:
        _label_dates(ax, dates, sims.shape[1])

    # Light gridlines behind the data, and remove the top/right borders for a cleaner look
    ax.grid(True, color='#e0e0e0', linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    # Legend for the reference lines
    ax.legend(loc='upper left', frameon=False)


# Colorbar explaining the path colors (uses the same colormap and normalization as the paths)
def _add_colorbar(fig, ax, cmap, norm):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label('Terminal Value', fontsize=11)
    cbar.outline.set_visible(False)


# Fan chart of simulated portfolio paths
def plot_simulated_paths(sims, method='Monte Carlo', dates=None, figsize=(12, 6.5), dpi=120):
    """Fan chart of simulated portfolio paths.

    Draws every path, colored by its terminal value (viridis: low is purple,
    high is yellow), with the median path and the starting value marked.

    Parameters
    ----------
    sims : array-like of shape (n_sims, n_periods + 1)
        One path per row, starting value in column 0, as returned by the
        `portfolio_forecast.forecast` simulators.
    method : str, default 'Monte Carlo'
        Name of the simulation method, shown in the title.
    dates : array-like of datetimes, optional
        X-axis labels, one per column of `sims`: the last historical date
        (the starting value) followed by the simulated periods, e.g. from
        `portfolio_forecast.utils.next_trading_dates`. None labels the
        periods 0 to n_periods.
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
        If `dates` does not have one entry per column of `sims`.

    Notes
    -----
    Periods stay evenly spaced on the x-axis even with `dates`, so weekends,
    holidays and overnight gaps do not stretch the paths; only the tick
    labels become dates.

    See Also
    --------
    plot_path_comparison : Several methods side by side.
    """
    sims = np.asarray(sims)

    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(sims[:, -1].min(), sims[:, -1].max())

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Small margin on y around the paths
    _draw_paths(ax, sims, method, cmap, norm, ylim=(sims.min() * 0.98, sims.max() * 1.02), dates=dates)
    _add_colorbar(fig, ax, cmap, norm)

    fig.tight_layout()
    return fig, ax


# Side-by-side fan charts for comparing simulation methods
def plot_path_comparison(sims_by_method, dates=None, figsize=None, dpi=120):
    """Fan charts of several simulation methods, side by side.

    Each method gets its own panel, drawn as in `plot_simulated_paths`. All
    panels share the y-axis and one color scale, so a height or a color means
    the same portfolio value in every panel and the spreads compare directly.

    Parameters
    ----------
    sims_by_method : dict of str to array-like
        Method name to its sims array (as passed to `plot_simulated_paths`),
        in the order the panels should appear.
    dates : array-like of datetimes, optional
        X-axis labels shared by every panel; see `plot_simulated_paths`. All
        sims arrays must then have ``len(dates)`` columns.
    figsize : tuple of float, optional
        Figure size in inches. Defaults to ``(6.5 * n_methods, 6)``.
    dpi : int, default 120
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes
        One axes per method, in the order of `sims_by_method`.

    See Also
    --------
    plot_simulated_paths : One method's fan chart.
    """
    sims_by_method = {method: np.asarray(sims) for method, sims in sims_by_method.items()}
    all_sims = list(sims_by_method.values())
    n = len(all_sims)

    # One color scale spanning every method's terminal values
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(min(s[:, -1].min() for s in all_sims), max(s[:, -1].max() for s in all_sims))
    # One y-range spanning every method's paths (with a small margin)
    ylim = (min(s.min() for s in all_sims) * 0.98, max(s.max() for s in all_sims) * 1.02)

    # One row of panels; constrained layout leaves room for the shared colorbar
    fig, axes = plt.subplots(1, n, figsize=figsize or (6.5 * n, 6), dpi=dpi, sharey=True,
                             layout='constrained', squeeze=False)
    axes = axes[0]

    for ax, (method, sims) in zip(axes, sims_by_method.items(), strict=True):
        _draw_paths(ax, sims, method, cmap, norm, ylim, dates=dates)
        # Narrower panels: put the path count on its own line
        ax.set_title(f'{method}\n{sims.shape[0]:,} simulated paths', fontsize=13,
                     fontweight='bold', loc='left')

    # The y-axis is shared, so only the first panel needs its label
    for ax in axes[1:]:
        ax.set_ylabel('')

    # One colorbar for all panels
    _add_colorbar(fig, list(axes), cmap, norm)

    return fig, axes
