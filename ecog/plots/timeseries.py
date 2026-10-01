"""
Binned time series of a long-format table (columns timeSeg, mouse, condition, <value>), as returned by
ecog.aggregate.aggregateAligned: per condition, the across-animal mean (of each animal's per-bin median)
with a SEM band; a single recording is binned directly with its own spread.
"""
import numpy as np
import matplotlib.pyplot as plt

from ecog.plots.timeline import annotateTimeline


def _binTimeseries(df, value_col: str, binning_s, error_metric=None):
    """
    Bin `df` ('timeSeg', 'mouse', 'condition', `value_col`) into
    `binning_s`-second bins per condition, returning one row per
    (timeBin, condition) with 'mean'/'std'/'count'.

    When only one 'mouse' contributes to `df` (a single recording), bins are
    aggregated directly over its raw segments -- 'count' is that recording's
    own segment count within the bin.

    When several mice contribute (aligned/averaged data from
    aggregate_aligned()), each mouse's own *median* within a bin is computed
    first -- median rather than mean so that the handful of raw segments a
    single mouse contributes to a bin can't be dominated by one or two
    extreme values (e.g. a band ratio spiking when its noise-band denominator
    happens to be near zero for one STFT segment); a mouse that happens to
    land more raw segments in a bin (e.g. an edge effect of alignment offset)
    also doesn't outweigh one that lands fewer. 'mean'/'count' are then the
    across-mice mean/count (count = number of mice in that bin) of those
    per-mouse bin medians.

    `error_metric` controls what the returned 'std' column actually holds:
    'std' is the raw standard deviation of whatever was aggregated into each
    bin; 'sem' (`std / sqrt(count)`) is the standard error of the *mean*
    plotted for that bin, which shrinks as more samples back it up rather
    than describing how spread out those samples are. Left at the default
    (None), it's chosen automatically per case: 'sem' when several mice
    contribute -- the plotted value there is itself an average of each
    mouse's own median, so SEM reflects how precisely that average is known,
    which is what "the spread of the means" calls for -- and 'std' for a
    single recording, where each bin's raw segments aren't being reported as
    an average of samples, so 'std' keeps meaning "how noisy is this window."
    Pass 'std' or 'sem' explicitly to override either case.
    """
    binned = df.copy()
    binned['timeBin'] = (binned['timeSeg'] // binning_s) * binning_s

    across_mice = binned['mouse'].nunique() > 1
    if across_mice:
        binned = binned.groupby(['timeBin', 'condition', 'mouse'])[value_col].median().reset_index()
    if error_metric is None:
        error_metric = 'sem' if across_mice else 'std'

    grouped = binned.groupby(['timeBin', 'condition'])[value_col].agg(['mean', 'std', 'count']).reset_index()
    if error_metric == 'sem':
        grouped['std'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['std'] = grouped['std'].fillna(0)
    return grouped


def plotBinnedTimeseries(
        df, value_col: str, ylabel: str, binning_s=300, smoothing_s=None, events=None,
        tick_step=300, colors=None, line_colors=None, xlim=None, ylim=None, figsize=(5, 3), ax=None,
        legend_pos="upper left", legend_bbox=(1, 1), show_band=True, title="", error_metric=None,
        x_limits:tuple|None=None
        ):
    """
    Bin a long-format dataframe (as returned by getRatioData/getBSRData --
    columns 'timeSeg', 'mouse', 'condition', `value_col`) into
    `binning_s`-second bins and plot each condition's mean over time, +/-
    shaded error where more than one segment/mouse contributes to a bin.
    Works for a single recording or several aligned/averaged recordings alike
    -- see `_binTimeseries` for how the two cases differ (within-recording
    spread vs. across-animal spread) and what `error_metric` ('std'/'sem',
    default auto-selected per case) controls.

    `smoothing_s` optionally applies an exponentially-weighted moving average
    across bins (independently per condition) before plotting the mean line,
    with a halflife of `smoothing_s` seconds -- a bin `smoothing_s` seconds
    in the past contributes half as much as the current one, decaying
    further back, so a single stale/noisy bin doesn't linger and drag on
    later averages the way a flat window would. This is a smoothing pass
    over the already-binned means, independent of `binning_s`: e.g.
    binning_s=300 (5 min bins) with smoothing_s=600 still updates every 5
    minutes but each point leans more on the last ~10 minutes than on data
    from an hour ago. Only the plotted mean line is smoothed -- the shaded
    error band still reflects each bin's own raw (unsmoothed) value, so it
    isn't blurred too.
    Assumes bins are evenly spaced (irregular gaps aren't specially handled).

    `line_colors` optionally maps a condition to its line/band color (e.g.
    `line_colors={"SAL": "tab:blue", "CNO": "tab:orange"}`); a condition not
    given a color there keeps matplotlib's default color-cycle choice --
    either way the shaded error band always matches its own line's actual
    color. `colors` is unrelated -- it's forwarded to annotateTimeline for
    the vertical event lines, not the data.

    `xlim`/`ylim` optionally fix the axis ranges instead of autoscaling to
    the data (`xlim` is applied before annotateTimeline computes its ticks,
    so custom ticks still span the requested range).

    `show_band=False` hides the shaded error band entirely, plotting just the
    mean line.

    Shared by plotRatioSNR (delta/alpha SNR), plotBandPower and
    plotBSRTimeseries so the binning/plotting logic isn't duplicated per metric.
    """
    grouped = _binTimeseries(df, value_col, binning_s, error_metric=error_metric)
    if x_limits:
        grouped = grouped.query(f"timeBin>={x_limits[0]} & timeBin<={x_limits[1]}")

    if smoothing_s:
        halflife_bins = smoothing_s / binning_s
        for condition in grouped['condition'].unique():
            cond_rows = grouped[grouped['condition'] == condition].sort_values('timeBin')
            smoothed = cond_rows['mean'].ewm(halflife=halflife_bins).mean()
            grouped.loc[smoothed.index, 'mean'] = smoothed.values

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    ax.margins(x=0.05)

    line_colors = line_colors or {}
    conditions = grouped['condition'].unique()
    for condition in conditions:
        condition_data = grouped[grouped['condition'] == condition]
        n = condition_data['count'].max()
        label = f"{condition} (n={n})" if n > 1 else condition
        line, = ax.plot(condition_data['timeBin'], condition_data['mean'], label=label, 
                         marker='', lw=3,
                         #marker='o',
                         color=line_colors.get(condition)
                        )
        if show_band:
            ax.fill_between(
                condition_data['timeBin'],
                condition_data['mean'] - condition_data['std'],
                condition_data['mean'] + condition_data['std'],
                alpha=0.2, color=line.get_color(),
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)

    annotateTimeline(ax, events=events, tick_step=tick_step, colors=colors,
                      legend_pos=legend_pos, legend_bbox=legend_bbox)
    if not events and len(conditions) > 1:
        ax.legend(bbox_to_anchor=legend_bbox, loc=legend_pos)

    return ax


def plotRatioSNR(ratio_df, ylabel, value_col='ratio', **kwargs):
    """plotBinnedTimeseries over a getRatioData() dataframe -- used for both delta and alpha SNR."""
    return plotBinnedTimeseries(ratio_df, value_col=value_col, ylabel=ylabel, **kwargs)


def plotBandPower(power_df, ylabel="Band power", **kwargs):
    """plotBinnedTimeseries over a getRatioData(..., value_name='power') dataframe -- from bandPowerToCSV."""
    return plotBinnedTimeseries(power_df, value_col='power', ylabel=ylabel, **kwargs)


binTimeseries = _binTimeseries   # public name: the binned mean/error table behind plotBinnedTimeseries
