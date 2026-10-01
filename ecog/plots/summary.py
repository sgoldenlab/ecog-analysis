"""
Per-animal summary plot: one point per animal per condition (mean or median within a window), a centre +/- spread
marker per condition, and lines joining each animal across conditions (the designs here are paired).
"""
import numpy as np
import matplotlib.pyplot as plt

from ecog.stats import _stripRecordingSuffix


def _resolveRangeColors(color_spec, range_labels, fallback=None):
    """
    Normalize a bar_color/marker_color spec to one color per range_label, in
    range_labels order: None -> `fallback` repeated; a single color -> that
    color repeated; a list/tuple -> used positionally (must match
    len(range_labels)); a {range_label: color} dict -> looked up per label,
    falling back to `fallback` for a label not given one. Used for both a
    time-range comparison (plot_range_summary) and a condition comparison
    (plot_condition_summary) -- "range_labels" there is just whatever the
    x-axis categories are.
    """
    if color_spec is None:
        return [fallback] * len(range_labels)
    if isinstance(color_spec, dict):
        return [color_spec.get(rlabel, fallback) for rlabel in range_labels]
    if isinstance(color_spec, (list, tuple)):
        return list(color_spec)
    return [color_spec] * len(range_labels)


def plot_condition_stripplot(df, value_col, window, condition_order=None, ylabel="", title="",
                        figsize=(6, 4), box_color="tab:blue", box_edgecolor="black",
                        marker_color=None, marker_size=None, zero_line=True,
                        center="mean", line_color="gray"):
    """
    Strip plot comparing conditions within a single time window -- same
    per-mouse-first aggregation as plot_condition_summary(), but shows every
    animal's own value as a jittered point plus a central-tendency +/-
    spread marker, instead of a bar (mean +/- std rising from a hidden zero)
    or a box (median/IQR/whiskers): useful when the raw per-animal points
    themselves are the most informative thing to show, e.g. with few
    animals per condition, rather than summarizing their spread as a
    box/whiskers.

    `center` picks both each animal's own per-window aggregation and which
    statistic the summary marker shows:
    - "mean" (default): each point is that animal's mean within `window`;
      the marker shows mean +/- std.
    - "median": each point is that animal's median within `window` (more
      robust to a single skewed/spiky segment); the marker shows median +/-
      MAD (median absolute deviation, scaled by 1.4826 so it's comparable to
      std for roughly-normal data). Pairing median with std instead would
      still let one outlier segment blow up the error bar even though the
      point itself is robust to it, defeating the point of choosing median.

    Same arguments as plot_condition_summary() (df/value_col/window/
    condition_order/ylabel/title/figsize, styling via box_color/
    box_edgecolor/marker_color/marker_size, using the same per-condition
    color resolution via _resolveRangeColors()), plus `zero_line` (draws a
    dashed horizontal reference line at y=0). `box_width` is unused here
    (kept for a consistent signature with plot_condition_box()).

    Each animal's points are joined across conditions by a thin line in `line_color` (default gray; None
    gives each animal its own matplotlib cycle colour). Animals missing a condition are joined across the
    conditions they do have.

    Returns (ax, summary) -- one row per condition, with 'mean', 'median',
    'std', 'mad', and 'count' columns regardless of `center` (so both
    statistics are always available to inspect), plus 'count'.
    """
    if center not in ("mean", "median"):
        raise ValueError(f"center must be 'mean' or 'median', got {center!r}")

    lo, hi = window
    window_df = df[(df["timeSeg"] >= lo) & (df["timeSeg"] < hi)].assign(
        mouse=lambda x: _stripRecordingSuffix(x["mouse"])
    )
    condition_order = condition_order or list(dict.fromkeys(window_df["condition"]))

    per_mouse = window_df.groupby(["condition", "mouse"])[value_col].agg(center).reset_index()

    grouped = per_mouse.groupby("condition")[value_col]
    summary = grouped.agg(["mean", "median", "std", "count"]).reindex(condition_order)
    summary["mad"] = (
        grouped.apply(lambda s: (s - s.median()).abs().median() * 1.4826).reindex(condition_order)
    )

    center_col, spread_col = ("mean", "std") if center == "mean" else ("median", "mad")

    box_colors = _resolveRangeColors(box_color, condition_order, fallback=None)
    box_edgecolors = _resolveRangeColors(box_edgecolor, condition_order, fallback=None)
    marker_colors = (
        box_colors if marker_color is None
        else _resolveRangeColors(marker_color, condition_order, fallback="black")
    )
    marker_sizes = (
        [20] * len(condition_order) if marker_size is None
        else _resolveRangeColors(marker_size, condition_order, fallback=20)
    )

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(condition_order))
    if zero_line:
        ax.axhline(0, color="gray", linewidth=1, linestyle="--", zorder=0)

    for i, cond in enumerate(condition_order):
        center_val = summary.loc[cond, center_col]
        yerr = [[summary.loc[cond, spread_col]]] if summary.loc[cond, "count"] > 1 else None
        ax.errorbar(
            [x[i]], [center_val], yerr=yerr, fmt="_", markersize=20, markeredgewidth=2.5,
            capsize=8, elinewidth=2, capthick=2, color=box_edgecolors[i], zorder=4,
        )

    rng = np.random.default_rng(0)
    for i, cond in enumerate(condition_order):
        vals = per_mouse.loc[per_mouse["condition"] == cond, value_col]
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=marker_colors[i],
                   edgecolors=box_edgecolors[i], linewidths=1.5, zorder=3, s=marker_sizes[i])

    wide = per_mouse.pivot(index="mouse", columns="condition", values=value_col).reindex(columns=condition_order)
    for _, row in wide.iterrows():
        ok = row.notna().to_numpy()
        ax.plot(x[ok], row.to_numpy()[ok], color=line_color, alpha=0.4, zorder=10)

    ax.set_xticks(x, condition_order)
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # loop through all children artists and disable clipping
    for child in ax.get_children():
        if hasattr(child, "set_clip_on"):
            child.set_clip_on(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax, summary
