"""
Figures for epoch-table analyses (ecog.epochs.buildEpochTable): conventional band measures, the aperiodic
decomposition, a compact summary inset, and the burst-suppression control.

Every per-animal panel uses one convention: one marker per animal, thin gray lines joining the same animal
across conditions (paired design), and a black mean +/- SEM marker per condition. Figures are returned, not
saved; save with ecog.outputs.saveFigure (pass bbox_inches=None to keep the exact canvas size).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ecog.epochs import aperiodicCurve, meanSpectrumByCondition

CONDITION_COLORS = {"AwakeControl": "#7f7f7f", "No Induction": "#E377C2", "Induction": "#5F4EAB"}
CONDITION_LABELS = {"AwakeControl": "Awake\n(0%)", "No Induction": "0.3%\nalone",
                    "Induction": "0.3% after\ninduction"}
ANIMAL_MARKERS = ["o"]
GREY = "#8a8a8a"
STYLE = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6,
         "ytick.labelsize": 6, "legend.fontsize": 6, "axes.spines.top": False,
         "axes.spines.right": False, "figure.dpi": 110, "savefig.bbox": "tight",
         "lines.linewidth": 1.4}


def _conditionColors(colors, condition_order):
    """`colors` (default CONDITION_COLORS) with a tab10 colour filled in for any condition it does not name."""
    colors = dict(colors or CONDITION_COLORS)
    cycle = iter(plt.get_cmap("tab10").colors)
    for condition in condition_order:
        colors.setdefault(condition, next(cycle))
    return colors


def _markers(animals):
    return dict(zip(animals, ANIMAL_MARKERS * (1 + len(animals) // len(ANIMAL_MARKERS))))


def _statLabel(anova, metric, window_label):
    """'F(2,6)=79.1, p=4.9e-05' from the ANOVA table, or '' if absent."""
    row = anova[(anova["metric"] == metric) & (anova["window"] == window_label)]
    if row.empty or not np.isfinite(row["F"].iloc[0]):
        return ""
    r = row.iloc[0]
    return f"F({r.df_num:.0f},{r.df_den:.0f})={r.F:.1f}, p={r.p:.2g}, n={r.n_animals:.0f}"


def perAnimalPanel(ax, table, metric, window, condition_order, ylabel="", title="",
                   colors=None, sig_threshold=0.05, anova=None, window_label=None):
    """
    One paired per-animal panel: every animal's window mean in each condition,
    joined across conditions, with a mean +/- SEM marker. Animals missing a
    condition (e.g. excluded by an epoch filter) are simply not joined.
    """
    colors = _conditionColors(colors, condition_order)
    lo, hi = window
    w = table[(table["timeSeg"] >= lo) & (table["timeSeg"] < hi)]
    wide = (w.groupby(["condition", "animal_id"])[metric].mean()
             .unstack("condition").reindex(columns=condition_order))
    markers = _markers(list(wide.index))
    x = np.arange(len(condition_order))
    for animal, row in wide.iterrows():
        ok = row.notna().to_numpy()
        ax.plot(x[ok], row.to_numpy()[ok], color=GREY, lw=0.7, zorder=1)
        for i, condition in enumerate(condition_order):
            if pd.notna(row[condition]):
                ax.scatter(x[i], row[condition], marker=markers[animal], s=30,
                           facecolor=colors[condition], edgecolor="black", lw=0.6, zorder=3,
                           label=animal if i == 0 else None)
    for i, condition in enumerate(condition_order):
        v = wide[condition].dropna()
        if len(v):
            ax.errorbar(x[i], v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="_", ms=16, mew=2,
                        capsize=5, elinewidth=1.3, color="black", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in condition_order], fontsize=6)
    ax.set_ylabel(ylabel or metric)
    ax.margins(x=0.22, y=0.2)
    stat = _statLabel(anova, metric, window_label) if anova is not None else ""
    if title or stat:
        p = anova.loc[(anova["metric"] == metric) & (anova["window"] == window_label), "p"] if anova is not None else pd.Series(dtype=float)
        strong = bool(len(p)) and np.isfinite(p.iloc[0]) and p.iloc[0] < sig_threshold
        ax.set_title("\n".join(t for t in (title, stat) if t), loc="left", fontsize=7,
                     color="black" if strong else GREY)
    return wide


def timeCoursePanel(ax, table, metric, condition_order, bin_s=150, ylabel="", window=None,
                    colors=None, xlim=(-30, 60), labels=None):
    """Mean +/- SEM across animals of `metric` in `bin_s` bins, aligned to onset (minutes). `labels` renames conditions in the legend."""
    colors = _conditionColors(colors, condition_order)
    binned = (table.assign(_bin=(table["timeSeg"] // bin_s) * bin_s)
                   .groupby(["condition", "_bin", "animal_id"], as_index=False)[metric].mean())
    stats = binned.groupby(["condition", "_bin"])[metric].agg(["mean", "std", "count"]).reset_index()
    for condition in condition_order:
        s = stats[stats["condition"] == condition].sort_values("_bin")
        sem = s["std"] / np.sqrt(s["count"])
        ax.fill_between(s["_bin"] / 60, s["mean"] - sem, s["mean"] + sem,
                        color=colors[condition], alpha=0.18, lw=0)
        ax.plot(s["_bin"] / 60, s["mean"], color=colors[condition], label=(labels or {}).get(condition, condition))
    if window:
        ax.axvspan(window[0] / 60, window[1] / 60, color="tab:orange", alpha=0.10, lw=0)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Time relative to 0.3% onset (min)")
    ax.set_ylabel(ylabel or metric)
    ax.margins(y=0.1)


def figConventionalMeasures(table, window, condition_order, anova=None, window_label=None,
                            bands=("delta", "theta", "alpha")):
    """
    Context/control: the measures a reader expects. Top row absolute band power,
    middle row band share of total power, bottom row band power after the
    aperiodic background is removed.
    """
    rows = [("{}_abs", "{} power (V$^2$/Hz)"), ("{}_rel", "{} share of 1-20 Hz"),
            ("{}_periodic", "{} periodic (log$_{{10}}$)")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(rows), len(bands), figsize=(7.2, 6.4), sharex=True)
        for r, (template, ylabel) in enumerate(rows):
            for c, band in enumerate(bands):
                metric = template.format(band)
                perAnimalPanel(axes[r, c], table, metric, window, condition_order,
                               ylabel=ylabel.format(band.capitalize()),
                               anova=anova, window_label=window_label)
        fig.suptitle("Conventional band measures: absolute, relative and oscillatory power",
                     x=0.01, ha="left", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def figAperiodic(table, spectra, window, condition_order, anova=None, window_label=None,
                 fit_range=(1.0, 20.0), colors=None, panels="abc", labels=None):
    """
    Aperiodic result in up to three panels, chosen with `panels` (any of "a", "b", "c"):
    a  mean spectra per condition (log-log) with their fitted backgrounds (dashed)
    b  the exponent over time (mean +/- SEM across animals, analysis window shaded)
    c  each animal's exponent in the analysis window (paired panel, with ANOVA label if `anova` is given)
    Panels keep their place on a fixed 2 x 2 grid (a across the top, b bottom left, c bottom right), so a figure
    with only "a" and "b" has the same panel sizes as the full one. `labels` renames conditions in legends.
    """
    colors = _conditionColors(colors, condition_order)
    labels = labels or {}
    freqs, means = meanSpectrumByCondition(spectra, window=window)
    lo, hi = window
    fitted = (table[(table["timeSeg"] >= lo) & (table["timeSeg"] < hi)]
              .groupby(["condition", "animal_id"])[["exponent", "offset"]].mean()
              .groupby("condition").mean())
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 5.6))
        fig.set_layout_engine("none")   # spacing is set explicitly below; skip the automatic pass
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.3)

        if "a" in panels:
            ax_spec = fig.add_subplot(gs[0, :])
            m = (freqs >= 0.5) & (freqs <= 20)
            f_fit = freqs[(freqs >= fit_range[0]) & (freqs <= fit_range[1])]
            for condition in condition_order:
                if condition not in means:
                    continue
                ax_spec.plot(freqs[m], means[condition][m], color=colors[condition],
                             label=labels.get(condition, condition))
                ax_spec.plot(f_fit, aperiodicCurve(f_fit, fitted.loc[condition, "offset"],
                                                   fitted.loc[condition, "exponent"]),
                             color=colors[condition], ls="--", lw=1.0)
            ax_spec.set_xscale("log")
            ax_spec.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
            ax_spec.set_xlabel("Frequency (Hz)")
            ax_spec.set_ylabel("log$_{10}$ PSD (V$^2$/Hz)")
            ax_spec.set_title("a  Mean spectrum in the analysis window; dashed = fitted background",
                              loc="left", fontweight="bold")
            ax_spec.legend(frameon=False, loc="lower left")

        if "b" in panels:
            ax_time = fig.add_subplot(gs[1, 0])
            timeCoursePanel(ax_time, table, "exponent", condition_order, window=window,
                            ylabel="Aperiodic exponent", xlim=(-5, 60), labels=labels)
            ax_time.set_title("b  Exponent over time (mean \u00b1 SEM)", loc="left", fontweight="bold")
            ax_time.set_ylim(0, 2.3)

        if "c" in panels:
            ax_animal = fig.add_subplot(gs[1, 1])
            perAnimalPanel(ax_animal, table, "exponent", window, condition_order,
                           ylabel="Aperiodic exponent", title=r"$\bf{c\ \ \ Per\ mouse,\ analysis\ window}$",
                           anova=anova, window_label=window_label)
    return fig


def figSuppressionControl(table, window, condition_order, anova=None, window_label=None,
                          anova_clean=None, clean_threshold=1.0, colors=None):
    """
    Control: how much true (absolute-threshold) burst suppression there is, how it
    relates to the exponent, and the exponent comparison restricted to epochs with
    essentially none.
    """
    colors = _conditionColors(colors, condition_order)
    lo, hi = window
    w = table[(table["timeSeg"] >= lo) & (table["timeSeg"] < hi)]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
        timeCoursePanel(axes[0], table, "suppression_pct", condition_order, window=window,
                        ylabel="Suppression (%)", xlim=(-5,60))
        axes[0].set_title("a  Suppression over time", loc="left", fontweight="bold")

        for condition in condition_order:
            g = w[w["condition"] == condition]
            axes[1].scatter(g["suppression_pct"], g["exponent"], s=7, alpha=0.4,
                            color=colors[condition], lw=0)
        axes[1].axvline(clean_threshold, color="black", ls=":", lw=1)
        axes[1].set_xlabel("Suppression (%) in epoch")
        axes[1].set_ylabel("Aperiodic exponent")
        axes[1].set_title(f"b  Exponent vs suppression\n(dotted = {clean_threshold:g}% cut for panel c)",
                          loc="left", fontweight="bold")

        clean = table[table["suppression_pct"] <= clean_threshold]
        perAnimalPanel(axes[2], clean, "exponent", window, condition_order,
                       ylabel="Aperiodic exponent", title="c  Suppression-free epochs only",
                       anova=anova_clean if anova_clean is not None else anova,
                       window_label=window_label)
        fig.suptitle("Burst-suppression control (absolute 20 µV / 0.5 s criterion)",
                     x=0.01, ha="left", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def annotateSignificance(ax, pairwise, metric, window_label, condition_order, pairs=None,
                         p_col="p_adj", ns_label="n.s.", height=0.14, fontsize=6):
    """
    Draw significance brackets above a per-animal panel, reading p-values from the
    pairwise table `compareAcrossConditions` returns. `pairs` defaults to every
    contrast involving the last condition (the one the argument is about).
    """
    rows = pairwise[(pairwise["metric"] == metric) & (pairwise["window"] == window_label)]
    if rows.empty:
        return
    index = {c: i for i, c in enumerate(condition_order)}
    if pairs is None:
        pairs = [(c, condition_order[-1]) for c in condition_order[:-1]]
    pairs = sorted(pairs, key=lambda p: abs(index[p[0]] - index[p[1]]))   # narrowest bracket lowest
    lo, hi = ax.get_ylim()
    span = hi - lo
    for level, (c1, c2) in enumerate(pairs):
        row = rows[((rows["condition_1"] == c1) & (rows["condition_2"] == c2)) |
                   ((rows["condition_1"] == c2) & (rows["condition_2"] == c1))]
        if row.empty:
            continue
        p = float(row[p_col].iloc[0])
        label = ns_label if p >= 0.05 else (f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.0e}")
        x1, x2 = index[c1], index[c2]
        y = hi + span * (height * (level + 0.6))
        ax.plot([x1, x1, x2, x2], [y, y + span * 0.02, y + span * 0.02, y], lw=0.8, color="black")
        ax.text((x1 + x2) / 2, y + span * 0.03, label, ha="center", va="bottom", fontsize=fontsize,
                color="black" if p < 0.05 else GREY)
    ax.set_ylim(lo, hi + span * height * (len(pairs) + 1.2))


def figSummaryInset(table, window, condition_order, anova, pairwise, window_label, p_col="p_adj",
                    panels=(("exponent", "Aperiodic exponent", "Background slope"),
                            ("spectral_entropy", "Spectral entropy (0-1)", "Spectral entropy")),
                    exponent_ylim=(-0.15, 2), bracket_note=None):
    """
    Compact per-animal paired panels (x ticks removed) with significance brackets read from `pairwise`
    (column `p_col`: 'p_adj' = uncorrected for a single contrast, or e.g. 'q_measures' from
    ecog.stats.fdrAcrossMeasures). `panels` is a sequence of (metric, ylabel, title); the default keeps
    the aperiodic exponent and spectral entropy. Width scales with the number of conditions and panels.
    `bracket_note` is the figure title's description of the brackets; by default it states the correction
    that `p_col` and the number of conditions imply.
    """
    if bracket_note is None:
        bracket_note = ("none (single condition)" if len(condition_order) < 2
                        else "paired t-tests, BH across measures" if p_col == "q_measures"
                        else "paired t-tests, BH across condition pairs" if len(condition_order) > 2
                        else "paired t-test, uncorrected")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=((0.55 * len(condition_order)) * len(panels) + 0.5, 2))
        for ax, (metric, ylabel, title) in zip(np.atleast_1d(axes), panels):
            perAnimalPanel(ax, table, metric, window, condition_order,
                           ylabel=ylabel, title=title, anova=anova, window_label=window_label)
            ax.set_xticks([])
            annotateSignificance(ax, pairwise, metric, window_label, condition_order, p_col=p_col)
            ax.margins(x=0.5)
            if metric == "exponent" and exponent_ylim:
                ax.set_ylim(*exponent_ylim)
        fig.suptitle(f"Same mice, same 0.3% isoflurane, {window_label} after onset "
                     f"\u2014 brackets: {bracket_note}", x=0.01, ha="left", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
