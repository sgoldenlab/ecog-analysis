"""
Statistics for condition comparisons. Everything is per-animal first: each animal's value in a window is the
mean (or median) of its samples/epochs in that window, animals are matched across conditions by identity
(the 'mouse' column with its trailing '_R<n>' session suffix stripped), and tests are paired.

- compareConditionMeans: paired t-test for every condition pair (optional multiple-comparison correction
  across the pairs, `p_adjust`)
- compareConditionsANOVA: one-way repeated-measures ANOVA across conditions
- compareAcrossConditions: both of the above for every metric x window of an epoch table
- fdrAcrossMeasures: a correction across the MEASURES reported together (the relevant family when only two
  conditions are compared, so there is a single contrast per measure)
- perAnimalMeans: each animal's mean of a set of metrics in a window
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


def _stripRecordingSuffix(mouse_col):
    """
    Strip aggregate_aligned()'s trailing "_R<n>" session suffix off a
    'mouse' column, collapsing it to a true per-animal id. Used by every
    condition-comparison function below (plot_condition_summary,
    plot_condition_box, compareConditionMeans) so animals are matched across
    conditions by animal, not by session -- an animal whose recording_number
    happens to differ between conditions (e.g. its Induction session got
    logged as "_R2" while its Alone session is "_R1") would otherwise fail
    to pair/be double-counted purely because of that numbering, not because
    it's actually a different animal.
    """
    return mouse_col.str.replace(r"_R\d+$", "", regex=True)


def _preBinWindow(window_df, value_col, binning_s):
    """
    Collapse an already time-window-filtered df into one row per (mouse,
    condition, `binning_s`-second bin) by averaging `value_col` within each
    bin first. Used by compareConditionMeans()/compareConditionsANOVA()'s
    optional `binning_s` so each bin contributes equally to an animal's
    per-condition mean, rather than every raw segment -- without this, a
    period with denser raw segment coverage would otherwise outweigh a
    sparser one in that mean, even though both cover the same wall-clock
    time.
    """
    time_bin = (window_df["timeSeg"] // binning_s) * binning_s
    return (
        window_df.assign(timeBin=time_bin)
        .groupby(["mouse", "condition", "timeBin"], as_index=False)[value_col].mean()
    )


def compareConditionMeans(df, value_col, window, condition_order=None, p_adjust=None, alpha=0.05,
                           binning_s=None, center="mean"):
    """
    Paired comparison (paired t-test) of a value's per-mouse mean between
    every pair of conditions within a single time window -- the
    condition-axis counterpart to compare_range_means(), mirroring how
    plot_condition_summary() is plot_range_summary()'s condition-axis
    counterpart.

    `df` should already have 'condition' relabeled per condition (see
    pipeline.process_condition_batch), same as plot_condition_summary().
    `window` is a single (start, end) tuple in df['timeSeg'] units.
    `condition_order` fixes which conditions are compared and in what order
    (defaults to first appearance in `df`).

    Mice are matched across conditions by animal (via _stripRecordingSuffix(),
    dropping the trailing "_R<n>") rather than by the raw 'mouse' value, so an
    animal whose recording_number happens to differ between conditions still
    pairs correctly.

    Each animal's per-condition value is the plain mean of every raw
    'timeSeg' segment in `window` by default. Pass `binning_s` to instead
    average within `binning_s`-second bins first (see _preBinWindow()) and
    average *those* per animal -- in effect, weighting the window by time
    rather than by raw segment count. With evenly-spaced segments (the usual
    case) this barely changes anything; it matters if segment density is
    uneven within the window (e.g. near a recording's start/end, or after
    reprocessing part of a recording at a different resolution).

    `center` ("mean" default, or "median") swaps that per-animal aggregation
    for the animal's median within `window`/its bins instead -- more robust
    to a single skewed/spiky segment, and worth matching whatever `center` a
    companion plot_condition_stripplot() call uses so the plotted points and
    the tested numbers agree. Either way, the 'mean_1'/'mean_2' columns
    returned are always the plain across-animal mean *of* those per-animal
    values (a paired t-test tests whether the mean difference is zero, so
    reporting a median-of-medians here wouldn't match what's actually being
    tested) -- only which per-animal statistic feeds the test changes.

    With more than two conditions, the several pairwise rows this returns
    carry an inflated false-positive rate if read as independent tests --
    pass `p_adjust` (any method name `statsmodels.stats.multitest.
    multipletests` accepts, e.g. "holm" (default recommendation), "bonferroni",
    "fdr_bh") to add a 'p_adj' column (correcting across every non-NaN row
    returned, i.e. every pair in `condition_order`) and a boolean
    'significant' column (`p_adj < alpha`). Left at the default (None), the
    result is unchanged from before -- just the raw, uncorrected 'p' column
    -- consider running compareConditionsANOVA() first and only using this
    (with `p_adjust` set) as its post-hoc follow-up once that omnibus test
    is significant, rather than reading the raw 'p' column on its own.

    Returns one row per condition pair: 'n' (paired mice), each condition's
    own mean ('mean_1'/'mean_2'), their difference, the t-statistic, and the
    p-value (plus 'p_adj'/'significant' if `p_adjust` is given). With a single
    condition there is no pair, and the table is empty (same columns).
    """
    if center not in ("mean", "median"):
        raise ValueError(f"center must be 'mean' or 'median', got {center!r}")
    lo, hi = window
    window_df = df[(df["timeSeg"] >= lo) & (df["timeSeg"] < hi)].assign(
        mouse=lambda x: _stripRecordingSuffix(x["mouse"])
    )
    if binning_s:
        window_df = _preBinWindow(window_df, value_col, binning_s)
    condition_order = condition_order or list(dict.fromkeys(window_df["condition"]))

    per_mouse = {
        cond: window_df[window_df["condition"] == cond].groupby("mouse")[value_col].agg(center)
        for cond in condition_order
    }

    rows = []
    for c1, c2 in itertools.combinations(condition_order, 2):
        paired = pd.concat([per_mouse[c1], per_mouse[c2]], axis=1, keys=[c1, c2]).dropna()
        if len(paired) >= 2:
            t_stat, p_val = stats.ttest_rel(paired[c1], paired[c2])
        else:
            t_stat, p_val = np.nan, np.nan
        rows.append({
            "condition_1": c1, "condition_2": c2, "n": len(paired),
            "mean_1": paired[c1].mean() if len(paired) else np.nan,
            "mean_2": paired[c2].mean() if len(paired) else np.nan,
            "mean_diff": (paired[c1] - paired[c2]).mean() if len(paired) else np.nan,
            "t": t_stat, "p": p_val,
        })
    # a single condition has no pair to compare: an empty table with the usual columns
    result = pd.DataFrame(rows, columns=["condition_1", "condition_2", "n", "mean_1", "mean_2", "mean_diff", "t", "p"])

    if p_adjust:
        valid = result["p"].notna()
        result["p_adj"] = np.nan
        result["significant"] = False
        if valid.any():
            reject, p_corrected, _, _ = multipletests(result.loc[valid, "p"], alpha=alpha, method=p_adjust)
            result.loc[valid, "p_adj"] = p_corrected
            result.loc[valid, "significant"] = reject
    return result


def compareConditionsANOVA(df, value_col, window, condition_order=None, binning_s=None, center="mean"):
    """
    One-way repeated-measures ANOVA (statsmodels' AnovaRM) of a value's
    per-animal mean across conditions within a single time window -- the
    omnibus-test counterpart to compareConditionMeans()'s pairwise paired
    t-tests. Prefer this over compareConditionMeans() when comparing more
    than two conditions: one omnibus test on the condition effect avoids the
    inflated false-positive rate of testing every pair separately, and
    doesn't need a post-hoc multiple-comparisons correction the way
    compareConditionMeans()'s several pairwise rows do.

    Same aggregation as compareConditionMeans(): each animal's per-condition
    value is first averaged across every 'timeSeg' segment landing in
    `window`, matched across conditions by animal identity (mouse with the
    trailing "_R<n>" session suffix stripped -- see _stripRecordingSuffix()).
    `condition_order` fixes which conditions are compared (defaults to first
    appearance in `df`). `binning_s` behaves exactly as in
    compareConditionMeans() -- average within `binning_s`-second bins first,
    so the per-animal mean is weighted by time rather than raw segment count.
    `center` ("mean" default, or "median") also behaves exactly as in
    compareConditionMeans() -- use each animal's median instead (more robust
    to a single skewed/spiky segment), matching whatever `center` a
    companion plot_condition_stripplot() call uses.

    A repeated-measures ANOVA needs a *balanced* design -- exactly one value
    per animal per condition, no missing cells -- so an animal missing a
    value for any condition in `condition_order` (e.g. no recording for it
    in one condition yet) is dropped entirely (listwise deletion) rather
    than raising. Raises ValueError if fewer than 2 conditions or fewer than
    2 animals have a complete set of values across all of them.

    Returns (anova_table, included_animals, excluded_animals):
    - anova_table: statsmodels' result table for the condition effect (F
      statistic, numerator/denominator df, p-value).
    - included_animals / excluded_animals: which animals fed the test and
      which were dropped for lacking at least one condition.
    """
    if center not in ("mean", "median"):
        raise ValueError(f"center must be 'mean' or 'median', got {center!r}")
    lo, hi = window
    window_df = df[(df["timeSeg"] >= lo) & (df["timeSeg"] < hi)].assign(
        mouse=lambda x: _stripRecordingSuffix(x["mouse"])
    )
    if binning_s:
        window_df = _preBinWindow(window_df, value_col, binning_s)
    condition_order = condition_order or list(dict.fromkeys(window_df["condition"]))
    if len(condition_order) < 2:
        raise ValueError(f"Need at least 2 conditions to compare, got {condition_order}")
    window_df = window_df[window_df["condition"].isin(condition_order)]

    per_mouse = window_df.groupby(["mouse", "condition"])[value_col].agg(center).reset_index()
    wide = per_mouse.pivot(index="mouse", columns="condition", values=value_col)

    complete = wide.dropna(subset=condition_order)
    included_animals = sorted(complete.index)
    excluded_animals = sorted(set(wide.index) - set(included_animals))
    if len(included_animals) < 2:
        raise ValueError(
            f"Need at least 2 animals with a value for every condition in {condition_order}, "
            f"got {len(included_animals)} (excluded for missing data: {excluded_animals})"
        )

    long = complete[condition_order].reset_index().melt(
        id_vars="mouse", value_vars=condition_order, var_name="condition", value_name=value_col
    )
    anova_table = AnovaRM(long, depvar=value_col, subject="mouse", within=["condition"]).fit().anova_table
    return anova_table, included_animals, excluded_animals


def perAnimalMeans(df, metrics, window, condition_order=None):
    """Each animal's mean of every metric within `window`, one row per animal x condition."""
    lo, hi = window
    w = df[(df["timeSeg"] >= lo) & (df["timeSeg"] < hi)]
    out = w.groupby(["condition", "animal_id"], as_index=False)[list(metrics)].mean()
    if condition_order:
        out["condition"] = pd.Categorical(out["condition"], condition_order, ordered=True)
        out = out.sort_values(["condition", "animal_id"]).reset_index(drop=True)
    return out


def fdrAcrossMeasures(pairwise, window_label, metrics=None, method="fdr_bh", out_col="q_measures"):
    """
    Add a multiplicity correction across MEASURES rather than across condition pairs.

    `compareConditionMeans` corrects within a metric, across the pairwise condition
    contrasts — the right family when three conditions are compared on one measure.
    With only two conditions there is a single contrast per measure, so that family
    is trivial and the relevant multiplicity is the set of measures reported
    together in a figure. This applies `method` across `metrics` separately for each
    condition pair within `window_label`, writing the result to `out_col` (NaN for
    rows outside the selected metrics/window, so the column can be passed straight
    to `response_figures.annotateSignificance(..., p_col=out_col)`).

    Record which family you used: `p_adj` and `q_measures` answer different questions.
    """
    from statsmodels.stats.multitest import multipletests

    out = pairwise.copy()
    out[out_col] = np.nan
    metrics = list(metrics) if metrics is not None else sorted(out["metric"].unique())
    sel = (out["window"] == window_label) & out["metric"].isin(metrics)
    for _, group in out[sel].groupby(["condition_1", "condition_2"]):
        out.loc[group.index, out_col] = multipletests(group["p"].to_numpy(), method=method)[1]
    return out


def compareAcrossConditions(df, metrics, windows, condition_order, p_adjust="fdr_bh"):
    """
    Repeated-measures ANOVA across conditions plus FDR-corrected paired t-tests,
    for every metric x window, using the processing.py helpers (so the animal
    matching and per-mouse-first aggregation are identical to the rest of the
    codebase).

    `p_adjust` defaults to Benjamini-Hochberg ("fdr_bh"), matching the manuscript
    convention, and is applied across the pairwise condition contrasts within each
    metric x window. Pass "holm" for the more conservative family-wise correction;
    the `p_adj` column is named the same either way, so record which was used.

    `windows` is {label: (start, end)}. Returns (anova_df, pairwise_df); a metric
    whose window has too few complete animals is reported with NaN statistics and
    a `note`, rather than raising. With a single condition the ANOVA rows carry that
    note and the pairwise table is empty.
    """
    anova_rows, pairwise = [], []
    for metric in metrics:
        sub = df.dropna(subset=[metric])
        for label, (lo, hi) in windows.items():
            row = {"metric": metric, "window": label}
            try:
                table, included, excluded = compareConditionsANOVA(
                    sub, metric, (lo, hi), condition_order=condition_order)
                row.update({"F": float(table["F Value"].iloc[0]),
                            "df_num": float(table["Num DF"].iloc[0]),
                            "df_den": float(table["Den DF"].iloc[0]),
                            "p": float(table["Pr > F"].iloc[0]),
                            "n_animals": len(included),
                            "excluded": ",".join(excluded)})
            except ValueError as exc:
                row.update({"F": np.nan, "df_num": np.nan, "df_den": np.nan, "p": np.nan,
                            "n_animals": np.nan, "excluded": "", "note": str(exc)})
            anova_rows.append(row)
            pairwise.append(compareConditionMeans(sub, metric, (lo, hi),
                                                  condition_order=condition_order,
                                                  p_adjust=p_adjust).assign(metric=metric, window=label))
    return pd.DataFrame(anova_rows), pd.concat(pairwise, ignore_index=True)
