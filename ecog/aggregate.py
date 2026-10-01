"""
Combine per-recording band tables across animals onto a shared time axis.

`bandTable` reads one recording's band CSV (ecog.spectral.computeBandRatios / computeBandPower) as a long
table with columns timeSeg, mouse, condition, <value_name>, and shifts timeSeg so that a named event sits
at 0. `aggregateAligned` stacks several recordings under one condition label; the result feeds
ecog.plots.timeseries.plotBinnedTimeseries and the ecog.stats comparisons directly.
"""
import pandas as pd


def bandTable(rec, channel, work_dir, band="delta", kind="ratio", align_event=None, value_name=None):
    """
    One recording/channel's band CSV in long form. `align_event` (a key of rec.events) is subtracted from
    timeSeg so 0 = that event; None keeps absolute recording-clock time. `value_name` defaults to
    'ratio' (band SNR) or 'power'.
    """
    value_name = value_name or ("ratio" if kind == "ratio" else "power")
    path = rec.bandCSV(work_dir, channel, band, kind)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run computeBandRatios/computeBandPower first")
    wide = pd.read_csv(path)
    long = pd.melt(wide, id_vars=["timeSeg"], var_name="condition", value_name=value_name)
    long["mouse"] = rec.mouse_id
    if align_event:
        long["timeSeg"] -= rec.eventTime(align_event)
    return long[["timeSeg", "mouse", "condition", value_name]]


def aggregateAligned(entries, work_dir, band="delta", kind="ratio", align_event=None, condition_label=None,
                     value_name=None):
    """
    Stack bandTable over `entries` = [(Recording, channel), ...], aligned on `align_event`.
    `condition_label` replaces the 'condition' column (so two alignments of the same recording, e.g. SAL
    and CNO, or any renamed condition, can be told apart once combined).
    """
    tables = [bandTable(rec, ch, work_dir, band, kind, align_event, value_name) for rec, ch in entries]
    df = pd.concat(tables, ignore_index=True)
    return df.assign(condition=condition_label) if condition_label is not None else df
