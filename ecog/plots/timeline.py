"""
Time axes and spectrograms.

`plotRecordingSpectrogram(rec, channel, work_dir, ...)` is the one-call spectrogram for a Recording: it
reads the raw CSV, plots 10*log10 power (dB/Hz) against time, optionally re-zeroes the axis on a named event
(`align_event`), crops to a window, and marks all the recording's events with dashed lines.
Display defaults come from `rec.spectrogram` and can be overridden per call.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from ecog.spectral import _applyNotchFilters

fs = 400
n_fft = 10000
n_overlap = n_fft / 2


def annotateTimeline(
        ax, events=None, end_time=None, tick_step=300, colors=None, default_color='red',
        legend_pos="upper left", legend_bbox=(1, 1)
        ):
    """
    Convert a seconds x-axis to minutes and mark named events with vertical lines.

    `events` is a dict of {label: time_seconds}, in whatever time base the
    plotted x-axis itself uses -- absolute (recording-clock) time for a
    single recording (the whole recording is always extracted, so 'timeSeg'
    is absolute throughout -- see run_recording()), or event-relative time
    (0 = the aligned event) for an aggregate_aligned()'d cross-animal plot.
    Either way, `events` values are plotted directly with no offset, since
    they're already expressed in that same axis's own units.

    `colors` optionally maps a label to its vertical line's color; labels not
    given a color there fall back to `default_color`.
    """
    # Span the axis's actual range (not just from 0) so aligned/averaged plots
    # -- where pre-event time is negative -- still get labeled ticks there.
    x_min, x_max = ax.get_xlim()
    if end_time:
        x_max = end_time
    xticks = np.arange(np.ceil(x_min / tick_step) * tick_step, x_max, tick_step)
    ax.set_xticks(xticks, labels=[int(x / 60) for x in xticks])
    ax.set_xlabel("Time (minutes)")

    colors = colors or {}
    for label, event_time in (events or {}).items():
        ax.axvline(event_time, color=colors.get(label, default_color), linestyle='--', label=label)

    if events:
        ax.legend(bbox_to_anchor=legend_bbox, loc=legend_pos)


def plotSpectrogram(
        input_file: Path, condition, title, fs=fs, n_fft=n_fft,
        n_overlap=n_overlap, notch_freqs=None, notch_quality=30,
        clim=(-80, -100), figsize=(5, 2.5), freq_min=0, freq_max=12, tick_step=None,
        align_time=0,
        ):
    """
    `freq_min`/`freq_max` (Hz) crop the plotted frequency axis (default 0-12
    Hz; raise freq_max, e.g. to 30, to see higher bands like beta).

    `tick_step` (seconds) is optional: when given, the x-axis is switched
    from raw seconds to minute ticks spaced `tick_step` seconds apart, the
    same tick control plotRatioSNR/plotBSRTimeseries take -- via a plain
    annotateTimeline(ax, tick_step=tick_step) call with no events, so calling
    annotateTimeline again afterward to add event lines is still fine (it
    just recomputes the same ticks). Left at the default (None), the axis is
    unchanged: raw seconds, matplotlib's own ticks.

    `align_time` (seconds, absolute recording-clock time -- `input_file` is
    always the whole recording, see run_recording()) shifts the plotted axis
    so that moment sits at x=0 instead of the recording's own t=0 -- e.g.
    pass a recording's own onset time to plot it relative to onset rather
    than in absolute time. Left at the default (0), the axis is absolute.
    """
    ecog = pd.read_csv(input_file)

    data = ecog.filter(like=condition, axis=1).values.flatten()
    if notch_freqs:
        data = _applyNotchFilters(data, fs, notch_freqs, notch_quality)
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=n_fft, noverlap=n_overlap)
    t = t - align_time

    freq_range = (f >= freq_min) & (f <= freq_max)
    f_filtered = f[freq_range]
    Sxx_filtered = Sxx[freq_range, :]

    # clim is conventionally given as (high, low); matplotlib needs vmin <= vmax
    vmin, vmax = sorted(clim)

    plt.figure(figsize=figsize)

    # Plotted against 't' (seconds) instead of 't/60' (minutes) for short recordings
    mesh = plt.pcolormesh(t, f_filtered, 10 * np.log10(Sxx_filtered), shading='gouraud', vmin=vmin, vmax=vmax)
    mesh.set_rasterized(True)

    plt.title(title)
    plt.ylabel('Frequency (Hz)')
    if tick_step:
        annotateTimeline(plt.gca(), tick_step=tick_step)
    else:
        plt.xlabel('Time (seconds)')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    return plt.gcf()


def plotRecordingSpectrogram(rec, channel, work_dir, freq_min=None, freq_max=None, align_event=None,
                             start_time=None, end_time=None, figsize=None, clim=None, tick_step=None,
                             legend_pos="upper left", legend_bbox=(1, 1)):
    """
    Spectrogram of one channel of a Recording, with its events marked. Returns the Figure.

    Any argument left at None falls back to `rec.spectrogram[...]`, then to freq 0-20 Hz, figsize (5, 2.5),
    `rec.clim`, `rec.tick_step`.

    `align_event` (a key of rec.events, e.g. "onset") makes the x-axis time relative to that event
    (0 = the event); `start_time`/`end_time` are then also relative to it. Without it the axis is absolute
    recording time. Leave `end_time` None to show the whole recording.

    The event legend sits at `legend_bbox` (axes coordinates; default as in the original code, on top of the colour bar):
    pass e.g. `legend_bbox=(1.25, 1)` to move it clear of the colour bar.
    """
    s = {**{"freq_min": 0, "freq_max": 20, "figsize": (5, 2.5)}, **rec.spectrogram}
    freq_min = freq_min if freq_min is not None else s["freq_min"]
    freq_max = freq_max if freq_max is not None else s["freq_max"]
    figsize = tuple(figsize if figsize is not None else s["figsize"])
    align_event = align_event if align_event is not None else s.get("align_event")
    start_time = start_time if start_time is not None else s.get("start_time")
    end_time = end_time if end_time is not None else s.get("end_time")
    clim = clim if clim is not None else s.get("clim", rec.clim)
    tick_step = tick_step if tick_step is not None else s.get("tick_step", rec.tick_step)

    align_offset = rec.eventTime(align_event) if align_event else 0
    events = {name: t - align_offset for name, t in rec.events.items()}

    fig = plotSpectrogram(
        input_file=rec.rawCSV(work_dir, channel), condition=rec.condition,
        title=f"{rec.label} ch{channel} - Spectrogram", fs=rec.fs, n_fft=rec.n_fft, n_overlap=rec.n_fft / 2,
        notch_freqs=rec.notch_freqs, notch_quality=rec.notch_quality, figsize=figsize,
        freq_min=freq_min, freq_max=freq_max, tick_step=tick_step, clim=clim, align_time=align_offset,
    )
    ax = fig.axes[0]
    if end_time is not None:
        ax.set_xlim(start_time if start_time is not None else 0, end_time + 1)
    annotateTimeline(ax, events=events, tick_step=tick_step, colors=rec.event_colors, default_color="white",
                     legend_pos=legend_pos, legend_bbox=legend_bbox)
    return fig
