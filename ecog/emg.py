"""
EMG channel processing (index 2 in the Pinnacle .edf exports: 'EMG EMG', 400 Hz, volts).

Movement shows up as broadband high-frequency EMG power, so the EMG channel doubles as an artefact
regressor for the EEG epochs. Typical use:

    extractEMG(rec, work_dir)                              # .edf -> <work_dir>/<animal>/csvs/<label>_raw_ch2.csv
    emg = emgEpochTable(rec, work_dir, onset_event="onset")  # one row per 30 s epoch, same time grid as buildEpochTable
    fig = figEMG(rec, emg, work_dir)                       # 1 s RMS envelope, epoch means, flagged movement
    table = attachEMG(epoch_table, emg)                    # add emg_* columns to an EEG epoch table

Cleaning (`cleanEMG`, on by default): in the Pinnacle exports the EMG channel is dominated by narrowband
interference, not muscle -- 60 Hz mains plus further lines (in the EmergRev files 98% of the 20-150 Hz power sits in
lines and the raw 1 s RMS is a constant ~345 uV whatever the animal does). Before the band-pass, the mains and
its harmonics are removed with +/-`mains_halfwidth_hz` band-stops, then any remaining in-band spectral line more
than `line_prominence` x the median PSD is notched. The lines removed and the raw line share are kept in
`table.attrs` and reported; check them before reading the envelope as movement.

Definitions (all on the cleaned, band-passed signal, `band` Hz, zero-phase 4th-order Butterworth):
- emg_rms_uv: RMS amplitude (uV) in `window_s` windows, then averaged within the epoch
- emg_power_db: mean power in the band (dB re 1 uV^2) over the epoch
- moving: epoch RMS above a threshold -- `threshold_uv` if given, else median + `n_mad` * 1.4826 * MAD of the epoch
  RMS values inside `baseline_window` (seconds on the epoch axis, default the whole recording). The default is
  relative to each recording, so state the choice when reporting.
"""
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, sosfiltfilt, welch

from ecog.cache import isStale, writeSettings
from ecog.edf import ecogToCSV

EMG_CHANNEL = 2


def extractEMG(rec, work_dir, channel=EMG_CHANNEL, overwrite=False):
    """EMG channel of a Recording to its raw CSV (same layout as the EEG channels); returns the path."""
    edf = rec.edfPath()
    path = rec.rawCSV(work_dir, channel)
    settings = {"edf": edf.name, "edf_bytes": edf.stat().st_size, "notch_freqs": None}
    existed = path.exists()
    redo = overwrite or isStale(path, settings)
    ecogToCSV(edf.parent, output_folder=Path(work_dir) / rec.animal_id / "csvs", animal_id=rec.animal_id,
              channel_index=[channel], notch_freqs=None, file_filter=edf.name, condition=rec.condition,
              recording_number=rec.recording_number, overwrite=redo, allow_emg=True)
    if redo or not existed:
        writeSettings(path, settings)
    return path


def _spectralLines(x, fs, band, prominence, nperseg=8192):
    """Frequencies (Hz) of in-band PSD peaks above `prominence` x the in-band median PSD, strongest first."""
    fr, P = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    in_band = (fr >= band[0]) & (fr <= band[1])
    med = np.median(P[in_band])
    peaks, _ = find_peaks(P, prominence=prominence * med)
    peaks = [i for i in peaks if band[0] - 5 <= fr[i] <= min(band[1] + 5, 0.95 * fs / 2)]
    return fr, P, sorted(peaks, key=lambda i: -P[i])


def cleanEMG(x, fs=400, band=(20, 150), mains=60, mains_halfwidth_hz=5.0, line_prominence=20.0,
             line_width_hz=1.0, max_lines=20):
    """
    Remove narrowband interference from an EMG trace (volts) and band-pass it.
    Returns (signal in uV, info) with info = {"raw_line_share", "mains_removed_hz", "lines_removed_hz"}:
    raw_line_share is the fraction of the raw in-band power lying within +/-1 Hz of detected lines.
    """
    y = np.asarray(x, dtype=float) - np.mean(x)
    fr, P, raw_peaks = _spectralLines(y, fs, band, line_prominence)
    in_band = (fr >= band[0]) & (fr <= band[1])
    near_line = np.zeros_like(fr, dtype=bool)
    for i in raw_peaks:
        near_line |= np.abs(fr - fr[i]) < 1.0
    share = float(P[in_band & near_line].sum() / P[in_band].sum())

    mains_removed = []
    if mains:
        h = 1
        while mains * h + mains_halfwidth_hz < 0.95 * fs / 2:
            sos = butter(4, [mains * h - mains_halfwidth_hz, mains * h + mains_halfwidth_hz], btype="bandstop",
                         fs=fs, output="sos")
            y = sosfiltfilt(sos, y)
            mains_removed.append(mains * h)
            h += 1
    fr, P, peaks = _spectralLines(y, fs, band, line_prominence)
    lines = []
    for i in peaks[:max_lines]:
        f0 = float(fr[i])
        b, a = iirnotch(f0, f0 / line_width_hz, fs)
        y = filtfilt(b, a, y)
        lines.append(round(f0, 1))
    sos = butter(4, [band[0], min(band[1], 0.95 * fs / 2)], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, y) * 1e6, {"raw_line_share": share, "mains_removed_hz": mains_removed,
                                      "lines_removed_hz": sorted(lines)}


def emgEnvelope(x, fs=400, band=(20, 150), window_s=1.0, clean=True, edge_s=1.0, **clean_kwargs):
    """
    RMS envelope of an EMG trace (volts): (window start times in s, RMS in uV per `window_s` window,
    the processed signal in uV, cleaning info). `clean=False` skips interference removal (band-pass only).
    The first and last `edge_s` seconds are set to NaN (zero-phase filters leave a transient there); windows are
    averaged with NaNs ignored.
    """
    if clean:
        y, info = cleanEMG(x, fs, band, **clean_kwargs)
    else:
        x = np.asarray(x, dtype=float)
        sos = butter(4, [band[0], min(band[1], 0.95 * fs / 2)], btype="bandpass", fs=fs, output="sos")
        y, info = sosfiltfilt(sos, x - np.mean(x)) * 1e6, {}
    if edge_s:
        k = int(edge_s * fs)
        y = y.copy()
        y[:k] = np.nan
        y[len(y) - k:] = np.nan
    n = int(window_s * fs)
    n_win = len(y) // n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rms = np.sqrt(np.nanmean(y[: n_win * n].reshape(n_win, n) ** 2, axis=1))
    return np.arange(n_win) * window_s, rms, y, info


def emgEpochTable(rec, work_dir, channel=EMG_CHANNEL, onset_event=None, epoch_s=30, band=(20, 150), window_s=1.0,
                  threshold_uv=None, n_mad=3.0, baseline_window=None, clean=True, edge_s=1.0, **clean_kwargs):
    """
    Per-epoch EMG summary for one Recording (raw EMG CSV must exist: see extractEMG).
    `onset_event` (key of rec.events) sets timeSeg = epoch centre relative to that event, matching
    ecog.epochs.buildEpochTable; None keeps absolute recording time.

    Columns: recording, condition, animal_id, mouse, channel, timeSeg, emg_rms_uv, emg_power_db, moving.
    `moving` is emg_rms_uv > threshold. The threshold is `threshold_uv` if given, else
    median + n_mad * 1.4826 * MAD of the epoch RMS values within `baseline_window` (epoch-axis seconds; default the
    whole recording) -- a rule relative to that recording, so report it. attrs carry threshold_uv, window_s and the
    cleaning info of cleanEMG.
    """
    path = rec.rawCSV(work_dir, channel)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run extractEMG first")
    x = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)
    t_win, rms, y, info = emgEnvelope(x, rec.fs, band, window_s, clean=clean, edge_s=edge_s, **clean_kwargs)
    if clean and info["raw_line_share"] > 0.5:
        print(f"note: {rec.label} EMG ch{channel}: {info['raw_line_share']:.0%} of raw {band[0]}-{band[1]} Hz power is in "
              f"narrowband lines; mains + {len(info['lines_removed_hz'])} extra lines removed before the envelope")
    onset = rec.eventTime(onset_event) if onset_event else 0.0

    win_per_epoch = int(round(epoch_s / window_s))
    n_epochs = len(rms) // win_per_epoch
    if n_epochs == 0:
        raise ValueError(f"{rec.label}: recording shorter than one {epoch_s}s epoch")
    epoch_rms = np.nanmean(rms[: n_epochs * win_per_epoch].reshape(n_epochs, win_per_epoch), axis=1)
    centres = np.arange(n_epochs) * epoch_s + epoch_s / 2 - onset
    n_samp = int(epoch_s * rec.fs)
    power = np.nanmean(y[: n_epochs * n_samp].reshape(n_epochs, n_samp) ** 2, axis=1)

    if threshold_uv is None:
        ref = epoch_rms if baseline_window is None else epoch_rms[
            (centres >= baseline_window[0]) & (centres < baseline_window[1])]
        if len(ref) == 0:
            raise ValueError(f"{rec.label}: baseline_window {baseline_window} contains no epochs")
        threshold_uv = float(np.median(ref) + n_mad * 1.4826 * np.median(np.abs(ref - np.median(ref))))

    table = pd.DataFrame({
        "recording": path.stem, "condition": rec.condition, "animal_id": rec.animal_id, "mouse": rec.mouse_id,
        "channel": channel, "timeSeg": centres, "emg_rms_uv": epoch_rms,
        "emg_power_db": 10 * np.log10(power), "moving": epoch_rms > threshold_uv,
    })
    table.attrs["threshold_uv"] = threshold_uv
    table.attrs["window_s"] = window_s
    table.attrs.update(info)
    return table


def attachEMG(table, emg, columns=("emg_rms_uv", "emg_power_db", "moving")):
    """
    Add EMG columns to an EEG epoch table (ecog.epochs.buildEpochTable) by matching animal session ('mouse')
    and epoch time. Both must use the same epoch length and onset event; unmatched epochs get NaN.
    """
    right = emg[["mouse", "timeSeg", *columns]].assign(_k=lambda d: d["timeSeg"].round(3)).drop(columns="timeSeg")
    out = table.assign(_k=table["timeSeg"].round(3)).merge(right, on=["mouse", "_k"], how="left")
    return out.drop(columns="_k")


def figEMG(rec, emg, work_dir, channel=EMG_CHANNEL, band=(20, 150), window_s=1.0, xlim_min=None, figsize=(6, 2.8),
           color="#444444", clean=True):
    """
    EMG time course: 1 s RMS envelope (thin), epoch means (thick), the movement threshold (dashed) and epochs flagged
    as moving (shaded). Time axis in minutes relative to the same event as `emg['timeSeg']`.
    """
    import matplotlib.pyplot as plt

    x = pd.read_csv(rec.rawCSV(work_dir, channel)).iloc[:, 0].to_numpy(dtype=float)
    t_win, rms, _, _ = emgEnvelope(x, rec.fs, band, window_s, clean=clean)
    epoch_s = float(np.median(np.diff(emg["timeSeg"]))) if len(emg) > 1 else 30.0
    offset = emg["timeSeg"].iloc[0] - epoch_s / 2          # recording time 0 on the emg['timeSeg'] axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot((t_win + window_s / 2 + offset) / 60, rms, color=color, lw=0.4, alpha=0.5, label=f"{window_s:g} s RMS")
    ax.plot(emg["timeSeg"] / 60, emg["emg_rms_uv"], color="black", lw=1.6, label=f"{epoch_s:g} s epoch mean")
    ax.axhline(emg.attrs.get("threshold_uv", np.nan), color="tab:red", ls="--", lw=1, label="movement threshold")
    for t in emg.loc[emg["moving"], "timeSeg"]:
        ax.axvspan((t - epoch_s / 2) / 60, (t + epoch_s / 2) / 60, color="tab:orange", alpha=0.18, lw=0)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(f"EMG RMS, {band[0]}-{band[1]} Hz (uV)")
    ax.set_title(f"{rec.label} EMG (shaded = moving epochs)")
    if xlim_min:
        ax.set_xlim(*xlim_min)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
    return fig
