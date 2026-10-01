"""
Spectrogram-based band measures, one value per STFT segment (n_fft samples, 50% overlap;
10000 samples at 400 Hz = 25 s windows every 12.5 s).

- band SNR ("ratio"): mean power in a signal band / mean power in flanking noise bands
- band power: plain mean power in a band (V^2/Hz)

`computeBandRatios(rec, work_dir, bands=("delta",))` writes one CSV per band and channel next to the
raw CSVs (see ecog.config for the layout); `bandRatioToCSV`/`bandPowerToCSV` are the general entry
points for any other band.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import spectrogram, iirnotch, filtfilt

from ecog.cache import isStale, writeSettings
from ecog.config import outputFilename, _recordingLabelFromRawStem

# Defaults for the low-level functions below (every analysis here passes its own values from the Recording).
fs = 400
n_fft = 10000
n_overlap = n_fft / 2

# Band SNR definitions: signal band vs flanking noise bands (Hz).
BAND_RATIOS = {
    "delta": {"signal": (1.5, 4), "noise": [(0.5, 1.5), (4, 7)]},
    "theta": {"signal": (4, 8), "noise": [(2, 4), (8, 13)]},
    "alpha": {"signal": (8, 13), "noise": [(4, 8), (13, 20)]},
    # Capped at 24 Hz with a single lower flank: the EmergRev acquisition low-pass sits near 25 Hz.
    "beta": {"signal": (13, 24), "noise": [(8, 13)]},
}


def _applyNotchFilters(data, fs, notch_freqs, notch_quality=200):
    """Cascade of IIR notch filters, one per frequency in notch_freqs, to remove narrowband artifacts."""
    filtered = data
    for freq in notch_freqs:
        b, a = iirnotch(freq, notch_quality, fs)
        filtered = filtfilt(b, a, filtered)
    return filtered


def _bandMask(f, lo, hi, exclude_lo=False, exclude_hi=False):
    lo_cmp = (f > lo) if exclude_lo else (f >= lo)
    hi_cmp = (f < hi) if exclude_hi else (f <= hi)
    return lo_cmp & hi_cmp


def _noiseBandMask(f, band, signal_band):
    """A noise band's edge is exclusive where it touches the signal band, so the two never double-count a bin."""
    lo, hi = band
    return _bandMask(f, lo, hi, exclude_lo=(lo == signal_band[1]), exclude_hi=(hi == signal_band[0]))


def _spectrogramMetricToCSV(
        input_file: Path, output_folder: Path, output_label: str, metric_fn,
        fs=fs, n_fft=n_fft, n_overlap=n_overlap, notch_freqs=None, notch_quality=30, overwrite=True,
        ):
    """
    Shared scaffold behind bandRatioToCSV/bandPowerToCSV: for each condition
    column, notch-filter (if requested), take its spectrogram, and reduce
    (f, Sxx) to one value per time segment via `metric_fn(f, Sxx) -> 1D
    array` -- the ratio (signal/noise power) or plain band power, depending
    on which public function called this. Writes 'timeSeg' + one column per
    condition to `output_folder / output_label`.
    """
    output_path = output_folder / output_label
    if not overwrite and output_path.exists():
        return output_path

    output_folder.mkdir(parents=True, exist_ok=True)
    ecog = pd.read_csv(input_file)

    df = pd.DataFrame()
    time_array = None  # Placeholder to capture the time array

    for column in ecog:
        signal = ecog[column].to_numpy()
        if notch_freqs:
            signal = _applyNotchFilters(signal, fs, notch_freqs, notch_quality)
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=n_fft, noverlap=n_overlap)
        time_array = t  # Store 't' (time in seconds for each segment)

        df[column] = metric_fn(f, Sxx)

    # Insert the actual time array as the first column called 'timeSeg'
    if time_array is not None:
        df.insert(0, 'timeSeg', time_array)

    # Added index=False to prevent writing an unnamed index column to the CSV
    df.to_csv(output_path, index=False)
    return output_path


def bandRatioToCSV(
        input_file: Path, output_folder: Path, signal_band, noise_bands, output_suffix: str,
        channel_index=0, fs=fs, n_fft=n_fft, n_overlap=n_overlap,
        notch_freqs=None, notch_quality=30,
        output_label: str|None = None, overwrite=True
        ):
    """
    For each condition column, the ratio of mean power in `signal_band` (Hz)
    to mean power in `noise_bands` (a list of (lo, hi) Hz tuples) gives that
    band's SNR over time. Backs deltaRatioToCSV/alphaRatioToCSV, but is
    itself the general entry point for any other band (beta, gamma, theta,
    ...) -- call it directly with that band's own signal_band/noise_bands
    and an output_suffix to name it, e.g.:

        bandRatioToCSV(raw_csv_path, output_folder, signal_band=(13, 30),
                        noise_bands=[(8, 13), (30, 40)], output_suffix="beta_ratio")

    See bandPowerToCSV for plain band power (no noise ratio) instead.
    """
    if not output_label:
        output_label = outputFilename(_recordingLabelFromRawStem(input_file.stem), channel_index, output_suffix)

    def metric(f, Sxx):
        signal_mask = _bandMask(f, *signal_band)
        noise_mask = np.zeros_like(f, dtype=bool)
        for band in noise_bands:
            noise_mask |= _noiseBandMask(f, band, signal_band)
        return Sxx[signal_mask, :].mean(axis=0) / Sxx[noise_mask, :].mean(axis=0)

    return _spectrogramMetricToCSV(
        input_file, output_folder, output_label, metric,
        fs=fs, n_fft=n_fft, n_overlap=n_overlap, notch_freqs=notch_freqs, notch_quality=notch_quality,
        overwrite=overwrite,
    )


def bandPowerToCSV(
        input_file: Path, output_folder: Path, band, output_suffix: str = "band_power",
        channel_index=0, fs=fs, n_fft=n_fft, n_overlap=n_overlap,
        notch_freqs=None, notch_quality=30,
        output_label: str|None = None, overwrite=True
        ):
    """
    Plain mean power (linear, matching spectrogram()'s units -- convert with
    10*np.log10(...) for dB) within `band` (lo, hi Hz) over time, for each
    condition column -- the absolute-power counterpart to bandRatioToCSV's
    signal/noise ratio, e.g. to see a band's power over time rather than its
    SNR relative to neighboring bands:

        bandPowerToCSV(raw_csv_path, output_folder, band=(13, 30), output_suffix="beta_power")

    Read back with getRatioData(..., value_name="power") (its column is
    'ratio' by default, which doesn't fit a plain power value) and plot with
    plotBandPower().
    """
    if not output_label:
        output_label = outputFilename(_recordingLabelFromRawStem(input_file.stem), channel_index, output_suffix)

    def metric(f, Sxx):
        return Sxx[_bandMask(f, *band), :].mean(axis=0)

    return _spectrogramMetricToCSV(
        input_file, output_folder, output_label, metric,
        fs=fs, n_fft=n_fft, n_overlap=n_overlap, notch_freqs=notch_freqs, notch_quality=notch_quality,
        overwrite=overwrite,
    )


def computeBandRatios(rec, work_dir, bands=("delta",), channels=None, overwrite=False, definitions=None):
    """
    Band SNR CSVs for one Recording (raw CSVs must exist: see ecog.edf.extractRecording).

    `bands` are keys of `definitions` (default BAND_RATIOS; a Recording's `extra["bands"]` can override
    single bands as {"beta": {"signal": [13, 30], "noise": [[8, 13], [30, 45]]}}).
    Returns {band: {channel: csv_path}}.
    """
    definitions = {**(definitions or BAND_RATIOS), **{k: v for k, v in rec.extra.get("bands", {}).items()}}
    channels = channels if channels is not None else rec.channels
    paths = {}
    for band in bands:
        if band not in definitions:
            raise KeyError(f"unknown band '{band}' (defined: {sorted(definitions)})")
        d = definitions[band]
        for ch in channels:
            raw = rec.rawCSV(work_dir, ch)
            if not raw.exists():
                raise FileNotFoundError(f"{raw} missing -- run extractRecording first")
            out = rec.bandCSV(work_dir, ch, band)
            settings = {"signal": list(d["signal"]), "noise": [list(b) for b in d["noise"]], "fs": rec.fs,
                        "n_fft": rec.n_fft, "notch_freqs": rec.notch_freqs, "raw_bytes": raw.stat().st_size}
            redo = overwrite or isStale(out, settings)
            existed = out.exists()
            paths.setdefault(band, {})[ch] = bandRatioToCSV(
                raw, output_folder=out.parent, channel_index=ch,
                signal_band=tuple(d["signal"]), noise_bands=[tuple(b) for b in d["noise"]],
                output_suffix=f"{band}_ratio", fs=rec.fs, n_fft=rec.n_fft, n_overlap=rec.n_fft / 2,
                notch_freqs=rec.notch_freqs, notch_quality=rec.notch_quality,
                output_label=out.name, overwrite=redo,
            )
            if redo or not existed:
                writeSettings(out, settings)
    return paths


def computeBandPower(rec, work_dir, bands, channels=None, overwrite=False):
    """Plain band-power CSVs. `bands` is {name: (lo, hi)}; returns {band: {channel: csv_path}}."""
    channels = channels if channels is not None else rec.channels
    paths = {}
    for name, (lo, hi) in bands.items():
        for ch in channels:
            raw, out = rec.rawCSV(work_dir, ch), rec.bandCSV(work_dir, ch, name, "power")
            settings = {"band": [lo, hi], "fs": rec.fs, "n_fft": rec.n_fft, "notch_freqs": rec.notch_freqs,
                        "raw_bytes": raw.stat().st_size}
            redo = overwrite or isStale(out, settings)
            existed = out.exists()
            paths.setdefault(name, {})[ch] = bandPowerToCSV(
                raw, output_folder=out.parent, band=(lo, hi), channel_index=ch, fs=rec.fs, n_fft=rec.n_fft,
                n_overlap=rec.n_fft / 2, notch_freqs=rec.notch_freqs, notch_quality=rec.notch_quality,
                output_label=out.name, overwrite=redo,
            )
            if redo or not existed:
                writeSettings(out, settings)
    return paths
