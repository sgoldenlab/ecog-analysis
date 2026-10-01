"""
Burst suppression: samples held below an amplitude threshold for a minimum duration.

`_detectSuppression` is the shared detector. `burstSuppressionRatioToCSV` writes the pipeline-style
BSR (% of 60 s epochs suppressed, optionally with a threshold relative to the recording's own baseline SD);
ecog.epochs.suppressionByEpoch applies an ABSOLUTE microvolt threshold per 30 s epoch, the version to use
for cross-condition statements.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ecog.config import outputFilename

fs = 400


def _detectSuppression(data, fs, amplitude_threshold=20, min_duration=0.5, scale_to_uv=1e6,
                        baseline_std_pct=None, baseline_duration=60):
    """
    Label each sample as suppressed (True) or burst (False).

    `data` is assumed to be in volts, as written by ecogToCSV; it's scaled by
    `scale_to_uv` before thresholding so `amplitude_threshold` can be given in uV.
    Pass scale_to_uv=1 if `data` is already in uV.
    `fs` is the sampling rate (Hz) of `data`.
    `min_duration` is how long (s) the signal must stay continuously below
    the threshold to count as suppression, so brief normal dips aren't
    miscounted (the spec's default is 0.5s).

    If `baseline_std_pct` is given, the threshold is that percentage of the
    standard deviation over the first `baseline_duration` seconds instead of
    the fixed `amplitude_threshold`.
    """
    data_uv = data * scale_to_uv

    if baseline_std_pct is not None:
        baseline_samples = int(baseline_duration * fs)
        threshold = (baseline_std_pct / 100) * np.std(data_uv[:baseline_samples])
    else:
        threshold = amplitude_threshold

    below_threshold = np.abs(data_uv) < threshold
    min_samples = int(min_duration * fs)

    # Walk contiguous runs (not sample-by-sample) so a run only counts as
    # suppression once it's held below threshold for min_duration.
    change_points = np.flatnonzero(np.diff(below_threshold.astype(int)) != 0) + 1
    run_starts = np.concatenate(([0], change_points))
    run_ends = np.concatenate((change_points, [len(below_threshold)]))

    suppressed = np.zeros_like(below_threshold, dtype=bool)
    for start, end in zip(run_starts, run_ends):
        if below_threshold[start] and (end - start) >= min_samples:
            suppressed[start:end] = True

    return suppressed


def _epochBSR(suppressed, fs, epoch_duration=60, epoch_overlap=0):
    """
    Percentage of each epoch spent suppressed, plus the epoch start times (s).

    `suppressed` is the boolean per-sample mask from `_detectSuppression`,
    sampled at `fs` Hz.
    `epoch_duration` is the window length (s) each BSR percentage is averaged
    over (the spec's "epoch", e.g. 60s for a 1-minute BSR).
    `epoch_overlap` is how much (s) consecutive epochs overlap; 0 gives
    non-overlapping epochs, a value > 0 gives a sliding window instead.
    """
    epoch_samples = int(epoch_duration * fs)
    step_samples = epoch_samples - int(epoch_overlap * fs)
    if step_samples <= 0:
        raise ValueError("epoch_overlap must be smaller than epoch_duration")

    starts = np.arange(0, max(len(suppressed) - epoch_samples, 0) + 1, step_samples)
    bsr = np.array([suppressed[start:start + epoch_samples].mean() * 100 for start in starts])
    time_array = starts / fs

    return time_array, bsr


def burstSuppressionRatioToCSV(input_file: Path, output_folder: Path, channel_index=0, fs=fs,
                                amplitude_threshold=20, min_duration=0.5, scale_to_uv=1e6,
                                baseline_std_pct=None, baseline_duration=60, epoch_duration=60,
                                epoch_overlap=0, output_label: str | None = None, overwrite=True):
    """
    Compute Burst Suppression Ratio (%) for every condition column in a raw
    CSV (as written by ecogToCSV) and write it to a '<recording_id>_bsr.csv'.

    `input_file` is a raw per-channel CSV; each column is one condition and
    gets its own BSR column in the output.
    `output_folder` is where the output is written; `channel_index` only
    affects the output filename (`input_file` is already one channel's data),
    so pass whatever channel `input_file` was actually extracted from.
    `fs` is the sampling rate (Hz) of `input_file`.
    `amplitude_threshold`, `min_duration`, `scale_to_uv`, `baseline_std_pct`,
    `baseline_duration` are passed straight to `_detectSuppression` -- see
    there for what each controls.
    `epoch_duration`, `epoch_overlap` are passed straight to `_epochBSR`.
    `overwrite=False` skips recomputation if the output file already exists.
    """
    input_file, output_folder = Path(input_file), Path(output_folder)
    if not output_label:
        output_label = outputFilename(input_file.stem.replace(f"_raw_ch{channel_index}", ""), channel_index, "bsr")
    output_path = output_folder / output_label
    if not overwrite and output_path.exists():
        return output_path

    output_folder.mkdir(parents=True, exist_ok=True)
    ecog = pd.read_csv(input_file)

    df = pd.DataFrame()
    time_array = None  # Placeholder to capture the time array

    for column in ecog:
        suppressed = _detectSuppression(
            ecog[column].to_numpy(), fs, amplitude_threshold, min_duration,
            scale_to_uv, baseline_std_pct, baseline_duration
        )
        time_array, bsr = _epochBSR(suppressed, fs, epoch_duration, epoch_overlap)
        df[column] = bsr

    # Insert the actual time array as the first column called 'timeSeg'
    if time_array is not None:
        df.insert(0, 'timeSeg', time_array)

    df.to_csv(output_path, index=False)
    return output_path
