"""
EDF -> raw CSV extraction.

`extractRecording(rec, work_dir)` converts one Recording's .edf into one single-column raw CSV
per requested channel (the whole recording, volts, column named after the condition) under
`<work_dir>/<animal_id>/csvs/`. Existing CSVs are reused unless `overwrite=True`.
Every downstream function (spectrogram, band ratios, epoch table, burst suppression) reads those CSVs.
"""
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ecog.cache import isStale, writeSettings
from ecog.config import channelIndices, recordingLabel

# '<animal_id>_<condition>_R<recording_number>' -- only used when ecogToCSV has to infer them from a file name
_RECORDING_FILENAME_PATTERN = re.compile(r"^(?P<condition>.+)_R(?P<recording_number>\d+)$")


def _parseRecordingFilename(file_stem: str, animal_id: str):
    """Split a '<animal_id>_<condition>_R<recording_number>' filename into its parts."""
    stem = file_stem.removeprefix(f"{animal_id}_")
    match = _RECORDING_FILENAME_PATTERN.match(stem)
    if not match:
        raise ValueError(
            f"Filename '{file_stem}' doesn't match '{animal_id}_<condition>_R<recording_number>'"
        )
    return match.group("condition"), match.group("recording_number")


def ecogToCSV(raw_edf_folder: Path, output_folder: Path, animal_id: str | None = None, channel_index=0,
              notch_freqs=None, notch_quality=200, file_filter=None, start_time=None, end_time=None,
              overwrite=True, condition=None, recording_number=None, allow_emg=False):
    """
    Convert every .edf in `raw_edf_folder` to a raw CSV, one per (recording,
    condition, channel) triple, named
    '<animal_id>_<condition>_R<recording_number>_raw_ch<channel_index>.csv'.

    `raw_edf_folder` is either an animal-specific raw_files folder (e.g.
    data/Pilots/PEEG00/raw_files) -- in which case `animal_id` defaults to
    its parent folder name -- or a shared central folder holding .edf's for
    many animals (e.g. data/Raw EDFs), in which case `animal_id` must be
    passed explicitly since it can't be inferred from the folder. Each .edf's
    own condition/recording number normally come from its filename
    ('<animal_id>_<condition>_R<recording_number>.edf').

    Recordings whose source .edf doesn't follow that naming (e.g. an export
    named by acquisition software rather than by this convention) can pass
    `condition`/`recording_number` explicitly instead -- this skips filename
    parsing entirely and uses the given values for every matched file, so
    `file_filter` should narrow the match to exactly one .edf in that case.

    `channel_index` may be a single int or a list of ints (e.g. [0, 1] to get
    both EEG channels most of these recordings have); each requested channel
    gets its own output CSV, and a given .edf is only read/cropped once no
    matter how many of its channels are requested.

    `overwrite=False` skips a channel's output entirely (without reading its
    .edf, if every requested channel for that file already has one) if that
    output CSV already exists.
    """
    animal_id = animal_id or Path(raw_edf_folder).parent.name
    channels = channelIndices(channel_index)

    raw_edf_folder, output_folder = Path(raw_edf_folder), Path(output_folder)
    files = list(raw_edf_folder.glob("*.edf"))
    if file_filter:
        exact = [f for f in files if f.name == file_filter]
        files = exact or [f for f in files if file_filter in f.name]
    if not files:
        raise ValueError(f"No .edf files found in {raw_edf_folder} matching filter '{file_filter}'")
    if (condition is not None or recording_number is not None) and len(files) != 1:
        raise ValueError(
            f"condition/recording_number were given explicitly, but file_filter '{file_filter}' "
            f"matched {len(files)} files in {raw_edf_folder} -- narrow file_filter to exactly one .edf"
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    for file in files:
        file_condition, file_recording_number = (
            (condition, recording_number) if condition is not None or recording_number is not None
            else _parseRecordingFilename(file.stem, animal_id)
        )
        label = recordingLabel(animal_id, file_condition, file_recording_number)
        output_paths = {ch: output_folder / f"{label}_raw_ch{ch}.csv" for ch in channels}

        pending_channels = channels if overwrite else [ch for ch in channels if not output_paths[ch].exists()]
        if not pending_channels:
            continue

        data = mne.io.read_raw_edf(file, preload=True)

        if start_time is not None or end_time is not None:
            data.crop(tmin=start_time if start_time is not None else 0.0, tmax=end_time)

        for ch in pending_channels:
            if not 0 <= ch < len(data.ch_names):
                raise ValueError(f"{file.name}: channel_index {ch} out of range (channels: {data.ch_names})")
            if "EMG" in data.ch_names[ch].upper() and not allow_emg:
                print(f"note: {file.name} channel {ch} is '{data.ch_names[ch]}' (EMG); pass allow_emg=True to silence")
            channel_data = data.copy()
            if notch_freqs:
                # MNE has no direct Q parameter; notch_widths is its bandwidth knob,
                # so convert using the same Q = freq / bandwidth definition iirnotch uses.
                notch_widths = np.asarray(notch_freqs) / notch_quality
                channel_data.notch_filter(freqs=notch_freqs, picks=[channel_data.ch_names[ch]], notch_widths=notch_widths)

            # Extract data without the hardcoded 1-hour np.pad()
            ecog = channel_data.pick([channel_data.ch_names[ch]]).get_data()[0]
            pd.DataFrame({file_condition: ecog}).to_csv(output_paths[ch], index=False)


def _rawSettings(rec, edf):
    return {"edf": edf.name, "edf_bytes": edf.stat().st_size, "notch_freqs": rec.notch_freqs,
            "notch_quality": rec.notch_quality if rec.notch_freqs else None}


def extractRecording(rec, work_dir, overwrite=False, allow_emg=False):
    """
    Run ecogToCSV for one Recording; returns {channel: raw_csv_path}. Existing CSVs are reused unless
    `overwrite=True` or they were made from a different EDF / notch setting (see ecog.cache).
    """
    edf = rec.edfPath()
    settings = _rawSettings(rec, edf)
    paths = {ch: rec.rawCSV(work_dir, ch) for ch in rec.channels}
    existed = {ch: p.exists() for ch, p in paths.items()}
    stale = [ch for ch, p in paths.items() if isStale(p, settings)]
    for channels, force in ((stale, True), ([c for c in rec.channels if c not in stale], overwrite)):
        if channels:
            ecogToCSV(
                edf.parent, output_folder=Path(work_dir) / rec.animal_id / "csvs", animal_id=rec.animal_id,
                channel_index=channels, notch_freqs=rec.notch_freqs, notch_quality=rec.notch_quality,
                file_filter=edf.name, condition=rec.condition, recording_number=rec.recording_number,
                overwrite=force, allow_emg=allow_emg,
            )
    for ch, path in paths.items():
        if overwrite or ch in stale or not existed[ch]:
            writeSettings(path, settings)
    return paths
