"""
Recording description and params.json loading.

A `Recording` is everything the processing functions need to know about one .edf file:
who it is (animal, condition, recording number), where the file is, which channel(s) to
use, named event times (absolute recording-clock seconds), and the display settings used
for its spectrogram. Build one directly in a notebook, or load a set from a params.json:

    rec = Recording("PEEG01M", "Induction", 2, edf="PEEG01M_Induction_2.edf",
                    edf_folder="raw_edfs", channel_index=0, events={"onset": 1800})
    recs = loadRecordings("params.json")          # {(animal_id, recording_key): Recording}

params.json shape (values resolve defaults -> animals.<id> -> animals.<id>.recordings.<key>,
each level overriding the last):

    {"defaults": {...},
     "animals": {"<animal_id>": {"channel_index": 0,
                                 "recordings": {"<condition>_R<n>": {"edf_filename": "...",
                                                                      "events": {"onset": 1800}}}}}}

`raw_edf_folder` (default "raw_edfs") is resolved relative to the params.json file.
Keys the loader does not know are kept in `Recording.extra`.

Output layout under a `work_dir` (the processing cache):

    <work_dir>/<animal_id>/csvs/<label>_raw_ch<c>.csv
    <work_dir>/<animal_id>/<band>/<label>_ch<c>_<band>_ratio.csv      (band SNR)
    <work_dir>/<animal_id>/<band>_power/<label>_ch<c>_<band>_power.csv
    <work_dir>/<animal_id>/bsr/<label>_ch<c>_bsr.csv

with label = `<animal_id>_<condition>_R<recording_number>`.
"""
import json
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

_RECORDING_KEY_PATTERN = re.compile(r"^(?P<condition>.+)_R(?P<recording_number>\d+)$")


def channelIndices(channel_index):
    """Normalize a channel_index (a single int, or a list/tuple of ints) to a list of ints."""
    return [int(channel_index)] if isinstance(channel_index, (int, float)) else [int(c) for c in channel_index]


def recordingLabel(animal_id: str, condition: str, recording_number):
    """Canonical '<animal_id>_<condition>_R<recording_number>' id used to name every file written for one recording."""
    return f"{animal_id}_{condition}_R{recording_number}"


def outputFilename(label: str, channel_index, suffix: str, ext="csv"):
    """'<label>_ch<channel_index>_<suffix>.<ext>' -- the naming convention shared by every derived output."""
    return f"{label}_ch{channel_index}_{suffix}.{ext}"


_RAW_CSV_SUFFIX_PATTERN = re.compile(r"_raw_ch\d+$")


def _recordingLabelFromRawStem(stem: str):
    """Recover a recordingLabel() from a raw CSV's filename stem ('<label>_raw_ch<c>')."""
    return _RAW_CSV_SUFFIX_PATTERN.sub("", stem)


@dataclass
class Recording:
    animal_id: str
    condition: str                      # raw CSV column name, e.g. "Induction", "SAL-CNO", "ISO"
    recording_number: int | str = 1
    edf: str | Path = ""                # file name (looked up in edf_folder) or a full path to the .edf
    edf_folder: str | Path | None = None
    channel_index: int | list = 0       # channel(s) to extract; 0/1 = EEG, 2 = EMG in the Pinnacle exports
    events: dict = field(default_factory=dict)   # {name: seconds}, absolute recording-clock time
    fs: int = 400                       # sampling rate (Hz)
    n_fft: int = 10000                  # spectrogram window (samples); 10000 at 400 Hz = 25 s, 50% overlap
    notch_freqs: list | None = None
    notch_quality: float = 300
    clim: tuple = (-80, -100)           # spectrogram colour limits (dB), (high, low)
    tick_step: int = 600                # x tick spacing (s) on time axes
    event_colors: dict = field(default_factory=dict)
    spectrogram: dict = field(default_factory=dict)   # display defaults: freq_min, freq_max, align_event, start_time, end_time, figsize
    extra: dict = field(default_factory=dict)         # any other params.json keys (e.g. per-recording band overrides)

    def __post_init__(self):
        self.recording_number = str(self.recording_number)
        self.edf_folder = Path(os.path.abspath(self.edf_folder)) if self.edf_folder is not None else None   # absolute, '..' collapsed
        self.events = dict(self.events or {})

    # ---- identity
    @property
    def label(self):
        return recordingLabel(self.animal_id, self.condition, self.recording_number)

    @property
    def mouse_id(self):
        """'<animal_id>_R<recording_number>' -- the 'mouse' column of aggregated tables."""
        return f"{self.animal_id}_R{self.recording_number}"

    @property
    def channels(self):
        return channelIndices(self.channel_index)

    def eventTime(self, name):
        if name not in self.events:
            raise KeyError(f"{self.label}: no event '{name}' (has {sorted(self.events)})")
        return self.events[name]

    # ---- source file
    def edfPath(self):
        """Resolve the .edf: a full path, else an exact name in edf_folder, else a unique name containing `edf`."""
        direct = Path(self.edf)
        if direct.is_file():
            return direct
        folder = self.edf_folder if self.edf_folder is not None else Path(".")
        exact = folder / direct.name
        if exact.is_file():
            return exact
        matches = [f for f in folder.glob("*.edf") if direct.name and direct.name in f.name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"{self.label}: no .edf matching '{self.edf}' in {folder.resolve()}")
        raise ValueError(f"{self.label}: '{self.edf}' matches {len(matches)} files in {folder.resolve()}: "
                         f"{[m.name for m in matches]}")

    # ---- derived-file paths (see module docstring)
    def rawCSV(self, work_dir, channel):
        return Path(work_dir) / self.animal_id / "csvs" / f"{self.label}_raw_ch{channel}.csv"

    def bandCSV(self, work_dir, channel, band, kind="ratio"):
        """kind 'ratio' -> <band>/..._<band>_ratio.csv; kind 'power' -> <band>_power/..._<band>_power.csv."""
        if kind == "ratio":
            return Path(work_dir) / self.animal_id / band / outputFilename(self.label, channel, f"{band}_ratio")
        return Path(work_dir) / self.animal_id / f"{band}_power" / outputFilename(self.label, channel, f"{band}_power")

    def bsrCSV(self, work_dir, channel):
        return Path(work_dir) / self.animal_id / "bsr" / outputFilename(self.label, channel, "bsr")

    # ---- checks
    def validate(self, check_edf=True):
        """Raise one ValueError listing every problem (missing file, channel out of range, bad events)."""
        problems = []
        for name, t in self.events.items():
            if not isinstance(t, (int, float)):
                problems.append(f"event '{name}' = {t!r} is not a number of seconds")
        if check_edf:
            try:
                edf = self.edfPath()
            except (FileNotFoundError, ValueError) as exc:
                problems.append(str(exc))
            else:
                import mne
                header = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
                for ch in self.channels:
                    if not 0 <= ch < len(header.ch_names):
                        problems.append(f"channel_index {ch} out of range for {edf.name} (channels: {header.ch_names})")
        if problems:
            raise ValueError(f"Recording {self.label}:\n  - " + "\n  - ".join(problems))
        return self


_RECORDING_FIELDS = {f.name for f in fields(Recording)}


def loadParams(path) -> dict:
    """Parse a params.json into a plain dict."""
    with open(path) as f:
        return json.load(f)


def getRecordingParams(config: dict, animal_id: str, recording_key: str, **overrides) -> dict:
    """Merge defaults -> animals.<animal_id> -> animals.<animal_id>.recordings.<recording_key> -> overrides."""
    animals = config.get("animals", {})
    if animal_id not in animals:
        raise KeyError(f"No animals.{animal_id} entry in params config")
    animal_cfg = animals[animal_id]
    recordings = animal_cfg.get("recordings", {})
    if recording_key not in recordings:
        raise KeyError(f"No animals.{animal_id}.recordings.{recording_key} entry in params config "
                       f"(has {sorted(recordings)})")
    merged = {**config.get("defaults", {}),
              **{k: v for k, v in animal_cfg.items() if k != "recordings"},
              **recordings[recording_key], **overrides}
    match = _RECORDING_KEY_PATTERN.match(recording_key)
    if not match:
        raise ValueError(f"Recording key '{recording_key}' doesn't match '<condition>_R<recording_number>'")
    merged.setdefault("condition", match.group("condition"))
    merged["recording_number"] = match.group("recording_number")
    merged["animal_id"] = animal_id
    return merged


def recordingFromParams(params: dict, base_dir=None) -> Recording:
    """Build a Recording from one merged params dict (see getRecordingParams)."""
    base_dir = Path(base_dir) if base_dir else Path(".")
    kwargs = {k: v for k, v in params.items() if k in _RECORDING_FIELDS}
    extra = {k: v for k, v in params.items()
             if k not in _RECORDING_FIELDS and k not in ("edf_filename", "raw_edf_folder", "recording_key")}
    kwargs["edf"] = params.get("edf_filename") or params.get("edf") or ""
    folder = params.get("raw_edf_folder", params.get("edf_folder", "raw_edfs"))
    folder = Path(folder)
    kwargs["edf_folder"] = folder if folder.is_absolute() else base_dir / folder
    kwargs["extra"] = {**params.get("extra", {}), **extra}
    if "clim" in kwargs:
        kwargs["clim"] = tuple(kwargs["clim"])
    return Recording(**kwargs)


def loadRecordings(params_path, keys=None):
    """
    Load every recording in a params.json as {(animal_id, recording_key): Recording}.
    `keys` optionally restricts to a list of (animal_id, recording_key) pairs, in that order.
    """
    params_path = Path(params_path)
    config = loadParams(params_path)
    base_dir = params_path.parent
    if keys is None:
        keys = [(a, k) for a, cfg in config["animals"].items() for k in cfg.get("recordings", {})]
    return {(a, k): recordingFromParams(getRecordingParams(config, a, k), base_dir) for a, k in keys}
