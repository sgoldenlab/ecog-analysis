# ecog-analysis

ECoG/EEG analysis for mouse recordings (Pinnacle `.edf` exports): a shared, tested library (`ecog/`), the final replicable
analyses of the completed experiments (`Completed_Analyses/`), and notebooks for processing new recordings. It was
refactored from `eeg-analysis`; see `docs/migration_map.md` for where every function went and what changed.

## Layout

```
ecog/                         shared library (import ecog)
  config.py    Recording object, params.json loading
  edf.py       .edf -> raw CSV            emg.py      EMG channel (index 2)
  spectral.py  band SNR / band power      bursts.py   burst suppression
  aggregate.py align + stack recordings   epochs.py   aperiodic fit, band/state metrics, epoch table
  stats.py     paired t, RM-ANOVA, FDR    outputs.py  never-overwrite saving, run record
  cache.py     stale-cache guard          plots/      spectrogram, time series, per-animal panels, figures
Completed_Analyses/
  LSL-DREADD/                 delta SNR after SAL / CNO / ISO ON            (run_analysis.py, params.json, raw_edfs/)
  Emergence/
    raw_edfs/                 the 12 recordings shared by both runs
    ThreeCondition_ReviewerResponse/   AwakeControl, No Induction, Induction (reviewer response)
    TwoCondition_Manuscript/           AwakeControl vs Induction (manuscript)
notebooks/
  fundamentals/   01 edf -> spectrogram, 02 band ratios + epoch table, 03 EMG channel
  analyses/       04 conventional band measures, 05 aperiodic exponent + entropy, 06 combine across animals
docs/migration_map.md         old -> new function map, behaviour differences
tests/                        synthetic-data tests (pytest)
```

## Setup

Tested with Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0.5, statsmodels 0.15, matplotlib 3.11, mne 1.13, fooof 1.1.1
(spectral parameterisation is reported as `specparam/FOOOF v1.1.1`, Donoghue et al. 2020).

```
uv venv && uv pip install -e ".[notebooks,excel,dev]"      # or: pip install -e ".[notebooks,excel,dev]"
```

The final scripts also run without installing (they put the repo root on `sys.path`), as long as the dependencies are present.

## Running a completed analysis

```
python Completed_Analyses/LSL-DREADD/run_analysis.py
python Completed_Analyses/Emergence/ThreeCondition_ReviewerResponse/run_analysis.py
python Completed_Analyses/Emergence/TwoCondition_Manuscript/run_analysis.py
```

Each folder is driven by its own `params.json` (only that analysis' animals and recordings; schema in `ecog/config.py`) and
the settings block at the top of its script. Results go to `outputs/` next to the script, processing caches to `cache/`
(both git-ignored). Outputs are never overwritten: a second run writes `name_1.svg`, `name_2.csv`, ... while caches are
reused (delete `cache/` or pass `overwrite=True` to rebuild). Every run writes `run_info.json` (package versions, settings).
The first run takes about 40-60 s; later runs 12-18 s.

Expected key results (use them to check an installation):

| analysis | check | value |
|---|---|---|
| LSL-DREADD | strip-plot window means (10-20 min), SAL / CNO / ISO ON | 1.070 / 3.536 / 6.311 |
| LSL-DREADD | RM-ANOVA, delta SNR | F(2,8) = 11.19, p = 0.0048 |
| Emergence, three conditions | exponent, 30-55 min, RM-ANOVA | F(2,6) = 79.13, p = 4.9e-05 |
| Emergence, three conditions | exponent means, awake / alone / induced | 0.482 / 0.603 / 1.042 |
| Emergence, two conditions | exponent, paired | t(3) = -12.79, p = 0.0010, q across 2 measures = 0.0021 |
| Emergence, two conditions | spectral entropy, paired | p = 0.021, q = 0.021 |

Raw data: the `.edf` files are in each analysis' `raw_edfs/` folder and are git-ignored (`*.edf`). Remove that line from
`.gitignore`, or `git lfs track "*.edf"`, if the recordings should be versioned.

## Processing a new recording

Open `notebooks/fundamentals/01_edf_to_spectrogram.ipynb`. Each notebook starts with a short variable block (paths,
channel, events, bands, windows as plain Python values; no JSON to edit), then one cell per step. In code:

```python
from ecog import Recording, extractRecording, plotRecordingSpectrogram, computeBandRatios, saveFigure

rec = Recording("PEEG05M", "Induction", 1, edf="my_recording.edf", edf_folder="raw_edfs",
                channel_index=0, events={"onset": 1800})
extractRecording(rec, "cache")                                    # .edf -> raw CSV (reused if present)
fig = plotRecordingSpectrogram(rec, 0, "cache", align_event="onset", start_time=-300, end_time=3600)
saveFigure(fig, "outputs/PEEG05M_spectrogram.svg")                # never overwrites
computeBandRatios(rec, "cache", ["delta"])                        # band SNR CSV
```

`Recording.validate()` checks the file, the channel numbers and the events before a long run.

## Tests

`pytest tests` runs synthetic-data checks of saving, config, band SNR, the epoch table and its cache, statistics, the strip
plot and the EMG cleaning. Agreement with the eeg-analysis results (the table above) was checked on the real recordings.
