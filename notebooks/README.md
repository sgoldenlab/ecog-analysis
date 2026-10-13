# Notebooks

Step-by-step versions of the pipeline in `ecog/`. Each notebook starts with one **Variables** cell (paths, recordings, channels, events, bands, windows, conditions); edit that cell and run the notebook top to bottom. No JSON needs editing, and no notebook depends on another.

| Notebook | Purpose | Main outputs |
|---|---|---|
| [fundamentals/01_edf_to_spectrogram](fundamentals/01_edf_to_spectrogram.ipynb) | `.edf` to raw CSV to time-frequency spectrogram of one recording, with event lines | spectrogram png/svg |
| [fundamentals/02_band_ratios_and_epoch_table](fundamentals/02_band_ratios_and_epoch_table.ipynb) | Process only missing outputs; band SNR ratios (delta/theta/alpha/beta, overridable); per-epoch table | band figure, epoch table csv, fit quality csv |
| [fundamentals/03_emg_channel](fundamentals/03_emg_channel.ipynb) | EMG channel (index 2): line removal, epoch table, movement flag, attach to an EEG epoch table | EMG figure, EMG and merged epoch tables |
| [analyses/04_conventional_band_measures](analyses/04_conventional_band_measures.ipynb) | Absolute / share / periodic delta, theta, alpha per animal and condition, with statistics | figure, ANOVA, pairwise, per-animal, fit quality tables |
| [analyses/05_aperiodic_exponent_and_entropy](analyses/05_aperiodic_exponent_and_entropy.ipynb) | Aperiodic exponent and spectral entropy; FDR across the two measures | aperiodic figure, summary inset, tables (incl. fit quality) |
| [analyses/06_combine_across_animals](analyses/06_combine_across_animals.ipynb) | Aligned band SNR time courses, per-animal window summary, RM-ANOVA and post-hoc | time course, strip plot, aligned/stat tables |

## Conventions

- **Working folder.** Run each notebook with its own folder as the working directory (Jupyter does this by default; in VS Code set `jupyter.notebookFileRoot` to `${fileDirname}`). `ecog/` is found by walking up from there; nothing needs installing.
- **Example data.** `EDF_FOLDER` defaults to `../../Completed_Analyses/Emergence/raw_edfs` (EmergRev recordings). Point it at any folder of `.edf` files.
- **Recordings.** Notebooks 01-03 describe one recording with plain variables. Notebooks 04-06 list recordings in `RECORDINGS` (one dict each: `animal_id`, `condition`, `recording_number`, `edf`, `channel_index`, `events`) and group them in `CONDITIONS = {label: [(index or dict, channel), ...]}`; the key order is the plotting order. To process a new recording, add a dict and reference it in `CONDITIONS`.
- **Outputs.** Figures and tables go to `outputs/<notebook name>/` next to the notebook and are never overwritten: an existing name gets `_1`, `_2`, ... appended (`run_info.json` records versions and settings).
- **Caches.** Raw CSVs, band CSVs and epoch tables go to `cache/` and are reused. Set the `OVERWRITE_*` flag of the Variables cell to recompute (necessary after changing a band definition, a notch filter or `n_fft`; cached files are not compared with those settings). The epoch table is rebuilt automatically when its recordings, onset events or epoch settings change. Notebooks 04 and 05 share `cache/epochs/`.
- **Fit quality.** Notebooks 02, 04 and 05 show and save the quality of the aperiodic fit (`ecog.fitQuality`): mean and SD across animals of each animal's mean `fit_r2` and `fit_error`, and the median and minimum `fit_r2` over epochs. Notebooks 04 and 05 use the primary window and also add both columns to `per_animal_primary_window.csv`. R2 falls for a flat spectrum (exponent near 0) even when the fit is good, so read it with `fit_error`.
- **Stored output.** Notebooks are committed without outputs; `outputs/` and `cache/` are git-ignored.
