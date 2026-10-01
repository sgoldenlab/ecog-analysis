# LSL-DREADD: delta SNR after SAL, CNO and ISO ON

Delta-band (1.5-4 Hz vs 0.5-1.5 / 4-7 Hz flanks) SNR of the EEG in five mice after saline (SAL) and CNO
injection (same SAL-CNO recording, aligned on each injection) and after isoflurane onset (ISO ON, separate
recording), compared in the 10-20 min window after each event.

## Run

    python run_analysis.py        # from this folder, or by path; needs the repo's `ecog/` package (found automatically)

Uses numpy, pandas, scipy, matplotlib, statsmodels and mne (no installation of `ecog` required).

## What the script does

1. Loads `params.json` (5 animals, 10 recordings, one analysed channel per recording, events, spectrogram colour limits and display window).
2. Per recording: EDF -> raw CSV, spectrogram (svg, 500 dpi), delta SNR CSV, and the unaligned delta table.
3. Aligns the recordings on SAL, CNO (both from the SAL-CNO recordings) and ISO ON, concatenates them, and plots the mean delta SNR over time (60 s bins, 120 s smoothing, mean +/- SEM across animals).
4. Strip plot of the per-animal 10-20 min mean (gray lines join each animal across conditions) with its summary table.
5. Repeated-measures ANOVA across SAL / CNO / ISO ON and post-hoc paired t-tests (`POSTHOC_P_ADJUST`).
6. Writes `run_info.json` (versions, settings).

## Folder layout

    params.json        recordings and display settings
    run_analysis.py    the analysis
    raw_edfs/          the 10 EDFs used (copied from eeg-analysis\data\Raw EDFs)
    cache/             raw CSVs and delta SNR CSVs, reused when present (gitignored)
    outputs/           deliverables (gitignored); existing files are never overwritten, a later run adds _1, _2, ...
        spectrograms/<label>_ch<c>_spectrogram.svg
        individual_delta_snr/<label>_ch<c>_delta_ratio.csv      unaligned long table (timeSeg, mouse, condition, ratio)
        combined_delta.csv                                      SAL / CNO / ISO ON, time re-zeroed on the event
        conditions_delta_line.svg, conditions_delta_line_summary.csv
        conditions_delta_strip.svg, conditions_delta_strip_summary.csv
        anova_omnibus.csv, posthoc_paired_t.csv, run_info.json

## Expected key numbers (checked against the original notebook outputs)

Strip summary (per-animal mean over 600-1200 s, n = 5 animals per condition):

| condition | mean | median | std | mad |
|---|---|---|---|---|
| SAL | 1.070079 | 1.102040 | 0.325427 | 0.226172 |
| CNO | 3.535982 | 4.176726 | 1.742685 | 2.323091 |
| ISO ON | 6.311453 | 5.679991 | 3.018030 | 4.021984 |

Repeated-measures ANOVA (60 s bins, 600-1200 s): F(2, 8) = 11.191395, p = 0.004807, all 5 animals included.

Post-hoc paired t-tests (Benjamini-Hochberg, `fdr_bh`):

| pair | raw p | p_adj |
|---|---|---|
| SAL vs CNO | 0.041855 | 0.041855 |
| SAL vs ISO ON | 0.022746 | 0.041855 |
| CNO vs ISO ON | 0.040919 | 0.041855 |

Combined table rows: SAL 2032, CNO 2032, ISO ON 2972. Per-recording delta CSVs equal the original `data\LSL\<animal>\delta\*.csv` exactly.

## Data provenance

The EDFs were copied unchanged from `eeg-analysis\data\Raw EDFs`. The SAL-CNO recording of C00391034-1L is stored there
as `LSL_C00391034-1L_SAL_CNO_1_2026-08-26_13_00_30_export.edf` (byte-identical to the copy in `data\LSL\C00391034-1L\raw_files`).
Channels: C00391035-1L and C00391036-1R were analysed on a different channel in the SAL-CNO and ISO recordings (see `channel_index` in `params.json`).

## Differences from the notebook (`analyze_LSL_EEG.ipynb`)

- Post-hoc correction is `fdr_bh`, which is what the notebook cell ran; its markdown text says Holm. Set `POSTHOC_P_ADJUST = "holm"` for Holm.
- Strip-plot connecting lines are gray (notebook: one colour per animal).
- Two `edf_filename` entries in `params.json` use the real file name (the original had no `LSL_` prefix for C00391034-1L SAL-CNO and no `.edf` extension for C00391036-1R ISO; the original code matched by substring).
- Figures are saved as svg only; no z-score analysis, bar/box plots, within-condition plots or per-animal trace plots.
- Spectrograms keep the library's event legend, which sits over the colour bar as it did in the notebook.
