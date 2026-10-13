# Emergence - three conditions (reviewer response)

Same four mice at a matched 0.3% isoflurane, three conditions: **AwakeControl** (0% for 60 min), **No Induction**
(0% -> 0.3% for 60 min, recording key `Alone_R1`) and **Induction** (2.5% induction -> ~1.2% for 30 min -> 0.3% for 60 min).
Question: is 0.3% after induction the same state as 0.3% in a naive animal? Conventional band measures do not separate
the two 0.3% conditions; the aperiodic background (exponent) does.

## Run

    python run_analysis.py

Needs `numpy scipy pandas matplotlib mne fooof statsmodels` (no installation of the repo itself; the script finds `ecog/`
in a parent folder). First run ~1-2 min (EDF -> CSV, per-epoch fooof fits); later runs reuse `cache/`. Deliverables in
`outputs/` are never overwritten: a repeated run writes `_1`, `_2`, ... copies.

## Layout

    params.json        4 animals x 3 conditions (12 recordings): EDF name, analysed channel, onset, spectrogram display settings
    run_analysis.py    the whole analysis (settings at the top)
    ../raw_edfs/       the 12 EDFs (shared with TwoCondition_Manuscript; git-ignored by *.edf)
    cache/             raw CSVs and derived/epoch_table.csv + epoch_spectra.npz (created at run time, git-ignored)
    outputs/           created at run time, git-ignored:
      spectrograms/<animal>_<condition>_R<n>_ch<c>_spectrogram.svg   0-20 Hz, onset-aligned, 0-3600 s, dpi 500
      epoch_table.csv                                   30-s epochs: aperiodic fit, band powers, state metrics, suppression
      conventional_band_measures.png                    absolute / share / periodic band power (dpi 300)
      aperiodic_exponent.png/.svg                       panels a (spectra + fitted background) and b (exponent over time), dpi 600
      summary_inset.png/.svg                            background slope and spectral entropy, paired per animal, dpi 600
      response_stats_anova.csv, response_stats_pairwise.csv
      response_per_animal_primary_window.csv            per-animal means of every tested measure, plus fit_r2 and fit_error
      response_fit_quality.csv                          aperiodic fit R2 and error per condition, primary window
      run_info.json                                     date, package versions, settings

Statistics: per-animal mean in the window, repeated-measures ANOVA across conditions, paired t-tests between conditions,
`p_adj` = Benjamini-Hochberg across the pairwise contrasts within metric x window (`P_ADJUST = "fdr_bh"`; the original
notebook text said Holm, the code ran BH). Primary window `30-55 min` = 1800-3350 s after onset (recordings end at ~3350 s).

Fit quality (`ecog.fitQuality`, not tested): per condition, the mean and SD across animals of each animal's mean
`fit_r2` and `fit_error` over its epochs in the primary window, and the median and minimum `fit_r2` over those epochs.
R2 falls for a flat spectrum (exponent near 0) even when the model follows the data; `fit_error` (mean absolute
error, log10 power) does not depend on the slope, so the two are reported together.

The burst-suppression control (exponent on epochs with <= 1% suppression, `figSuppressionControl`, suppression-free ANOVA
export) is kept in `run_analysis.py` as a commented-out block (section e); uncomment it to run it.

## Expected key numbers (window 30-55 min unless stated)

- 1862 epochs, 12 recordings (AwakeControl 520, No Induction 588, Induction 754); `epoch_table.csv` is identical to
  `eeg-analysis/data/EmergRev/derived/epoch_table.csv` (max abs difference 0).
- Exponent ANOVA: F(2,6) = 79.12629, p = 4.9e-05, n = 4 animals; 40-55 min F = 33.97866; whole hour 0-55 min F = 89.87804.
- Exponent pairwise: AwakeControl vs No Induction 0.4824 vs 0.6032, t = -2.2202, p = 0.1130; AwakeControl vs Induction
  p = 0.0010, p_adj = 0.0027; No Induction vs Induction p = 0.0018, p_adj = 0.0027.
- Group mean (SEM) of per-animal means: exponent 0.482 (0.171) / 0.603 (0.143) / 1.042 (0.132); spectral entropy
  0.923 (0.027) / 0.914 (0.032) / 0.864 (0.024) (AwakeControl / No Induction / Induction); entropy ANOVA F(2,6) = 8.977, p = 0.0157.
- Conventional band measures (absolute, share, periodic power of delta/theta/alpha): no ANOVA p < 0.05 across the three conditions.

## Data provenance

EDFs copied unchanged from `eeg-analysis/data/Raw EDFs` (`EmergRev_3Perc_<condition>_<animal>_<date>_export.edf`;
file names are listed in `params.json`). ECoG channel per animal: ch0 for PEEG01M, PEEG02M, PEEG04F; ch1 for PEEG03F.
PEEG01M Induction is `Induction_R2` (session of 2026-09-01); its first session (R1) is not used.
Onset = switch to 0.3% (s on the recording clock): AwakeControl 0; Alone 900 (PEEG01M, PEEG04F), 300 (PEEG02M, PEEG03F);
Induction 1800 (1860 for PEEG04F).

**PEEG03F Alone_R1 onset note:** `eeg-analysis/params.json` has `events.onset = 0` for this recording, but the cached
epoch table and all reported results used 300 s (same protocol as PEEG02M Alone_R1). `params.json` here uses 300 and
carries a `_note` key on that recording saying so.
