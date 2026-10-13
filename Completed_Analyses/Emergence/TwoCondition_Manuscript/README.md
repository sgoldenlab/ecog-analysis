# Emergence - two conditions (manuscript)

AwakeControl (0% for 60 min) versus Induction (2.5% induction -> ~1.2% for 30 min -> 0.3% for 60 min), same four mice.
The No Induction (`Alone`) recordings are not used or processed here. Statistics are computed for these two conditions only.

## Run

    python run_analysis.py

Needs `numpy scipy pandas matplotlib mne fooof statsmodels`. First run ~1 min (EDF -> CSV, per-epoch fooof fits); later
runs reuse `cache/`. Deliverables in `outputs/` are never overwritten: a repeated run writes `_1`, `_2`, ... copies.

## Layout

    params.json        4 animals x 2 conditions (8 recordings): EDF name, analysed channel, onset, spectrogram display settings
    run_analysis.py    the whole analysis (settings at the top)
    ../raw_edfs/       EDFs (shared with ThreeCondition_ReviewerResponse; this folder reads 8 of the 12; git-ignored by *.edf)
    cache/             raw CSVs and derived/epoch_table.csv + epoch_spectra.npz (created at run time, git-ignored)
    outputs/           same files as the three-condition folder (spectrograms/, epoch_table.csv, conventional_band_measures.png,
                       aperiodic_exponent.png/.svg, summary_inset.png/.svg, response_stats_anova.csv,
                       response_stats_pairwise.csv, response_per_animal_primary_window.csv, response_fit_quality.csv,
                       run_info.json)

Differences from the three-condition script:

- `response_stats_pairwise.csv` has an extra column `q_measures`: Benjamini-Hochberg across the two reported measures
  (`REPORTED = ["exponent", "spectral_entropy"]`, primary window only; NaN elsewhere).
- With one contrast per measure `p_adj` equals the uncorrected `p`. The summary inset keeps `p_col="p_adj"` (the agreed
  convention for the main-figure panels), so its brackets show the uncorrected paired t-test.
- `DISPLAY_LABELS = {}` renames conditions in the legend of the aperiodic figure, e.g.
  `{"AwakeControl": "Open Field", "Induction": "Emergence"}`; empty = the internal names.

The burst-suppression control (exponent on epochs with <= 1% suppression, `figSuppressionControl`, suppression-free ANOVA
export) is kept in `run_analysis.py` as a commented-out block (section e); uncomment it to run it.

## Expected key numbers (window 30-55 min unless stated)

- 1274 epochs, 8 recordings (AwakeControl 520, Induction 754); identical to the AwakeControl + Induction rows of
  `eeg-analysis/data/EmergRev/derived/epoch_table.csv` (max abs difference 0).
- Exponent: RM-ANOVA F(1,3) = 163.56729, p = 0.001031 (40-55 min p = 0.000091; whole hour p = 0.001553); paired t-test
  0.4824 vs 1.0420, t = -12.7893, p = p_adj = 0.001031; q_measures = 0.0021.
- Spectral entropy: F(1,3) = 19.778, p = 0.0211; 0.9229 vs 0.8638, t = 4.4473; q_measures = 0.0211.
- Conventional band measures (delta/theta/alpha absolute, share, periodic power): only delta share is significant
  (`delta_rel`, F(1,3) = 50.4, p = 0.0057); all other ANOVA p > 0.05.
- `response_stats_anova.csv` / `response_stats_pairwise.csv` (51 rows each) match the stored
  `eeg-analysis/data/EmergRev/derived/response_stats_{anova,pairwise}.csv` to <= 2e-12.
- Group mean (SEM) of per-animal means: exponent 0.482 (0.171) / 1.042 (0.132); spectral entropy 0.923 (0.027) / 0.864 (0.024).

## Data provenance

EDFs copied unchanged from `eeg-analysis/data/Raw EDFs` (`EmergRev_3Perc_<condition>_<animal>_<date>_export.edf`;
file names are listed in `params.json`). ECoG channel per animal: ch0 for PEEG01M, PEEG02M, PEEG04F; ch1 for PEEG03F.
PEEG01M Induction is `Induction_R2` (session of 2026-09-01); its first session (R1) is not used.
Onset = switch to 0.3% (s on the recording clock): AwakeControl 0; Induction 1800 (1860 for PEEG04F).

**PEEG03F Alone_R1 onset note:** the `Alone` recordings are not used in this folder. In the sibling three-condition
`params.json`, PEEG03F Alone_R1 uses onset 300 s (the original `eeg-analysis/params.json` had 0; all reported results used 300).
