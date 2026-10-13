"""Emergence, three conditions (AwakeControl, No Induction, Induction): reviewer-response analysis. Run: python run_analysis.py"""
import matplotlib; matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # closes figures after saving
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent  # this script folder
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "ecog").is_dir())  # repo root (contains ecog/)
sys.path.insert(0, str(ROOT))  # makes `ecog` importable without installing

import ecog  # shared analysis library
from ecog.plots import plotRecordingSpectrogram, figConventionalMeasures, figAperiodic, figSummaryInset  # , figSuppressionControl

# ---- settings
PARAMS = HERE / "params.json"  # recordings, channels, onsets, spectrogram display settings
OUT = HERE / "outputs"  # deliverables; files are never overwritten (_1, _2, ... are appended)
CACHE = HERE / "cache"  # raw CSVs and epoch-table cache; reused when present
RECORDINGS = {  # displayed condition -> [(animal, recording key in params.json)]
    "AwakeControl": [("PEEG01M", "AwakeControl_R1"), ("PEEG02M", "AwakeControl_R1"), ("PEEG03F", "AwakeControl_R1"), ("PEEG04F", "AwakeControl_R1")],
    "No Induction": [("PEEG01M", "Alone_R1"), ("PEEG02M", "Alone_R1"), ("PEEG03F", "Alone_R1"), ("PEEG04F", "Alone_R1")],
    "Induction": [("PEEG01M", "Induction_R2"), ("PEEG02M", "Induction_R1"), ("PEEG03F", "Induction_R1"), ("PEEG04F", "Induction_R1")],
}
ORDER = ["AwakeControl", "No Induction", "Induction"]  # condition order in tables and figures
PRIMARY = "30-55 min"  # analysis window shown in the figures
WINDOWS = {PRIMARY: (1800, 3350), "40-55 min": (2400, 3300), "whole hour 0-55 min": (0, 3350)}  # seconds after onset
METRICS = ["exponent", "offset", "total_power",
           "delta_abs", "theta_abs", "alpha_abs",
           "delta_rel", "theta_rel", "alpha_rel",
           "delta_periodic", "theta_periodic", "alpha_periodic",
           "median_freq", "spectral_entropy", "sef95", "suppression_pct", "mean_abs_uv"]  # epoch-table measures tested
FIT_QUALITY = ["fit_r2", "fit_error"]  # aperiodic fit quality, exported per animal (not tested)
P_ADJUST = "fdr_bh"  # pairwise correction across contrasts within metric x window (Benjamini-Hochberg)
INSET_BBOX = None  # inset canvas: None = exact figure size (as in eeg-analysis; long titles are clipped); "tight" keeps the full title
SUMMARY_PANELS = (("exponent", "Aperiodic exponent", "Background slope"),
                  ("spectral_entropy", "Spectral entropy (0-1)", "Spectral entropy"))  # inset panels: (metric, y label, title)

# ---- a. EDF -> raw CSV and time-frequency spectrogram per recording
recs = ecog.loadRecordings(PARAMS)  # {(animal, key): Recording}
for (animal, key), rec in recs.items():
    ecog.extractRecording(rec, CACHE)  # writes the raw CSV of the analysed channel (reused if present)
    for ch in rec.channels:  # analysed channel of this animal
        fig = plotRecordingSpectrogram(rec, ch, CACHE)  # 0-20 Hz, aligned to onset, 0-3600 s (params.json)
        ecog.saveFigure(fig, OUT / "spectrograms" / f"{rec.label}_ch{ch}_spectrogram.svg", dpi=500)  # svg, tight bbox
        plt.close(fig)  # frees the figure

# ---- b. per-epoch table (aperiodic fit, band powers, state metrics)
BATCHES = {cond: [(recs[(a, k)], recs[(a, k)].channels[0]) for a, k in keys] for cond, keys in RECORDINGS.items()}  # {condition: [(Recording, channel)]}
table, spectra = ecog.buildEpochTable(BATCHES, CACHE, cache=CACHE / "derived", onset_event="onset")  # epoch table + spectra, cached
ecog.saveTable(table, OUT / "epoch_table.csv")  # epoch table as a deliverable

# ---- c. conventional band measures
anova, pairwise = ecog.compareAcrossConditions(table, METRICS, WINDOWS, ORDER, p_adjust=P_ADJUST)  # RM-ANOVA + paired t-tests per metric x window
fig = figConventionalMeasures(table, WINDOWS[PRIMARY], ORDER, anova=anova, window_label=PRIMARY)  # absolute / relative / periodic band power
ecog.saveFigure(fig, OUT / "conventional_band_measures.png", dpi=300, bbox_inches=None)  # png, no tight cropping
plt.close(fig)  # frees the figure

# ---- d. aperiodic decomposition (panels a and b)
fig = figAperiodic(table, spectra, WINDOWS[PRIMARY], ORDER, anova=anova, window_label=PRIMARY, panels="ab")  # a: spectra + fits, b: exponent over time
ecog.saveFigure(fig, OUT / "aperiodic_exponent", dpi=600, bbox_inches=None, formats=("png", "svg"))  # png + svg, no tight cropping
plt.close(fig)  # frees the figure
print(anova[anova.metric == "exponent"][["window", "F", "df_num", "df_den", "p", "n_animals"]].round(5).to_string(index=False))  # exponent ANOVA, all windows
print(pairwise[(pairwise.metric == "exponent") & (pairwise.window == PRIMARY)]
      [["condition_1", "condition_2", "n", "mean_1", "mean_2", "t", "p", "p_adj"]].round(4).to_string(index=False))  # exponent pairwise, primary window

# ---- e. burst-suppression control (uncomment to run): exponent comparison on epochs with <= 1% suppression
# sup_by_animal = (ecog.perAnimalMeans(table, ["suppression_pct"], WINDOWS[PRIMARY], ORDER)
#                  .pivot(index="animal_id", columns="condition", values="suppression_pct"))  # mean % of epoch suppressed per animal
# clean = table[table.suppression_pct <= 1.0]  # epochs with essentially no suppression
# anova_clean, pairwise_clean = ecog.compareAcrossConditions(clean, ["exponent"], WINDOWS, ORDER, p_adjust=P_ADJUST)  # exponent statistics on clean epochs
# fig = figSuppressionControl(table, WINDOWS[PRIMARY], ORDER, anova=anova, anova_clean=anova_clean, window_label=PRIMARY)  # suppression vs exponent
# ecog.saveFigure(fig, OUT / "suppression_control.png", dpi=300, bbox_inches=None)  # png, no tight cropping
# plt.close(fig)  # frees the figure
# ecog.saveTable(anova_clean, OUT / "response_stats_anova_suppression_free.csv")  # suppression-free ANOVA export
# print(sup_by_animal.round(2).to_string())  # suppression per animal
# print(anova_clean[anova_clean.window == PRIMARY][["F", "df_num", "df_den", "p", "n_animals", "excluded"]].round(4).to_string(index=False))

# ---- f. summary metrics, inset (background slope and spectral entropy)
fig = figSummaryInset(table, WINDOWS[PRIMARY], ORDER, anova, pairwise, PRIMARY, panels=SUMMARY_PANELS)  # paired panels with significance brackets
ecog.saveFigure(fig, OUT / "summary_inset", dpi=600, bbox_inches=INSET_BBOX, formats=("png", "svg"))  # png + svg
plt.close(fig)  # frees the figure
summary_metrics = [m for m, _, _ in SUMMARY_PANELS]  # metrics shown in the inset
print(ecog.perAnimalMeans(table, summary_metrics, WINDOWS[PRIMARY], ORDER)
      .groupby("condition", observed=True)[summary_metrics].agg(["mean", "sem"]).round(3).to_string())  # group mean and SEM across animals

# ---- g. exports
ecog.saveTable(anova, OUT / "response_stats_anova.csv")  # RM-ANOVA, every metric x window
ecog.saveTable(pairwise, OUT / "response_stats_pairwise.csv")  # paired t-tests, every metric x window x contrast
ecog.saveTable(ecog.perAnimalMeans(table, METRICS + FIT_QUALITY, WINDOWS[PRIMARY], ORDER), OUT / "response_per_animal_primary_window.csv")  # per-animal means + fit quality, primary window
fit_quality = ecog.fitQuality(table, WINDOWS[PRIMARY], ORDER)  # fit R2 and error per condition: mean/SD across animals, median/min over epochs
ecog.saveTable(fit_quality, OUT / "response_fit_quality.csv")  # aperiodic fit quality, primary window
print(fit_quality.round(3).to_string(index=False))  # fit quality per condition

# ---- h. run record
ecog.writeRunInfo(OUT, {"params": PARAMS.name, "recordings": RECORDINGS, "order": ORDER, "primary": PRIMARY, "windows": WINDOWS,
                        "metrics": METRICS, "p_adjust": P_ADJUST, "summary_panels": SUMMARY_PANELS})  # timestamp, versions, settings
exp = anova[(anova.metric == "exponent") & (anova.window == PRIMARY)].iloc[0]  # exponent ANOVA, primary window
print(f"exponent {PRIMARY}: F({exp.df_num:.0f},{exp.df_den:.0f}) = {exp.F:.5f}, p = {exp.p:.2g}, n = {exp.n_animals:.0f} animals")  # key number
