"""LSL-DREADD: delta SNR after SAL / CNO (SAL-CNO recordings) and ISO ON (ISO recordings). Run: python run_analysis.py"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "ecog").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ecog
from ecog.plots import plotRecordingSpectrogram, plotRatioSNR, binTimeseries, plot_condition_stripplot

# ---- settings
PARAMS = HERE / "params.json"                  # recordings, events, clim, spectrogram display settings
CACHE = HERE / "cache"                         # raw CSVs and delta SNR CSVs (reused when present)
OUT = HERE / "outputs"                         # deliverables (never overwritten)
SAL_CNO_RECORDINGS = [                         # (animal, recording key, channel) of the SAL-CNO recordings
    ("C00391034-1L", "SAL-CNO_R1", 1), ("C00391035-1L", "SAL-CNO_R1", 1), ("C00391035-2R", "SAL-CNO_R1", 1),
    ("C00391036-1L", "SAL-CNO_R1", 0), ("C00391036-1R", "SAL-CNO_R1", 0),
]
ISO_RECORDINGS = [                             # (animal, recording key, channel) of the ISO recordings
    ("C00391034-1L", "ISO_R1", 1), ("C00391035-2R", "ISO_R1", 1), ("C00391035-1L", "ISO_R1", 0),
    ("C00391036-1R", "ISO_R1", 1), ("C00391036-1L", "ISO_R1", 0),
]
CONDITION_BATCHES = {                          # condition -> recordings and the event placed at t = 0
    "SAL": {"recordings": SAL_CNO_RECORDINGS, "align_event": "SAL"},
    "CNO": {"recordings": SAL_CNO_RECORDINGS, "align_event": "CNO"},
    "ISO ON": {"recordings": ISO_RECORDINGS, "align_event": "ISO ON"},
}
CONDITION_COLORS = {"SAL": "gray", "CNO": "#9C1717", "ISO ON": "#5F4EAB"}   # line / marker colour per condition
BINNING_S = 60                                 # time bin (s) for the line plot and the stats
SMOOTHING_S = 120                              # EWMA half-life (s) of the plotted mean line only
TICK_STEP = 600                                # x tick spacing (s)
LINE_XLIM, LINE_YLIM = (-600, 1200), (0.5, 10)  # line plot axis ranges (s, SNR)
RESPONSE_WINDOW = (600, 1200)                  # post-event window (s) for strip plot and stats
POSTHOC_P_ADJUST = "fdr_bh"                    # what the notebook ran (its markdown says Holm; "holm" is the alternative)
PLOT_FORMATS = ("svg",)                        # figure file formats
DPI = 500                                      # figure dpi (spectrogram raster)

# ---- recordings
recs = ecog.loadRecordings(PARAMS)             # {(animal, key): Recording}
for batch in CONDITION_BATCHES.values():       # channel lists above must agree with params.json
    for a, k, ch in batch["recordings"]:
        assert recs[(a, k)].channels == [ch], f"{a} {k}: channel {ch} not in params.json ({recs[(a, k)].channels})"
all_entries = list(dict.fromkeys((a, k, ch) for b in CONDITION_BATCHES.values() for a, k, ch in b["recordings"]))   # each recording once

# ---- a/b: raw CSV, spectrogram and delta SNR per recording
for a, k, ch in all_entries:
    rec = recs[(a, k)]
    ecog.extractRecording(rec, CACHE)                                 # edf -> raw CSV (cached)
    fig = plotRecordingSpectrogram(rec, ch, CACHE)                    # time-frequency plot with the params.json display settings
    ecog.saveFigure(fig, OUT / "spectrograms" / f"{rec.label}_ch{ch}_spectrogram", dpi=DPI, formats=PLOT_FORMATS)
    plt.close(fig)
    ecog.computeBandRatios(rec, CACHE, ["delta"])                     # delta SNR CSV (cached)
    ecog.saveTable(ecog.bandTable(rec, ch, CACHE, "delta"),           # unaligned (recording-clock) delta table
                   OUT / "individual_delta_snr" / f"{rec.label}_ch{ch}_delta_ratio.csv")

# ---- c: combined delta table and mean delta SNR line plot
combined = pd.concat([                                                # long table (timeSeg, mouse, condition, ratio), one block per condition
    ecog.aggregateAligned([(recs[(a, k)], ch) for a, k, ch in b["recordings"]], CACHE, "delta",
                          align_event=b["align_event"], condition_label=name)
    for name, b in CONDITION_BATCHES.items()], ignore_index=True)
ecog.saveTable(combined, OUT / "combined_delta.csv")
order = list(CONDITION_COLORS)                                        # condition order for plots and stats

ax = plotRatioSNR(combined, ylabel="Delta SNR (mean +/- sem)", events={"onset": 0}, colors={"onset": "black"},
                  binning_s=BINNING_S, smoothing_s=SMOOTHING_S, tick_step=TICK_STEP, line_colors=CONDITION_COLORS,
                  show_band=True, x_limits=LINE_XLIM, figsize=(4, 3))  # condition means +/- SEM across animals
ax.spines["top"].set_visible(False)                                   # open frame
ax.spines["right"].set_visible(False)
ax.set_xlim(*LINE_XLIM)
ax.set_ylim(*LINE_YLIM)
ax.set_xticks(np.arange(LINE_XLIM[0], LINE_XLIM[1] + 1, 300), np.arange(LINE_XLIM[0] // 60, LINE_XLIM[1] // 60 + 1, 5))   # minutes every 5 min
ax.axvspan(*RESPONSE_WINDOW, 0, 1, color="tab:orange", alpha=0.1, zorder=6)   # analysis window
ecog.saveFigure(ax.figure, OUT / "conditions_delta_line", dpi=DPI, formats=PLOT_FORMATS)
plt.close(ax.figure)
ecog.saveTable(binTimeseries(combined, "ratio", BINNING_S), OUT / "conditions_delta_line_summary.csv")   # binned mean/error behind the line

# ---- d: strip plot, one point per animal
window_min = f"{RESPONSE_WINDOW[0] // 60}-{RESPONSE_WINDOW[1] // 60}"   # window label in minutes
ax, strip_summary = plot_condition_stripplot(
    combined, value_col="ratio", window=RESPONSE_WINDOW, condition_order=order, ylabel="Delta SNR (mean per animal)",
    title=f"Delta SNR by condition across animals\n({window_min} min post-onset mean)",
    marker_color=CONDITION_COLORS, marker_size=80, figsize=(2, 4), zero_line=False, center="mean",
    line_color="gray")                                                # per-animal connecting lines in gray
ax.axhline(1, 0, 1, ls=":", c="tab:gray", lw=2, zorder=0)             # SNR = 1 reference
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
ax.tick_params(width=1.5)
ax.set_title(ax.get_title(), pad=20)
ax.set_ylim(-0.2, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_xlim([-0.5, 2.25])
ecog.saveFigure(ax.figure, OUT / "conditions_delta_strip", dpi=DPI, formats=PLOT_FORMATS)
plt.close(ax.figure)
ecog.saveTable(strip_summary.rename_axis("condition").reset_index(), OUT / "conditions_delta_strip_summary.csv")

# ---- e: statistics (window mean per animal, 60 s bins)
anova, included, excluded = ecog.compareConditionsANOVA(               # repeated-measures ANOVA across the three conditions
    combined, value_col="ratio", window=RESPONSE_WINDOW, condition_order=order, binning_s=BINNING_S, center="mean")
anova = anova.rename_axis("effect").reset_index().assign(
    n_included=len(included), included="; ".join(included), excluded="; ".join(excluded))   # table plus which animals fed it
ecog.saveTable(anova, OUT / "anova_omnibus.csv")
posthoc = ecog.compareConditionMeans(                                  # pairwise paired t-tests with multiple-comparison correction
    combined, value_col="ratio", window=RESPONSE_WINDOW, condition_order=order, p_adjust=POSTHOC_P_ADJUST,
    binning_s=BINNING_S, center="mean")
ecog.saveTable(posthoc, OUT / "posthoc_paired_t.csv")
print("strip plot summary (per-animal window means)\n", strip_summary.to_string())
print("\nrepeated-measures ANOVA\n", anova.to_string(index=False))
print(f"\npost-hoc paired t-tests (p_adjust={POSTHOC_P_ADJUST})\n", posthoc.to_string(index=False))

# ---- g: run record
settings = {"params": PARAMS.name, "condition_batches": CONDITION_BATCHES, "condition_colors": CONDITION_COLORS,
            "binning_s": BINNING_S, "smoothing_s": SMOOTHING_S, "response_window": RESPONSE_WINDOW,
            "posthoc_p_adjust": POSTHOC_P_ADJUST, "plot_formats": PLOT_FORMATS, "dpi": DPI}
ecog.writeRunInfo(OUT, settings)

# ---- key numbers
f = anova.iloc[0]                                                      # omnibus effect row
print(f"\nkey numbers: ANOVA F({f['Num DF']:.0f}, {f['Den DF']:.0f}) = {f['F Value']:.6f}, p = {f['Pr > F']:.6f}, n = {f['n_included']}")
print(posthoc.assign(pair=posthoc["condition_1"] + " vs " + posthoc["condition_2"]).set_index("pair")[["p", "p_adj"]].round(6).to_string())
