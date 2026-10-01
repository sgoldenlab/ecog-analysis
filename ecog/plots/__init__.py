"""Plotting: spectrograms/timelines, binned time series, per-animal summaries, epoch-table figures."""
from ecog.plots.timeline import annotateTimeline, plotSpectrogram, plotRecordingSpectrogram
from ecog.plots.timeseries import plotBinnedTimeseries, plotRatioSNR, plotBandPower, binTimeseries
from ecog.plots.summary import plot_condition_stripplot
from ecog.plots.epochs import (
    STYLE, CONDITION_COLORS, CONDITION_LABELS, perAnimalPanel, timeCoursePanel, figConventionalMeasures,
    figAperiodic, figSuppressionControl, annotateSignificance, figSummaryInset,
)
