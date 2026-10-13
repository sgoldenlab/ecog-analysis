"""
ecog -- ECoG/EEG analysis toolkit for the lab.

Typical flow (see notebooks/fundamentals):

    rec = Recording(...)                          # or loadRecordings("params.json")
    extractRecording(rec, work_dir)               # .edf -> raw CSV
    plotRecordingSpectrogram(rec, 0, work_dir)    # time-frequency spectrogram
    computeBandRatios(rec, work_dir, ["delta"])   # band SNR CSVs
    buildEpochTable(batches, work_dir, cache=...) # aperiodic / band / state metrics per 30 s epoch
    compareAcrossConditions(table, metrics, windows, order)   # paired stats

Deliverables are saved with ecog.outputs (never overwritten; a cumulative index is appended).
"""
from ecog.config import Recording, loadParams, loadRecordings, getRecordingParams, recordingFromParams
from ecog.edf import extractRecording, ecogToCSV
from ecog.spectral import computeBandRatios, computeBandPower, bandRatioToCSV, bandPowerToCSV, BAND_RATIOS
from ecog.bursts import burstSuppressionRatioToCSV
from ecog.aggregate import bandTable, aggregateAligned
from ecog.epochs import (buildEpochTable, recordingEpochTable, epochTableFromCSV, epochPSDs, fitAperiodicEpochs,
                         stateMetrics, suppressionByEpoch, meanSpectrumByCondition, aperiodicCurve)
from ecog.stats import (compareConditionMeans, compareConditionsANOVA, compareAcrossConditions,
                        fdrAcrossMeasures, perAnimalMeans, fitQuality)
from ecog.emg import extractEMG, cleanEMG, emgEnvelope, emgEpochTable, attachEMG, figEMG
from ecog.plots import plotRecordingSpectrogram
from ecog.outputs import saveFigure, saveTable, writeRunInfo, uniquePath
