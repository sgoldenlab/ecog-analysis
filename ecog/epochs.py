"""
Epoch-wise spectral parameterisation: aperiodic (1/f) vs periodic decomposition, state summary
metrics, and absolute-threshold burst suppression.

Each recording is cut into consecutive non-overlapping epochs (default 30 s); every epoch gets a Welch
spectrum (4 s sub-windows, 50% overlap -> 0.25 Hz resolution) from which we take:

- absolute band power, band share of total power, and periodic band power (power above the fitted background)
- the aperiodic background: `exponent` (steepness) and `offset` (height), fitted with fooof over `FIT_RANGE`
- state summaries: median frequency, SEF95, spectral entropy (normalised Shannon entropy over FIT_RANGE)
- % of the epoch suppressed under an absolute microvolt criterion

Epoch times are expressed relative to a named event of each recording (default "onset"), so conditions
are compared at matched time since the switch. `buildEpochTable` runs a whole {condition: [(Recording, channel)]}
design and caches the result.

Requires `fooof` (>=1.1); `specparam` 2.x is accepted as a fallback.
"""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from ecog.bursts import _detectSuppression

FS = 400
EPOCH_S = 30
WELCH_WINDOW_S = 4
FIT_RANGE = (1.0, 20.0)          # upper limit set by the ~20-25 Hz acquisition low-pass
BANDS = {"delta": (1.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0)}
FOOOF_SETTINGS = dict(peak_width_limits=(1.0, 8.0), max_n_peaks=6,
                      min_peak_height=0.10, peak_threshold=2.0, aperiodic_mode="fixed")
SUPPRESSION_UV = 20              # absolute threshold, uV
SUPPRESSION_MIN_S = 0.5


def epochPSDs(x, fs=FS, epoch_s=EPOCH_S, welch_window_s=WELCH_WINDOW_S):
    """
    Welch power spectrum of every non-overlapping `epoch_s` epoch of `x`.

    Returns (freqs, psd[n_epochs, n_freqs], starts_s). Within an epoch the
    Welch estimate averages `welch_window_s` sub-windows overlapping by 50%,
    so the frequency resolution is 1/welch_window_s Hz and each epoch's
    spectrum is already smoothed across sub-windows.
    """
    n_per_epoch = int(epoch_s * fs)
    n_epochs = len(x) // n_per_epoch
    if n_epochs == 0:
        raise ValueError(f"recording shorter than one {epoch_s}s epoch ({len(x)/fs:.1f}s)")
    seg = np.asarray(x[: n_epochs * n_per_epoch], dtype=float).reshape(n_epochs, n_per_epoch)
    nperseg = int(welch_window_s * fs)
    freqs, psd = welch(seg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=1)
    return freqs, psd, np.arange(n_epochs) * epoch_s


def _makeModel():
    """fooof 1.x preferred (stable); specparam 2.x renamed FOOOF -> SpectralModel."""
    try:
        from fooof import FOOOF
        return lambda: FOOOF(verbose=False, **FOOOF_SETTINGS)
    except ImportError:                                            # pragma: no cover
        try:
            from specparam import SpectralModel
        except ImportError as exc:
            raise ImportError("spectral parameterisation needs `fooof` (pip install fooof) "
                              "or `specparam`") from exc
        return lambda: SpectralModel(verbose=False, **FOOOF_SETTINGS)


def _modelResults(model):
    """Pull (aperiodic_params, r2, error, peak_params) from either API version."""
    try:                                                           # fooof 1.x
        return model.aperiodic_params_, model.r_squared_, model.error_, np.atleast_2d(model.peak_params_)
    except AttributeError:                                         # specparam 2.x
        r = model.results
        return r.aperiodic_params, r.r_squared, r.error, np.atleast_2d(r.peak_params)


def aperiodicCurve(freqs, offset, exponent):
    """The fitted background in log10 power, for plotting alongside a spectrum."""
    return offset - np.log10(np.asarray(freqs, dtype=float) ** exponent)


def fitAperiodicEpochs(freqs, psd, fit_range=FIT_RANGE, bands=BANDS):
    """
    Fit background + peaks to every epoch's spectrum; one row per epoch.

    Columns: exponent, offset (the background), fit_r2/fit_error (how well the
    model describes the spectrum), total_power (summed over `fit_range`), and
    per band `<band>_abs` (absolute power), `<band>_rel` (share of total),
    `<band>_periodic` (mean power above the fitted background, i.e. the
    oscillation with the background removed), plus the largest fitted peak in
    that band (`_peak_cf` centre frequency, `_peak_pw` power, `_peak_bw` width)
    and how many peaks fell in it (`_n_peaks`).
    """
    make = _makeModel()
    mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    f_fit = freqs[mask]
    rows = []
    for i in range(psd.shape[0]):
        model = make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(freqs, psd[i], fit_range)
        ap, r2, err, peaks = _modelResults(model)
        offset, exponent = float(ap[0]), float(ap[-1])
        flat = np.log10(psd[i, mask]) - aperiodicCurve(f_fit, offset, exponent)
        total = float(psd[i, mask].sum())
        row = {"exponent": exponent, "offset": offset, "fit_r2": float(r2), "fit_error": float(err),
               "total_power": total}
        for band, (lo, hi) in bands.items():
            b = (f_fit >= lo) & (f_fit < hi)
            row[f"{band}_abs"] = float(psd[i, mask][b].sum())
            row[f"{band}_rel"] = float(psd[i, mask][b].sum() / total)
            row[f"{band}_periodic"] = float(flat[b].mean())
            in_band = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] < hi)] if peaks.size else peaks
            row[f"{band}_n_peaks"] = int(len(in_band))
            if len(in_band):
                cf, pw, bw = in_band[np.argmax(in_band[:, 1])]
                row[f"{band}_peak_cf"], row[f"{band}_peak_pw"], row[f"{band}_peak_bw"] = float(cf), float(pw), float(bw)
            else:
                row[f"{band}_peak_cf"] = row[f"{band}_peak_pw"] = row[f"{band}_peak_bw"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def stateMetrics(freqs, psd, band=FIT_RANGE):
    """
    Single-number summaries of each epoch's spectral shape, over `band`.

    - median_freq: frequency below which half the power lies.
    - sef95: spectral edge frequency, below which 95% of the power lies.
    - spectral_entropy: Shannon entropy of the normalised spectrum, divided by
      log(n_bins) so 1 = flat/unstructured and lower = power concentrated in
      fewer frequencies.

    All three are bounded by `band`: with a 20 Hz ceiling, sef95 saturates near
    the ceiling and is much less informative than it would be over 0.5-45 Hz.
    """
    mask = (freqs >= band[0]) & (freqs <= band[1])
    f_b, P = freqs[mask], psd[:, mask]
    p_norm = P / P.sum(axis=1, keepdims=True)
    cum = np.cumsum(p_norm, axis=1)
    return pd.DataFrame({
        "median_freq": f_b[np.argmax(cum >= 0.50, axis=1)],
        "sef95": f_b[np.argmax(cum >= 0.95, axis=1)],
        "spectral_entropy": -(p_norm * np.log(p_norm)).sum(axis=1) / np.log(p_norm.shape[1]),
    })


def suppressionByEpoch(x, fs=FS, epoch_s=EPOCH_S, amplitude_threshold=SUPPRESSION_UV,
                       min_duration=SUPPRESSION_MIN_S):
    """
    Percentage of each epoch spent suppressed, by an ABSOLUTE voltage criterion.

    Wraps processing._detectSuppression with a fixed uV threshold rather than
    the relative (percent-of-baseline-SD) threshold the pipeline's BSR CSVs use.
    The relative threshold is referenced to each recording's own first
    `bsr_baseline_duration` seconds, which sit at a different anaesthetic depth
    in each condition, and it scores 3-20% even in awake animals -- so it is not
    comparable across conditions. Use this for any cross-condition statement
    about burst suppression.
    """
    sup = _detectSuppression(np.asarray(x, dtype=float), fs,
                             amplitude_threshold=amplitude_threshold, min_duration=min_duration)
    n_per_epoch = int(epoch_s * fs)
    n_epochs = len(sup) // n_per_epoch
    return sup[: n_epochs * n_per_epoch].reshape(n_epochs, n_per_epoch).mean(axis=1) * 100


def rolloffSlope(freqs, psd, lo, hi):
    """Local log-log slope of a (mean) spectrum between `lo` and `hi` Hz."""
    m = (freqs >= lo) & (freqs <= hi)
    return float(np.polyfit(np.log10(freqs[m]), np.log10(np.asarray(psd)[m]), 1)[0])


def epochTableFromCSV(path, onset, condition, animal, channel, recording_number,
                      fs=FS, epoch_s=EPOCH_S, fit_range=FIT_RANGE, bands=BANDS,
                      welch_window_s=WELCH_WINDOW_S, suppression=True):
    """
    One raw CSV -> (per-epoch DataFrame, spectra dict).

    The DataFrame carries the identifying columns the ecog.stats helpers expect ('mouse', 'condition',
    'timeSeg'), with `timeSeg` giving the epoch centre relative to `onset` (seconds).
    """
    x = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)
    freqs, psd, starts = epochPSDs(x, fs=fs, epoch_s=epoch_s, welch_window_s=welch_window_s)
    table = pd.concat([fitAperiodicEpochs(freqs, psd, fit_range=fit_range, bands=bands),
                       stateMetrics(freqs, psd, band=fit_range)], axis=1)
    if suppression:
        table["suppression_pct"] = suppressionByEpoch(x, fs=fs, epoch_s=epoch_s)[: len(table)]
        table["mean_abs_uv"] = np.abs(
            x[: len(table) * int(epoch_s * fs)].reshape(len(table), int(epoch_s * fs)) * 1e6).mean(axis=1)
    timeSeg = starts + epoch_s / 2 - onset
    table.insert(0, "timeSeg", timeSeg)
    table.insert(0, "channel", channel)
    table.insert(0, "mouse", f"{animal}_R{recording_number}")
    table.insert(0, "animal_id", animal)
    table.insert(0, "condition", condition)
    table.insert(0, "recording", f"{Path(path).stem}")
    spectra = {"freqs": freqs, "psd": psd, "timeSeg": timeSeg,
               "condition": condition, "animal_id": animal}
    return table, spectra


def recordingEpochTable(rec, channel, work_dir, condition=None, onset_event="onset", **kwargs):
    """epochTableFromCSV for one Recording/channel (its raw CSV must exist); `condition` defaults to rec.condition."""
    kwargs.setdefault("fs", rec.fs)
    return epochTableFromCSV(
        rec.rawCSV(work_dir, channel), rec.eventTime(onset_event), condition or rec.condition,
        rec.animal_id, channel, rec.recording_number, **kwargs)


def _cacheSignature(batches, work_dir, onset_event, kwargs):
    """Hash of everything the epoch table depends on, so a changed design or setting is never served from cache."""
    items = []
    for condition, entries in batches.items():
        for rec, ch in entries:
            raw = rec.rawCSV(work_dir, ch)
            items.append([condition, rec.label, int(ch), rec.eventTime(onset_event),
                          raw.stat().st_size if raw.exists() else None])
    settings = {"fs": FS, "epoch_s": EPOCH_S, "welch_window_s": WELCH_WINDOW_S, "fit_range": FIT_RANGE,
                "bands": BANDS, "fooof": FOOOF_SETTINGS, "suppression": [SUPPRESSION_UV, SUPPRESSION_MIN_S],
                "kwargs": kwargs}
    return hashlib.sha1(json.dumps([items, settings], sort_keys=True, default=str).encode()).hexdigest()


def buildEpochTable(batches, work_dir, cache=None, force=False, onset_event="onset", **kwargs):
    """
    Run recordingEpochTable over {condition: [(Recording, channel), ...]} and concatenate.
    The condition key becomes the table's 'condition' (so e.g. a recording whose own condition is
    "Alone" can be reported as "No Induction").

    With `cache` set to a directory, the epoch table and epoch spectra are written there
    (epoch_table.csv / epoch_spectra.npz / epoch_cache.json) and reloaded on the next call, unless
    `force=True` or the recordings/settings changed since (the fits take a couple of minutes, the reload a second).
    Returns (table, spectra).
    """
    cache = Path(cache) if cache else None
    signature = _cacheSignature(batches, work_dir, onset_event, kwargs)
    if cache and not force and (cache / "epoch_table.csv").exists():
        stored = (cache / "epoch_cache.json")
        if stored.exists() and json.loads(stored.read_text()).get("signature") == signature:
            return pd.read_csv(cache / "epoch_table.csv"), loadSpectra(cache / "epoch_spectra.npz")
        print(f"epoch cache in {cache} does not match this design/settings -- recomputing")

    tables, spectra = [], {}
    for condition, entries in batches.items():
        for rec, channel in entries:
            one, spec = recordingEpochTable(rec, channel, work_dir, condition=condition,
                                            onset_event=onset_event, **kwargs)
            tables.append(one)
            spectra[one["recording"].iloc[0]] = spec
    table = pd.concat(tables, ignore_index=True)
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache / "epoch_table.csv", index=False)
        saveSpectra(spectra, cache / "epoch_spectra.npz")
        (cache / "epoch_cache.json").write_text(json.dumps({"signature": signature}))
    return table, spectra


def saveSpectra(spectra, path):
    arrays = {f"{k}__{field}": v[field] for k, v in spectra.items()
              for field in ("freqs", "psd", "timeSeg")}
    meta = {k: {"condition": v["condition"], "animal_id": v["animal_id"]} for k, v in spectra.items()}
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def loadSpectra(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return {k: {"freqs": z[f"{k}__freqs"], "psd": z[f"{k}__psd"], "timeSeg": z[f"{k}__timeSeg"], **v}
            for k, v in meta.items()}


def meanSpectrumByCondition(spectra, window=None, log=True):
    """
    Average spectrum per condition: mean across epochs within a recording
    (in log10 power, matching the domain the fits work in), then across
    recordings. `window` restricts to (start, end) seconds relative to onset.
    """
    per_condition = {}
    freqs = None
    for spec in spectra.values():
        freqs = spec["freqs"]
        t = np.asarray(spec["timeSeg"])
        sel = np.ones(len(t), dtype=bool) if window is None else (t >= window[0]) & (t < window[1])
        if not sel.any():
            continue
        values = np.log10(spec["psd"][sel]) if log else spec["psd"][sel]
        per_condition.setdefault(spec["condition"], []).append(values.mean(axis=0))
    return freqs, {c: np.mean(v, axis=0) for c, v in per_condition.items()}
