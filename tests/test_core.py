"""Synthetic-data checks of the shared library (no recordings needed): run `pytest tests` from the repo root."""
import json

import numpy as np
import pandas as pd
import pytest

import ecog
from ecog import Recording
from ecog.outputs import uniquePath, uniquePaths


def _recording(tmp_path, **kw):
    kw = {"animal_id": "A1", "condition": "Cond", "recording_number": 1, "edf": "x.edf", "events": {"onset": 100}, **kw}
    return Recording(edf_folder=tmp_path, **kw)


def _writeRaw(rec, tmp_path, x, channel=0):
    path = rec.rawCSV(tmp_path, channel)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({rec.condition: x}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- outputs
def test_uniquePath_appends_cumulative_index(tmp_path):
    p = tmp_path / "fig.svg"
    assert uniquePath(p) == p
    p.write_text("a")
    assert uniquePath(p).name == "fig_1.svg"
    (tmp_path / "fig_1.svg").write_text("b")
    assert uniquePath(p).name == "fig_2.svg"


def test_uniquePaths_share_one_index(tmp_path):
    (tmp_path / "f.svg").write_text("a")
    assert [q.name for q in uniquePaths(tmp_path / "f", ["svg", "png"])] == ["f_1.svg", "f_1.png"]


def test_saveTable_never_overwrites(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    first = ecog.saveTable(df, tmp_path / "t.csv")
    second = ecog.saveTable(df.assign(a=[3, 4]), tmp_path / "t.csv")
    assert first.name == "t.csv" and second.name == "t_1.csv"
    assert pd.read_csv(first)["a"].tolist() == [1, 2]


def test_saveFigure_formats_and_index(tmp_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    a = ecog.saveFigure(fig, tmp_path / "fig", formats=("svg", "png"))
    b = ecog.saveFigure(fig, tmp_path / "fig", formats=("svg", "png"))
    assert [p.name for p in a] == ["fig.svg", "fig.png"] and [p.name for p in b] == ["fig_1.svg", "fig_1.png"]
    plt.close(fig)


# ---------------------------------------------------------------- config
def test_recording_paths_and_ids(tmp_path):
    rec = _recording(tmp_path, channel_index=[0, 1])
    assert rec.label == "A1_Cond_R1" and rec.mouse_id == "A1_R1" and rec.channels == [0, 1]
    assert rec.rawCSV("w", 1).as_posix() == "w/A1/csvs/A1_Cond_R1_raw_ch1.csv"
    assert rec.bandCSV("w", 0, "delta").as_posix() == "w/A1/delta/A1_Cond_R1_ch0_delta_ratio.csv"
    with pytest.raises(KeyError):
        rec.eventTime("missing")


def test_loadRecordings_merges_defaults_animal_recording(tmp_path):
    cfg = {"defaults": {"fs": 400, "n_fft": 10000, "raw_edf_folder": "edfs", "clim": [-80, -100]},
           "animals": {"A1": {"channel_index": 1, "recordings": {
               "Ind_R2": {"edf_filename": "f.edf", "events": {"onset": 5}, "clim": [-90, -110], "note": "hi"}}}}}
    (tmp_path / "params.json").write_text(json.dumps(cfg))
    rec = ecog.loadRecordings(tmp_path / "params.json")[("A1", "Ind_R2")]
    assert (rec.condition, rec.recording_number, rec.channel_index) == ("Ind", "2", 1)
    assert rec.clim == (-90, -110) and rec.events == {"onset": 5} and rec.extra["note"] == "hi"
    assert rec.edf_folder == tmp_path / "edfs"


def test_edfPath_errors_are_specific(tmp_path):
    rec = _recording(tmp_path, edf="nope.edf")
    with pytest.raises(FileNotFoundError):
        rec.edfPath()
    (tmp_path / "a_export.edf").write_text("")
    (tmp_path / "b_export.edf").write_text("")
    with pytest.raises(ValueError):
        _recording(tmp_path, edf="_export").edfPath()


# ---------------------------------------------------------------- spectral / aggregate
def test_delta_ratio_and_alignment(tmp_path):
    fs = 400
    t = np.arange(0, 120, 1 / fs)
    rng = np.random.default_rng(0)
    x = 1e-5 * rng.standard_normal(len(t)) + 4e-5 * np.sin(2 * np.pi * 2.5 * t) * (t > 60)   # delta appears at 60 s
    rec = _recording(tmp_path, n_fft=2000)
    _writeRaw(rec, tmp_path, x)
    ecog.computeBandRatios(rec, tmp_path, ["delta"])
    df = ecog.bandTable(rec, 0, tmp_path, "delta", align_event="onset")
    assert list(df.columns) == ["timeSeg", "mouse", "condition", "ratio"] and df["mouse"].unique().tolist() == ["A1_R1"]
    assert df["timeSeg"].min() < 0
    early, late = df[df.timeSeg < -50]["ratio"].mean(), df[df.timeSeg > -30]["ratio"].mean()   # 100 s onset: -40 s = 60 s
    assert late > 5 * early


def test_band_cache_reused_until_settings_change(tmp_path):
    rec = _recording(tmp_path, n_fft=2000)
    _writeRaw(rec, tmp_path, 1e-5 * np.random.default_rng(0).standard_normal(400 * 60))
    out = ecog.computeBandRatios(rec, tmp_path, ["delta"])["delta"][0]
    first = out.stat().st_mtime_ns
    assert (out.parent / (out.name + ".settings.json")).exists()
    ecog.computeBandRatios(rec, tmp_path, ["delta"])
    assert out.stat().st_mtime_ns == first                                   # same settings: reused
    before = pd.read_csv(out)
    rec.extra["bands"] = {"delta": {"signal": [1.0, 3.0], "noise": [[4.0, 7.0]]}}
    ecog.computeBandRatios(rec, tmp_path, ["delta"])
    assert out.stat().st_mtime_ns != first and not before.equals(pd.read_csv(out))   # new definition: recomputed


def test_edf_folder_is_normalised(tmp_path):
    rec = Recording("A1", "Cond", 1, edf="x.edf", edf_folder=tmp_path / "sub" / "..")
    assert ".." not in rec.edf_folder.parts and rec.edf_folder == tmp_path


def test_unknown_band_is_rejected(tmp_path):
    rec = _recording(tmp_path)
    _writeRaw(rec, tmp_path, np.zeros(4000))
    with pytest.raises(KeyError):
        ecog.computeBandRatios(rec, tmp_path, ["gamma"])


# ---------------------------------------------------------------- epochs
def _pinkNoise(n, fs, exponent, rng):
    f = np.fft.rfftfreq(n, 1 / fs)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (-exponent / 2)
    spec = amp * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    return np.fft.irfft(spec, n) * 1e-5


def test_epoch_table_recovers_exponent_and_cache(tmp_path):
    fs, rng = 400, np.random.default_rng(1)
    shallow, steep = _recording(tmp_path, animal_id="A1"), _recording(tmp_path, animal_id="A1", condition="Steep")
    _writeRaw(shallow, tmp_path, _pinkNoise(fs * 300, fs, 1.0, rng))
    _writeRaw(steep, tmp_path, _pinkNoise(fs * 300, fs, 2.0, rng))
    batches = {"Shallow": [(shallow, 0)], "Steep": [(steep, 0)]}
    table, spectra = ecog.buildEpochTable(batches, tmp_path, cache=tmp_path / "c")
    means = table.groupby("condition")["exponent"].mean()
    assert abs(means["Shallow"] - 1.0) < 0.35 and abs(means["Steep"] - 2.0) < 0.35
    assert {"delta_abs", "delta_rel", "delta_periodic", "spectral_entropy", "median_freq", "suppression_pct"} <= set(table)
    assert table["timeSeg"].min() == 15 - 100
    again, _ = ecog.buildEpochTable(batches, tmp_path, cache=tmp_path / "c")        # served from cache
    pd.testing.assert_frame_equal(table, again)
    # a changed onset invalidates the cache instead of returning stale epoch times
    shallow.events["onset"] = 0
    changed, _ = ecog.buildEpochTable(batches, tmp_path, cache=tmp_path / "c")
    assert changed.loc[changed["condition"] == "Shallow", "timeSeg"].min() == 15


# ---------------------------------------------------------------- stats
def _pairedTable(effect=1.0, n=5, seed=0, rename_session=False):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.normal(0, 1)
        for cond, shift in (("A", 0.0), ("B", effect)):
            session = "R2" if (rename_session and cond == "B" and i == 0) else "R1"
            for t in range(10):
                rows.append({"timeSeg": t * 100.0, "mouse": f"M{i}_{session}", "animal_id": f"M{i}", "condition": cond,
                             "m": base + shift + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_paired_tests_match_animals_across_sessions():
    df = _pairedTable(effect=2.0, rename_session=True)
    res = ecog.compareConditionMeans(df, "m", (0, 1000), condition_order=["A", "B"])
    assert res["n"].iloc[0] == 5 and res["p"].iloc[0] < 0.001            # M0_R1 vs M0_R2 still pair
    anova, included, excluded = ecog.compareConditionsANOVA(df, "m", (0, 1000), condition_order=["A", "B"])
    assert len(included) == 5 and not excluded and anova["Pr > F"].iloc[0] < 0.001


def test_compareAcrossConditions_and_fdr_across_measures():
    df = _pairedTable(effect=2.0).assign(n=lambda d: np.random.default_rng(3).normal(size=len(d)))
    anova, pairwise = ecog.compareAcrossConditions(df, ["m", "n"], {"w": (0, 1000)}, ["A", "B"])
    out = ecog.fdrAcrossMeasures(pairwise, "w", ["m", "n"])
    assert set(out["metric"]) == {"m", "n"} and out["q_measures"].notna().all()
    assert (out["q_measures"] >= out["p"] - 1e-12).all()


def test_stripplot_gray_lines_and_missing_animal():
    import matplotlib.pyplot as plt
    from ecog.plots import plot_condition_stripplot
    df = _pairedTable()
    df = df[~((df["animal_id"] == "M0") & (df["condition"] == "B"))]      # M0 missing a condition
    ax, summary = plot_condition_stripplot(df, "m", (0, 1000), condition_order=["A", "B"])
    import matplotlib.colors as mc
    joined = {mc.to_rgba(l.get_color()) for l in ax.lines if len(l.get_xdata()) == 2}
    assert joined == {mc.to_rgba("gray")} and summary.loc["B", "count"] == 4
    plt.close("all")


# ---------------------------------------------------------------- emg
def test_emg_cleaning_removes_mains_and_flags_bursts(tmp_path):
    fs, rng = 400, np.random.default_rng(2)
    t = np.arange(0, 300, 1 / fs)
    burst = ((t // 30) % 5 == 3).astype(float)                              # every 5th epoch is a movement burst
    x = 1e-5 * rng.standard_normal(len(t)) * (1 + 6 * burst) + 3e-4 * np.sin(2 * np.pi * 60 * t)
    rec = _recording(tmp_path, channel_index=2)
    _writeRaw(rec, tmp_path, x, channel=2)
    emg = ecog.emgEpochTable(rec, tmp_path, channel=2)
    assert emg.attrs["raw_line_share"] > 0.5 and 60 in emg.attrs["mains_removed_hz"]
    assert emg.loc[emg["moving"], "timeSeg"].tolist() == [30 * k + 15 for k in range(10) if k % 5 == 3]
