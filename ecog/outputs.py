"""
Saving helpers.

Deliverables (figures, stats tables, aggregate CSVs) are always written and are
never overwritten: if the target already exists, a cumulative index is appended
(`name.svg` -> `name_1.svg` -> `name_2.svg` ...). Processing caches (raw CSV, band
CSVs, epoch table) are different: they are reused when present and only rebuilt
when a function is called with `overwrite=True` / `force=True`.
"""
import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path


def uniquePath(path):
    """`path` if it does not exist yet, else `<stem>_1<suffix>`, `<stem>_2<suffix>`, ... (first free index)."""
    path = Path(path)
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def uniquePaths(base, exts):
    """
    One path per extension in `exts` for the same output, sharing one index so that
    `fig.svg` and `fig.png` stay paired (`fig_1.svg` / `fig_1.png`). `base` has no extension.
    """
    base = Path(base)
    exts = [e if e.startswith(".") else f".{e}" for e in exts]
    i = 0
    while True:
        stem = base.name if i == 0 else f"{base.name}_{i}"
        paths = [base.with_name(stem + e) for e in exts]
        if not any(p.exists() for p in paths):
            return paths
        i += 1


def saveFigure(fig, path, dpi=300, bbox_inches="tight", formats=None):
    """
    Save `fig` (a matplotlib Figure, or None for the current figure) without overwriting.

    `path` may carry an extension (`outputs/line.svg`); pass `formats=("svg", "png")` to write
    several formats of the same figure under one shared index (the extension of `path`, if any,
    is then ignored). Returns the list of paths written.
    """
    import matplotlib.pyplot as plt

    fig = fig if fig is not None else plt.gcf()
    path = Path(path)
    if formats:
        targets = uniquePaths(path.with_suffix("") if path.suffix else path, formats)
    else:
        targets = [uniquePath(path)]
    targets[0].parent.mkdir(parents=True, exist_ok=True)
    for target in targets:
        fig.savefig(target, dpi=dpi, bbox_inches=bbox_inches)
    return targets


def saveTable(df, path, index=False, **kwargs):
    """Write a DataFrame to csv (or xlsx, by extension) without overwriting; returns the path written."""
    path = uniquePath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(path, index=index, **kwargs)
    else:
        df.to_csv(path, index=index, **kwargs)
    return path


def writeRunInfo(output_dir, settings=None, packages=("numpy", "scipy", "pandas", "matplotlib", "mne",
                                                       "fooof", "specparam", "statsmodels")):
    """
    Record when/with what a set of outputs was produced: timestamp, Python, installed package
    versions and the `settings` dict (anything JSON-serialisable, other values are stringified).
    """
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    info = {"created": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "platform": platform.platform(), "packages": versions, "settings": settings or {}}
    path = uniquePath(Path(output_dir) / "run_info.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path
