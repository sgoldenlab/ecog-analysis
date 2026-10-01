"""
Stale-cache guard for processing CSVs.

Raw CSVs and band CSVs are reused when they exist (`overwrite=False`). Each one written by this library gets a small
`<name>.csv.settings.json` sidecar recording what it was made from (EDF, notch filter, band definition, n_fft, ...).
If the settings requested now differ from the sidecar, the file is recomputed and a note is printed, so changing a band
definition or a filter never silently reuses an old file. A CSV without a sidecar (made elsewhere) is trusted.
"""
import json
from pathlib import Path


def _sidecar(csv_path):
    return Path(str(csv_path) + ".settings.json")


def _normalised(settings):
    return json.loads(json.dumps(settings, default=str, sort_keys=True))


def isStale(csv_path, settings):
    """True when `csv_path` exists, has a sidecar, and the sidecar differs from `settings`."""
    csv_path, side = Path(csv_path), _sidecar(csv_path)
    if not csv_path.exists() or not side.exists():
        return False
    stale = json.loads(side.read_text()) != _normalised(settings)
    if stale:
        print(f"note: {csv_path.name} was made with different settings -- recomputing")
    return stale


def writeSettings(csv_path, settings):
    """Record `settings` next to `csv_path`."""
    _sidecar(csv_path).write_text(json.dumps(_normalised(settings), indent=1))
