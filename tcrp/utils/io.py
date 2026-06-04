"""I/O helpers for persisting experiment results."""
from __future__ import annotations

import datetime
import json
from pathlib import Path


def save_results(
    results: dict,
    name: str,
    out_dir: str | Path = "results",
    path: str | Path | None = None,
) -> Path:
    """Serialise results to a timestamped JSON file.

    Parameters
    ----------
    results : dict
        Experiment results (must be JSON-serialisable).
    name : str
        Human-readable run identifier used in the auto-generated filename.
    out_dir : str | Path
        Directory to write to when path is not given (created if absent).
    path : str | Path | None
        Explicit output path; overrides out_dir + auto-naming when provided.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    if path is None:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = Path(out_dir) / f"{name}_{ts}.json"
    else:
        dest = Path(path)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=2))
    return dest


def now_iso(timespec: str = "seconds") -> str:
    """Return the current local datetime as an ISO-8601 string."""
    return datetime.datetime.now().isoformat(timespec=timespec)


def ts_tag() -> str:
    """Return a compact timestamp string suitable for filenames (YYYYMMDD_HHMMSS)."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
