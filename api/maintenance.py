"""
Maintenance helpers for the prediction output directory.

Prediction CSVs are written to outputs/predictions/ on every /predict-file
call. Without housekeeping that directory grows forever, so we delete files
older than a retention window (on startup and after each new write).
"""

import time
from pathlib import Path
from typing import Optional, Union


def cleanup_old_predictions(
    directory: Union[str, Path],
    max_age_hours: float,
    now: Optional[float] = None,
    pattern: str = "*.csv",
) -> int:
    """
    Delete prediction files older than `max_age_hours`.

    Args:
        directory: folder to clean.
        max_age_hours: files whose mtime is older than this are removed.
        now: current epoch seconds (injectable for testing; defaults to time.time()).
        pattern: glob of files to consider.

    Returns:
        Number of files deleted. Missing directory => 0. Individual files that
        cannot be removed are skipped silently so cleanup never breaks a request.
    """

    directory = Path(directory)

    if not directory.exists():
        return 0

    if max_age_hours is None or max_age_hours < 0:
        return 0

    if now is None:
        now = time.time()

    cutoff = now - max_age_hours * 3600.0
    removed = 0

    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue
        try:
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                removed += 1
        except OSError:
            # File may have been removed concurrently or be locked; ignore.
            continue

    return removed
