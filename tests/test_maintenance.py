"""
Unit tests for api/maintenance.py — prediction-output retention cleanup.

Uses tmp_path and an injected `now` so file ages are deterministic (no sleeps).
"""

import os
import time

import pytest

from api.maintenance import cleanup_old_predictions


def _make_file(directory, name, mtime):
    path = directory / name
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


class TestCleanupOldPredictions:
    def test_removes_only_expired_files(self, tmp_path):
        now = 1_000_000.0
        old = _make_file(tmp_path, "old.csv", mtime=now - 48 * 3600)   # 48h old
        fresh = _make_file(tmp_path, "fresh.csv", mtime=now - 1 * 3600)  # 1h old

        removed = cleanup_old_predictions(tmp_path, max_age_hours=24, now=now)

        assert removed == 1
        assert not old.exists()
        assert fresh.exists()

    def test_returns_zero_for_missing_directory(self, tmp_path):
        missing = tmp_path / "nope"
        assert cleanup_old_predictions(missing, max_age_hours=24) == 0

    def test_only_matches_pattern(self, tmp_path):
        now = 1_000_000.0
        csv = _make_file(tmp_path, "a.csv", mtime=now - 100 * 3600)
        txt = _make_file(tmp_path, "a.txt", mtime=now - 100 * 3600)

        removed = cleanup_old_predictions(tmp_path, max_age_hours=1, now=now)

        assert removed == 1
        assert not csv.exists()
        assert txt.exists()  # not matched by *.csv

    def test_negative_age_is_noop(self, tmp_path):
        now = 1_000_000.0
        f = _make_file(tmp_path, "a.csv", mtime=now - 100 * 3600)
        assert cleanup_old_predictions(tmp_path, max_age_hours=-1, now=now) == 0
        assert f.exists()

    def test_nothing_to_remove(self, tmp_path):
        now = 1_000_000.0
        _make_file(tmp_path, "fresh.csv", mtime=now)
        assert cleanup_old_predictions(tmp_path, max_age_hours=24, now=now) == 0
