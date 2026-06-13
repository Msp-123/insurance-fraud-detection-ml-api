"""
Unit tests for api/security.py — the optional API-key dependency.

These don't need the model or the running app; they call the dependency
function directly with a monkeypatched environment.
"""

import pytest
from fastapi import HTTPException

from api.security import require_api_key, auth_enabled, API_KEY_ENV_VAR


class TestAuthEnabled:
    def test_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        assert auth_enabled() is False

    def test_enabled_with_env(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "k")
        assert auth_enabled() is True


class TestRequireApiKey:
    def test_allows_anything_when_disabled(self, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        # No key configured -> any header (even None) passes.
        assert require_api_key(x_api_key=None) is None
        assert require_api_key(x_api_key="whatever") is None

    def test_rejects_missing_key_when_enabled(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "secret")
        with pytest.raises(HTTPException) as exc:
            require_api_key(x_api_key=None)
        assert exc.value.status_code == 401

    def test_rejects_wrong_key(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "secret")
        with pytest.raises(HTTPException) as exc:
            require_api_key(x_api_key="wrong")
        assert exc.value.status_code == 401

    def test_accepts_correct_key(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "secret")
        assert require_api_key(x_api_key="secret") is None
