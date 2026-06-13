"""
Optional API-key authentication.

Auth is OFF by default so local development and the test suite keep working
out of the box. It is enabled only when the `API_KEY` environment variable is
set; in that case every protected endpoint requires a matching `X-API-Key`
request header.
"""

import os
from typing import Optional

from fastapi import Header, HTTPException, status


API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_ENV_VAR = "API_KEY"


def auth_enabled() -> bool:
    """True when an API key is configured in the environment."""
    return bool(os.getenv(API_KEY_ENV_VAR))


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER_NAME)
) -> None:
    """
    FastAPI dependency enforcing the API key when auth is enabled.

    - No `API_KEY` env set  -> auth disabled, request always allowed.
    - `API_KEY` set         -> the `X-API-Key` header must match exactly.
    """

    expected = os.getenv(API_KEY_ENV_VAR)

    if not expected:
        return  # auth disabled

    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": API_KEY_HEADER_NAME},
        )
