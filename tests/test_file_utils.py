"""
Unit tests for api/file_utils.py

Covers extension parsing, input validation, and reading uploaded CSV/Excel
content into a DataFrame (including error cases). Uploads are simulated with
Starlette's UploadFile over an in-memory buffer.
"""

import asyncio
from io import BytesIO

import pandas as pd
import pytest
from starlette.datastructures import UploadFile, Headers

from api.file_utils import (
    get_file_extension,
    validate_file_prediction_input,
    read_uploaded_file_to_dataframe,
    SUPPORTED_FILE_EXTENSIONS,
)


def _make_upload(filename, content: bytes) -> UploadFile:
    """Build a Starlette UploadFile from raw bytes with a forced filename."""
    headers = Headers({"content-disposition": f'form-data; name="file"; filename="{filename}"'})
    return UploadFile(file=BytesIO(content), filename=filename, headers=headers)


class TestGetFileExtension:
    def test_lowercases(self):
        assert get_file_extension("DATA.CSV") == ".csv"
        assert get_file_extension("sheet.XLSX") == ".xlsx"

    def test_no_extension(self):
        assert get_file_extension("noext") == ""

    def test_supported_set(self):
        assert ".csv" in SUPPORTED_FILE_EXTENSIONS
        assert ".xlsx" in SUPPORTED_FILE_EXTENSIONS
        assert ".xls" in SUPPORTED_FILE_EXTENSIONS


class TestValidateFilePredictionInput:
    def test_valid_frame_passes(self):
        df = pd.DataFrame({"a": [1, 2]})
        validate_file_prediction_input(df)  # should not raise

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError):
            validate_file_prediction_input(pd.DataFrame())


class TestReadUploadedFile:
    def test_reads_csv(self):
        content = b"a,b\n1,2\n3,4\n"
        upload = _make_upload("data.csv", content)
        df = asyncio.run(read_uploaded_file_to_dataframe(upload))
        assert df.shape == (2, 2)
        assert list(df.columns) == ["a", "b"]

    def test_reads_xlsx(self):
        buf = BytesIO()
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(buf, index=False)
        upload = _make_upload("data.xlsx", buf.getvalue())
        df = asyncio.run(read_uploaded_file_to_dataframe(upload))
        assert df.shape == (2, 2)

    def test_unsupported_extension_raises(self):
        upload = _make_upload("data.txt", b"hello")
        with pytest.raises(ValueError, match="Unsupported file format"):
            asyncio.run(read_uploaded_file_to_dataframe(upload))

    def test_empty_file_raises(self):
        upload = _make_upload("data.csv", b"")
        with pytest.raises(ValueError, match="empty"):
            asyncio.run(read_uploaded_file_to_dataframe(upload))

    def test_latin1_fallback(self):
        # Bytes that are invalid UTF-8 but valid latin-1 should still parse.
        content = "a,b\n1,café\n".encode("latin1")
        upload = _make_upload("data.csv", content)
        df = asyncio.run(read_uploaded_file_to_dataframe(upload))
        assert df.shape == (1, 2)
