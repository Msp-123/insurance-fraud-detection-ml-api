"""
API integration tests using FastAPI's TestClient.

The TestClient triggers the startup event, which loads the model artifacts.
If artifacts are missing, the whole module is skipped.
"""

import pytest

from conftest import ARTIFACTS_AVAILABLE

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_AVAILABLE,
    reason="Model artifacts required to start the API.",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app

    # Context manager form runs startup/shutdown events (loads artifacts).
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_root_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["preprocessor_loaded"] is True

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is True


class TestPredict:
    def test_predict_returns_valid_response(self, client, real_sample_claim):
        resp = client.post("/predict", json=real_sample_claim)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["prediction"] in (0, 1)
        assert body["risk_level"] in {"Low", "Medium", "High"}

    def test_predict_explain_returns_reasons(self, client, real_sample_claim):
        resp = client.post("/predict-explain", json=real_sample_claim)
        assert resp.status_code == 200
        body = resp.json()
        assert "top_reasons" in body
        assert isinstance(body["top_reasons"], list)

    def test_batch_predict(self, client, real_sample_claim):
        resp = client.post(
            "/batch-predict",
            json={"claims": [real_sample_claim, real_sample_claim]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert len(body["results"]) == 2

    def test_predict_with_empty_body_uses_defaults(self, client):
        # All fields optional -> empty body is accepted (extra="allow", Optional).
        resp = client.post("/predict", json={})
        # Either a valid prediction or a controlled 500; must not be a crash/422.
        assert resp.status_code in (200, 500)


class TestPredictFile:
    def test_predict_file_csv_roundtrip(self, client, real_sample_claim):
        import io
        import pandas as pd

        df = pd.DataFrame([real_sample_claim, real_sample_claim])
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        resp = client.post(
            "/predict-file",
            files={"file": ("claims.csv", csv_bytes, "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["input_rows"] == 2
        assert body["result_count"] == 2
        assert "download_url" in body

        # The generated file should be downloadable.
        download = client.get(body["download_url"])
        assert download.status_code == 200

    def test_predict_file_rejects_unsupported_extension(self, client):
        resp = client.post(
            "/predict-file",
            files={"file": ("data.txt", b"hello", "text/plain")},
        )
        # file_utils raises ValueError -> handler returns 500.
        assert resp.status_code == 500

    def test_download_missing_file_returns_404(self, client):
        resp = client.get("/download-predictions/does_not_exist.csv")
        assert resp.status_code == 404
