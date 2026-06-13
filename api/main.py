import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas import (
    ClaimRequest,
    PredictionResponse,
    BatchPredictionRequest,
    HealthResponse,
    ExplanationResponse,
)

from api.model_service import fraud_model_service

from api.file_utils import (
    read_uploaded_file_to_dataframe,
    validate_file_prediction_input,
)
from api.security import require_api_key, auth_enabled
from api.maintenance import cleanup_old_predictions

# Model-layer config (src/ is added to sys.path by api.model_service import).
from config import MAX_UPLOAD_SIZE_MB, PREDICTION_RETENTION_HOURS


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("fraud_api")


# =========================================================
# Paths and operational settings
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTION_OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"
PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Runtime-overridable operational settings (env wins over config defaults).
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", MAX_UPLOAD_SIZE_MB)) * 1024 * 1024
RETENTION_HOURS = float(os.getenv("PREDICTION_RETENTION_HOURS", PREDICTION_RETENTION_HOURS))

# Comma-separated list of allowed CORS origins. Default "*" (open).
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]


# =========================================================
# Lifespan (replaces deprecated @app.on_event("startup"))
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts ...")
    fraud_model_service.load_artifacts()
    logger.info(
        "Artifacts loaded (model=%s, preprocessor=%s, threshold=%s).",
        fraud_model_service.model is not None,
        fraud_model_service.preprocessor is not None,
        fraud_model_service.threshold,
    )

    removed = cleanup_old_predictions(PREDICTION_OUTPUT_DIR, RETENTION_HOURS)
    if removed:
        logger.info("Removed %d expired prediction file(s) on startup.", removed)

    logger.info("API ready. Auth %s.", "ENABLED" if auth_enabled() else "disabled")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Vehicle Insurance Claim Fraud Detection API",
    description=(
        "An explainable vehicle insurance claim fraud scoring API "
        "using XGBoost, feature engineering, threshold tuning and FastAPI."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# =========================================================
# Middleware
# =========================================================

# CORS. Credentials cannot be combined with the "*" wildcard, so only enable
# them when explicit origins are configured.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, status and latency."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# =========================================================
# Helpers
# =========================================================

def pydantic_to_dict(model):
    """
    Pydantic v1/v2 compatible conversion.
    Removes None values from request body.
    """

    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)

    return model.dict(exclude_none=True)


def _health_payload() -> dict:
    return {
        "status": "ok",
        "model_loaded": fraud_model_service.model is not None,
        "preprocessor_loaded": fraud_model_service.preprocessor is not None,
        "threshold": fraud_model_service.threshold,
    }


# =========================================================
# Health endpoints (no auth)
# =========================================================

@app.get("/", response_model=HealthResponse)
def root():
    return _health_payload()


@app.get("/health", response_model=HealthResponse)
def health():
    return _health_payload()


# =========================================================
# Prediction endpoints (auth-protected when API_KEY is set)
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict_claim(request: ClaimRequest):
    try:
        claim_data = pydantic_to_dict(request)
        return fraud_model_service.predict(claim_data)
    except Exception:
        # Log the full traceback server-side; never leak internals to the client.
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.post(
    "/predict-explain",
    response_model=ExplanationResponse,
    dependencies=[Depends(require_api_key)],
)
def predict_claim_with_explanation(request: ClaimRequest):
    try:
        claim_data = pydantic_to_dict(request)
        return fraud_model_service.predict_with_explanation(claim_data)
    except Exception:
        logger.exception("Prediction explanation failed.")
        raise HTTPException(status_code=500, detail="Prediction explanation failed.")


@app.post("/batch-predict", dependencies=[Depends(require_api_key)])
def batch_predict_claims(request: BatchPredictionRequest):
    try:
        results = fraud_model_service.predict_batch(request.claims)
        return {"count": len(results), "results": results}
    except Exception:
        logger.exception("Batch prediction failed.")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")


@app.post("/predict-file", dependencies=[Depends(require_api_key)])
async def predict_file(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    preview_rows: int = Form(20),
):
    """
    Predict fraud risk for claims uploaded as CSV or Excel file.

    Instead of returning all prediction rows as JSON, this endpoint:
    - scores the full uploaded file,
    - saves the prediction result as a CSV file,
    - returns summary statistics and a small preview.

    Supported formats:
    - .csv
    - .xlsx
    - .xls
    """

    try:
        df = await read_uploaded_file_to_dataframe(
            file=file,
            sheet_name=sheet_name,
            max_size_bytes=MAX_UPLOAD_BYTES,
        )

        validate_file_prediction_input(df)

        claims = df.to_dict(orient="records")
        results = fraud_model_service.predict_batch(claims)
        result_df = pd.DataFrame(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]

        output_filename = f"fraud_predictions_{timestamp}_{unique_id}.csv"
        output_path = PREDICTION_OUTPUT_DIR / output_filename

        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # Housekeeping: drop expired prediction files.
        cleanup_old_predictions(PREDICTION_OUTPUT_DIR, RETENTION_HOURS)

        fraud_count = int((result_df["prediction"] == 1).sum())
        not_fraud_count = int((result_df["prediction"] == 0).sum())

        risk_level_counts = (
            result_df["risk_level"].value_counts().to_dict()
            if "risk_level" in result_df.columns
            else {}
        )

        preview_df = result_df.head(preview_rows).copy()
        preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

        return {
            "filename": file.filename,
            "input_rows": int(df.shape[0]),
            "input_columns": int(df.shape[1]),
            "result_count": int(result_df.shape[0]),
            "fraud_count": fraud_count,
            "not_fraud_count": not_fraud_count,
            "fraud_rate_percent": round(fraud_count / len(result_df) * 100, 2),
            "risk_level_counts": risk_level_counts,
            "output_file": output_filename,
            "download_url": f"/download-predictions/{output_filename}",
            "preview_rows": int(preview_rows),
            "preview": preview_df.to_dict(orient="records"),
        }

    except ValueError as e:
        # Client errors (bad format, empty file, too large, no rows/columns).
        logger.warning("Rejected prediction file: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception:
        logger.exception("File prediction failed.")
        raise HTTPException(status_code=500, detail="File prediction failed.")


@app.get("/download-predictions/{file_name}", dependencies=[Depends(require_api_key)])
def download_predictions(file_name: str):
    """
    Download prediction result CSV file.
    """

    safe_file_name = Path(file_name).name
    file_path = PREDICTION_OUTPUT_DIR / safe_file_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Prediction file not found.")

    return FileResponse(
        path=file_path,
        filename=safe_file_name,
        media_type="text/csv",
    )
