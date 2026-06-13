"""
Shared pytest configuration and fixtures.

This file:
- Makes the `src/` package importable from tests (same way api/model_service.py does).
- Detects whether the trained model artifacts exist so that integration tests
  can be skipped gracefully on a clean checkout (artifacts are gitignored).
- Provides small reusable fixtures (a sample raw claim, a tiny raw dataframe).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup: make `src` imports work (mirrors api/model_service.py)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Make `api` importable as a package (api/main.py does `from api.schemas import ...`)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ---------------------------------------------------------------------------
# Artifact availability
# ---------------------------------------------------------------------------

from config import (  # noqa: E402  (import after sys.path tweak)
    MODEL_PATH,
    PREPROCESSOR_PATH,
    PREPROCESSED_DATA_PATH,
    FEATURE_NAMES_PATH,
    RAW_FEATURE_COLUMNS_PATH,
)

ARTIFACTS_AVAILABLE = all(
    p.exists()
    for p in [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        PREPROCESSED_DATA_PATH,
        FEATURE_NAMES_PATH,
        RAW_FEATURE_COLUMNS_PATH,
    ]
)

# Reusable skip marker for tests that need a trained model on disk.
requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_AVAILABLE,
    reason=(
        "Trained model artifacts not found. "
        "Run preprocessing.py, train.py and threshold_tuning.py first."
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_claim():
    """
    A single realistic raw claim record (matches fraud_oracle.csv columns).

    Kept fully literal so unit tests don't depend on the dataset being present.
    """

    return {
        "Month": "Dec",
        "WeekOfMonth": 5,
        "DayOfWeek": "Wednesday",
        "Make": "Honda",
        "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Tuesday",
        "MonthClaimed": "Jan",
        "WeekOfMonthClaimed": 1,
        "Sex": "Female",
        "MaritalStatus": "Single",
        "Age": 21,
        "Fault": "Policy Holder",
        "PolicyType": "Sport - Liability",
        "VehicleCategory": "Sport",
        "VehiclePrice": "more than 69000",
        "PolicyNumber": 1,
        "RepNumber": 12,
        "Deductible": 300,
        "DriverRating": 1,
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "none",
        "AgeOfVehicle": "3 years",
        "AgeOfPolicyHolder": "26 to 30",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "1 year",
        "NumberOfCars": "3 to 4",
        "Year": 1994,
        "BasePolicy": "Liability",
    }


@pytest.fixture
def raw_claims_df(sample_claim):
    """A small two-row raw dataframe built from the sample claim."""

    second = dict(sample_claim)
    second.update(
        {
            "Fault": "Third Party",
            "PoliceReportFiled": "Yes",
            "WitnessPresent": "Yes",
            "Age": 45,
            "AgeOfVehicle": "7 years",
            "PastNumberOfClaims": "more than 4",
            "BasePolicy": "All Perils",
        }
    )
    return pd.DataFrame([sample_claim, second])


@pytest.fixture(scope="session")
def real_sample_claim():
    """
    A real raw claim taken from the saved test split, when artifacts exist.

    Returns None when artifacts are missing so integration tests can skip.
    """

    if not ARTIFACTS_AVAILABLE:
        return None

    import joblib

    data = joblib.load(PREPROCESSED_DATA_PATH)
    if "X_test_raw" not in data:
        return None

    return data["X_test_raw"].iloc[0].to_dict()
