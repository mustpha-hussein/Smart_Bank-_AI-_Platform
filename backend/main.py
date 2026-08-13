"""SmartBank AI prediction service.

Run from the repository root with: uvicorn backend.main:app --reload
"""
from __future__ import annotations

import __main__
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from afrip_pipeline_classes import (
    DropColumnsTransformer,
    FeatureEngineeringTransformer,
    FinalPreprocessor,
    FrequencyEncoder,
    GenderEncoder,
    ThresholdClassifier,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

if not (MODELS / "loan" / "loan_model.pkl").exists():
    raise FileNotFoundError(
        f"Model file not found: {MODELS / 'loan' / 'loan_model.pkl'}"
    )
DEFAULT_LOAN_THRESHOLD = 0.5


class LoanRequest(BaseModel):
    person_age: int = Field(ge=18, le=110)
    person_gender: Literal["female", "male"]
    person_education: Literal["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    person_income: float = Field(gt=0, le=1_000_000)
    person_emp_exp: int = Field(ge=0, le=90)
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    loan_amnt: float = Field(gt=0)
    loan_intent: Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    loan_int_rate: float = Field(ge=0, le=100)
    loan_percent_income: float = Field(ge=0, le=1)
    cb_person_cred_hist_length: int = Field(ge=0, le=90)
    credit_score: int = Field(ge=300, le=850)
    previous_loan_defaults_on_file: Literal["Yes", "No"]

    def to_frame(self) -> pd.DataFrame:
        payload = self.model_dump()
        payload["person_gender"] = "female" if payload["person_gender"] == "female" else "male"
        return pd.DataFrame([payload])


class FraudRequest(BaseModel):
    amount: float = Field(gt=0, alias="amt")
    category: Literal[
        "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
        "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
        "shopping_net", "shopping_pos", "travel",
    ]
    merchant: str = Field(min_length=1, max_length=160)
    gender: Literal["F", "M"]
    job: str = Field(min_length=1, max_length=120)
    state: str = Field(min_length=2, max_length=2)
    date: date
    time: time
    date_of_birth: date

    @field_validator("state")
    @classmethod
    def normalize_state(cls, value: str) -> str:
        return value.upper()

    def to_frame(self) -> pd.DataFrame:
        transaction_time = datetime.combine(self.date, self.time)
        return pd.DataFrame([{
            "trans_date_trans_time": transaction_time.isoformat(sep=" "),
            "dob": self.date_of_birth.isoformat(),
            "amt": self.amount,
            "category": self.category,
            "merchant": self.merchant,
            "gender": self.gender,
            "job": self.job,
            "state": self.state,
            # The serialized feature transformer calculates distance before
            # later dropping these geographic columns. The UI does not collect
            # location data, so a neutral same-location pair is supplied.
            "lat": 0.0,
            "long": 0.0,
            "merch_lat": 0.0,
            "merch_long": 0.0,
        }])


def _restore_legacy_fraud_pickle_symbols() -> None:
    """The fraud pipeline was serialized from a notebook (__main__)."""
    for cls in (
        FeatureEngineeringTransformer, DropColumnsTransformer, GenderEncoder,
        FrequencyEncoder, FinalPreprocessor, ThresholdClassifier,
    ):
        setattr(__main__, cls.__name__, cls)


def _load_models():
    try:
        loan_artifact = joblib.load(MODELS / "loan" / "loan_model.pkl")
        if isinstance(loan_artifact, dict):
            loan = loan_artifact.get("model")
            threshold = loan_artifact.get("threshold", DEFAULT_LOAN_THRESHOLD)
        else:
            loan = loan_artifact
            threshold = DEFAULT_LOAN_THRESHOLD

        if loan is None or not hasattr(loan, "predict_proba"):
            raise ValueError("Loan artifact must contain a model with predict_proba().")
        if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
            raise ValueError("Loan decision threshold must be a number between 0 and 1.")

        _restore_legacy_fraud_pickle_symbols()
        fraud = joblib.load(MODELS / "fraud" / "afrip_fraud_pipelinee.pkl")
        return loan, float(threshold), fraud
    except Exception as exc:  # pragma: no cover - startup failure is surfaced clearly
        raise RuntimeError(f"Unable to load SmartBank model artifacts: {exc}") from exc


loan_model, loan_threshold, fraud_model = _load_models()
app = FastAPI(title="SmartBank AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "models": {"loan": "ready", "fraud": "ready"},
        "loan_decision_threshold": loan_threshold,
    }


@app.post("/api/loan/predict")
def predict_loan(request: LoanRequest):
    try:
        probability = float(loan_model.predict_proba(request.to_frame())[0, 1])
        approved = bool(probability >= loan_threshold)
        return {
            "approved": approved,
            "decision": "Eligible" if approved else "Review required",
            "approval_probability": round(probability, 4),
            "decision_threshold": loan_threshold,
            "confidence": round(max(probability, 1 - probability), 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Loan prediction failed.") from exc


@app.post("/api/fraud/predict")
def predict_fraud(request: FraudRequest):
    try:
        probability = float(fraud_model.predict_proba(request.to_frame())[0, 1])
        flagged = bool(fraud_model.predict(request.to_frame())[0])
        return {
            "flagged": flagged,
            "decision": "Potential fraud" if flagged else "Low risk",
            "fraud_probability": round(probability, 4),
            "confidence": round(max(probability, 1 - probability), 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Fraud prediction failed.") from exc


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
