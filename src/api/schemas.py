from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field

from typing import Optional, List

class LoanApplicationRequest(BaseModel):
    no_of_dependents: int = Field(..., ge=0)
    education: Literal["Graduate", "Not Graduate"]
    self_employed: Literal["Yes", "No"]
    income_annum: float = Field(..., ge=0)
    loan_amount: float = Field(..., ge=0)
    loan_term: float = Field(..., gt=0)
    cibil_score: float = Field(..., ge=0)
    residential_assets_value: float = Field(..., ge=0)
    commercial_assets_value: float = Field(..., ge=0)
    luxury_assets_value: float = Field(..., ge=0)
    bank_asset_value: float = Field(..., ge=0)


class PredictionResult(BaseModel):
    predicted_label_numeric: int
    prediction: str
    probability: float
    threshold: float


class LocalExplanation(BaseModel):
    top_positive: Dict[str, float]
    top_negative: Dict[str, float]


class PredictResponse(BaseModel):
    prediction_result: PredictionResult


class ExplainResponse(BaseModel):
    prediction_result: PredictionResult
    local_explanation: LocalExplanation


class SummarizeResponse(BaseModel):
    prediction_result: PredictionResult
    local_explanation: LocalExplanation
    summary: str


class HealthResponse(BaseModel):
    status: str
    message: str

class SavedRecordResponse(BaseModel):
    id: int
    created_at: str
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float
    predicted_label_numeric: int
    prediction: str
    probability: float
    threshold: float
    local_explanation_json: Optional[str] = None
    summary: Optional[str] = None


class HistoryListResponse(BaseModel):
    records: List[SavedRecordResponse]