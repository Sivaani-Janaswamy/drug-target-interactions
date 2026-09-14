from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=2000)
    protein_sequence: str = Field(min_length=1, max_length=100000)
    model: str = Field(min_length=1)
    drug_name: str = Field(default="Custom drug", max_length=120)
    target_name: str = Field(default="Custom target", max_length=120)


class FeatureResponse(BaseModel):
    name: str
    technical_name: str
    value: float
    direction: Literal["positive", "negative"]
    explanation: str
    chemical_context: dict[str, Any] | None = None


class PredictionResponse(BaseModel):
    drug: dict[str, Any]
    target: dict[str, Any]
    model: str
    score: float
    label: str
    color: str
    explanation: str
    gauge_percent: float
    uncertainty: float | None
    features: list[FeatureResponse]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    context: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str


class ErrorResponse(BaseModel):
    detail: str
