from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.chat_service import get_chat_answer
from backend.config import get_settings
from backend.prediction_service import get_api_benchmarks, get_api_presets, predict_binding
from backend.schemas import ChatRequest, ChatResponse, ErrorResponse, PredictionRequest, PredictionResponse

settings = get_settings()
app = FastAPI(title="DTI-ML API", version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "dti-ml-api", "version": settings.api_version}


@app.get("/api/presets")
def presets() -> list[dict]:
    return get_api_presets()


@app.get("/api/benchmarks")
def benchmarks() -> list[dict]:
    return get_api_benchmarks()


@app.post("/api/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}})
def predict(payload: PredictionRequest) -> dict:
    try:
        return predict_binding(
            payload.smiles,
            payload.protein_sequence,
            payload.model,
            payload.drug_name,
            payload.target_name,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.") from error


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    client_id = request.client.host if request.client else "anonymous"
    return ChatResponse(answer=get_chat_answer(payload.question, payload.context, client_id))
