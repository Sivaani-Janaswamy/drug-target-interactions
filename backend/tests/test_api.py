from fastapi.testclient import TestClient

from backend.main import app
from backend import prediction_service
from app.helpers import MockPredictor


client = TestClient(app)
VALID_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
VALID_PAYLOAD = {
    "smiles": "CCO",
    "protein_sequence": VALID_SEQUENCE,
    "model": "Random Forest",
    "drug_name": "Test drug",
    "target_name": "Test target",
}


def test_health_includes_version():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()


def test_presets_have_separate_metadata():
    response = client.get("/api/presets")
    assert response.status_code == 200
    assert response.json()[0]["drug_name"]
    assert response.json()[0]["target_name"]


def test_benchmarks_have_expected_columns():
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    assert {"Algorithm", "Split", "RMSE", "Pearson r", "CI"}.issubset(response.json()[0])


def test_prediction_response_shape(monkeypatch):
    monkeypatch.setattr(prediction_service, "compute_shap_attributions", lambda model, features: [[0.2] * 1071])
    monkeypatch.setattr(prediction_service, "get_cached_model", lambda name: (MockPredictor(name), True))
    response = client.post("/api/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert {"score", "label", "gauge_percent", "uncertainty", "drug", "target", "features"}.issubset(body)

def test_invalid_smiles_returns_400():
    payload = {**VALID_PAYLOAD, "smiles": "not-a-smiles"}
    assert client.post("/api/predict", json=payload).status_code == 400


def test_invalid_protein_returns_400():
    payload = {**VALID_PAYLOAD, "protein_sequence": "ABCZ"}
    assert client.post("/api/predict", json=payload).status_code == 400


def test_unsupported_model_returns_400():
    payload = {**VALID_PAYLOAD, "model": "Unknown"}
    assert client.post("/api/predict", json=payload).status_code == 400


def test_chat_cache_and_refusal(monkeypatch):
    cached = client.post("/api/chat", json={"question": "Explain SHAP simply", "context": {}})
    refused = client.post("/api/chat", json={"question": "Give me medical advice", "context": {}})
    assert cached.status_code == 200
    assert "SHAP" in cached.json()["answer"]
    assert refused.status_code == 200
    assert "only help" in refused.json()["answer"]


def test_chat_fallback_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/api/chat", json={"question": "What is a kinase?", "context": {}})
    assert response.status_code == 200
    assert "couldn't reach" in response.json()["answer"]
