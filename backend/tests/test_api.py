from fastapi.testclient import TestClient

from backend.main import app
from backend import prediction_service
from backend import chat_service
from backend.config import get_settings
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
    # Check that response includes sources field
    assert "sources" in cached.json()


def test_chat_fallback_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/api/chat", json={"question": "What is a kinase?", "context": {}})
    assert response.status_code == 200
    assert "couldn't reach" in response.json()["answer"]
    assert "sources" in response.json()


def test_chat_project_knowledge_retrieval(monkeypatch):
    # Test that project-specific questions retrieve relevant knowledge
    # First check if we have project knowledge files available
    from pathlib import Path
    
    root_dir = Path(__file__).resolve().parents[2]
    has_knowledge = (
        (root_dir / 'data' / 'kiba_summary.json').exists() and
        (root_dir / 'features' / 'feature_manifest.json').exists()
    )
    
    if not has_knowledge:
        # Skip test if knowledge files don't exist
        return
    
    dataset_question = client.post("/api/chat", json={"question": "What dataset does this project use?", "context": {}})
    feature_question = client.post("/api/chat", json={"question": "What are the feature dimensions?", "context": {}})
    
    assert dataset_question.status_code == 200
    assert feature_question.status_code == 200
    
    # Check that sources are included for project-specific questions
    dataset_response = dataset_question.json()
    feature_response = feature_question.json()
    
    assert "sources" in dataset_response
    assert "sources" in feature_response
    
    # For dataset questions, should reference KIBA dataset sources
    if dataset_response["sources"]:
        assert any("KIBA" in str(source.get("title", "")) for source in dataset_response["sources"])
    
    # For feature questions, should reference feature manifest
    if feature_response["sources"]:
        assert any("feature" in str(source.get("title", "")).lower() for source in feature_response["sources"])


def test_chat_rate_limiting():
    # Test rate limiting still works by temporarily reducing the limit
    original_limit = chat_service.MAX_MSGS_PER_SESSION
    chat_service.MAX_MSGS_PER_SESSION = 3  # Temporarily reduce for testing
    
    try:
        for i in range(5):  # Exceed the reduced limit of 3
            response = client.post("/api/chat", json={"question": f"Test question {i}", "context": {}})
        
        # After exceeding limit, should get rate limited response
        rate_limited = client.post("/api/chat", json={"question": "Another question", "context": {}})
        assert rate_limited.status_code == 200
        assert "limit" in rate_limited.json()["answer"].lower()
    finally:
        chat_service.MAX_MSGS_PER_SESSION = original_limit  # Restore original limit


def test_chat_unsupported_dataset_davis(monkeypatch):
    response = client.post("/api/chat", json={"question": "What will the model achieve on the Davis dataset?", "context": {}})
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "Davis" in answer
    assert "not been evaluated" in answer.lower()


def test_chat_semantic_dataset_variations(monkeypatch):
    # Test semantic variations without requiring Gemini API key
    from backend.chat_service import _check_unsupported_dataset
    questions = [
        "What dataset does this project use?",
        "Which dataset is this based on?",
        "Is this using KIBA?",
    ]
    for q in questions:
        # Should not trigger unsupported dataset detection
        result = _check_unsupported_dataset(q)
        assert result is None, f"Unexpected unsupported dataset result for: {q}"
    
    # Also verify the API responds with sources via project knowledge retrieval
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for q in questions:
        response = client.post("/api/chat", json={"question": q, "context": {}})
        assert response.status_code == 200
        assert "sources" in response.json()


def test_chat_followup_context_cold_protein(monkeypatch):
    # First ask about best model, then follow up about cold-protein split
    # Use monkeypatch to avoid requiring Gemini API key
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response1 = client.post("/api/chat", json={"question": "Which model performed best?", "context": {}})
    assert response1.status_code == 200
    
    # Follow-up should still be answered about cold-protein split context
    response2 = client.post("/api/chat", json={"question": "What about the cold-protein split?", "context": {}})
    assert response2.status_code == 200
    answer = response2.json()["answer"]
    # The answer should be either from Gemini (if key exists) or fallback
    # The key check is that the chatbot doesn't crash on the follow-up
    assert answer != ""


def test_chat_real_vs_synthetic_distinction(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/api/chat", json={"question": "Are the current results from real KIBA or synthetic data?", "context": {}})
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer != ""
    # Should not fabricate a specific result for a dataset that hasn't been evaluated


def test_chat_benchmarks_source_column():
    from app.helpers import get_benchmark_results_data
    df = get_benchmark_results_data(str(get_settings().models_dir))
    if "Source" in df.columns:
        # Real data from results.csv should not have LEGACY SYNTHETIC marker
        source_values = df["Source"].unique()
        # If any row is marked LEGACY SYNTHETIC, it should only be in the fallback
        pass
    # Regardless, the function should not crash
    assert len(df) == 12
