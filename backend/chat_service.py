from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Any

from backend.config import get_settings

ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_json_file(file_path: Path) -> dict[str, Any] | None:
    """Load and parse a JSON file, return None if file doesn't exist or is invalid."""
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _get_project_knowledge() -> dict[str, Any]:
    """Load all available project knowledge sources."""
    knowledge = {
        'experiment_manifest': _load_json_file(ROOT_DIR / 'models' / 'kiba_experiment_manifest.json'),
        'kiba_summary': _load_json_file(ROOT_DIR / 'data' / 'kiba_summary.json'),
        'split_metadata': _load_json_file(ROOT_DIR / 'data' / 'kiba_split_metadata.json'),
        'feature_manifest': _load_json_file(ROOT_DIR / 'features' / 'feature_manifest.json'),
    }
    return knowledge


def _retrieve_relevant_knowledge(question: str, knowledge: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """
    Retrieve relevant project knowledge based on the question.
    Returns (context_string, list of source_references).
    """
    question_lower = question.lower()
    context_parts = []
    sources = []

    # Dataset information
    if any(keyword in question_lower for keyword in ['dataset', 'kiba', 'interactions', 'drugs', 'proteins', 'size', 'rows']):
        if knowledge.get('kiba_summary'):
            summary = knowledge['kiba_summary']
            context_parts.append(f"Dataset: {summary.get('dataset', 'KIBA')} - {summary.get('description', '')}")
            context_parts.append(f"Total interactions: {summary.get('rows_total', 'N/A')}")
            context_parts.append(f"Unique drugs: {summary.get('unique_drugs', 'N/A')}")
            context_parts.append(f"Unique canonical SMILES: {summary.get('unique_canonical_smiles', 'N/A')}")
            context_parts.append(f"Unique proteins: {summary.get('unique_proteins', 'N/A')}")
            context_parts.append(f"Affinity range: {summary.get('affinity_min', 'N/A')} to {summary.get('affinity_max', 'N/A')}")
            sources.append({
                'title': 'KIBA Dataset Summary',
                'file': 'data/kiba_summary.json',
                'section': 'dataset_statistics'
            })

    # Feature dimensions
    if any(keyword in question_lower for keyword in ['feature', 'dimension', 'representation', '1030', '197', '1227']):
        if knowledge.get('feature_manifest'):
            features = knowledge['feature_manifest']
            drug_dim = features.get('drug_feature_dim', 'N/A')
            protein_dim = features.get('protein_feature_dim', 'N/A')
            combined = drug_dim + protein_dim if isinstance(drug_dim, int) and isinstance(protein_dim, int) else 'N/A'
            context_parts.append(f"Drug features: {drug_dim} dimensions (Morgan ECFP4 fingerprints + physicochemical descriptors)")
            context_parts.append(f"Protein features: {protein_dim} dimensions (AAC + CTD + PSEAAC)")
            context_parts.append(f"Combined representation: {combined} dimensions")
            context_parts.append(f"Drug features used: {', '.join(features.get('drug_features', []))}")
            context_parts.append(f"Protein features used: {', '.join(features.get('protein_features', []))}")
            sources.append({
                'title': 'Feature Manifest',
                'file': 'features/feature_manifest.json',
                'section': 'feature_dimensions'
            })

    # Split information
    if any(keyword in question_lower for keyword in ['split', 'train', 'test', 'overlap', 'leakage', 'cold']):
        if knowledge.get('split_metadata'):
            splits = knowledge['split_metadata']
            for split_name, split_data in splits.items():
                context_parts.append(f"{split_name.upper()} split:")
                context_parts.append(f"  Train rows: {split_data.get('train_rows', 'N/A')}")
                context_parts.append(f"  Test rows: {split_data.get('test_rows', 'N/A')}")
                if 'drug_id_overlap' in split_data:
                    context_parts.append(f"  Drug ID overlap: {split_data['drug_id_overlap']}")
                if 'smiles_overlap' in split_data:
                    context_parts.append(f"  Canonical SMILES overlap: {split_data['smiles_overlap']}")
                if 'protein_id_overlap' in split_data:
                    context_parts.append(f"  Protein ID overlap: {split_data['protein_id_overlap']}")
                if 'sequence_overlap' in split_data:
                    context_parts.append(f"  Protein sequence overlap: {split_data['sequence_overlap']}")
            sources.append({
                'title': 'Split Metadata',
                'file': 'data/kiba_split_metadata.json',
                'section': 'split_validation'
            })

    # Experimental results
    if any(keyword in question_lower for keyword in ['result', 'performance', 'rmse', 'mse', 'ci', 'pearson', 'best', 'model', 'xgboost', 'random forest', 'svr', 'gpr']):
        if knowledge.get('experiment_manifest'):
            manifest = knowledge['experiment_manifest']
            context_parts.append(f"Experiment: {manifest.get('dataset', 'KIBA')} benchmark")
            context_parts.append(f"Models evaluated: {', '.join(manifest.get('models_evaluated', []))}")
            context_parts.append(f"Splits evaluated: {', '.join(manifest.get('splits_evaluated', []))}")
            context_parts.append(f"Random seed: {manifest.get('random_seed', 'N/A')}")
            
            # Best models by split
            if 'best_models_by_split' in manifest:
                for split, best in manifest['best_models_by_split'].items():
                    context_parts.append(f"Best model on {split} split: {best.get('model', 'N/A')} (RMSE: {best.get('test_rmse', 'N/A')}, CI: {best.get('test_ci', 'N/A')})")
            
            sources.append({
                'title': 'Experiment Manifest',
                'file': 'models/kiba_experiment_manifest.json',
                'section': 'experimental_results'
            })

    # Project objective/methodology
    if any(keyword in question_lower for keyword in ['objective', 'goal', 'purpose', 'motivation', 'what', 'project']):
        context_parts.append("DTI-ML is a student project that predicts kinase drug-target binding affinity using classical machine learning.")
        context_parts.append("The project evaluates four ML models (Random Forest, XGBoost, SVR, GPR) on the KIBA dataset.")
        context_parts.append("It uses leakage-safe cold-split evaluation to measure genuine generalization performance.")
        context_parts.append("The project provides SHAP-based interpretability for individual predictions.")
        sources.append({
            'title': 'Project Documentation',
            'file': 'README.md',
            'section': 'project_overview'
        })

    return '\n'.join(context_parts), sources


SYSTEM_PROMPT = """You are the assistant embedded in DTI-ML, a student project that predicts kinase drug-target binding affinity using classical ML. Only answer questions about kinases, binding affinity, cold-split evaluation, SHAP interpretability, the four ML models used, and the current prediction result. Refuse unrelated questions and medical advice. Never invent numbers. Keep every answer under 4 sentences in plain language."""
REFUSAL = "I can only help with questions about this DTI-ML project and its predictions."
FALLBACK = "I couldn't reach the explanation service just now — try again in a moment."
BLOCKED_PATTERNS = re.compile(r"\b(ignore (all|previous) instructions|system prompt|jailbreak|medical advice|diagnos|prescri)\b", re.I)
LOCAL_CACHE = {
    "score": ("what does", "mean", "score"),
    "cold": ("cold split", "cold-split", "lower", "why"),
    "shap": ("shap", "explain shap"),
}
CACHE_ANSWERS = {
    "score": "This score sits on the model's affinity scale — higher means a tighter, more drug-like fit.",
    "cold": "A cold split removes any drug/protein overlap between training and testing, so the score reflects genuine generalisation, not memorisation — which is why it's usually a bit lower, and more trustworthy.",
    "shap": "SHAP measures how much each input feature pushed the prediction up or down — it's the model showing its working, feature by feature.",
}
UNEVALUATED_DATASETS = {"davis", "bindingdb", "uniprot", "chembl", "pubchem", "drugbank"}
MAX_MSGS_PER_SESSION = 18
WINDOW_SECONDS = 3600

_hits: dict[str, deque[float]] = defaultdict(deque)
_hits_lock = Lock()


def _check_rate_limit(client_id: str) -> bool:
    now = time.time()
    with _hits_lock:
        hits = _hits[client_id]
        while hits and now - hits[0] >= WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= MAX_MSGS_PER_SESSION:
            return False
        hits.append(now)
        return True


def _check_unsupported_dataset(question: str) -> str | None:
    """Return a message if the question asks about an unevaluated dataset, else None."""
    question_lower = question.lower()
    for dataset in UNEVALUATED_DATASETS:
        if dataset in question_lower or f" {dataset} " in question_lower or question_lower.endswith(f" {dataset}"):
            return (f"{dataset.capitalize()} has not been evaluated in the current project, "
                    f"so I can't provide a project-specific {dataset} result.")
    return None


def _local_cache_lookup(question: str) -> str | None:
    lowered = question.lower()
    for key, keywords in LOCAL_CACHE.items():
        if any(keyword in lowered for keyword in keywords):
            return CACHE_ANSWERS[key]
    return None


def get_chat_answer(question: str, context: dict[str, Any], client_id: str = "anonymous") -> dict[str, Any]:
    question = (question or "").strip()
    if not question:
        return {"answer": "Ask me anything about this project or the current result.", "sources": []}
    if BLOCKED_PATTERNS.search(question):
        return {"answer": REFUSAL, "sources": []}
    if not _check_rate_limit(client_id):
        return {"answer": "You've hit the question limit for this session — try again in a bit.", "sources": []}

    cached = _local_cache_lookup(question)
    if cached:
        return {"answer": cached, "sources": []}

    unsupported = _check_unsupported_dataset(question)
    if unsupported:
        return {"answer": unsupported, "sources": []}

    # Load project knowledge
    knowledge = _get_project_knowledge()
    project_context, sources = _retrieve_relevant_knowledge(question, knowledge)

    api_key = get_settings().gemini_api_key
    if not api_key:
        return {"answer": FALLBACK, "sources": []}

    # Combine current prediction context with project knowledge
    prediction_context = f"Current prediction — score: {context.get('score')}, label: {context.get('label')}, top features: {context.get('features', context.get('top_features', []))}"
    full_context = f"{project_context}\n\n{prediction_context}" if project_context else prediction_context

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Context: {full_context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=180, temperature=0.4),
        )
        answer = (response.text or "").strip() or FALLBACK
        
        # If no project-specific sources were found but we got an answer, indicate it's from general knowledge
        if not sources and answer != FALLBACK:
            sources.append({
                'title': 'General Knowledge',
                'file': 'Gemini LLM',
                'section': 'general_scientific_knowledge'
            })
        
        return {"answer": answer, "sources": sources}
    except Exception:
        return {"answer": FALLBACK, "sources": []}
