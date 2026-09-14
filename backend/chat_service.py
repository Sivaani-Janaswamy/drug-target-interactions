from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from backend.config import get_settings

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

_hits: dict[str, deque[float]] = defaultdict(deque)
_hits_lock = Lock()
MAX_MSGS_PER_SESSION = 18
WINDOW_SECONDS = 3600


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


def _local_cache_lookup(question: str) -> str | None:
    lowered = question.lower()
    for key, keywords in LOCAL_CACHE.items():
        if any(keyword in lowered for keyword in keywords):
            return CACHE_ANSWERS[key]
    return None


def get_chat_answer(question: str, context: dict[str, Any], client_id: str = "anonymous") -> str:
    question = (question or "").strip()
    if not question:
        return "Ask me anything about this project or the current result."
    if BLOCKED_PATTERNS.search(question):
        return REFUSAL
    if not _check_rate_limit(client_id):
        return "You've hit the question limit for this session — try again in a bit."

    cached = _local_cache_lookup(question)
    if cached:
        return cached

    api_key = get_settings().gemini_api_key
    if not api_key:
        return FALLBACK

    context_str = f"Current prediction — score: {context.get('score')}, label: {context.get('label')}, top features: {context.get('features', context.get('top_features', []))}"
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Context: {context_str}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=180, temperature=0.4),
        )
        return (response.text or "").strip() or FALLBACK
    except Exception:
        return FALLBACK
