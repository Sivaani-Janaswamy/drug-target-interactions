# Chatbot Feature — Architecture Spec

> Documentation only. No implementation in this task.

**Note on framework:** the app is built with **Streamlit** (`app/streamlit_app.py`), not
Flask with static HTML/JS. There is no `dti-ml-frontend-mockup.html` in this repo. This
spec's contract is written framework-agnostically (request/response shape) so it can be
implemented either as a Streamlit-native chat block (`st.chat_input`/`session_state`,
calling the Gemini SDK directly in-process) or, if the project moves to a separate
backend later, as the Flask route shown below. The implementation task should confirm
which of these applies before writing code.

## Provider

Google Gemini Flash free tier, using the official Gemini API. No self-hosted model, no
new infra — one outbound HTTPS call per user question.

**Why this stays deployment-light:** no model weights are bundled or hosted by us; the
only additions to our deployment are one environment variable (API key) and one
lightweight HTTP client dependency (`google-genai`).

## Contract

```
POST /api/chat
Request:  { "question": string,
            "context": { "score": float, "label": string,
                          "top_features": [{ "name": string, "direction": "pos"|"neg",
                                              "weight": float, "explanation": string }] } }
Response: { "answer": string }
```

If implemented in Streamlit instead of a REST route, this becomes a function
`get_chat_answer(question: str, context: dict) -> str` with the same input/output shape,
called directly from the chat UI callback rather than over HTTP.

## Guardrails

- Fixed system prompt scoping the assistant to DTI-ML/kinase/SHAP concepts and the
  current prediction only; refuses off-topic and medical-advice questions.
- Server-side keyword filter as a second layer before the API call.
- `max_output_tokens` capped (~150–200) per response.
- Per-session rate limit (~15–20 messages) to protect the free quota.
- Local scripted-answer cache checked before calling the live API for the most common
  questions (score meaning, cold-split rationale, SHAP explanation).
- Graceful fallback message on API error/timeout — chat never breaks the page.

## Implementation steps (for a future task)

1. Provision free Gemini API key, store as env var (`GEMINI_API_KEY`).
2. Write grounding system prompt.
3. Implement the chat contract above, in whichever form fits the app's actual framework
   at the time (Streamlit chat block or a Flask/FastAPI route).
4. Wire the current prediction's score/label/top_features into the chat context.
5. Add rate limiting + error fallback.
6. Test locally, then add the key as a deployment secret.

Do not write the actual implementation as part of this spec — this document is
planning only.
