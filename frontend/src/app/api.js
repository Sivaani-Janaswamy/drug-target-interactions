import { API_BASE } from './constants';

async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || 'The request could not be completed.');
  return body;
}

export const getPresets = () => request('/presets');
export const getBenchmarks = () => request('/benchmarks');
export const predict = (payload) => request('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
export const askChat = (question, context) => request('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, context }) });
