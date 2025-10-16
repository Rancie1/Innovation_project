const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.headers.get('content-type')?.includes('application/json') ? res.json() : res.text();
}

export const api = {
  health: () => request('/health'),
  listModels: () => request('/models'),
  selectModel: (modelName) => request('/model', { method: 'PUT', body: JSON.stringify({ model_name: modelName }) }),
  predict: (code) => request('/predict', { method: 'POST', body: JSON.stringify({ code }) }),
};
