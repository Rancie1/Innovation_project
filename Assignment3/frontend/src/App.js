import './App.css';
import { useEffect, useMemo, useState } from 'react';
import { api } from './api';

function App() {
  const [code, setCode] = useState('');
  const [models, setModels] = useState([]);
  const [currentModel, setCurrentModel] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const isCodeValid = useMemo(() => code.trim().length >= 5, [code]);

  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        const data = await api.listModels();
        if (!isMounted) return;
        setModels(data.available_models || []);
        setCurrentModel(data.current_model || '');
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error(e);
      }
    })();
    return () => { isMounted = false; };
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');
    setResult(null);
    if (!isCodeValid) {
      setError('Please enter at least 5 characters of code.');
      return;
    }
    setLoading(true);
    try {
      const data = await api.predict(code);
      setResult(data);
    } catch (e) {
      setError('Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  async function handleModelChange(e) {
    const name = e.target.value;
    setCurrentModel(name);
    try {
      await api.selectModel(name);
    } catch (e) {
      setError('Failed to switch model.');
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1>Assignment 3 — Code Predictor</h1>
        <p className="subtitle">Submit a code snippet to get a prediction from the backend model.</p>
      </header>

      <main className="content">
        <form className="card" onSubmit={handleSubmit} noValidate>
          <div className="row">
            <label htmlFor="model" className="label">Model</label>
            <select id="model" className="select" value={currentModel} onChange={handleModelChange} aria-label="Select model">
              {models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          <div className="row">
            <label htmlFor="code" className="label">Code Snippet</label>
            <textarea
              id="code"
              className={`textarea ${!isCodeValid && code ? 'invalid' : ''}`}
              value={code}
              onChange={(e) => setCode(e.target.value)}
              placeholder="Paste or type your code here..."
              rows={10}
              aria-invalid={!isCodeValid && !!code}
              aria-describedby="code-help"
              required
            />
            <div id="code-help" className="help">Minimum 5 characters.</div>
          </div>

          {error && <div role="alert" className="error">{error}</div>}

          <div className="actions">
            <button type="submit" className="button" disabled={!isCodeValid || loading}>
              {loading ? 'Predicting…' : 'Predict'}
            </button>
          </div>
        </form>

        {result && (
          <section className="card result">
            <h2>Result</h2>
            <pre className="pre">
{JSON.stringify(result, null, 2)}
            </pre>
          </section>
        )}
      </main>

      <footer className="footer">Backend: http://localhost:8000 • Frontend: http://localhost:3000</footer>
    </div>
  );
}

export default App;
