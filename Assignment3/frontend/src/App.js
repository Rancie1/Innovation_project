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
  const [taskType, setTaskType] = useState('severity'); // 'severity' | 'binary'

  // Filter using backend IDs (underscored)
  const severityModels = useMemo(() => models.filter(m => m === 'logistic_regression' || m === 'random_forest'), [models]);
  const binaryModels = useMemo(() => models.filter(m => m === 'model1_logreg' || m === 'model1_random_forest'), [models]);
  const visibleModels = taskType === 'severity' ? severityModels : binaryModels;

  // Display names with spaces while keeping IDs under the hood
  const displayName = (modelId) => {
    if (modelId === 'model1_logreg') return 'logistic regression';
    if (modelId === 'model1_random_forest') return 'random forest';
    return String(modelId).replace(/_/g, ' ');
  };

  const isCodeValid = useMemo(() => code.trim().length >= 5, [code]);

  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        const data = await api.listModels();
        if (!isMounted) return;
        const available = data.available_models || [];
        setModels(available);
        // Choose initial task/model
        const hasSeverity = available.some(m => m === 'logistic_regression' || m === 'random_forest');
        const initialTask = hasSeverity ? 'severity' : 'binary';
        setTaskType(initialTask);
        const candidates = hasSeverity
          ? available.filter(m => m === 'logistic_regression' || m === 'random_forest')
          : available.filter(m => m === 'model1_logreg' || m === 'model1_random_forest');
        const chosen = candidates[0] || available[0] || '';
        if (chosen) {
          setCurrentModel(chosen);
          try { await api.selectModel(chosen); } catch {}
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error(e);
      }
    })();
    return () => { isMounted = false; };
  }, []);

  async function applyModel(name) {
    setCurrentModel(name);
    try {
      await api.selectModel(name);
    } catch (e) {
      setError('Failed to switch model.');
    }
  }

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
    await applyModel(name);
  }

  async function handleTaskToggle(nextTask) {
    if (taskType === nextTask) return;
    setTaskType(nextTask);
    // pick first available model for that task and apply
    const candidates = (nextTask === 'severity') ? severityModels : binaryModels;
    const chosen = candidates[0] || '';
    if (chosen) {
      await applyModel(chosen);
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
            <div className="segmented" role="tablist" aria-label="Task type">
              <button type="button" role="tab" aria-selected={taskType==='severity'} className={`chip ${taskType==='severity' ? 'active' : ''}`} onClick={() => handleTaskToggle('severity')}>
                Severity type
              </button>
              <button type="button" role="tab" aria-selected={taskType==='binary'} className={`chip ${taskType==='binary' ? 'active' : ''}`} onClick={() => handleTaskToggle('binary')}>
                Binary
              </button>
            </div>
          </div>

          <div className="row">
            <label htmlFor="model" className="label">Model</label>
            <select id="model" className="select" value={currentModel} onChange={handleModelChange} aria-label="Select model">
              {visibleModels.map((m) => (
                <option key={m} value={m}>{displayName(m)}</option>
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
