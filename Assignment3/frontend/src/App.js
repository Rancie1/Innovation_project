import React, { useState, useEffect } from 'react';
import axios from 'axios';
import VulnerabilityDashboard from './components/VulnerabilityDashboard';
import CodeInputForm from './components/CodeInputForm';
import ThemeToggle from './components/ThemeToggle';
import './App.css';

/**
 * Main Application Component
 * 
 * Handles:
 * - Theme switching (dark/light mode)
 * - API communication with FastAPI backend
 * - State management for predictions and history
 * - Error handling and loading states
 */
function App() {
  // Theme state - default to dark mode
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    return savedTheme || 'dark';
  });

  // Prediction state
  const [predictionData, setPredictionData] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Model state
  const [availableModels, setAvailableModels] = useState([]);
  const [currentModel, setCurrentModel] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
    fetchAvailableModels();
  }, []);

  /**
   * Check if backend is running
   */
  const checkBackendHealth = async () => {
    try {
      const response = await axios.get('http://localhost:8000/health', {
        timeout: 5000
      });
      if (response.data.status === 'ok') {
        setBackendStatus('connected');
      }
    } catch (err) {
      setBackendStatus('disconnected');
      setError('Backend server is not running. Please start the FastAPI server on port 8000.');
    }
  };

  /**
   * Fetch available models from backend
   */
  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get('http://localhost:8000/models');
      setAvailableModels(response.data.available_models || []);
      setCurrentModel(response.data.current_model);
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  };

  /**
   * Switch to a different model
   */
  const handleModelChange = async (modelName) => {
    try {
      setLoading(true);
      await axios.put('http://localhost:8000/model', {
        model_name: modelName
      });
      setCurrentModel(modelName);
      setError(null);
    } catch (err) {
      setError(`Failed to switch model: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Submit code for vulnerability prediction
   */
  const handleCodeSubmit = async (code) => {
    // Validate input
    if (!code || code.trim().length === 0) {
      setError('Please enter some code to analyze');
      return;
    }

    if (code.trim().length < 10) {
      setError('Code is too short. Please enter at least 10 characters.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/predict', {
        code: code
      });

      const result = response.data;

      // Check for backend errors in response
      if (result.prediction && result.prediction.error) {
        setError(result.prediction.error);
        setLoading(false);
        return;
      }

      // Format prediction data for dashboard
      const formattedData = {
        prediction: result.prediction,
        model_name: result.model_name
      };

      setPredictionData(formattedData);

      // Add to history
      const historyEntry = {
        timestamp: new Date().toLocaleTimeString(),
        category: result.prediction.predicted_category || 'Unknown',
        confidence: ((result.prediction.confidence || 0) * 100).toFixed(1)
      };

      setPredictionHistory(prev => [...prev, historyEntry]);
      
    } catch (err) {
      if (err.response) {
        // Backend returned an error
        setError(`Prediction failed: ${err.response.data.detail || err.message}`);
      } else if (err.request) {
        // Request made but no response
        setError('No response from server. Is the backend running on port 8000?');
      } else {
        // Something else happened
        setError(`Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  /**
   * Toggle between dark and light themes
   */
  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'dark' ? 'light' : 'dark');
  };

  /**
   * Clear current prediction and history
   */
  const handleClearResults = () => {
    setPredictionData(null);
    setPredictionHistory([]);
    setError(null);
  };

  return (
    <div className={`app-container ${theme}`}>
      {/* Header with theme toggle */}
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <h1 className="app-title">
              <span className="terminal-symbol">🛡️</span> VULNERABILITY DETECTOR
            </h1>
            <div className="status-badge">
              <span className={`status-indicator ${backendStatus}`}></span>
              <span className="status-text">
                {backendStatus === 'connected' ? 'Backend Connected' : 
                 backendStatus === 'disconnected' ? 'Backend Disconnected' : 
                 'Checking...'}
              </span>
            </div>
          </div>
          <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
        </div>
      </header>

      <main className="app-main">
        {/* Model selector */}
        {availableModels.length > 0 && (
          <div className="model-selector-container">
            <label htmlFor="model-select">Active Model:</label>
            <select 
              id="model-select"
              value={currentModel || ''} 
              onChange={(e) => handleModelChange(e.target.value)}
              disabled={loading}
              className="model-select"
            >
              {availableModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠</span>
            <span className="error-message">{error}</span>
            <button 
              className="error-close"
              onClick={() => setError(null)}
              aria-label="Close error"
            >
              ×
            </button>
          </div>
        )}

        {/* Code input form */}
        <CodeInputForm 
          onSubmit={handleCodeSubmit}
          loading={loading}
          onClear={handleClearResults}
          hasResults={predictionData !== null}
        />

        {/* Results dashboard */}
        {predictionData && (
          <VulnerabilityDashboard 
            predictionData={predictionData}
            predictionHistory={predictionHistory}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>AI-Powered Vulnerability Detection System | Powered by FastAPI & React</p>
      </footer>
    </div>
  );
}

export default App;
