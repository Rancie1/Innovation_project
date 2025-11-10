import React, { useState, useEffect, useCallback } from 'react';
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
  
  // Classification type state - 'severity' or 'binary'
  const [classificationType, setClassificationType] = useState(() => {
    const savedType = localStorage.getItem('classificationType');
    return savedType || 'severity';
  });

  // Info modal state
  const [infoModal, setInfoModal] = useState(null); // 'severity', 'binary', or null

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Save classification type to localStorage
  useEffect(() => {
    localStorage.setItem('classificationType', classificationType);
  }, [classificationType]);

  /**
   * Format model name for display
   */
  const formatModelName = (modelName) => {
    const nameMap = {
      'logistic_regression': 'Logistic Regression',
      'random_forest': 'Random Forest',
      'model1_logreg': 'Logistic Regression',
      'model1_random_forest': 'Random Forest'
    };
    return nameMap[modelName] || modelName;
  };

  /**
   * Switch to a different model
   */
  const handleModelChange = useCallback(async (modelName) => {
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
  }, []);

  /**
   * Filter models based on classification type
   */
  const getFilteredModels = useCallback((allModels) => {
    if (classificationType === 'severity') {
      return allModels.filter(model => 
        model === 'logistic_regression' || model === 'random_forest'
      );
    } else {
      return allModels.filter(model => 
        model === 'model1_logreg' || model === 'model1_random_forest'
      );
    }
  }, [classificationType]);

  /**
   * Fetch available models from backend
   */
  const fetchAvailableModels = useCallback(async () => {
    try {
      const response = await axios.get('http://localhost:8000/models');
      const allModels = response.data.available_models || [];
      setAvailableModels(allModels);
      
      // Filter models based on current classification type
      const filteredModels = getFilteredModels(allModels);
      
      // If current model is not in filtered list, switch to first available
      const currentModelFromBackend = response.data.current_model;
      if (filteredModels.length > 0) {
        if (filteredModels.includes(currentModelFromBackend)) {
          setCurrentModel(currentModelFromBackend);
        } else {
          // Auto-select first model of the current classification type
          await handleModelChange(filteredModels[0]);
        }
      } else {
        setCurrentModel(null);
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  }, [getFilteredModels, handleModelChange]);

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

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
    fetchAvailableModels();
  }, [fetchAvailableModels]);

  // Refetch and filter models when classification type changes
  useEffect(() => {
    if (availableModels.length > 0) {
      const filteredModels = getFilteredModels(availableModels);
      if (filteredModels.length > 0) {
        // Auto-select first model of the new classification type
        if (!filteredModels.includes(currentModel)) {
          handleModelChange(filteredModels[0]);
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classificationType]);


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
        category: result.prediction.predicted_category || 
                  result.prediction.predicted_label_name || 
                  result.prediction.predicted_label || 
                  'Unknown',
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
   * Switch classification type
   */
  const handleClassificationTypeChange = (type) => {
    setClassificationType(type);
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
        {/* Classification Type Selector */}
        <div className="classification-type-container">
          <label className="classification-type-label">Classification Type:</label>
          <div className="classification-type-buttons">
            <div className="classification-type-group">
              <button
                type="button"
                className={`classification-type-btn ${classificationType === 'severity' ? 'active' : ''}`}
                onClick={() => handleClassificationTypeChange('severity')}
                disabled={loading}
              >
                Severity Type Classification
              </button>
              <button
                type="button"
                className="info-btn"
                onClick={() => setInfoModal(infoModal === 'severity' ? null : 'severity')}
                disabled={loading}
                aria-label="Info about Severity Type Classification"
                title="Info about Severity Type Classification"
              >
                ?
              </button>
            </div>
            <div className="classification-type-group">
              <button
                type="button"
                className={`classification-type-btn ${classificationType === 'binary' ? 'active' : ''}`}
                onClick={() => handleClassificationTypeChange('binary')}
                disabled={loading}
              >
                Binary Classification
              </button>
              <button
                type="button"
                className="info-btn"
                onClick={() => setInfoModal(infoModal === 'binary' ? null : 'binary')}
                disabled={loading}
                aria-label="Info about Binary Classification"
                title="Info about Binary Classification"
              >
                ?
              </button>
            </div>
          </div>
        </div>

        {/* Info Modal */}
        {infoModal && (
          <div className="info-modal-overlay" onClick={() => setInfoModal(null)}>
            <div className="info-modal" onClick={(e) => e.stopPropagation()}>
              <button
                className="info-modal-close"
                onClick={() => setInfoModal(null)}
                aria-label="Close info"
              >
                ×
              </button>
              <h3 className="info-modal-title">
                {infoModal === 'severity' ? 'Severity Type Classification' : 'Binary Classification'}
              </h3>
              <div className="info-modal-content">
                {infoModal === 'severity' ? (
                  <>
                    <p><strong>Input:</strong></p>
                    <p>Code snippets that may contain various types of security vulnerabilities. The model analyzes the code to identify specific Common Weakness Enumeration (CWE) categories.</p>
                    <p><strong>Output:</strong></p>
                    <p>Predicts the specific CWE vulnerability category (e.g., CWE-79, CWE-89, CWE-22) with confidence scores and probability distributions across all possible categories. Provides detailed multi-class classification results showing which vulnerability type is most likely present.</p>
                    <p><strong>Models Available:</strong></p>
                    <ul>
                      <li>Logistic Regression</li>
                      <li>Random Forest</li>
                    </ul>
                  </>
                ) : (
                  <>
                    <p><strong>Input:</strong></p>
                    <p>Code snippets that need to be classified as either safe or unsafe. The model performs a binary classification to determine if the code contains any security vulnerability.</p>
                    <p><strong>Output:</strong></p>
                    <p>Classifies code as either <strong>Safe</strong> or <strong>Unsafe</strong> with confidence scores and probability distributions for both classes. Provides a simple binary decision indicating whether the code is vulnerable or not, without specifying the exact vulnerability type.</p>
                    <p><strong>Models Available:</strong></p>
                    <ul>
                      <li>Logistic Regression</li>
                      <li>Random Forest</li>
                    </ul>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Model selector */}
        {availableModels.length > 0 && (() => {
          const filteredModels = getFilteredModels(availableModels);
          return filteredModels.length > 0 ? (
            <div className="model-selector-container">
              <label htmlFor="model-select">Active Model:</label>
              <select 
                id="model-select"
                value={currentModel || ''} 
                onChange={(e) => handleModelChange(e.target.value)}
                disabled={loading}
                className="model-select"
              >
                {filteredModels.map(model => (
                  <option key={model} value={model}>{formatModelName(model)}</option>
                ))}
              </select>
            </div>
          ) : (
            <div className="model-selector-container">
              <div className="no-models-message">
                No {classificationType === 'severity' ? 'severity type' : 'binary'} models available.
                {classificationType === 'severity' 
                  ? ' Please ensure logistic_regression or random_forest models are loaded.'
                  : ' Please ensure model1_logreg or model1_random_forest models are loaded.'}
              </div>
            </div>
          );
        })()}

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
