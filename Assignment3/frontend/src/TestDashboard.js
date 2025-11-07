import React, { useState, useEffect } from 'react';
import VulnerabilityDashboard from './components/VulnerabilityDashboard';
import './TestDashboard.css';

function TestDashboard() {
  const [codeInput, setCodeInput] = useState('');
  const [predictionData, setPredictionData] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_URL = 'http://localhost:8000';

  const sampleCode = `def get_user(username):
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()`;

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!codeInput.trim()) {
      setError('Please enter some code');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: codeInput })
      });

      if (!response.ok) throw new Error('Prediction failed');

      const result = await response.json();
      setPredictionData(result);

      const historyEntry = {
        timestamp: predictionHistory.length + 1,
        confidence: result.prediction.confidence 
          ? (result.prediction.confidence * 100).toFixed(2) 
          : 0,
        category: result.prediction.predicted_category || 'Unknown'
      };
      
      setPredictionHistory(prev => [...prev, historyEntry]);
    } catch (err) {
      setError('Failed to get prediction. Make sure backend is running on port 8000.');
    } finally {
      setIsLoading(false);
    }
  };

  const loadSample = () => {
    setCodeInput(sampleCode);
  };

  return (
    <div className="test-container">
      <header className="test-header">
        <h1>Vulnerability Detection Dashboard - Test Page</h1>
        <p>Data Visualization Component Testing</p>
      </header>

      <div className="test-content">
        <div className="input-section">
          <h2>Input Code for Analysis</h2>
          
          <form onSubmit={handleSubmit}>
            <textarea
              value={codeInput}
              onChange={(e) => setCodeInput(e.target.value)}
              placeholder="Paste code here..."
              rows="10"
              className="code-input"
            />
            
            <div className="button-group">
              <button 
                type="button" 
                onClick={loadSample}
                className="btn-secondary"
              >
                Load Sample Code
              </button>
              <button 
                type="submit" 
                className="btn-primary"
                disabled={isLoading}
              >
                {isLoading ? 'Analyzing...' : 'Analyze Code'}
              </button>
            </div>
          </form>

          {error && (
            <div className="error-box">
              {error}
            </div>
          )}
        </div>

        <div className="visualization-section">
          <VulnerabilityDashboard
            predictionData={predictionData}
            predictionHistory={predictionHistory}
          />
        </div>
      </div>
    </div>
  );
}

export default TestDashboard;
