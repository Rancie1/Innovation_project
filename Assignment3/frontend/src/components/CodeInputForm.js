import React, { useState } from 'react';
import './CodeInputForm.css';

/**
 * CodeInputForm Component
 * 
 * Form for submitting code snippets for vulnerability analysis
 * Features:
 * - Code textarea with syntax highlighting styles
 * - Character and line count display
 * - Input validation
 * - Loading states
 * - Clear functionality
 */
const CodeInputForm = ({ onSubmit, loading, onClear, hasResults }) => {
  const [code, setCode] = useState('');
  const [validationError, setValidationError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Reset validation error
    setValidationError('');

    // Validate input
    if (!code.trim()) {
      setValidationError('Please enter code to analyze');
      return;
    }

    if (code.trim().length < 10) {
      setValidationError('Code must be at least 10 characters long');
      return;
    }

    // Submit if validation passes
    onSubmit(code);
  };

  const handleClear = () => {
    setCode('');
    setValidationError('');
    onClear();
  };

  const handleCodeChange = (e) => {
    setCode(e.target.value);
    // Clear validation error when user starts typing
    if (validationError) {
      setValidationError('');
    }
  };

  // Calculate stats
  const charCount = code.length;
  const lineCount = code ? code.split('\n').length : 0;
  const wordCount = code.trim() ? code.trim().split(/\s+/).length : 0;

  return (
    <div className="code-input-container">
      <div className="form-header">
        <h2 className="form-title">
          <span className="prompt-symbol">&gt;_</span> Code Analysis Input
        </h2>
        <div className="form-description">
          Enter or paste your code snippet below for vulnerability detection analysis
        </div>
      </div>

      <form onSubmit={handleSubmit} className="code-form">
        <div className="textarea-wrapper">
          <textarea
            className="code-textarea"
            value={code}
            onChange={handleCodeChange}
            placeholder="// Paste your code here...
// Example:
function validateInput(userInput) {
  eval(userInput); // Potential vulnerability
  return true;
}"
            disabled={loading}
            rows={15}
            spellCheck={false}
          />
          
          {/* Code statistics */}
          <div className="code-stats">
            <div className="stat-item">
              <span className="stat-label">Characters:</span>
              <span className="stat-value">{charCount}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Lines:</span>
              <span className="stat-value">{lineCount}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Words:</span>
              <span className="stat-value">{wordCount}</span>
            </div>
          </div>
        </div>

        {/* Validation error */}
        {validationError && (
          <div className="validation-error">
            <span className="error-icon">⚠</span>
            {validationError}
          </div>
        )}

        {/* Action buttons */}
        <div className="form-actions">
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={loading || !code.trim()}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              <>
                <span className="btn-icon">🔍</span>
                Analyze Code
              </>
            )}
          </button>

          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleClear}
            disabled={loading}
          >
            <span className="btn-icon">🗑️</span>
            Clear All
          </button>
        </div>

        {/* Helper text */}
        <div className="helper-text">
          <p>
            <strong>Tip:</strong> For best results, submit code snippets between 50-5000 characters. 
            The AI model will analyze your code for common vulnerability patterns.
          </p>
        </div>
      </form>
    </div>
  );
};

export default CodeInputForm;
