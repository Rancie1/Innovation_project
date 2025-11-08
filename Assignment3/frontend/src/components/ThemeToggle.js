import React from 'react';
import './ThemeToggle.css';

/**
 * ThemeToggle Component
 * 
 * Toggle switch for switching between dark and light themes
 * Features smooth animation and terminal-style icons
 */
const ThemeToggle = ({ theme, toggleTheme }) => {
  return (
    <div className="theme-toggle-container">
      <button 
        className="theme-toggle-button"
        onClick={toggleTheme}
        aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
      >
        <div className={`toggle-track ${theme}`}>
          <div className="toggle-thumb">
            {theme === 'dark' ? (
              <span className="icon">🌙</span>
            ) : (
              <span className="icon">☀️</span>
            )}
          </div>
        </div>
        <span className="toggle-label">
          {theme === 'dark' ? 'Dark Mode' : 'Light Mode'}
        </span>
      </button>
    </div>
  );
};

export default ThemeToggle;
