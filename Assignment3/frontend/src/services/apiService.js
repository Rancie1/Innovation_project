import axios from 'axios';

/**
 * API Service for Vulnerability Detection Backend
 * Handles all HTTP requests to the FastAPI server
 */

// Base URL for the FastAPI backend
const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 seconds timeout
});

/**
 * API Service Object
 * Contains all methods for interacting with the backend
 */
const apiService = {
  /**
   * Health check endpoint
   * @returns {Promise} Server health status
   */
  healthCheck: async () => {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  /**
   * Get list of available models
   * @returns {Promise} Object containing available models and current model
   */
  getModels: async () => {
    try {
      const response = await apiClient.get('/models');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch models:', error);
      throw error;
    }
  },

  /**
   * Select a model for predictions
   * @param {string} modelName - Name of the model to select
   * @returns {Promise} Confirmation of model selection
   */
  selectModel: async (modelName) => {
    try {
      const response = await apiClient.put('/model', {
        model_name: modelName
      });
      return response.data;
    } catch (error) {
      console.error('Failed to select model:', error);
      throw error;
    }
  },

  /**
   * Make a prediction on code snippet
   * @param {string} code - Code snippet to analyze
   * @returns {Promise} Prediction results with confidence and probabilities
   */
  predict: async (code) => {
    try {
      const response = await apiClient.post('/predict', {
        code: code
      });
      return response.data;
    } catch (error) {
      console.error('Prediction failed:', error);
      
      // Enhanced error handling with user-friendly messages
      if (error.response) {
        // Server responded with error status
        const errorMessage = error.response.data.detail || 'Prediction failed';
        throw new Error(errorMessage);
      } else if (error.request) {
        // Request made but no response received
        throw new Error('No response from server. Please check if the backend is running.');
      } else {
        // Error in request setup
        throw new Error('Failed to make prediction request');
      }
    }
  }
};

export default apiService;
