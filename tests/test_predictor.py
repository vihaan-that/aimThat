#!/usr/bin/env python3
"""
Tests for the NoScope9000 Predictor
"""
import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.noscope_predictor import NoScopePredictor

class TestNoScopePredictor(unittest.TestCase):
    """Tests for the NoScopePredictor class"""
    
    def setUp(self):
        """Set up a predictor instance for testing"""
        self.predictor = NoScopePredictor(model_type='ensemble')
    
    def test_standardize_features(self):
        """Test that feature standardization works correctly"""
        # Test with means and stds matching the defaults
        features = [20.5, 1.5, 75.3, 120.4, 0.0, 1.5, 15.2]
        standardized = self.predictor.standardize_features(features)
        # If we standardize with the same means, all values should be 0
        self.assertTrue(all(abs(x) < 1e-6 for x in standardized))
        
        # Test with different values
        features = [40.0, 3.0, 90.0, 180.0, 5.0, 3.0, 30.0]
        standardized = self.predictor.standardize_features(features)
        # Values should be non-zero after standardization
        self.assertTrue(any(abs(x) > 0.1 for x in standardized))
    
    def test_predict_returns_valid_result(self):
        """Test that predict method returns valid results"""
        features = [20.5, 1.5, 75.3, 120.4, 0.0, 1.5, 15.2]
        prediction, confidence = self.predictor.predict(features)
        
        # Prediction should be either 0 or 1
        self.assertIn(prediction, [0, 1])
        
        # Confidence should be between 0 and 1
        self.assertTrue(0.0 <= confidence <= 1.0)
    
    def test_predict_with_params(self):
        """Test the named parameter prediction method"""
        result = self.predictor.predict_with_params(
            distance=20.5, 
            elevation=1.5, 
            tilt_x=75.3, 
            tilt_y=120.4, 
            x_diff=0.0, 
            y_diff=1.5, 
            z_diff=15.2
        )
        
        # Check result structure
        self.assertIn('prediction', result)
        self.assertIn('hit', result)
        self.assertIn('confidence', result)
        self.assertIn('inputs', result)
        
        # Check types
        self.assertIsInstance(result['prediction'], int)
        self.assertIsInstance(result['hit'], bool)
        self.assertIsInstance(result['confidence'], float)

if __name__ == '__main__':
    unittest.main()
