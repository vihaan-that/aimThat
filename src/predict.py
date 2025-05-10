#!/usr/bin/env python3
"""
NoScope9000 Shot Prediction Script

This script loads the trained models and allows for prediction of shot outcomes
based on input parameters.
"""
import os
import sys
import pickle
import argparse

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_model(model_path):
    """Load a trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def standardize_features(features, means=None, stds=None):
    """Standardize input features (Z-score normalization)."""
    # Pre-computed means and stds from training data
    # These should be replaced with values from your training dataset
    if means is None:
        means = [20.5, 1.5, 75.3, 120.4, 0.0, 1.5, 15.2]
    if stds is None:
        stds = [15.2, 0.5, 20.1, 50.6, 25.3, 0.5, 20.8]
    
    standardized = []
    for i, feature in enumerate(features):
        if stds[i] != 0:
            standardized.append((feature - means[i]) / stds[i])
        else:
            standardized.append(0.0)
    return standardized

def predict_shot(model, features):
    """Predict if a shot will hit or miss given the input features."""
    # Standardize the features
    std_features = standardize_features(features)
    
    # Make prediction
    if isinstance(model, list):  # Ensemble model
        votes = []
        for m in model:
            pred = m.predict([std_features])[0]
            votes.append(pred)
        prediction = max(set(votes), key=votes.count)
        return prediction
    else:  # Single model
        return model.predict([std_features])[0]

def main():
    parser = argparse.ArgumentParser(description='Predict sniper shot outcomes using trained models.')
    parser.add_argument('--model', type=str, choices=['baseline', 'oversampled', 'feature', 'ensemble'], 
                        default='ensemble', help='Model to use for prediction')
    parser.add_argument('--distance', type=float, required=True, help='Distance from target')
    parser.add_argument('--elevation', type=float, required=True, help='Elevation difference')
    parser.add_argument('--tiltx', type=float, required=True, help='Gun tilt X')
    parser.add_argument('--tilty', type=float, required=True, help='Gun tilt Y')
    parser.add_argument('--xdiff', type=float, required=True, help='X difference')
    parser.add_argument('--ydiff', type=float, required=True, help='Y difference')
    parser.add_argument('--zdiff', type=float, required=True, help='Z difference')
    args = parser.parse_args()

    # Map the model choice to the actual model file
    model_files = {
        'baseline': '../models/rf_baseline_model.pkl',
        'oversampled': '../models/rf_oversampled_model.pkl',
        'feature': '../models/rf_feature_selected_model.pkl',
        'ensemble': '../models/ensemble_model.pkl'
    }

    # Load the model
    model_path = os.path.join(os.path.dirname(__file__), model_files[args.model])
    model = load_model(model_path)
    if model is None:
        print(f"Failed to load model from {model_path}")
        return 1

    # Prepare the input features
    features = [
        args.distance,
        args.elevation,
        args.tiltx,
        args.tilty,
        args.xdiff,
        args.ydiff,
        args.zdiff
    ]

    # Make prediction
    prediction = predict_shot(model, features)
    result = "HIT" if prediction == 1 else "MISS"
    
    print(f"\nPrediction: {result}")
    print("\nInput parameters:")
    print(f"  Distance from target: {args.distance}")
    print(f"  Elevation difference: {args.elevation}")
    print(f"  Gun tilt X: {args.tiltx}")
    print(f"  Gun tilt Y: {args.tilty}")
    print(f"  X difference: {args.xdiff}")
    print(f"  Y difference: {args.ydiff}")
    print(f"  Z difference: {args.zdiff}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
