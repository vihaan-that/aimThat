#!/usr/bin/env python3
"""
Visualization script for NoScope9000-ML-Analysis

This script generates visualizations of the model performance and feature importance.
"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

# Import custom classifier classes to ensure pickle can find them
# Use direct import with sys.path modification for proper module resolution
from RandomForest import RandomForestClassifier, DecisionTreeClassifier

# Add parent directory to the path
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

def generate_feature_importance(model_path, output_path):
    """Generate and save feature importance visualization."""
    # Load the model
    model = load_model(model_path)
    if model is None or not hasattr(model, 'feature_importances'):
        print("Model does not support feature importance calculation")
        return False
    
    # Get feature importance
    feature_names = [
        'DistanceFromTarget', 
        'ElevationDifference', 
        'GunTiltX', 
        'GunTiltY', 
        'XDifference', 
        'YDifference', 
        'ZDifference'
    ]
    importances = model.feature_importances(len(feature_names))
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance for Shot Prediction', fontsize=16)
    plt.bar(range(len(importances)), 
            [importances[i] for i in indices],
            align='center')
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45, 
               ha='right')
    plt.tight_layout()
    plt.ylabel('Relative Importance', fontsize=14)
    plt.xlabel('Features', fontsize=14)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance visualization saved to {output_path}")
    return True

def load_dataset(data_path):
    """Load and prepare the dataset."""
    try:
        df = pd.read_csv(data_path)
        # Split features and target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def generate_model_comparison(models_dict, X, y, output_path):
    """Generate and save model comparison visualization."""
    plt.figure(figsize=(10, 8))
    
    # Colors for different models
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    # Plot ROC curve for each model
    for (model_name, model_path), color in zip(models_dict.items(), colors):
        model = load_model(model_path)
        if model is None:
            continue
        
        # Make predictions
        if isinstance(model, list):  # Ensemble model
            y_scores = []
            for m in model:
                scores = []
                for x in X:
                    if hasattr(m, 'predict_proba'):
                        scores.append(m.predict([x])[0])
                    else:
                        scores.append(m.predict([x])[0])
                y_scores.append(scores)
            # Take average of scores - convert to numpy arrays first
            y_score = np.mean(np.array(y_scores), axis=0)
        else:  # Single model
            if hasattr(model, 'predict_proba'):
                y_score = [model.predict([x])[0] for x in X]
            else:
                y_score = [model.predict([x])[0] for x in X]
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison visualization saved to {output_path}")
    return True

def main():
    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'SniperDataset.csv')
    models_dir = os.path.join(base_dir, 'models')
    output_dir = os.path.join(base_dir, 'results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    X, y = load_dataset(data_path)
    if X is None or y is None:
        print("Failed to load dataset")
        return 1
    
    # Define models and output paths
    models = {
        'Baseline RF': os.path.join(models_dir, 'rf_baseline_model.pkl'),
        'Oversampled RF': os.path.join(models_dir, 'rf_oversampled_model.pkl'),
        'Feature RF': os.path.join(models_dir, 'rf_feature_selected_model.pkl'),
        'Ensemble': os.path.join(models_dir, 'ensemble_model.pkl')
    }
    
    # Generate feature importance visualization
    feature_importance_path = os.path.join(output_dir, 'feature_importance.png')
    generate_feature_importance(models['Baseline RF'], feature_importance_path)
    
    # Generate model comparison visualization
    model_comparison_path = os.path.join(output_dir, 'model_comparison.png')
    generate_model_comparison(models, X, y, model_comparison_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
