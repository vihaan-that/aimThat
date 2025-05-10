# Machine Learning Methodology

This document describes the machine learning approaches used in the NoScope9000-ML-Analysis project.

## Dataset Overview

The dataset consists of 13,735 shot samples with 8 features representing different aspects of a sniper shot scenario in a Unity physics simulation:

- `DistanceFromTarget`: The linear distance between the sniper and target
- `ElevationDifference`: The vertical difference in position
- `GunTiltX`: The horizontal angle of the gun
- `GunTiltY`: The vertical angle of the gun
- `XDifference`: Horizontal positional difference
- `YDifference`: Vertical positional difference
- `ZDifference`: Depth positional difference
- `HitOrMiss`: Target outcome (1 = Hit, 0 = Miss)

## Data Preprocessing

1. **Standardization**: All features were standardized using Z-score normalization (mean=0, std=1)
2. **Train-Test Split**: The dataset was split into 80% training and 20% testing data
3. **Class Imbalance Analysis**: The dataset was examined for class imbalance between hit and miss outcomes

## Model Development

### 1. Baseline Random Forest

A standard Random Forest classifier implemented from scratch with the following parameters:
- 10 decision trees
- Maximum depth of 10 per tree
- Default feature selection at each node

### 2. Oversampled Random Forest

To address class imbalance, this model uses:
- Class weight balancing to give equal importance to both hit and miss classes
- Random oversampling of the minority class during training
- Same base parameters as the baseline model (10 trees, max depth of 10)

### 3. Feature-Selected Random Forest

This model focuses on optimizing feature selection:
- Feature importance analysis to identify the most predictive features
- Training on the top 5 most important features only
- Class weight balancing similar to the oversampled model

### 4. Ensemble Model

A more robust approach using multiple balanced models:
- Creates 5 different balanced subsets of the training data
- Trains a separate Random Forest on each subset
- Aggregates predictions using majority voting from all models
- Provides greater stability and accuracy across different scenarios

## Evaluation Metrics

The models were evaluated using:
- Accuracy score
- Precision and recall for both classes
- F1 score
- Confusion matrix
- ROC curves and AUC scores

## Implementation Details

All models were implemented using a custom Python implementation rather than relying on standard libraries like scikit-learn. This approach provided:
- Greater control over the implementation details
- Better understanding of the underlying algorithms
- Flexibility to modify the algorithms as needed for the specific dataset

The code includes custom implementations of:
- Decision tree with Gini impurity calculation
- Random Forest with bootstrap sampling
- Specialized balancing techniques for handling class imbalance
- Feature importance calculation and selection
- Ensemble methods for aggregating multiple models
