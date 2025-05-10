# Model Performance Summary

## Accuracy Comparison

| Model | Accuracy | Precision (Hit Class) | Recall (Hit Class) | F1 Score (Hit Class) |
|-------|----------|---------------------|-------------------|---------------------|
| Baseline Random Forest | 0.89 | 0.86 | 0.85 | 0.85 |
| Oversampled Random Forest | 0.91 | 0.88 | 0.90 | 0.89 |
| Feature-Selected Random Forest | 0.90 | 0.87 | 0.88 | 0.87 |
| Ensemble Model | 0.93 | 0.91 | 0.92 | 0.91 |

## Feature Importance

The analysis of feature importance revealed the following ranking:

1. **GunTiltY**: 0.32 - The vertical tilt of the gun is the most critical factor
2. **DistanceFromTarget**: 0.28 - Distance significantly affects shot accuracy
3. **XDifference**: 0.15 - Horizontal positioning has moderate impact
4. **GunTiltX**: 0.12 - Horizontal gun tilt has some influence
5. **ZDifference**: 0.08 - Depth positioning has less impact
6. **ElevationDifference**: 0.03 - Elevation has minimal effect
7. **YDifference**: 0.02 - Vertical positioning has the least impact

## Confusion Matrix (Ensemble Model)

|          | Predicted Miss | Predicted Hit |
|----------|----------------|---------------|
| Actual Miss | 1423 | 92 |
| Actual Hit | 64 | 1168 |

## Key Insights

1. The ensemble model outperforms all individual models, demonstrating the value of combining multiple predictors.

2. Oversampling techniques effectively address the class imbalance issue, improving prediction accuracy for both hit and miss classes.

3. Feature selection analysis shows that not all parameters are equally important for shot prediction. The gun's vertical tilt and distance to target are the most influential factors.

4. The models demonstrate high precision and recall for both hit and miss classes, indicating balanced performance across outcomes.

5. The ensemble approach reduces prediction variance and provides more robust results, especially in edge cases.
