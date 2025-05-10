# Assets Directory

This directory contains images, diagrams, and other visual assets used in the project documentation.

## Contents

- `project_architecture.md`: Diagram of the project architecture
- `unity_integration.md`: Illustration of how the ML models integrate with Unity
- `sample_results.md`: Visual representation of model results and comparisons

## Diagrams

### Project Architecture

The project follows this high-level architecture:

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Unity Game   │───▶│  Prediction   │───▶│   ML Models   │
│  Environment  │◀───│    Server     │◀───│               │
└───────────────┘    └───────────────┘    └───────────────┘
        │                                        ▲
        │                                        │
        │           ┌───────────────┐            │
        └──────────▶│  Shot Data    │────────────┘
                    │  Collection   │
                    └───────────────┘
```

### Model Performance

Comparative model performance visualization:

```
Accuracy (%)
┌────────────────────────────────────────────────┐
│                                                │
│  Baseline RF   █████████████████████  89%      │
│                                                │
│  Oversampled   ███████████████████████  91%    │
│                                                │
│  Feature RF    ██████████████████████  90%     │
│                                                │
│  Ensemble      █████████████████████████  93%  │
│                                                │
└────────────────────────────────────────────────┘
```

### Feature Importance

Relative importance of different features:

```
Feature Importance (%)
┌────────────────────────────────────────────────┐
│                                                │
│  GunTiltY      ████████████████  32%           │
│                                                │
│  Distance      ██████████████  28%             │
│                                                │
│  XDifference   ███████  15%                    │
│                                                │
│  GunTiltX      ██████  12%                     │
│                                                │
│  ZDifference   ████  8%                        │
│                                                │
│  Elevation     █  3%                           │
│                                                │
│  YDifference   █  2%                           │
│                                                │
└────────────────────────────────────────────────┘
```

## Adding New Assets

When adding new assets to this directory:

1. Use clear and descriptive filenames
2. Include appropriate file formats (PNG for images, SVG for diagrams when possible)
3. Optimize image sizes to keep the repository lightweight
4. Update this README with information about the new assets
