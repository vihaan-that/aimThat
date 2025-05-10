# aimThat 🎯🤖

The machine learning component of the NoScope9000 sniper shot prediction simulator. This repository contains the ML models and analysis developed by the NoScope9000 team, complementing our [Unity simulation environment](https://github.com/AbhiramKothagundu/NoScope9000).

## 📋 Overview

As part of the broader NoScope9000 project, this repository contains the machine learning components that power our sniper shot prediction system. Our team developed both the Unity-based physics simulation (in the NoScope9000 repository) and the machine learning models (in this repository) to create a complete system that accurately predicts shot outcomes.

## 🔍 Key Features

- **Multiple Model Implementations**: Random Forest classifiers with various optimization techniques
- **Data Imbalance Handling**: Techniques to address the hit/miss class imbalance
- **Feature Importance Analysis**: Identification of the most critical factors for shot prediction
- **Ensemble Learning**: Voting-based ensemble model for improved prediction accuracy
- **Comprehensive Metrics**: Detailed performance evaluation including ROC curves, confusion matrices, and classification reports

## 🗂️ Repository Structure

```
├── data/                # Dataset files
├── models/              # Trained model files 
├── notebooks/           # Jupyter notebooks for analysis and visualization
├── src/                 # Source code for the ML implementation
├── docs/                # Documentation and resources
├── results/             # Visualization outputs and performance metrics
└── assets/              # Images and additional files
```

## 📊 Dataset

The dataset contains parameters that influence shot accuracy:
- `DistanceFromTarget` (float): Distance to target
- `ElevationDifference` (float): Difference in elevation
- `GunTiltX` (float): Horizontal gun tilt
- `GunTiltY` (float): Vertical gun tilt
- `XDifference`, `YDifference`, `ZDifference` (float): Position differences
- `HitOrMiss` (binary): Target outcome (1 = Hit, 0 = Miss)

## 🧠 Models and Results

| Model | Accuracy | Key Features |
|-------|----------|-------------|
| Baseline Random Forest | 0.89 | Basic implementation |
| Oversampled Random Forest | 0.91 | Balanced class representation |
| Feature-Selected Random Forest | 0.90 | Optimized feature subset |
| Ensemble Model | 0.93 | Majority voting from multiple models |

## 📈 Key Insights

- The most influential factors for shot prediction are GunTiltY, DistanceFromTarget, and XDifference
- Oversampling techniques significantly improved model performance on the minority class
- Ensemble methods provided the best overall prediction accuracy

## 🔧 Setup & Usage

1. Clone the repository
```bash
git clone https://github.com/yourusername/NoScope9000-ML-Analysis.git
cd NoScope9000-ML-Analysis
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the prediction models
```bash
python src/predict.py
```

## 📚 Documentation

For more detailed information, refer to the notebooks in the `notebooks/` directory and the documentation in the `docs/` folder.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

This project is a collaborative effort by our team:

- **Abhiram Kothagundu** - Unity Developmen
- **[Your Name]** - Machine Learning 
- **[Team Member]** - Data Analysis
- **[Team Member]** - Model Training


## 🔗 Project Components

- [NoScope9000](https://github.com/AbhiramKothagundu/NoScope9000) - The Unity simulation environment
- [aimThat](https://github.com/yourusername/aimThat) - The ML models and analysis (this repository)

Both repositories were developed in parallel by our team as part of a unified academic project to create an AI-powered sniper shot prediction system.
