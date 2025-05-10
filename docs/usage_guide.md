# NoScope9000-ML Usage Guide

This document provides instructions for using the trained models in the NoScope9000-ML-Analysis project.

## Quick Start

The simplest way to use the models is through the provided `predict.py` script:

```bash
cd NoScope9000-ML-Analysis
python src/predict.py --model ensemble \
                      --distance 20.5 \
                      --elevation 1.5 \
                      --tiltx 70.3 \
                      --tilty 150.4 \
                      --xdiff 10.2 \
                      --ydiff 1.4 \
                      --zdiff 18.3
```

This will output a prediction of whether the shot will hit or miss based on the provided parameters.

## Available Models

The project includes four trained models that you can use:

1. **baseline**: The basic Random Forest model (`--model baseline`)
2. **oversampled**: Random Forest with oversampling for class balance (`--model oversampled`)
3. **feature**: Random Forest trained on selected important features (`--model feature`)
4. **ensemble**: Voting ensemble of multiple models (default, `--model ensemble`)

## Integration with Unity

To integrate with the original NoScope9000 Unity project:

1. Set up a simple HTTP server to receive shot parameters:

```python
from flask import Flask, request, jsonify
import os
import sys
import pickle

app = Flask(__name__)

# Load the model (use absolute path)
model_path = "/path/to/NoScope9000-ML-Analysis/models/ensemble_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['distance'],
        data['elevation'],
        data['tiltx'],
        data['tilty'],
        data['xdiff'],
        data['ydiff'],
        data['zdiff']
    ]
    
    # Standardize features (using pre-computed means and stds)
    means = [20.5, 1.5, 75.3, 120.4, 0.0, 1.5, 15.2]
    stds = [15.2, 0.5, 20.1, 50.6, 25.3, 0.5, 20.8]
    std_features = []
    for i, feature in enumerate(features):
        if stds[i] != 0:
            std_features.append((feature - means[i]) / stds[i])
        else:
            std_features.append(0.0)
    
    # Make prediction
    if isinstance(model, list):  # Ensemble model
        votes = []
        for m in model:
            pred = m.predict([std_features])[0]
            votes.append(pred)
        prediction = max(set(votes), key=votes.count)
    else:  # Single model
        prediction = model.predict([std_features])[0]
    
    return jsonify({'prediction': int(prediction), 'hit': bool(prediction == 1)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

2. In Unity, send HTTP requests to this server before firing:

```csharp
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System.Text;

public class ShotPredictor : MonoBehaviour
{
    [SerializeField] private string serverUrl = "http://localhost:5000/predict";
    
    public IEnumerator PredictShot(float distance, float elevation, float tiltX, float tiltY, 
                                   float xDiff, float yDiff, float zDiff)
    {
        // Create JSON payload
        string json = JsonUtility.ToJson(new ShotData
        {
            distance = distance,
            elevation = elevation,
            tiltx = tiltX,
            tilty = tiltY,
            xdiff = xDiff,
            ydiff = yDiff,
            zdiff = zDiff
        });

        // Send request to prediction server
        using (UnityWebRequest request = new UnityWebRequest(serverUrl, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                string resultText = request.downloadHandler.text;
                PredictionResult result = JsonUtility.FromJson<PredictionResult>(resultText);
                
                Debug.Log("Shot prediction: " + (result.hit ? "HIT" : "MISS"));
                
                // Use the prediction to update UI or game logic
                // ...
            }
            else
            {
                Debug.LogError("Error: " + request.error);
            }
        }
    }
    
    [System.Serializable]
    private class ShotData
    {
        public float distance;
        public float elevation;
        public float tiltx;
        public float tilty;
        public float xdiff;
        public float ydiff;
        public float zdiff;
    }
    
    [System.Serializable]
    private class PredictionResult
    {
        public int prediction;
        public bool hit;
    }
}
```

## Using Models in Custom Applications

To load and use the models in your own Python application:

```python
import pickle

# Load the model
with open('path/to/models/ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define your features
features = [20.5, 1.5, 70.3, 150.4, 10.2, 1.4, 18.3]  # Example values

# Standardize features
# ... (standardization code)

# Make prediction
if isinstance(model, list):  # Ensemble model
    votes = []
    for m in model:
        pred = m.predict([features])[0]
        votes.append(pred)
    prediction = max(set(votes), key=votes.count)
else:
    prediction = model.predict([features])[0]

print("Hit" if prediction == 1 else "Miss")
```

## Advanced Usage: Model Retraining

If you want to retrain the models on new data:

1. Prepare your data in the same format as `SniperDataset.csv`
2. Run the training script:

```bash
cd NoScope9000-ML-Analysis
python src/RandomForest.py
```

This will retrain all models and save the new models to the `models/` directory.
