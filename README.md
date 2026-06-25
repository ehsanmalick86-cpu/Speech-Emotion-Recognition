# Speech Emotion Recognition (SER) using Machine Learning

A Python-based Machine Learning project built to recognize human emotions from speech audio files using the RAVDESS dataset.

## 🚀 How It Works
1. **Feature Extraction:** Uses `librosa` to process raw `.wav` audio files and extract **MFCCs** (Mel-Frequency Cepstral Coefficients) which represent vocal frequencies.
2. **AI Model:** Trains a Multi-Layer Perceptron Neural Network (`MLPClassifier` via Scikit-Learn) to learn patterns behind human emotions.
3. **Classification:** Detects and classifies speech into categories like *Happy, Sad, Angry, and Neutral*.

## 🛠️ Requirements & Libraries
To run this project locally, install the following packages:
```bash
pip install librosa numpy scikit-learn
