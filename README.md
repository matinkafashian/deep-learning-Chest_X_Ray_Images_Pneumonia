# 🩺 Chest X-Ray Pneumonia Detection using CNN (Keras + TensorFlow)

This project implements a deep learning pipeline to automatically detect pneumonia from chest X-ray images using a custom-built Convolutional Neural Network (CNN) with ResNet-style blocks.

## 📂 Dataset

- **Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Classes:
  - `NORMAL`: Healthy lungs
  - `PNEUMONIA`: Infected lungs (bacterial or viral)

## 🧠 Model Architecture

- Custom CNN built with:
  - `Conv2D`, `BatchNormalization`, `ReLU`, `Dropout`
  - ResNet-like residual blocks with skip connections
- Input shape: `(224, 224, 1)` — grayscale X-ray images
- Final activation: `sigmoid` (binary classification)
- Optimizer: Adam with `learning_rate=1e-4`
- Loss function: `binary_crossentropy`

## 📈 Training & Validation

- Training with data generators (`ImageDataGenerator`) with rescaling
- Validation split: 10%
- Callbacks:
  - `EarlyStopping`: to prevent overfitting
  - `ModelCheckpoint`: to save the best model based on validation accuracy

## ✅ Results

| Metric          | Value     |
|----------------|-----------|
| Train Accuracy | ~97%      |
| Val Accuracy   | ~94%      |
| Test Accuracy  | ~90%+     |

_(Results may vary slightly depending on run conditions and random seed.)_

## 🖼 Sample Predictions

> *[Optional section: You can add matplotlib images or Grad-CAM heatmaps here]*

## 🚀 Getting Started

### 🔧 Requirements
```bash
tensorflow
numpy
matplotlib
