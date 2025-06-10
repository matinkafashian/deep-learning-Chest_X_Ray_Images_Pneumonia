
# Pneumonia Detection using Deep Learning

This repository contains a Deep Learning project for detecting Pneumonia from Chest X-Ray images using the **Chest_X_Ray_Images (Pneumonia)** dataset. The project leverages **EfficientNetB0** for feature extraction and includes data augmentation techniques to improve model performance, especially with limited dataset sizes.

## Project Overview
- **Objective**: Build a binary classification model to detect Pneumonia (Normal vs. Pneumonia) from X-Ray images.
- **Techniques**: Feature Extraction with EfficientNetB0, Data Augmentation, ImageDataGenerator.
- **Dataset**: Chest_X_Ray_Images (Pneumonia) dataset with approximately 50 images per split (train, validation, test).
- **Tools**: TensorFlow, Keras, Python.

## Features
- Implementation of EfficientNetB0 for feature extraction with a custom top layer.
- Data augmentation to generate additional images and address dataset size limitations.
- Train, validation, and test data processing with optimized batch sizes.
- Model saving and evaluation metrics.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pneumonia-detection.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset is placed in the correct directory (e.g., `/path/to/Chest_X_Ray_Images`).

## Usage
1. Run the data augmentation script to generate additional images:
   ```bash
   python augment_data.py
   ```
2. Train the model using the main script:
   ```bash
   python train_model.py
   ```
3. Evaluate the model or make predictions with the saved model.

## Files
- `augment_data.py`: Script to generate augmented images.
- `train_model.py`: Main script for training the EfficientNetB0 model.
- `requirements.txt`: List of required Python packages.
- `pneumonia_resnet_model.h5`: Saved model file (after training).

## Results
- Initial accuracy may vary due to limited data; augmentation improves performance.
- Further tuning (e.g., Fine-tuning ResNet layers) can be explored for better results.

## Contributing
Feel free to fork this repository, submit issues, or create pull requests. Suggestions for improving model accuracy or adding new features are welcome!

## Contact
For questions or collaboration, reach out at [kafashianmatin@gmail.com]
