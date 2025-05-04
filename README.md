# Diabetic_retinopathy_Detection
Building a Machine Learning Model for Predicting Diabetic Retinopathy

# Diabetic Retinopathy Detection

This repository contains a complete pipeline for detecting diabetic retinopathy from retinal images using deep learning.

## 📁 Project Structure

```
diabetic_retinopathy_detection/
├── data/
│   ├── raw/                   # Dataset files
│   │   ├── train.zip.001
│   │   ├── train.zip.002
│   │   ├── ... (all zip parts)
│   │   └── trainLabels.csv.zip
│   └── processed/             # Processed data
│       ├── train/             # Extracted training images
│       └── test/              # Extracted test images
├── src/
│   ├── data_preparation.py    # Data preprocessing scripts
│   ├── model.py               # Model architecture
│   ├── train.py               # Model training
│   └── predict.py             # Inference script
├── outputs/
│   ├── models/                # Saved model checkpoints
│   ├── logs/                  # Training logs
│   └── submissions/           # Prediction outputs
├── config.py                  # Configuration file
└── requirements.txt           # Python dependencies
```

## 🖥️ Windows Directory Setup

If you're on Windows, you can create the necessary directory structure by running the following in Command Prompt or Git Bash:

```bash
mkdir -p diabetic_retinopathy_detection/data/{raw,processed/{train,test}}
mkdir -p diabetic_retinopathy_detection/src
mkdir -p diabetic_retinopathy_detection/outputs/{models,logs,submissions}
```

## 📦 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install tensorflow==2.12.0
pip install pandas==2.0.3
pip install opencv-python-headless==4.7.0.72
pip install albumentations==1.3.0
pip install scikit-learn==1.3.0
pip install tqdm==4.66.1
pip install matplotlib==3.7.1
```
