# Diabetic Retinopathy Detection

This repository contains a complete pipeline for detecting diabetic retinopathy from retinal images using machine learning.

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
│   ├── config.py              # Configuration file
│   ├── model.py               # Model architecture
│   ├── train.py               # Model training
│   └── predict.py             # Inference script
├── outputs/
│   ├── models/                # Saved model checkpoints
│   ├── logs/                  # Training logs
│   └── submissions/           # Prediction outputs
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

## 💻 3. Command-Line Workflow (This one is just for Windows, idk if you use Windows or nah)

### A. Open Command Prompt

```cmd
cd C:\DIABETIC_RETINOPATHY
```

### B. Set Up Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### C. Prepare Data

Place all ZIP files in `data/raw/`, then run:

```cmd
python src/data_preparation.py
```

### D. Start Training

```cmd
python src/train.py
```

### E. Evaluate Model

```cmd
python src/evaluate.py
```

### F. Generate Predictions

```cmd
python src/predict.py
```

## 🛠️  Issues I ran into and how to fix

| **Issue**              | **Solution**                                                                 |
|------------------------|------------------------------------------------------------------------------|
| CUDA Out of Memory     | Reduce `BATCH_SIZE` in `config.py` (e.g., set to 4 or 8)                      |
| Missing ZIP Parts      | Ensure all required ZIP parts are present in `data/raw/`                     |
| DLL Load Failed        | Install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| Slow Training          | Disable GPU by setting `CUDA_VISIBLE_DEVICES=-1`                             |
| File Path Errors       | Use absolute file paths in `config.py`                                       |
| Corrupt ZIP Files      | Re-download the affected ZIP parts                                            |
