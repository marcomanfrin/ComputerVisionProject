# Computer Vision Project

> Exam project for the Computer Vision course — full pipeline from data acquisition to evaluation.

---

## Overview

This project develops a complete Computer Vision application addressing a real-world problem.  
The pipeline covers image acquisition and preprocessing, feature engineering, model training (both classical and deep learning), post-processing, and quantitative evaluation.

---

## Project Structure

```
ComputerVisionProject/
├── src/                  # Source code (modules)
├── data/
│   ├── raw/              # Original, unmodified data
│   ├── processed/        # Preprocessed / augmented data
│   └── annotations/      # Labels and annotation files
├── results/
│   ├── models/           # Saved model checkpoints
│   ├── plots/            # Training curves, confusion matrices, etc.
│   └── metrics/          # JSON / CSV evaluation reports
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt
└── environment.yml
```

---

## Setup

### Option A — pip + venv

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate cv-project
```

### Requirements

| Package | Version |
|---|---|
| Python | 3.11+ |
| PyTorch | 2.1+ |
| OpenCV | 4.8+ |
| scikit-learn | 1.3+ |
| NumPy / Pandas / Matplotlib | latest stable |

---

## Pipeline Architecture

```
Raw Images
    │
    ▼
┌─────────────────────────┐
│  1. Preprocessing        │  Noise reduction, normalization, augmentation
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  2. Feature Extraction   │  Handcrafted (HOG/SIFT) or learned (CNN/ViT backbone)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  3. Core Model           │  Classification / Detection / Segmentation
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  4. Post-processing      │  NMS, morphological ops, confidence thresholding
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  5. Evaluation           │  Metrics, plots, error analysis
└─────────────────────────┘
```

---

## Running the Project

```bash
# Preprocess data
python src/preprocess.py --input data/raw --output data/processed

# Train the model
python src/train.py --config configs/default.yaml

# Evaluate
python src/evaluate.py --checkpoint results/models/best.pth
```

> If using Google Colab: *(link to notebook will be added here)*

---

## Results

| Metric | Value |
|---|---|
| — | — |

> Results will be populated after training. Full analysis is available in the [Technical Analysis Document](results/report.pdf).

---

## Evaluation Metrics

Metrics are chosen according to the task type:

- **Classification** — Accuracy, Precision, Recall, F1-score, Confusion Matrix  
- **Detection** — mAP, IoU  
- **Segmentation** — mIoU, Dice Coefficient  
- **Generation** — FID, Inception Score

---

## Technical Analysis Document

A PDF report (max 10 pages) is included in `results/` and covers:

1. Problem statement and motivation  
2. Methodology — algorithms and architecture choices  
3. Experimental results — tables and plots  
4. Error analysis — failure cases and root causes  
5. Ethical considerations — bias and privacy

---

## License

This project is for academic purposes only.
