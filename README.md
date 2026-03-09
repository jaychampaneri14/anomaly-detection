# Anomaly Detection System

Multi-method anomaly detection comparing Isolation Forest, Autoencoder, LOF, and One-Class SVM.

## Methods
- **Isolation Forest** — tree-based partitioning
- **Local Outlier Factor (LOF)** — density-based
- **One-Class SVM** — boundary-based
- **Autoencoder** — deep learning reconstruction error

## Features
- Trained on normal data only (unsupervised)
- Synthetic multivariate data with point, contextual, and collective anomalies
- AUC-ROC and AUC-PR comparison across all methods
- Score distribution visualization
- PCA 2D anomaly projection

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output
- `anomaly_scores.png` — PCA visualization with anomaly scores
- `score_distributions.png` — score histograms per method
