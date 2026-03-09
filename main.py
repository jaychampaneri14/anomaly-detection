"""
Anomaly Detection System
Multi-method anomaly detection: Isolation Forest, Autoencoder, LOF, One-Class SVM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


def generate_multivariate_anomaly_data(n_normal=2000, n_anomaly=100, n_features=10, seed=42):
    """Generate multivariate time-series data with injected anomalies."""
    np.random.seed(seed)
    # Normal: correlated multivariate Gaussian
    cov_base = np.random.randn(n_features, n_features)
    cov = cov_base @ cov_base.T + np.eye(n_features) * 0.5
    X_normal = np.random.multivariate_normal(np.zeros(n_features), cov, n_normal)

    # Anomalies: various types
    n_each = n_anomaly // 4
    # Point anomalies
    X_point = np.random.multivariate_normal(np.ones(n_features) * 5, np.eye(n_features) * 2, n_each)
    # Contextual anomalies
    X_ctx   = np.random.multivariate_normal(np.zeros(n_features), np.eye(n_features) * 0.1, n_each)
    X_ctx[:, 0] += 8
    # Collective anomalies
    X_coll  = np.random.randn(n_each, n_features) * 0.3
    X_coll += np.random.choice([-6, 6], size=(n_each, n_features))
    # Near-boundary anomalies
    X_near  = np.random.randn(n_anomaly - 3*n_each, n_features) * 3

    X_anomaly = np.vstack([X_point, X_ctx, X_coll, X_near])
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * len(X_anomaly))
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


class Autoencoder(nn.Module):
    """Deep autoencoder for anomaly detection via reconstruction error."""
    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        with torch.no_grad():
            recon = self(x)
            return ((x - recon) ** 2).mean(dim=1).numpy()


def train_autoencoder(X_normal_scaled, epochs=50, latent_dim=4):
    """Train autoencoder on normal data only."""
    X_t = torch.FloatTensor(X_normal_scaled)
    loader = DataLoader(TensorDataset(X_t), batch_size=64, shuffle=True)
    model = Autoencoder(X_normal_scaled.shape[1], latent_dim)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (batch,) in loader:
            opt.zero_grad()
            recon = model(batch)
            loss  = nn.MSELoss()(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Autoencoder Epoch {epoch+1}/{epochs}: Loss={total_loss/len(loader):.6f}")
    return model


def run_all_detectors(X_train, X_test, y_test, scaler):
    """Run all anomaly detectors and compare."""
    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)
    results = {}

    # Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X_tr_s)
    scores_iso = -iso.score_samples(X_te_s)
    results['IsolationForest'] = scores_iso

    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
    lof.fit(X_tr_s)
    scores_lof = -lof.score_samples(X_te_s)
    results['LOF'] = scores_lof

    # One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    ocsvm.fit(X_tr_s)
    scores_svm = -ocsvm.score_samples(X_te_s)
    results['OneClassSVM'] = scores_svm

    # Autoencoder
    print("\n--- Training Autoencoder ---")
    ae = train_autoencoder(X_tr_s, epochs=50)
    ae.eval()
    scores_ae = ae.reconstruction_error(torch.FloatTensor(X_te_s))
    results['Autoencoder'] = scores_ae

    # Evaluate
    print("\n--- Detection Performance (AUC-ROC / AUC-PR) ---")
    for name, scores in results.items():
        auc_roc = roc_auc_score(y_test, scores)
        auc_pr  = average_precision_score(y_test, scores)
        print(f"  {name:20s}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")

    return results, iso, ae, scaler


def plot_anomaly_scores(X_test, y_test, scores, name='IsolationForest', save_path='anomaly_scores.png'):
    """2D PCA projection colored by anomaly score."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_test)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sc = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=scores, cmap='RdYlGn_r', s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax1, label='Anomaly Score')
    ax1.set_title(f'Anomaly Scores — {name}')
    ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
    colors = ['green' if l == 0 else 'red' for l in y_test]
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=20, alpha=0.7)
    ax2.set_title('True Labels (green=normal, red=anomaly)')
    ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Anomaly score plot saved to {save_path}")


def plot_score_distributions(results, y_test, save_path='score_distributions.png'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, scores) in zip(axes.ravel(), results.items()):
        scores_n = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        ax.hist(scores_n[y_test==0], bins=40, alpha=0.6, color='green', label='Normal', density=True)
        ax.hist(scores_n[y_test==1], bins=40, alpha=0.6, color='red',   label='Anomaly', density=True)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.set_xlabel('Normalized Score')
    plt.suptitle('Anomaly Score Distributions by Method')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Distribution plot saved to {save_path}")


def main():
    print("=" * 60)
    print("ANOMALY DETECTION SYSTEM")
    print("=" * 60)

    X, y = generate_multivariate_anomaly_data(2000, 100, n_features=10)
    print(f"Dataset: {len(X)} samples, {y.sum()} anomalies ({y.mean():.1%})")

    # Train on clean data only (unsupervised)
    X_normal = X[y == 0]
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train_normal  = X_train[y_train == 0]

    scaler = StandardScaler()
    scaler.fit(X_train_normal)

    results, iso_model, ae_model, _ = run_all_detectors(X_train_normal, X_test, y_test, scaler)

    # Best model threshold
    best_name   = max(results, key=lambda k: roc_auc_score(y_test, results[k]))
    best_scores = results[best_name]
    threshold   = np.percentile(best_scores, 95)
    y_pred      = (best_scores >= threshold).astype(int)
    print(f"\nBest detector: {best_name}")
    print(f"Classification Report (threshold={threshold:.4f}):")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

    plot_anomaly_scores(scaler.transform(X_test), y_test, best_scores, best_name)
    plot_score_distributions(results, y_test)

    print("\n✓ Anomaly Detection System complete!")


if __name__ == '__main__':
    main()
