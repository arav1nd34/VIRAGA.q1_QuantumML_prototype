"""
Classical baseline models on PCA-REDUCED features.
Label direction: 1 = malignant (disease present), 0 = benign (healthy).
(sklearn's built-in dataset ships the opposite way round — flipped here
to match standard disease-detection convention: positive class = disease.)

Trains on the SAME reduced feature space your quantum circuit will use
(n_components = n_qubits) — the fair comparison point for QSVM/VQC later.

Outputs two CSV files:
  - predictions_with_pca.csv : per-sample true label, predicted label, predicted probability, model
  - metrics_with_pca.csv     : accuracy, precision, recall, f1, auc, train_time per model
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import pandas as pd
import numpy as np
import time

N_COMPONENTS = 4  # match this to your quantum circuit's qubit count

# ---- Load ----
data = load_breast_cancer()
X = data.data
y_raw = data.target  # sklearn: 0 = malignant, 1 = benign
y = 1 - y_raw         # flipped: 1 = malignant (disease), 0 = benign (healthy)
print("Label direction: 1 = malignant (disease), 0 = benign (healthy)")

# ---- Split (before scaling/PCA, to avoid leakage) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---- Scale ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---- PCA reduction ----
pca = PCA(n_components=N_COMPONENTS)
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)
np.save("X_train_pca.npy", X_train_p)
np.save("X_test_pca.npy", X_test_p)
variance_retained = pca.explained_variance_ratio_.sum()
print(f"Components: {N_COMPONENTS} | Variance retained: {variance_retained:.4f}")
if variance_retained < 0.80:
    print("WARNING: <80% variance retained — consider raising N_COMPONENTS.")

# ---- Train + evaluate ----
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="rbf", probability=True),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
}

prediction_rows = []
metric_rows = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train_p, y_train)
    train_time = time.time() - start

    preds = model.predict(X_test_p)
    probs = model.predict_proba(X_test_p)[:, 1]  # probability of class 1 = malignant

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\n=== {name} (PCA={N_COMPONENTS}) ===")
    print(f"Train time: {train_time:.3f}s | Acc: {acc:.4f} | AUC: {auc:.4f}")

    metric_rows.append({
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1_score": f1, "auc": auc, "train_time_sec": train_time,
        "pca_components": N_COMPONENTS, "variance_retained": variance_retained
    })

    for true_label, pred_label, prob in zip(y_test, preds, probs):
        prediction_rows.append({
            "model": name,
            "true_label": true_label,      # 1 = malignant, 0 = benign
            "predicted_label": pred_label,
            "predicted_probability_malignant": prob
        })

# ---- Save CSVs ----
pd.DataFrame(prediction_rows).to_csv("predictions_with_pca.csv", index=False)
pd.DataFrame(metric_rows).to_csv("metrics_with_pca.csv", index=False)
print("\nSaved: predictions_with_pca.csv, metrics_with_pca.csv")
