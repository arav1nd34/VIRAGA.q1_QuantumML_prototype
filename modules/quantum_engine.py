import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import time

# Qiskit imports
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

def run_quantum_pipeline(n_qubits=4, random_state=42):
    """
    Runs the Qiskit VQC pipeline with PCA reduction matching the qubit count.
    Saves predictions and metrics to CSV files.
    """
    algorithm_globals.random_seed = random_state
    
    print(f"[*] Initializing Quantum Pipeline with {n_qubits} qubits...")

    # 1. Load Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    # Invert target so 1 = malignant (disease positive), matching classical pipeline fix
    y = 1 - y 

    # 2. Train-Test Split (Preventing data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # 3. Standardization
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. PCA Dimensionality Reduction (to match qubit count)
    pca = PCA(n_components=n_qubits, random_state=random_state)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    
    variance_retained = pca.explained_variance_ratio_.sum()
    print(f"[+] PCA Complete. Retained variance: {variance_retained:.4f}")

    # 5. Quantum Circuit Construction (Feature Map + Ansatz)
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='full')
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2)
    optimizer = COBYLA(maxiter=50)  # Fast convergence for hackathon runtime

    # 6. Variational Quantum Classifier (VQC)
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        num_qubits=n_qubits
    )

    print("[*] Training VQC model (this may take a moment)...")
    start_time = time.time()
    vqc.fit(X_train_p, y_train)
    train_time = time.time() - start_time
    print(f"[+] Training finished in {train_time:.2f} seconds.")

    # 7. Predictions & Probabilities
    preds = vqc.predict(X_test_p)
    # VQC outputs one-hot or class probabilities depending on setup; handling probabilities safely:
    raw_probs = vqc.predict_proba(X_test_p)
    probs_malignant = raw_probs[:, 1] if raw_probs.shape[1] > 1 else raw_probs[:, 0]

    # 8. Calculate Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs_malignant)

    metrics_dict = {
        "model": ["VQC_Quantum"],
        "accuracy": [acc],
        "precision": [prec],
        "recall": [rec],
        "f1_score": [f1],
        "auc": [auc],
        "train_time_sec": [train_time]
    }

    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv("metrics_quantum.csv", index=False)

    # 9. Save Predictions Output
    df_preds = pd.DataFrame({
        "model": "VQC_Quantum",
        "true_label": y_test,
        "predicted_label": preds,
        "predicted_probability_malignant": probs_malignant
    })
    df_preds.to_csv("predictions_quantum.csv", index=False)

    print("[+] Quantum pipeline outputs saved successfully to 'metrics_quantum.csv' and 'predictions_quantum.csv'!")
    return df_metrics, df_preds

if __name__ == "__main__":
    run_quantum_pipeline()
