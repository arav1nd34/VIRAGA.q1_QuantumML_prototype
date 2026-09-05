import time
import os
import json

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

N_QUBITS = 4
N_LAYERS = 4
FEATURE_MAP_REPS = 1
MAX_ITER = 600
RANDOM_STATE = 42
SKIP_TRAINING_IF_SAVED = True
WEIGHTS_FILE = "trained_vqc_4q.npz"


def load_and_split_data():
    data = load_breast_cancer()
    X = data.data
    y = 1 - data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def preprocess_for_quantum(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # loads the same top-N features classical_engine.py already selected via
    # Random Forest importance, so both pipelines compare on identical inputs
    with open("selected_features.json") as f:
        selected = json.load(f)
    top_indices = selected["indices"]
    print(f"Using pre-selected features: {selected['names']}")

    X_train_sel = X_train_scaled[:, top_indices]
    X_test_sel = X_test_scaled[:, top_indices]

    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_angles = angle_scaler.fit_transform(X_train_sel)
    X_test_angles = angle_scaler.transform(X_test_sel)
    X_test_angles = np.clip(X_test_angles, 0, np.pi)

    return X_train_angles, X_test_angles


def add_feature_map(circuit, features):
    for _ in range(FEATURE_MAP_REPS):
        for q in range(N_QUBITS):
            circuit.h(q)
        for q in range(N_QUBITS):
            circuit.rz(2 * features[q], q)
        for q in range(N_QUBITS):
            j = (q + 1) % N_QUBITS
            pair_angle = 2 * (np.pi - features[q]) * (np.pi - features[j])
            circuit.cx(q, j)
            circuit.rz(pair_angle, j)
            circuit.cx(q, j)


def add_variational_layers(circuit, weights):
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            circuit.ry(weights[layer, q], q)
        for q in range(N_QUBITS):
            circuit.cx(q, (q + 1) % N_QUBITS)


def build_circuit(features, weights):
    circuit = QuantumCircuit(N_QUBITS)
    add_feature_map(circuit, features)
    add_variational_layers(circuit, weights)
    return circuit


READOUT_OPERATOR = SparsePauliOp("I" * (N_QUBITS - 1) + "Z")


def measure_expectation(circuit):
    statevector = Statevector.from_instruction(circuit)
    return statevector.expectation_value(READOUT_OPERATOR).real


def predict_probability(features, weights, bias):
    circuit = build_circuit(features, weights)
    expectation = measure_expectation(circuit)
    return 1 / (1 + np.exp(-(expectation + bias)))


def unpack_params(flat_params):
    weights = flat_params[:-1].reshape(N_LAYERS, N_QUBITS)
    bias = flat_params[-1]
    return weights, bias


def compute_loss(flat_params, X, y):
    # class-weighted: malignant cases are rarer (~37%) than benign (~63%),
    # unweighted loss collapses recall toward zero without this
    weights, bias = unpack_params(flat_params)
    malignant_weight = 0.5 / max(y.mean(), 1e-6)
    benign_weight = 0.5 / max(1 - y.mean(), 1e-6)

    total_loss = 0.0
    for features, label in zip(X, y):
        p = predict_probability(features, weights, bias)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        sample_weight = malignant_weight if label == 1 else benign_weight
        total_loss += -sample_weight * (label * np.log(p) + (1 - label) * np.log(1 - p))
    return total_loss / len(X)


def train_qml_model(X_train, y_train):
    if SKIP_TRAINING_IF_SAVED and os.path.exists(WEIGHTS_FILE):
        print(f"Found {WEIGHTS_FILE} — loading saved weights instead of retraining.")
        saved = np.load(WEIGHTS_FILE)
        return saved["weights"], float(saved["bias"]), 0.0

    rng = np.random.default_rng(RANDOM_STATE)
    n_weights = N_LAYERS * N_QUBITS
    initial_weights = rng.uniform(0, 2 * np.pi, n_weights)
    initial_bias = np.array([0.0])
    initial_params = np.concatenate([initial_weights, initial_bias])

    initial_loss = compute_loss(initial_params, X_train, y_train)
    print(f"Initial loss (before any training): {initial_loss:.4f}")
    print(f"Parameters to optimize: {len(initial_params)} "
          f"(COBYLA needs ~{len(initial_params) + 1} evals just to build its simplex)")

    start_time = time.time()
    result = minimize(
        compute_loss, initial_params, args=(X_train, y_train),
        method="COBYLA", options={"maxiter": MAX_ITER},
    )
    train_time = time.time() - start_time

    print(f"Final loss (after {result.nfev} evaluations, max {MAX_ITER} iterations): {result.fun:.4f}")
    print(f"COBYLA stop reason: {result.message}")
    if result.fun >= initial_loss * 0.98:
        print("WARNING: final loss barely improved on initial loss — "
              "optimizer likely got stuck (flat landscape / barren plateau).")

    weights, bias = unpack_params(result.x)

    np.savez(WEIGHTS_FILE, weights=weights, bias=bias)
    print(f"Saved trained weights/bias to {WEIGHTS_FILE}")

    return weights, bias, train_time


def find_best_threshold(y_true, probabilities, min_precision=0.0):
    # sweep candidate thresholds on TRAINING data only — fitting this against
    # test labels would be leakage, same as fitting model weights on test data
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold, best_f1 = 0.5, -1.0

    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        if prec < min_precision:
            continue
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    return best_threshold, best_f1


def evaluate_on_test_set(weights, bias, X_train, y_train, X_test, y_test):
    train_probabilities = np.array(
        [predict_probability(features, weights, bias) for features in X_train]
    )
    best_threshold, train_f1 = find_best_threshold(y_train, train_probabilities)
    print(f"\nThreshold selected on training data: {best_threshold:.2f} "
          f"(training F1 at this threshold: {train_f1:.3f})")

    probabilities = []
    for features in X_test:
        probabilities.append(predict_probability(features, weights, bias))
    probabilities = np.array(probabilities)
    predictions = (probabilities >= best_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "auc": roc_auc_score(y_test, probabilities),
    }

    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(cm)
    print(f"Predicted-label distribution: {np.bincount(predictions)}")
    print(f"True-label distribution:      {np.bincount(y_test)}")

    return predictions, probabilities, metrics


def save_results(y_test, predictions, probabilities, metrics, train_time):
    predictions_df = pd.DataFrame({
        "model": "VQC_Quantum_4q_fixed",
        "true_label": y_test,
        "predicted_label": predictions,
        "predicted_probability_malignant": probabilities,
    })
    predictions_df.to_csv("predictions_quantum_4q_fixed.csv", index=False)

    metrics_df = pd.DataFrame([{
        "model": "VQC_Quantum_4q_fixed",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "auc": metrics["auc"],
        "train_time_sec": train_time,
    }])
    metrics_df.to_csv("metrics_quantum_4q_fixed.csv", index=False)

    print("\nSaved predictions_quantum_4q_fixed.csv and metrics_quantum_4q_fixed.csv")


def main():
    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_and_split_data()

    print("Preprocessing (scale + feature selection + angle-scaling)...")
    X_train_angles, X_test_angles = preprocess_for_quantum(X_train, X_test)

    print(f"Training the {N_QUBITS}-qubit quantum circuit (nearest-neighbor entanglement, "
          f"max {MAX_ITER} iterations)...")
    weights, bias, train_time = train_qml_model(X_train_angles, y_train)

    print("\nEvaluating on the test set...")
    predictions, probabilities, metrics = evaluate_on_test_set(
        weights, bias, X_train_angles, y_train, X_test_angles, y_test
    )

    print(f"\nAccuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 score:  {metrics['f1_score']:.3f}")
    print(f"AUC:       {metrics['auc']:.3f}")
    print(f"Train time: {train_time:.1f}s")

    save_results(y_test, predictions, probabilities, metrics, train_time)


if __name__ == "__main__":
    main()    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # loads the same top-N features classical_engine.py already selected via
    # Random Forest importance, so both pipelines compare on identical inputs
    with open("selected_features.json") as f:
        selected = json.load(f)
    top_indices = selected["indices"]
    print(f"Using pre-selected features: {selected['names']}")

    X_train_sel = X_train_scaled[:, top_indices]
    X_test_sel = X_test_scaled[:, top_indices]

    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_angles = angle_scaler.fit_transform(X_train_sel)
    X_test_angles = angle_scaler.transform(X_test_sel)
    X_test_angles = np.clip(X_test_angles, 0, np.pi)

    return X_train_angles, X_test_angles


def add_feature_map(circuit, features):
    for _ in range(FEATURE_MAP_REPS):
        for q in range(N_QUBITS):
            circuit.h(q)
        for q in range(N_QUBITS):
            circuit.rz(2 * features[q], q)
        for q in range(N_QUBITS):
            j = (q + 1) % N_QUBITS
            pair_angle = 2 * (np.pi - features[q]) * (np.pi - features[j])
            circuit.cx(q, j)
            circuit.rz(pair_angle, j)
            circuit.cx(q, j)


def add_variational_layers(circuit, weights):
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            circuit.ry(weights[layer, q], q)
        for q in range(N_QUBITS):
            circuit.cx(q, (q + 1) % N_QUBITS)


def build_circuit(features, weights):
    circuit = QuantumCircuit(N_QUBITS)
    add_feature_map(circuit, features)
    add_variational_layers(circuit, weights)
    return circuit


READOUT_OPERATOR = SparsePauliOp("I" * (N_QUBITS - 1) + "Z")


def measure_expectation(circuit):
    statevector = Statevector.from_instruction(circuit)
    return statevector.expectation_value(READOUT_OPERATOR).real


def predict_probability(features, weights, bias):
    circuit = build_circuit(features, weights)
    expectation = measure_expectation(circuit)
    return 1 / (1 + np.exp(-(expectation + bias)))


def unpack_params(flat_params):
    weights = flat_params[:-1].reshape(N_LAYERS, N_QUBITS)
    bias = flat_params[-1]
    return weights, bias


def compute_loss(flat_params, X, y):
    # class-weighted: malignant cases are rarer (~37%) than benign (~63%),
    # unweighted loss collapses recall toward zero without this
    weights, bias = unpack_params(flat_params)
    malignant_weight = 0.5 / max(y.mean(), 1e-6)
    benign_weight = 0.5 / max(1 - y.mean(), 1e-6)

    total_loss = 0.0
    for features, label in zip(X, y):
        p = predict_probability(features, weights, bias)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        sample_weight = malignant_weight if label == 1 else benign_weight
        total_loss += -sample_weight * (label * np.log(p) + (1 - label) * np.log(1 - p))
    return total_loss / len(X)


def train_qml_model(X_train, y_train):
    if SKIP_TRAINING_IF_SAVED and os.path.exists(WEIGHTS_FILE):
        print(f"Found {WEIGHTS_FILE} — loading saved weights instead of retraining.")
        saved = np.load(WEIGHTS_FILE)
        return saved["weights"], float(saved["bias"]), 0.0

    rng = np.random.default_rng(RANDOM_STATE)
    n_weights = N_LAYERS * N_QUBITS
    initial_weights = rng.uniform(0, 2 * np.pi, n_weights)
    initial_bias = np.array([0.0])
    initial_params = np.concatenate([initial_weights, initial_bias])

    initial_loss = compute_loss(initial_params, X_train, y_train)
    print(f"Initial loss (before any training): {initial_loss:.4f}")
    print(f"Parameters to optimize: {len(initial_params)} "
          f"(COBYLA needs ~{len(initial_params) + 1} evals just to build its simplex)")

    start_time = time.time()
    result = minimize(
        compute_loss, initial_params, args=(X_train, y_train),
        method="COBYLA", options={"maxiter": MAX_ITER},
    )
    train_time = time.time() - start_time

    print(f"Final loss (after {result.nfev} evaluations, max {MAX_ITER} iterations): {result.fun:.4f}")
    print(f"COBYLA stop reason: {result.message}")
    if result.fun >= initial_loss * 0.98:
        print("WARNING: final loss barely improved on initial loss — "
              "optimizer likely got stuck (flat landscape / barren plateau).")

    weights, bias = unpack_params(result.x)

    np.savez(WEIGHTS_FILE, weights=weights, bias=bias)
    print(f"Saved trained weights/bias to {WEIGHTS_FILE}")

    return weights, bias, train_time


def find_best_threshold(y_true, probabilities, min_precision=0.0):
    # sweep candidate thresholds on TRAINING data only — fitting this against
    # test labels would be leakage, same as fitting model weights on test data
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold, best_f1 = 0.5, -1.0

    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        if prec < min_precision:
            continue
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    return best_threshold, best_f1


def evaluate_on_test_set(weights, bias, X_train, y_train, X_test, y_test):
    train_probabilities = np.array(
        [predict_probability(features, weights, bias) for features in X_train]
    )
    best_threshold, train_f1 = find_best_threshold(y_train, train_probabilities)
    print(f"\nThreshold selected on training data: {best_threshold:.2f} "
          f"(training F1 at this threshold: {train_f1:.3f})")

    probabilities = []
    for features in X_test:
        probabilities.append(predict_probability(features, weights, bias))
    probabilities = np.array(probabilities)
    predictions = (probabilities >= best_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "auc": roc_auc_score(y_test, probabilities),
    }

    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(cm)
    print(f"Predicted-label distribution: {np.bincount(predictions)}")
    print(f"True-label distribution:      {np.bincount(y_test)}")

    return predictions, probabilities, metrics


def save_results(y_test, predictions, probabilities, metrics, train_time):
    predictions_df = pd.DataFrame({
        "model": "VQC_Quantum_4q_fixed",
        "true_label": y_test,
        "predicted_label": predictions,
        "predicted_probability_malignant": probabilities,
    })
    predictions_df.to_csv("predictions_quantum_4q_fixed.csv", index=False)

    metrics_df = pd.DataFrame([{
        "model": "VQC_Quantum_4q_fixed",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "auc": metrics["auc"],
        "train_time_sec": train_time,
    }])
    metrics_df.to_csv("metrics_quantum_4q_fixed.csv", index=False)

    print("\nSaved predictions_quantum_4q_fixed.csv and metrics_quantum_4q_fixed.csv")


def main():
    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_and_split_data()

    print("Preprocessing (scale + feature selection + angle-scaling)...")
    X_train_angles, X_test_angles = preprocess_for_quantum(X_train, X_test)

    print(f"Training the {N_QUBITS}-qubit quantum circuit (nearest-neighbor entanglement, "
          f"max {MAX_ITER} iterations)...")
    weights, bias, train_time = train_qml_model(X_train_angles, y_train)

    print("\nEvaluating on the test set...")
    predictions, probabilities, metrics = evaluate_on_test_set(
        weights, bias, X_train_angles, y_train, X_test_angles, y_test
    )

    print(f"\nAccuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 score:  {metrics['f1_score']:.3f}")
    print(f"AUC:       {metrics['auc']:.3f}")
    print(f"Train time: {train_time:.1f}s")

    save_results(y_test, predictions, probabilities, metrics, train_time)


if __name__ == "__main__":
    main()
