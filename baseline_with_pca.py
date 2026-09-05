from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import pandas as pd
import numpy as np
import time
import joblib
import json

N_SELECTED_FEATURES = 4

data = load_breast_cancer()
X = data.data
y_raw = data.target
y = 1 - y_raw
print("Label direction: 1 = malignant (disease), 0 = benign (healthy)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

selector = RandomForestClassifier(n_estimators=200, random_state=42)
selector.fit(X_train_s, y_train)
importances = selector.feature_importances_
top_indices = np.argsort(importances)[::-1][:N_SELECTED_FEATURES]
selected_feature_names = [data.feature_names[i] for i in top_indices]
print(f"Selected top {N_SELECTED_FEATURES} features: {selected_feature_names}")

X_train_sel = X_train_s[:, top_indices]
X_test_sel = X_test_s[:, top_indices]

np.save("X_train_selected.npy", X_train_sel)
np.save("X_test_selected.npy", X_test_sel)
with open("selected_features.json", "w") as f:
    json.dump({"indices": top_indices.tolist(), "names": selected_feature_names}, f)

models = {
    "LogReg": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

prediction_rows = []
metric_rows = []
for name, model in models.items():
    start = time.time()
    model.fit(X_train_sel, y_train)
    train_time = time.time() - start

    joblib.dump(model, f"model_{name}.pkl")

    preds = model.predict(X_test_sel)
    probs = model.predict_proba(X_test_sel)[:, 1]
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"\n=== {name} (top {N_SELECTED_FEATURES} features) ===")
    print(f"Train time: {train_time:.3f}s | Acc: {acc:.4f} | AUC: {auc:.4f}")
    metric_rows.append({
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1_score": f1, "auc": auc, "train_time_sec": train_time,
        "n_selected_features": N_SELECTED_FEATURES
    })
    for true_label, pred_label, prob in zip(y_test, preds, probs):
        prediction_rows.append({
            "model": name,
            "true_label": true_label,
            "predicted_label": pred_label,
            "predicted_probability_malignant": prob
        })

pd.DataFrame(prediction_rows).to_csv("predictions_with_pca.csv", index=False)
pd.DataFrame(metric_rows).to_csv("metrics_with_pca.csv", index=False)
print("\nSaved: predictions_with_pca.csv, metrics_with_pca.csv, model_<name>.pkl, selected_features.json")
