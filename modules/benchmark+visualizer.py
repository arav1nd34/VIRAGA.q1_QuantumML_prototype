import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

CLASSICAL_FILE = "classical_results.csv"
QUANTUM_FILE = "quantum_results.csv"
OUTPUT_FOLDER = "benchmark_results"

METRICS_FILE = os.path.join(OUTPUT_FOLDER, "benchmark_metrics.csv")
SUMMARY_FILE = os.path.join(OUTPUT_FOLDER, "benchmark_summary.txt")
METRIC_PLOT_FILE = os.path.join(OUTPUT_FOLDER, "metric_comparison.png")
ROC_PLOT_FILE = os.path.join(OUTPUT_FOLDER, "roc_curve.png")


def load_results(file_path):
    if not os.path.exists(file_path):
        print("ERROR: File not found:", file_path)
        return None
    try:
        return pd.read_csv(file_path)
    except Exception as error:
        print("ERROR: Could not read CSV:", error)
        return None


def check_required_columns(data, file_name):
    required_columns = ["model", "y_true", "y_pred", "y_score"]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        print(f"ERROR: {file_name} is missing columns: {', '.join(missing)}")
        return False
    return True


def prepare_prediction_data(model_data):
    # Standardize data types safely
    y_true = np.asarray(model_data["y_true"], dtype=int)
    y_pred = np.asarray(model_data["y_pred"], dtype=int)
    y_score = np.asarray(model_data["y_score"], dtype=float)
    return y_true, y_pred, y_score


def calculate_confusion_values(y_true, y_pred):
    # Ensure matrix is strictly 2x2 even if class predictions are missing
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return tn, fp, fn, tp


def calculate_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = calculate_confusion_values(y_true, y_pred)
    denominator = tp + fn
    return tp / denominator if denominator > 0 else np.nan


def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = calculate_confusion_values(y_true, y_pred)
    denominator = tn + fp
    return tn / denominator if denominator > 0 else np.nan


def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def calculate_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan


def calculate_model_metrics(model_data, model_name, source):
    y_true, y_pred, y_score = prepare_prediction_data(model_data)

    return {
        "source": source,
        "model": model_name,
        "accuracy": calculate_accuracy(y_true, y_pred),
        "sensitivity": calculate_sensitivity(y_true, y_pred),
        "specificity": calculate_specificity(y_true, y_pred),
        "roc_auc": calculate_roc_auc(y_true, y_score)
    }


def calculate_all_metrics(data, source):
    results = []
    model_names = data["model"].dropna().unique()

    for model_name in model_names:
        model_data = data[data["model"] == model_name].copy()
        if len(model_data) == 0:
            continue
        results.append(calculate_model_metrics(model_data, model_name, source))

    return results


def create_metrics_dataframe(results):
    columns = ["source", "model", "accuracy", "sensitivity", "specificity", "roc_auc"]
    return pd.DataFrame(results, columns=columns)


def create_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_metrics(metrics_data):
    metrics_data.to_csv(METRICS_FILE, index=False)
    print("Saved metrics:", METRICS_FILE)


def print_metrics(metrics_data):
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    print(metrics_data.to_string(index=False))


def find_best_model(metrics_data, metric):
    valid_data = metrics_data.dropna(subset=[metric])
    if len(valid_data) == 0:
        return None
    best_index = valid_data[metric].idxmax()
    return valid_data.loc[best_index]


def create_summary(metrics_data):
    metrics = ["accuracy", "sensitivity", "specificity", "roc_auc"]
    lines = ["CLASSICAL ML VS QUANTUM ML BENCHMARK SUMMARY", "=" * 55]

    for metric in metrics:
        best = find_best_model(metrics_data, metric)
        if best is None:
            lines.append(f"{metric}: No valid result")
        else:
            lines.append(
                f"{metric.upper()}: {best['model']} ({best['source']}) = {best[metric]:.4f}"
            )

    return lines


def save_summary(lines):
    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")
    print("Saved summary:", SUMMARY_FILE)


def print_summary(lines):
    print("\n" + "=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


def create_metric_comparison_plot(metrics_data):
    plot_data = metrics_data.copy()
    plot_data["model_label"] = plot_data["source"] + " - " + plot_data["model"].astype(str)

    long_data = plot_data.melt(
        id_vars=["model_label"],
        value_vars=["accuracy", "sensitivity", "specificity"],
        var_name="metric",
        value_name="score"
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(data=long_data, x="model_label", y="score", hue="metric")
    plt.title("Accuracy, Sensitivity and Specificity")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(METRIC_PLOT_FILE, dpi=300)
    plt.close()
    print("Saved plot:", METRIC_PLOT_FILE)


def create_roc_curve_plot(classical_data, quantum_data):
    plt.figure(figsize=(9, 7))
    all_sources = [("Classical ML", classical_data), ("Quantum ML", quantum_data)]

    for source, data in all_sources:
        for model_name in data["model"].dropna().unique():
            model_data = data[data["model"] == model_name].copy()
            y_true, _, y_score = prepare_prediction_data(model_data)

            auc_value = calculate_roc_auc(y_true, y_score)
            if np.isnan(auc_value):
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            label = f"{source} - {model_name} (AUC={auc_value:.3f})"
            plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PLOT_FILE, dpi=300)
    plt.close()
    print("Saved plot:", ROC_PLOT_FILE)


def create_confusion_matrix_plots(classical_data, quantum_data):
    all_sources = [("Classical ML", classical_data), ("Quantum ML", quantum_data)]

    for source, data in all_sources:
        for model_name in data["model"].dropna().unique():
            model_data = data[data["model"] == model_name].copy()
            y_true, y_pred, _ = prepare_prediction_data(model_data)

            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            safe_source = source.replace(" ", "_")
            safe_model = str(model_name).replace(" ", "_").replace("/", "_")
            file_name = f"confusion_matrix_{safe_source}_{safe_model}.png"
            file_path = os.path.join(OUTPUT_FOLDER, file_name)

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"]
            )
            plt.title(f"{source} - {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(file_path, dpi=300)
            plt.close()
            print("Saved plot:", file_path)


def run_benchmark():
    print("=" * 70)
    print("CLASSICAL ML VS QUANTUM ML BENCHMARK")
    print("=" * 70)

    create_output_folder()

    classical_data = load_results(CLASSICAL_FILE)
    quantum_data = load_results(QUANTUM_FILE)

    if classical_data is None or quantum_data is None:
        return

    if not check_required_columns(classical_data, CLASSICAL_FILE) or \
       not check_required_columns(quantum_data, QUANTUM_FILE):
        return

    classical_results = calculate_all_metrics(classical_data, "Classical ML")
    quantum_results = calculate_all_metrics(quantum_data, "Quantum ML")
    combined_results = classical_results + quantum_results

    metrics_data = create_metrics_dataframe(combined_results)
    save_metrics(metrics_data)
    print_metrics(metrics_data)

    summary_lines = create_summary(metrics_data)
    print_summary(summary_lines)
    save_summary(summary_lines)

    create_metric_comparison_plot(metrics_data)
    create_roc_curve_plot(classical_data, quantum_data)
    create_confusion_matrix_plots(classical_data, quantum_data)

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
