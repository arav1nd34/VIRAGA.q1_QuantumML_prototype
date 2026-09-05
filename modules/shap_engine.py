import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP module is not installed. Run 'pip install shap' to proceed.")

OUTPUT_FOLDER = "benchmark_results"


def ensure_output_dir():
    """Creates output folder if it doesn't exist."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def calculate_and_save_shap(
    model, 
    X_train, 
    X_test, 
    feature_names=None, 
    model_name="Model", 
    model_type="tree"
):
    """
    Computes SHAP values and generates plots for a given model.
    
    Parameters:
    -----------
    model : object
        The trained estimator (Scikit-Learn, XGBoost, Qiskit, PennyLane, etc.).
    X_train : np.ndarray or pd.DataFrame
        Background dataset for SHAP baseline explanation.
    X_test : np.ndarray or pd.DataFrame
        Target evaluation set to compute SHAP values on.
    feature_names : list, optional
        Names of the input features.
    model_name : str
        Label used for saving file artifacts.
    model_type : str
        Type of explainer strategy: 'tree', 'linear', or 'kernel' (default/quantum safe).
    """
    if not SHAP_AVAILABLE:
        print("ERROR: SHAP is not installed. Exiting.")
        return

    ensure_output_dir()
    
    # 1. Prepare Feature Names
    if feature_names is None:
        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns.tolist()
        else:
            feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

    # Convert DataFrame to Numpy if necessary
    X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    print(f"\nComputing SHAP values for: {model_name}...")

    # 2. Select appropriate Explainer
    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test_np)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X_train_np)
            shap_values = explainer(X_test_np)
        else:
            # KernelExplainer works for model-agnostic setups, including Quantum ML pipelines
            # Summarize background dataset to speed up Kernel SHAP calculation
            background = shap.kmeans(X_train_np, min(25, X_train_np.shape[0]))
            
            # Use predict_proba for probabilities if available, else predict
            pred_func = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            explainer = shap.KernelExplainer(pred_func, background)
            shap_values = explainer(X_test_np)

        # 3. Handle multidimensional outputs (Binary classification 2D output handling)
        if isinstance(shap_values, list):
            # If shap returns list of matrices for each class, take positive class (1)
            raw_values = shap_values[1].values if hasattr(shap_values[1], "values") else shap_values[1]
        elif hasattr(shap_values, "values"):
            raw_values = shap_values.values
            if raw_values.ndim == 3:  # (samples, features, classes)
                raw_values = raw_values[:, :, 1]
        else:
            raw_values = np.array(shap_values)

        # 4. Save SHAP Values to CSV
        safe_name = str(model_name).replace(" ", "_").replace("/", "_")
        csv_file = os.path.join(OUTPUT_FOLDER, f"shap_values_{safe_name}.csv")
        
        shap_df = pd.DataFrame(raw_values, columns=feature_names)
        shap_df.to_csv(csv_file, index=False)
        print(f"Saved SHAP values CSV: {csv_file}")

        # 5. Generate and Save Summary Dot Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            raw_values, 
            X_test_np, 
            feature_names=feature_names, 
            show=False
        )
        dot_plot_file = os.path.join(OUTPUT_FOLDER, f"shap_summary_{safe_name}.png")
        plt.title(f"SHAP Summary Plot - {model_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(dot_plot_file, dpi=300)
        plt.close()
        print(f"Saved SHAP Summary Plot: {dot_plot_file}")

        # 6. Generate and Save Global Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            raw_values, 
            X_test_np, 
            feature_names=feature_names, 
            plot_type="bar", 
            show=False
        )
        bar_plot_file = os.path.join(OUTPUT_FOLDER, f"shap_importance_{safe_name}.png")
        plt.title(f"Feature Importance Bar Plot - {model_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(bar_plot_file, dpi=300)
        plt.close()
        print(f"Saved SHAP Importance Plot: {bar_plot_file}")

    except Exception as e:
        print(f"ERROR: Failed to compute SHAP for {model_name}. Details: {e}")

if __name__ == "__main__":
    from qml_4qubit import load_and_split_data, preprocess_for_quantum, predict_probability

    feature_names = ["PC1", "PC2", "PC3", "PC4"]

    X_train_pca = np.load("X_train_pca.npy")
    X_test_pca = np.load("X_test_pca.npy")

    for model_name, model_type in [("LogReg", "linear"), ("SVM", "kernel"), ("XGBoost", "tree")]:
        model = joblib.load(f"model_{model_name}.pkl")
        calculate_and_save_shap(
            model=model,
            X_train=X_train_pca,
            X_test=X_test_pca,
            feature_names=feature_names,
            model_name=model_name,
            model_type=model_type
        )

    X_train_raw, X_test_raw, y_train, y_test = load_and_split_data()
    X_train_angles, X_test_angles = preprocess_for_quantum(X_train_raw, X_test_raw)

    saved = np.load("trained_vqc_4q.npz")
    qml_model = QMLModelWrapper(saved["weights"], float(saved["bias"]), predict_probability)
    calculate_and_save_shap(
        model=qml_model,
        X_train=X_train_angles,
        X_test=X_test_angles,
        feature_names=feature_names,
        model_name="VQC_Quantum_4q_fixed",
        model_type="kernel"
    )
