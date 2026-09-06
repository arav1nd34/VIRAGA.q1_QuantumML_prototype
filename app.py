import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from qiskit.quantum_info import Statevector

from qml_4qubit import (
    load_and_split_data, preprocess_for_quantum,
    predict_probability, build_circuit, N_QUBITS
)

st.set_page_config(
    page_title="Hybrid QML Diagnostics",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BENCHMARK_DIR = "benchmark_results"

BENIGN_COLOR = "#A8D8B9"
MALIGNANT_COLOR = "#F4A9A8"
ACCENT_COLOR = "#A9C6F4"
NEUTRAL_DARK = "#3B3B4F"

st.markdown("""
<style>
    .stApp { background-color: #FAF9FB; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: #3B3B4F; }
    .stButton>button { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "LANDING"


def inject_gpu_continuous_rotation(auto_rotate: bool):
    """Animates the 3D camera via requestAnimationFrame, independent of the plotted data."""
    if not auto_rotate:
        return
    components.html(
        """
        <script>
        (function() {
            let angle = 0;
            const radius = 2.1;
            const eyeZ = 0.85;
            const speed = 0.012;
            function step() {
                const plots = window.parent.document.querySelectorAll('.js-plotly-plot');
                plots.forEach(plot => {
                    if (plot && plot._fullLayout && plot._fullLayout.scene) {
                        angle += speed;
                        const x = radius * Math.cos(angle);
                        const y = radius * Math.sin(angle);
                        window.parent.Plotly.relayout(plot, {
                            'scene.camera.eye': { x: x, y: y, z: eyeZ }
                        });
                    }
                });
                window._orbitLoop = requestAnimationFrame(step);
            }
            if (window._orbitLoop) cancelAnimationFrame(window._orbitLoop);
            setTimeout(() => { window._orbitLoop = requestAnimationFrame(step); }, 300);
        })();
        </script>
        """,
        height=0, width=0
    )


@st.cache_resource
def load_everything():
    with open("selected_features.json") as f:
        selected = json.load(f)

    classical_models = {
        name: joblib.load(f"model_{name}.pkl")
        for name in ["LogReg", "SVM", "XGBoost"]
    }

    X_test_sel = np.load("X_test_selected.npy")

    saved_qml = np.load("trained_vqc_4q.npz")
    qml_weights, qml_bias = saved_qml["weights"], float(saved_qml["bias"])

    X_train_raw, X_test_raw, y_train, y_test = load_and_split_data()
    _, X_test_angles = preprocess_for_quantum(X_train_raw, X_test_raw)

    return {
        "feature_names": selected["names"],
        "classical_models": classical_models,
        "X_test_sel": X_test_sel,
        "X_test_angles": X_test_angles,
        "y_test": y_test,
        "qml_weights": qml_weights,
        "qml_bias": qml_bias,
    }


def get_predictions_for_patient(ctx, idx):
    features_sel = ctx["X_test_sel"][idx]
    features_angles = ctx["X_test_angles"][idx]

    rows = []
    for name, model in ctx["classical_models"].items():
        prob = model.predict_proba(features_sel.reshape(1, -1))[0, 1]
        rows.append({"model": name, "probability": prob, "diagnosis": "Malignant" if prob >= 0.5 else "Benign"})

    qml_prob = predict_probability(features_angles, ctx["qml_weights"], ctx["qml_bias"])
    rows.append({"model": "VQC (Quantum)", "probability": qml_prob, "diagnosis": "Malignant" if qml_prob >= 0.5 else "Benign"})

    return pd.DataFrame(rows), features_sel, features_angles


ctx = load_everything()

# ==========================================
# PAGE 1: LANDING
# ==========================================
if st.session_state["current_page"] == "LANDING":
    st.caption("HYBRID QUANTUM-CLASSICAL ML DIAGNOSTICS PLATFORM")
    st.title("Choose a Diagnosis Module")
    st.write("Pick a module to run the hybrid quantum-classical pipeline on real test-set patients.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("🟢 **ACTIVE**")
            st.subheader("Cancer Detection")
            st.write("Breast cancer diagnosis via classical ML + a 4-qubit VQC, benchmarked head-to-head.")
            if st.button("Launch Cancer Module →", use_container_width=True):
                st.session_state["current_page"] = "CANCER_DASHBOARD"
                st.rerun()
    with col2:
        with st.container(border=True):
            st.markdown("⚪ *IN DEVELOPMENT*")
            st.subheader("Cardiovascular Risk")
            st.write("Not built for this prototype — future extension of the same pipeline.")
            if st.button("Access Cardio Pipeline", use_container_width=True):
                st.session_state["current_page"] = "UNDER_DEVELOPMENT"
                st.rerun()
    with col3:
        with st.container(border=True):
            st.markdown("⚪ *IN DEVELOPMENT*")
            st.subheader("Neurological Screening")
            st.write("Not built for this prototype — future extension of the same pipeline.")
            if st.button("Access Neuro Pipeline", use_container_width=True):
                st.session_state["current_page"] = "UNDER_DEVELOPMENT"
                st.rerun()

# ==========================================
# PAGE 2: CANCER DASHBOARD
# ==========================================
elif st.session_state["current_page"] == "CANCER_DASHBOARD":
    if st.button("← Back to Module Selector"):
        st.session_state["current_page"] = "LANDING"
        st.rerun()

    with st.sidebar:
        st.title("Diagnostics")
        view_selection = st.radio(
            "View", ["Command HUD", "Test-Set Overview", "Quantum State Map", "Benchmark & Explainability"]
        )
        st.divider()
        auto_rotate = st.toggle("Auto-Rotate 3D View", value=False)
        st.divider()
        st.subheader("Patient Selection")
        n_patients = len(ctx["y_test"])
        patient_index = st.slider("Test-set patient index", 0, n_patients - 1, 0)
        st.caption(f"{n_patients} patients available in the held-out test set.")

    preds_df, features_sel, features_angles = get_predictions_for_patient(ctx, patient_index)
    true_label = ctx["y_test"][patient_index]
    avg_risk = preds_df["probability"].mean() * 100
    agreement = (preds_df["diagnosis"] == preds_df["diagnosis"].mode()[0]).mean() * 100

    # ---------- VIEW 1: COMMAND HUD ----------
    if view_selection == "Command HUD":
        st.subheader(f"Clinical HUD — Patient #{patient_index}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("True Diagnosis", "Malignant" if true_label == 1 else "Benign")
        m2.metric("Avg. Predicted Risk", f"{avg_risk:.1f}%")
        m3.metric("Model Agreement", f"{agreement:.0f}%")
        m4.metric("Models Run", "4 (3 classical + 1 quantum)")

        if avg_risk > 50:
            st.error(f"⚠️ Elevated malignancy risk ({avg_risk:.1f}% average across models).")
        else:
            st.success(f"✅ Low malignancy risk ({avg_risk:.1f}% average across models).")

        col_left, col_right = st.columns([1.3, 1])

        with col_left:
            st.markdown("**Model-by-model prediction**")
            fig_preds = px.bar(
                preds_df, x="model", y="probability", color="diagnosis",
                color_discrete_map={"Malignant": MALIGNANT_COLOR, "Benign": BENIGN_COLOR},
                range_y=[0, 1]
            )
            fig_preds.update_layout(margin=dict(l=0, r=0, b=0, t=10), yaxis_title="P(malignant)")
            st.plotly_chart(fig_preds, use_container_width=True)

        with col_right:
            st.markdown("**Patient feature values**")
            feat_df = pd.DataFrame({
                "Feature": ctx["feature_names"],
                "Value": features_sel.round(3)
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.markdown("**Per-patient SHAP attribution (VQC)**")
        shap_path = os.path.join(BENCHMARK_DIR, "shap_values_VQC_Quantum_4q_fixed.csv")
        if os.path.exists(shap_path):
            shap_df = pd.read_csv(shap_path)
            if patient_index < len(shap_df):
                row = shap_df.iloc[patient_index]
                attr_df = pd.DataFrame({"Feature": row.index, "Attribution": row.values})
                fig_shap = px.bar(attr_df, x="Attribution", y="Feature", orientation="h",
                                   color_discrete_sequence=[NEUTRAL_DARK])
                fig_shap.update_layout(margin=dict(l=0, r=0, b=0, t=10))
                st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.warning("Run shap_engine.py to generate per-patient attributions.")

    # ---------- VIEW 2: TEST-SET OVERVIEW ----------
    elif view_selection == "Test-Set Overview":
        st.subheader("Test-Set Overview (Real Data)")

        y_test = ctx["y_test"]
        n = len(y_test)
        malignant_count = int((y_test == 1).sum())

        b1, b2, b3 = st.columns(3)
        b1.metric("Total Patients", n)
        b2.metric("Malignant (True)", malignant_count)
        b3.metric("Benign (True)", n - malignant_count)

        feat_names = ctx["feature_names"]
        fx, fy = st.columns(2)
        x_feat = fx.selectbox("X-axis feature", feat_names, index=0)
        y_feat = fy.selectbox("Y-axis feature", feat_names, index=1)

        df_all = pd.DataFrame(ctx["X_test_sel"], columns=feat_names)
        df_all["Diagnosis"] = np.where(y_test == 1, "Malignant", "Benign")
        df_all["Patient"] = np.arange(n)

        fig_scatter = px.scatter(
            df_all, x=x_feat, y=y_feat, color="Diagnosis", hover_data=["Patient"],
            color_discrete_map={"Malignant": MALIGNANT_COLOR, "Benign": BENIGN_COLOR}
        )
        fig_scatter.add_vline(x=features_sel[feat_names.index(x_feat)], line_dash="dash", line_color=ACCENT_COLOR)
        fig_scatter.add_hline(y=features_sel[feat_names.index(y_feat)], line_dash="dash", line_color=ACCENT_COLOR)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(f"Dashed lines mark currently selected patient (#{patient_index}).")

        st.dataframe(df_all, use_container_width=True, hide_index=True)

    # ---------- VIEW 3: QUANTUM STATE MAP ----------
    elif view_selection == "Quantum State Map":
        st.subheader("Quantum State Map (Real Circuit Output)")
        st.caption(
            f"Actual statevector of the trained {N_QUBITS}-qubit circuit for this patient — "
            "not a synthetic visualization."
        )

        circuit = build_circuit(features_angles, ctx["qml_weights"])
        statevector = Statevector.from_instruction(circuit)
        amplitudes = statevector.data
        probabilities = np.abs(amplitudes) ** 2
        basis_labels = [format(i, f"0{N_QUBITS}b") for i in range(2 ** N_QUBITS)]

        fig_state = go.Figure(data=[go.Scatter3d(
            x=amplitudes.real,
            y=amplitudes.imag,
            z=probabilities,
            mode="markers+text",
            text=basis_labels,
            textposition="top center",
            marker=dict(
                size=6 + probabilities * 40,
                color=probabilities,
                colorscale=[[0, "#D9E8F5"], [1, MALIGNANT_COLOR]],
                opacity=0.85
            )
        )])
        fig_state.update_layout(
            height=600,
            margin=dict(l=0, r=0, b=0, t=20),
            scene=dict(
                xaxis_title="Re(amplitude)",
                yaxis_title="Im(amplitude)",
                zaxis_title="Probability",
                xaxis=dict(gridcolor="#E2E2EC"),
                yaxis=dict(gridcolor="#E2E2EC"),
                zaxis=dict(gridcolor="#E2E2EC"),
            )
        )
        st.plotly_chart(fig_state, use_container_width=True, key=f"state_{patient_index}")
        inject_gpu_continuous_rotation(auto_rotate)

        top_states = pd.DataFrame({"basis state": basis_labels, "probability": probabilities}) \
            .sort_values("probability", ascending=False).head(5)
        st.markdown("**Most probable basis states**")
        st.dataframe(top_states, use_container_width=True, hide_index=True)

    # ---------- VIEW 4: BENCHMARK & EXPLAINABILITY ----------
    elif view_selection == "Benchmark & Explainability":
        st.subheader("Classical vs Quantum — Benchmark")

        metrics_path = os.path.join(BENCHMARK_DIR, "benchmark_metrics.csv")
        if os.path.exists(metrics_path):
            st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)
        else:
            st.warning("Run benchmark+visualize.py first.")

        col1, col2 = st.columns(2)
        metric_plot = os.path.join(BENCHMARK_DIR, "metric_comparison.png")
        roc_plot = os.path.join(BENCHMARK_DIR, "roc_curve.png")
        if os.path.exists(metric_plot):
            col1.image(metric_plot, caption="Accuracy / Sensitivity / Specificity")
        if os.path.exists(roc_plot):
            col2.image(roc_plot, caption="ROC Curves")

        st.divider()
        st.subheader("Global Feature Importance (SHAP)")
        model_choice = st.selectbox("Model", ["LogReg", "SVM", "XGBoost", "VQC_Quantum_4q_fixed"])
        bar_img = os.path.join(BENCHMARK_DIR, f"shap_importance_{model_choice}.png")
        summary_img = os.path.join(BENCHMARK_DIR, f"shap_summary_{model_choice}.png")
        c1, c2 = st.columns(2)
        if os.path.exists(summary_img):
            c1.image(summary_img, caption="SHAP Summary")
        if os.path.exists(bar_img):
            c2.image(bar_img, caption="Feature Importance")

# ==========================================
# PAGE 3: UNDER DEVELOPMENT
# ==========================================
elif st.session_state["current_page"] == "UNDER_DEVELOPMENT":
    st.warning("🚧 **Module Under Development**")
    st.write("Not part of this prototype's scope — a future extension of the same hybrid pipeline.")
    if st.button("← Return to Diagnostic Selector"):
        st.session_state["current_page"] = "LANDING"
        st.rerun()

