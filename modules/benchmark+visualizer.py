"""
===============================================================================
 qbench.py — Benchmarking + Visualization + Explainability backend
             for a Hybrid Quantum / Classical ML disease-detection platform.

 SINGLE FILE. Copy-paste it, run it.

     pip install numpy pandas scipy scikit-learn matplotlib shap tabulate
     python qbench.py                                # demo -> results/ + report/
     python qbench.py --data breast_cancer_wisconsin.csv
     python qbench.py --results results/ --out report/ --no-demo   # your CSVs

 WHAT IT DOES  (the six backend stages)
   1. INGEST      read plain CSV produced by the ML and QML pipelines
   2. VALIDATE    prove every model was scored on identical CV folds
   3. SCORE       ROC-AUC, PR-AUC, sensitivity, specificity, PPV, NPV, F1, MCC,
                  Brier, sensitivity@95%-specificity, bootstrap 95% CIs
   4. TEST        DeLong (AUC), McNemar (paired labels), Wilcoxon (folds),
                  all Holm-corrected for multiple comparisons
   5. VISUALIZE   13 matplotlib figures incl. a one-slide dashboard
   6. EXPLAIN     SHAP feature attributions — model-agnostic, so it explains
                  the QUANTUM model exactly as it explains the classical one

 DATASET
   Reads breast_cancer_wisconsin.csv from the working directory if present,
   otherwise falls back to sklearn's built-in copy. Point --data at any binary
   -outcome CSV to swap datasets; --target-col names the label column.

 THE CSV CONTRACT (give this to your teammates on day one)
   results/predictions.csv — one row per (model, fold, test sample):
     model            str    short name; appears on every plot
     family           str    classical | quantum | hybrid
     fold             int    0..k-1
     test_index       int    row index into the ORIGINAL dataset
     y_true           int    0/1, where 1 = disease
     y_pred           int    0/1 hard label
     y_score          float  P(y=1); required for ROC / PR / DeLong
     train_time_s     float  repeat the fold's value on every row
     inference_time_s float  repeat the fold's value on every row

   Optional sidecars in the same folder:
     runs_meta.csv        model,family,key,value      (n_qubits, depth, shots...)
     history.csv          model,epoch,train_loss      -> convergence plot
     learning_curves.csv  model,n_train,mean,std      -> sample-efficiency plot

 FOUR RULES THAT MAKE THE BENCHMARK REAL
   1. Every model uses the SAME StratifiedKFold(n_splits=5, shuffle=True,
      random_state=42). Stage 2 refuses to run otherwise.
   2. Class 1 = disease. (sklearn's breast-cancer set labels benign as 1: flip it.)
   3. Scaler / PCA / feature selection are fit INSIDE the training fold only.
      Fitting on the full dataset leaks test information and produces a fake
      quantum "win" that any judge will catch.
   4. Same feature budget. If the QSVM sees PCA->6 features, at least one
      classical baseline must also see PCA->6 features, or you are comparing
      dimensionality reduction rather than quantum vs classical.
===============================================================================
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats as sps
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, roc_curve)

warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except ImportError:                                   # explainability degrades gracefully
    HAS_SHAP = False

SEED = 42


# =============================================================================
# 1 + 2.  SCHEMA — CSV ingest, and the split-integrity check
# =============================================================================

VALID_FAMILIES = ("classical", "quantum", "hybrid")
PRED_COLUMNS = ["model", "family", "fold", "test_index", "y_true", "y_pred",
                "y_score", "train_time_s", "inference_time_s"]
RESERVED_FILES = ("runs_meta.csv", "history.csv", "learning_curves.csv")


def _arr(x) -> np.ndarray:
    return np.asarray(x).ravel()


def _coerce(v: Any) -> Any:
    """Turn a CSV string back into int / float / bool / None where possible."""
    if not isinstance(v, str):
        return None if (isinstance(v, float) and np.isnan(v)) else v
    s = v.strip()
    if s in ("", "nan", "NaN", "None"):
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


@dataclass
class FoldResult:
    """Predictions for ONE cross-validation fold."""
    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
    train_time_s: float = 0.0
    inference_time_s: float = 0.0
    test_index: Optional[np.ndarray] = None

    def __post_init__(self):
        self.y_true = _arr(self.y_true).astype(int)
        self.y_pred = _arr(self.y_pred).astype(int)
        self.y_score = _arr(self.y_score).astype(float)
        n = len(self.y_true)
        if not (len(self.y_pred) == len(self.y_score) == n):
            raise ValueError(f"fold {self.fold}: length mismatch")
        if self.test_index is not None:
            self.test_index = _arr(self.test_index).astype(int)
            if len(self.test_index) != n:
                raise ValueError(f"fold {self.fold}: test_index length != n")
        bad = set(np.unique(self.y_true)) - {0, 1}
        if bad:
            raise ValueError(f"fold {self.fold}: y_true must be 0/1, saw {bad}. "
                             "Encode the disease class as 1.")


@dataclass
class ModelRun:
    """All folds for ONE model, plus the metadata that makes comparison fair."""
    name: str
    family: str
    folds: List[FoldResult]
    meta: Dict[str, Any] = field(default_factory=dict)
    history: Optional[Dict[str, List[float]]] = None
    learning_curve: Optional[Dict[str, List[float]]] = None

    def __post_init__(self):
        if self.family not in VALID_FAMILIES:
            raise ValueError(f"family must be one of {VALID_FAMILIES}, got {self.family!r}")
        if not self.folds:
            raise ValueError(f"{self.name}: no folds supplied")

    # pooled out-of-fold arrays
    @property
    def y_true(self):  return np.concatenate([f.y_true for f in self.folds])
    @property
    def y_pred(self):  return np.concatenate([f.y_pred for f in self.folds])
    @property
    def y_score(self): return np.concatenate([f.y_score for f in self.folds])

    @property
    def test_index(self):
        if any(f.test_index is None for f in self.folds):
            return None
        return np.concatenate([f.test_index for f in self.folds])

    @property
    def train_time_s(self):
        return float(np.sum([f.train_time_s for f in self.folds]))

    @property
    def inference_time_s(self):
        return float(np.sum([f.inference_time_s for f in self.folds]))

    # ---- CSV out ----
    def to_frame(self, include_meta: bool = False) -> pd.DataFrame:
        parts = []
        for f in self.folds:
            n = len(f.y_true)
            parts.append(pd.DataFrame({
                "model": self.name, "family": self.family, "fold": f.fold,
                "test_index": (f.test_index if f.test_index is not None
                               else np.full(n, -1, dtype=int)),
                "y_true": f.y_true, "y_pred": f.y_pred, "y_score": f.y_score,
                "train_time_s": f.train_time_s,
                "inference_time_s": f.inference_time_s}))
        df = pd.concat(parts, ignore_index=True)[PRED_COLUMNS]
        if include_meta:
            for k, v in self.meta.items():
                df[f"meta_{k}"] = str(v) if isinstance(v, (dict, list, tuple)) else v
        return df

    def history_frame(self) -> Optional[pd.DataFrame]:
        if not self.history or "train_loss" not in self.history:
            return None
        h = {k: list(v) for k, v in self.history.items() if v is not None}
        n = len(h["train_loss"])
        h.setdefault("epoch", list(range(1, n + 1)))
        return pd.DataFrame({"model": self.name,
                             **{k: v for k, v in h.items() if len(v) == n}})

    def learning_curve_frame(self) -> Optional[pd.DataFrame]:
        if not self.learning_curve or "n_train" not in self.learning_curve:
            return None
        return pd.DataFrame({"model": self.name,
                             **{k: list(v) for k, v in self.learning_curve.items()
                                if v is not None}})

    def save_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.to_frame(include_meta=True).to_csv(path, index=False)
        stem = path[:-4] if path.lower().endswith(".csv") else path
        hf, lf = self.history_frame(), self.learning_curve_frame()
        if hf is not None:
            hf.to_csv(f"{stem}.history.csv", index=False)
        if lf is not None:
            lf.to_csv(f"{stem}.learning_curve.csv", index=False)
        return path

    # ---- CSV in ----
    @classmethod
    def from_frame(cls, df, name=None, meta=None, history=None, learning_curve=None):
        missing = [c for c in ("fold", "y_true", "y_pred", "y_score") if c not in df.columns]
        if missing:
            raise ValueError(f"predictions CSV missing required columns: {missing}")
        name = name or (df["model"].iloc[0] if "model" in df else "unnamed")
        family = df["family"].iloc[0] if "family" in df else "classical"
        meta = dict(meta or {})
        for c in df.columns:
            if c.startswith("meta_"):
                meta.setdefault(c[5:], _coerce(df[c].iloc[0]))
        folds = []
        for k, g in df.groupby("fold", sort=True):
            ti = g["test_index"].to_numpy() if "test_index" in g else None
            if ti is not None and (ti < 0).any():
                ti = None
            folds.append(FoldResult(
                fold=int(k), y_true=g["y_true"].to_numpy(),
                y_pred=g["y_pred"].to_numpy(), y_score=g["y_score"].to_numpy(),
                train_time_s=float(g["train_time_s"].iloc[0]) if "train_time_s" in g else 0.0,
                inference_time_s=float(g["inference_time_s"].iloc[0]) if "inference_time_s" in g else 0.0,
                test_index=ti))
        return cls(str(name), str(family), folds, meta, history, learning_curve)

    @classmethod
    def load_csv(cls, path: str) -> "ModelRun":
        df = pd.read_csv(path)
        stem = path[:-4] if path.lower().endswith(".csv") else path
        hist = lc = None
        if os.path.exists(f"{stem}.history.csv"):
            h = pd.read_csv(f"{stem}.history.csv").drop(columns=["model"], errors="ignore")
            hist = {c: h[c].dropna().tolist() for c in h.columns}
        if os.path.exists(f"{stem}.learning_curve.csv"):
            l = pd.read_csv(f"{stem}.learning_curve.csv").drop(columns=["model"], errors="ignore")
            lc = {c: l[c].tolist() for c in l.columns}
        return cls.from_frame(df, history=hist, learning_curve=lc)


def save_runs(runs: Sequence[ModelRun], directory: str) -> Dict[str, str]:
    """Write predictions.csv (+ runs_meta / history / learning_curves sidecars)."""
    os.makedirs(directory, exist_ok=True)
    written = {}
    p = os.path.join(directory, "predictions.csv")
    pd.concat([r.to_frame() for r in runs], ignore_index=True).to_csv(p, index=False)
    written["predictions"] = p

    meta_rows = [{"model": r.name, "family": r.family, "key": k,
                  "value": str(v) if isinstance(v, (dict, list, tuple)) else v}
                 for r in runs for k, v in r.meta.items()]
    if meta_rows:
        p = os.path.join(directory, "runs_meta.csv")
        pd.DataFrame(meta_rows).to_csv(p, index=False); written["meta"] = p

    hs = [h for h in (r.history_frame() for r in runs) if h is not None]
    if hs:
        p = os.path.join(directory, "history.csv")
        pd.concat(hs, ignore_index=True).to_csv(p, index=False); written["history"] = p

    ls = [l for l in (r.learning_curve_frame() for r in runs) if l is not None]
    if ls:
        p = os.path.join(directory, "learning_curves.csv")
        pd.concat(ls, ignore_index=True).to_csv(p, index=False); written["learning_curves"] = p
    return written


def _sidecars(directory: str):
    meta, hist, lc = {}, {}, {}
    mp = os.path.join(directory, "runs_meta.csv")
    if os.path.exists(mp):
        for model, g in pd.read_csv(mp).groupby("model"):
            meta[model] = {r.key: _coerce(r.value) for r in g.itertuples()}
    hp = os.path.join(directory, "history.csv")
    if os.path.exists(hp):
        for model, g in pd.read_csv(hp).groupby("model"):
            g = g.drop(columns=["model"])
            hist[model] = {c: g[c].dropna().tolist() for c in g.columns}
    lp = os.path.join(directory, "learning_curves.csv")
    if os.path.exists(lp):
        for model, g in pd.read_csv(lp).groupby("model"):
            g = g.drop(columns=["model"])
            lc[model] = {c: g[c].tolist() for c in g.columns}
    return meta, hist, lc


def load_dir(directory: str) -> List[ModelRun]:
    """Load a results folder: either a predictions.csv bundle, or one CSV per model."""
    pp = os.path.join(directory, "predictions.csv")
    if os.path.exists(pp):
        preds = pd.read_csv(pp)
        meta, hist, lc = _sidecars(directory)
        return [ModelRun.from_frame(g, name=n, meta=meta.get(n),
                                    history=hist.get(n), learning_curve=lc.get(n))
                for n, g in preds.groupby("model", sort=False)]
    paths = sorted(p for p in glob.glob(os.path.join(directory, "*.csv"))
                   if not p.endswith((".history.csv", ".learning_curve.csv"))
                   and os.path.basename(p) not in RESERVED_FILES)
    if not paths:
        raise FileNotFoundError(f"no result CSVs found in {directory!r}")
    return [ModelRun.load_csv(p) for p in paths]


def assert_same_splits(runs: Sequence[ModelRun]) -> bool:
    """Stage 2. The check that separates a real benchmark from a coincidence."""
    ref = runs[0]
    if any(f.test_index is None for f in ref.folds):
        print("[qbench] WARNING: no test_index column — cannot verify identical splits; "
              "paired tests (McNemar) will be skipped.")
        return False
    for r in runs[1:]:
        if len(r.folds) != len(ref.folds):
            raise ValueError(f"{r.name} has {len(r.folds)} folds, {ref.name} has {len(ref.folds)}")
        for a, b in zip(ref.folds, r.folds):
            if b.test_index is None or not np.array_equal(np.sort(a.test_index),
                                                          np.sort(b.test_index)):
                raise ValueError(f"Split mismatch on fold {a.fold} between {ref.name!r} "
                                 f"and {r.name!r}. Use the SAME StratifiedKFold(seed).")
    return True


def blank_template(path: str = "predictions_template.csv") -> str:
    """Empty CSV with the correct header — hand this to your teammates."""
    pd.DataFrame(columns=PRED_COLUMNS).to_csv(path, index=False)
    return path


# =============================================================================
# 3.  METRICS — clinical scoring
# =============================================================================

PRIMARY_METRICS = ["roc_auc", "pr_auc", "sensitivity", "specificity",
                   "balanced_accuracy", "f1", "mcc", "accuracy"]

METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.",
    "sensitivity": "Sensitivity (recall)", "specificity": "Specificity",
    "ppv": "Precision (PPV)", "npv": "NPV", "f1": "F1", "mcc": "MCC",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
    "brier": "Brier (lower=better)", "sens_at_spec95": "Sensitivity @ 95% spec.",
}


def _safe_div(a, b):
    return float(a) / float(b) if b else float("nan")


def sensitivity_at_specificity(y_true, y_score, target_spec: float = 0.95) -> float:
    """Best sensitivity while holding specificity >= target. What screening optimizes."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= (1.0 - target_spec) + 1e-12
    return float(np.max(tpr[ok])) if ok.any() else float("nan")


def compute_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if both else float("nan"),
        "roc_auc": roc_auc_score(y_true, y_score) if both else float("nan"),
        "pr_auc": average_precision_score(y_true, y_score) if both else float("nan"),
        "brier": brier_score_loss(y_true, np.clip(y_score, 0, 1)) if both else float("nan"),
        "sens_at_spec95": sensitivity_at_specificity(y_true, y_score, 0.95),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def per_fold_table(run: ModelRun) -> pd.DataFrame:
    rows = []
    for f in run.folds:
        m = compute_metrics(f.y_true, f.y_pred, f.y_score)
        m.update(model=run.name, family=run.family, fold=f.fold,
                 train_time_s=f.train_time_s, inference_time_s=f.inference_time_s,
                 n_test=len(f.y_true))
        rows.append(m)
    return pd.DataFrame(rows)


def long_fold_table(runs: Sequence[ModelRun]) -> pd.DataFrame:
    return pd.concat([per_fold_table(r) for r in runs], ignore_index=True)


def bootstrap_ci(y_true, y_pred, y_score, metric="roc_auc", n_boot=1000,
                 alpha=0.05, seed=0) -> Tuple[float, float, float]:
    """Stratified bootstrap CI on the pooled out-of-fold predictions."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    point = compute_metrics(y_true, y_pred, y_score)[metric]
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        try:
            vals.append(compute_metrics(y_true[idx], y_pred[idx], y_score[idx])[metric])
        except Exception:
            pass
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if vals.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def summary_table(runs: Sequence[ModelRun], metrics=tuple(PRIMARY_METRICS),
                  n_boot=1000, seed=0) -> pd.DataFrame:
    rows = []
    for run in runs:
        ft = per_fold_table(run)
        row = {"model": run.name, "family": run.family, "n_folds": len(run.folds)}
        for m in metrics:
            row[f"{m}_mean"] = ft[m].mean()
            row[f"{m}_std"] = ft[m].std(ddof=1) if len(ft) > 1 else 0.0
        pt, lo, hi = bootstrap_ci(run.y_true, run.y_pred, run.y_score,
                                  "roc_auc", n_boot=n_boot, seed=seed)
        row.update(roc_auc_pooled=pt, roc_auc_ci_lo=lo, roc_auc_ci_hi=hi,
                   train_time_s=run.train_time_s, inference_time_s=run.inference_time_s,
                   n_qubits=run.meta.get("n_qubits"),
                   circuit_depth=run.meta.get("circuit_depth"),
                   n_params=run.meta.get("n_params"), shots=run.meta.get("shots"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def format_headline(df: pd.DataFrame, metrics=tuple(PRIMARY_METRICS)) -> pd.DataFrame:
    out = pd.DataFrame({"Model": df["model"], "Family": df["family"]})
    for m in metrics:
        out[METRIC_LABELS.get(m, m)] = [f"{a:.3f} ± {b:.3f}"
                                        for a, b in zip(df[f"{m}_mean"], df[f"{m}_std"])]
    out["Train time (s)"] = df["train_time_s"].map(lambda v: f"{v:.2f}")
    return out


# =============================================================================
# 4.  STATISTICS — is the difference real, or a lucky split?
# =============================================================================

def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, float); T2[J] = T
    return T2


def delong_test(y_true, score_a, score_b) -> Dict[str, float]:
    """Two-sided DeLong test for AUC(a) - AUC(b) on the same samples."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")          # positives first, stable
    m = int(y_true.sum())
    P = np.vstack([np.asarray(score_a, float)[order],
                   np.asarray(score_b, float)[order]])
    n = P.shape[1] - m
    k = P.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(P[r, :m]); ty[r] = _midrank(P[r, m:]); tz[r] = _midrank(P[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta_auc": diff,
                "z": float("nan"), "p_value": 1.0,
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    se = float(np.sqrt(var)); z = diff / se
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta_auc": diff,
            "z": float(z), "p_value": float(2 * sps.norm.sf(abs(z))),
            "ci_lo": diff - 1.96 * se, "ci_hi": diff + 1.96 * se}


def mcnemar_test(y_true, pred_a, pred_b, exact_threshold: int = 25) -> Dict[str, float]:
    """b = a-right/b-wrong, c = a-wrong/b-right. Exact binomial when b+c is small."""
    y_true, pred_a, pred_b = (np.asarray(v).astype(int) for v in (y_true, pred_a, pred_b))
    ca, cb = pred_a == y_true, pred_b == y_true
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb)); n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "test": "identical"}
    if n < exact_threshold:
        return {"b": b, "c": c, "statistic": float(min(b, c)),
                "p_value": float(sps.binomtest(b, n, 0.5).pvalue), "test": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n                      # continuity-corrected chi-square
    return {"b": b, "c": c, "statistic": float(stat),
            "p_value": float(sps.chi2.sf(stat, 1)), "test": "chi2_cc"}


def paired_fold_test(vals_a, vals_b) -> Dict[str, float]:
    a, b = np.asarray(vals_a, float), np.asarray(vals_b, float)
    d = a - b
    out = {"mean_diff": float(d.mean()), "n_folds": int(len(d)),
           "wins": int(np.sum(d > 0)), "losses": int(np.sum(d < 0))}
    if len(d) < 3 or np.allclose(d, 0):
        out.update(t_p=float("nan"), wilcoxon_p=float("nan"), cohens_d=float("nan"))
        return out
    out["t_p"] = float(sps.ttest_rel(a, b).pvalue)
    try:
        out["wilcoxon_p"] = float(sps.wilcoxon(a, b).pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    sd = d.std(ddof=1)
    out["cohens_d"] = float(d.mean() / sd) if sd > 0 else float("inf")
    return out


def holm_bonferroni(pvals, alpha: float = 0.05) -> np.ndarray:
    """Step-down adjusted p-values. Mandatory once you compare more than 2 models."""
    p = np.asarray(pvals, float)
    valid = np.isfinite(p)
    adj = np.full_like(p, np.nan)
    pv = p[valid]; n = len(pv)
    order = np.argsort(pv); running = 0.0; out = np.empty(n)
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pv[i])
        out[i] = min(1.0, running)
    adj[valid] = out
    return adj


def _align(a: ModelRun, b: ModelRun):
    """Align pooled predictions by dataset index so the tests are genuinely paired."""
    ia, ib = a.test_index, b.test_index
    if ia is None or ib is None:
        if len(a.y_true) != len(b.y_true):
            return None
        return a.y_true, a.y_pred, a.y_score, b.y_pred, b.y_score
    oa, ob = np.argsort(ia), np.argsort(ib)
    if not np.array_equal(ia[oa], ib[ob]):
        return None
    return a.y_true[oa], a.y_pred[oa], a.y_score[oa], b.y_pred[ob], b.y_score[ob]


def pairwise_comparisons(runs: Sequence[ModelRun], metric="roc_auc", alpha=0.05) -> pd.DataFrame:
    tabs = {r.name: per_fold_table(r) for r in runs}
    rows = []
    for a, b in combinations(runs, 2):
        row = {"model_a": a.name, "model_b": b.name,
               "family_a": a.family, "family_b": b.family}
        al = _align(a, b)
        if al is not None:
            y, pa, sa, pb, sb = al
            row.update({f"delong_{k}": v for k, v in delong_test(y, sa, sb).items()})
            row.update({f"mcnemar_{k}": v for k, v in mcnemar_test(y, pa, pb).items()})
        else:
            row.update(delong_p_value=np.nan, mcnemar_p_value=np.nan, delong_delta_auc=np.nan)
        row.update({f"fold_{k}": v for k, v in paired_fold_test(
            tabs[a.name][metric].values, tabs[b.name][metric].values).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ("delong_p_value", "mcnemar_p_value", "fold_wilcoxon_p"):
            if col in df:
                df[col + "_holm"] = holm_bonferroni(df[col].values, alpha)
        df["significant"] = df.get("delong_p_value_holm",
                                   pd.Series(np.nan, index=df.index)) < alpha
    return df


def best_vs_rest(runs: Sequence[ModelRun], champion: str, metric="roc_auc") -> pd.DataFrame:
    """Focused table: the best quantum/hybrid model vs every other model."""
    df = pairwise_comparisons(runs, metric=metric)
    sub = df[(df.model_a == champion) | (df.model_b == champion)].reset_index(drop=True).copy()
    flip = (sub.model_b == champion).to_numpy()

    def swap(ca, cb, negate=False):
        # copies are mandatory: .values on a .loc slice can alias the same block
        if ca not in sub or cb not in sub:
            return
        a = sub[ca].to_numpy(copy=True); b = sub[cb].to_numpy(copy=True)
        if negate:
            a, b = -a, -b
        sub.loc[flip, ca] = b[flip]
        sub.loc[flip, cb] = a[flip]

    swap("model_a", "model_b"); swap("family_a", "family_b")
    swap("delong_auc_a", "delong_auc_b"); swap("fold_wins", "fold_losses")
    swap("delong_ci_lo", "delong_ci_hi", negate=True)   # interval negates AND swaps ends
    for col in ("delong_delta_auc", "fold_mean_diff", "delong_z", "fold_cohens_d"):
        if col in sub:
            v = sub[col].to_numpy(copy=True)
            sub.loc[flip, col] = -v[flip]
    return sub


# =============================================================================
# 5.  PLOTS — matplotlib only
#     Colour encodes FAMILY (classical = cool, quantum/hybrid = warm) so the
#     story reads from three metres away. Every estimate carries an error bar.
# =============================================================================

CLASSICAL_COLORS = ["#3B6EA5", "#5C9EAD", "#7C90A0", "#2E4057", "#4F86C6"]
QUANTUM_COLORS = ["#D1495B", "#E8871A", "#8B3A9E", "#C1435F", "#F0A202"]
FAMILY_MARKER = {"classical": "o", "quantum": "D", "hybrid": "s"}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.bbox": "tight", "axes.grid": True,
        "grid.color": "#e3e3e3", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.edgecolor": "#444", "axes.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11.5, "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


def color_map(runs: Sequence[ModelRun]) -> Dict[str, str]:
    cmap, ci, qi = {}, 0, 0
    for r in runs:
        if r.family == "classical":
            cmap[r.name] = CLASSICAL_COLORS[ci % len(CLASSICAL_COLORS)]; ci += 1
        else:
            cmap[r.name] = QUANTUM_COLORS[qi % len(QUANTUM_COLORS)]; qi += 1
    return cmap


def _style_of(r: ModelRun):
    """Quantum models get thick solid lines; classical get thin dashed."""
    return (2.8, "-") if r.family != "classical" else (1.8, "--")


def _save(fig, outdir, name):
    if outdir is None:
        return fig
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _rotate(ax, labels, size=8.5):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=size)


def _mean_roc(run: ModelRun, grid: np.ndarray):
    tprs = []
    for f in run.folds:
        if len(np.unique(f.y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(f.y_true, f.y_score)
        tprs.append(np.interp(grid, fpr, tpr))
    if not tprs:
        return None, None, None
    T = np.vstack(tprs); T[:, 0] = 0.0
    mean = T.mean(axis=0); mean[-1] = 1.0
    return mean, T.std(axis=0), auc(grid, mean)


def plot_roc(runs, outdir=None, name="roc_curves.png"):
    set_style()
    grid = np.linspace(0, 1, 200); cmap = color_map(runs)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for r in runs:
        mean, sd, a = _mean_roc(r, grid)
        if mean is None:
            continue
        lw, ls = _style_of(r)
        ax.plot(grid, mean, color=cmap[r.name], lw=lw, ls=ls,
                label=f"{r.name} (AUC {a:.3f})")
        ax.fill_between(grid, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                        color=cmap[r.name], alpha=0.12, lw=0)
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1, label="Chance")
    ax.set(xlim=(-.01, 1.01), ylim=(-.01, 1.01), xlabel="1 − Specificity (FPR)",
           ylabel="Sensitivity (TPR)", title="ROC — mean across folds (±1 SD)")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, outdir, name)


def plot_pr(runs, outdir=None, name="pr_curves.png"):
    set_style(); cmap = color_map(runs)
    fig, ax = plt.subplots(figsize=(7.2, 6.2)); prev = None
    for r in runs:
        y, s = r.y_true, r.y_score
        prev = y.mean()
        prec, rec, _ = precision_recall_curve(y, s)
        lw, ls = _style_of(r)
        ax.plot(rec, prec, color=cmap[r.name], lw=lw, ls=ls,
                label=f"{r.name} (AP {average_precision_score(y, s):.3f})")
    if prev is not None:
        ax.axhline(prev, ls=":", color="grey", lw=1, label=f"No-skill ({prev:.2f})")
    ax.set(xlabel="Recall (Sensitivity)", ylabel="Precision (PPV)",
           xlim=(-.01, 1.01), ylim=(-.01, 1.05),
           title="Precision–Recall — pooled out-of-fold predictions")
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, outdir, name)


def plot_metric_bars(runs, metrics=("roc_auc", "sensitivity", "specificity", "f1"),
                     outdir=None, name="metric_bars.png"):
    set_style()
    df = long_fold_table(runs); cmap = color_map(runs)
    order = [r.name for r in runs]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4.8))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        g = df.groupby("model")[m]
        mean = g.mean().reindex(order).values
        ci = (1.96 * g.sem().reindex(order).fillna(0)).values
        ax.bar(range(len(order)), mean, yerr=ci, capsize=4,
               color=[cmap[k] for k in order], edgecolor="black", lw=0.7,
               error_kw=dict(ecolor="#333", lw=1.2))
        for i, v in enumerate(mean):
            ax.text(i, v + ci[i] + 0.012, f"{v:.3f}", ha="center",
                    fontsize=8.5, fontweight="bold")
        ax.set_ylim(max(0.0, float(np.nanmin(mean - ci)) - 0.08),
                    min(1.06, float(np.nanmax(mean + ci)) + 0.06))
        _rotate(ax, order)
        ax.set_title(METRIC_LABELS.get(m, m))
    fig.suptitle("Classical vs quantum — mean ± 95% CI across folds", fontsize=13, y=1.02)
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_fold_distribution(runs, metric="roc_auc", outdir=None, name="fold_distribution.png"):
    """Box + paired lines. The honest view of variance."""
    set_style()
    df = long_fold_table(runs); cmap = color_map(runs)
    order = [r.name for r in runs]
    data = [df[df.model == m][metric].values for m in order]
    fig, ax = plt.subplots(figsize=(1.5 * len(order) + 3.5, 5.4))
    bp = ax.boxplot(data, positions=range(len(order)), widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.6))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(cmap[m]); patch.set_alpha(0.65); patch.set_edgecolor("black")
    for fold, sub in df.groupby("fold"):
        sub = sub.set_index("model").reindex(order)
        ax.plot(range(len(order)), sub[metric].values, color="grey", alpha=0.35, lw=0.9, zorder=0)
    for i, m in enumerate(order):
        v = df[df.model == m][metric].values
        ax.scatter(np.full(len(v), i) + np.random.uniform(-.09, .09, len(v)),
                   v, color="#222", s=22, zorder=3)
    _rotate(ax, order)
    ax.set(ylabel=METRIC_LABELS.get(metric, metric),
           title=f"Per-fold {METRIC_LABELS.get(metric, metric)} "
                 f"(grey lines pair the same fold across models)")
    return _save(fig, outdir, name)


def plot_confusion_grid(runs, outdir=None, name="confusion_matrices.png"):
    set_style()
    k = len(runs); ncol = min(4, k); nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    labels = ["No disease", "Disease"]
    for ax, r in zip(axes, runs):
        cm = confusion_matrix(r.y_true, r.y_pred, labels=[0, 1])
        norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues" if r.family == "classical" else "Oranges",
                  vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j]:.2f}\n(n={cm[i, j]})", ha="center",
                        va="center", fontsize=9.5,
                        color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks([0, 1], labels, fontsize=8.5)
        ax.set_yticks([0, 1], labels, fontsize=8.5)
        ax.set(title=r.name, xlabel="Predicted", ylabel="True")
        ax.grid(False)
    for ax in axes[k:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices — row-normalised (pooled out-of-fold)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_radar(runs, metrics=("roc_auc", "pr_auc", "sensitivity", "specificity", "f1", "mcc"),
               outdir=None, name="radar_profile.png"):
    set_style()
    df = long_fold_table(runs).groupby("model")[list(metrics)].mean()
    cmap = color_map(runs)
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw=dict(polar=True))
    for r in runs:
        v = df.loc[r.name, list(metrics)].tolist(); v += v[:1]
        lw, ls = _style_of(r)
        ax.plot(ang, v, lw=lw, ls=ls, color=cmap[r.name], label=r.name)
        ax.fill(ang, v, color=cmap[r.name], alpha=0.08)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=9)
    ax.set_ylim(max(0.0, float(np.nanmin(df.values)) - 0.12), 1.0)
    ax.set_title("Multi-metric profile (fold means)", pad=26)
    ax.legend(loc="upper right", bbox_to_anchor=(1.33, 1.12), fontsize=8.5)
    return _save(fig, outdir, name)


def plot_efficiency(runs, metric="roc_auc", outdir=None, name="efficiency_tradeoff.png"):
    set_style()
    df = long_fold_table(runs).groupby(["model", "family"], as_index=False).agg(
        score=(metric, "mean"), sd=(metric, "std"), t=("train_time_s", "mean"))
    cmap = color_map(runs)
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for _, row in df.iterrows():
        ax.errorbar(row.t, row.score, yerr=row.sd, fmt=FAMILY_MARKER.get(row.family, "o"),
                    ms=12, color=cmap[row.model], ecolor="#888", elinewidth=1, capsize=3,
                    markeredgecolor="black", markeredgewidth=0.8)
        ax.annotate(row.model, (row.t, row.score), textcoords="offset points",
                    xytext=(9, 6), fontsize=8.8)
    ax.set_xscale("log")
    ax.set(xlabel="Mean training time per fold (s, log scale)",
           ylabel=METRIC_LABELS.get(metric, metric),
           title="Performance vs computational cost\n"
                 "(upper-left is better; quantum models pay a simulation tax)")
    ax.legend(handles=[Line2D([], [], marker=FAMILY_MARKER[f], ls="", color="#444",
                              ms=10, label=f.capitalize())
                       for f in ("classical", "quantum", "hybrid") if f in set(df.family)],
              loc="lower right", fontsize=9)
    return _save(fig, outdir, name)


def plot_learning_curves(runs, outdir=None, name="learning_curves.png"):
    """The strongest legitimate QML argument: performance in the low-data regime."""
    have = [r for r in runs if r.learning_curve]
    if not have:
        return None
    set_style(); cmap = color_map(runs)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for r in have:
        lc = r.learning_curve
        n = np.asarray(lc["n_train"], float)
        mu = np.asarray(lc["mean"], float)
        sd = np.asarray(lc.get("std", np.zeros_like(mu)), float)
        lw, ls = _style_of(r)
        ax.plot(n, mu, marker=FAMILY_MARKER.get(r.family, "o"), lw=lw, ls=ls,
                color=cmap[r.name], label=r.name)
        ax.fill_between(n, mu - sd, mu + sd, color=cmap[r.name], alpha=0.13, lw=0)
    ax.set_xscale("log")
    ax.set(xlabel="Number of training samples", ylabel="ROC-AUC",
           title="Sample efficiency — performance in the low-data regime")
    ax.legend(fontsize=9, loc="lower right")
    return _save(fig, outdir, name)


def plot_convergence(runs, outdir=None, name="convergence.png"):
    have = [r for r in runs if r.history and "train_loss" in r.history]
    if not have:
        return None
    set_style(); cmap = color_map(runs)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for r in have:
        h = r.history
        ep = h.get("epoch") or list(range(1, len(h["train_loss"]) + 1))
        ax.plot(ep, h["train_loss"], color=cmap[r.name], lw=2, label=f"{r.name} — train")
        if h.get("val_loss"):
            ax.plot(ep, h["val_loss"], color=cmap[r.name], lw=1.6, ls="--",
                    alpha=0.8, label=f"{r.name} — val")
    ax.set(xlabel="Training iteration", ylabel="Loss",
           title="Variational circuit convergence")
    ax.legend(fontsize=9)
    return _save(fig, outdir, name)


def plot_calibration(runs, n_bins=10, outdir=None, name="calibration.png"):
    set_style(); cmap = color_map(runs)
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1.2, label="Perfectly calibrated")
    for r in runs:
        s = np.clip(r.y_score, 0, 1)
        if np.allclose(s.min(), s.max()):
            continue
        try:
            frac, mean_pred = calibration_curve(r.y_true, s, n_bins=n_bins,
                                                strategy="quantile")
        except Exception:
            continue
        lw, ls = _style_of(r)
        ax.plot(mean_pred, frac, marker=FAMILY_MARKER.get(r.family, "o"),
                color=cmap[r.name], lw=lw * 0.7, ls=ls, label=r.name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency of disease",
           xlim=(0, 1), ylim=(0, 1), title="Reliability diagram")
    ax.legend(fontsize=9, loc="upper left")
    return _save(fig, outdir, name)


def plot_significance(pairwise, runs, pcol="delong_p_value_holm",
                      outdir=None, name="significance_heatmap.png"):
    if pairwise.empty or pcol not in pairwise:
        return None
    set_style()
    names = [r.name for r in runs]
    P = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for _, row in pairwise.iterrows():
        P.loc[row.model_a, row.model_b] = P.loc[row.model_b, row.model_a] = row[pcol]
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 3, 1.15 * len(names) + 2.6))
    im = ax.imshow(P.values.astype(float), cmap="RdYlGn_r", vmin=0, vmax=0.20)
    for i in range(len(names)):
        for j in range(len(names)):
            v = P.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, "<0.001" if v < 1e-3 else f"{v:.3f}",
                        ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(names)), names, rotation=32, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_title("Pairwise DeLong test on ROC-AUC\n(green = significant, p < 0.05)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Holm-adjusted p", fraction=0.045)
    return _save(fig, outdir, name)


def plot_delta(champ, outdir=None, name="delta_vs_baselines.png", champion_name="Champion"):
    if champ.empty or "delong_delta_auc" not in champ:
        return None
    set_style()
    d = champ.dropna(subset=["delong_delta_auc"]).sort_values("delong_delta_auc")
    if d.empty:
        return None
    err = None
    if "delong_ci_lo" in d and d.delong_ci_lo.notna().all():
        err = np.vstack([d.delong_delta_auc - d.delong_ci_lo,
                         d.delong_ci_hi - d.delong_delta_auc])
        err = np.abs(err)
    fig, ax = plt.subplots(figsize=(7.8, 0.75 * len(d) + 2.6))
    ax.barh(d.model_b, d.delong_delta_auc, xerr=err, capsize=4, lw=0.7,
            color=["#2E8B57" if v > 0 else "#C0392B" for v in d.delong_delta_auc],
            edgecolor="black", error_kw=dict(ecolor="#333", lw=1.2))
    ax.axvline(0, color="black", lw=1.2)
    pcol = "delong_p_value_holm" if "delong_p_value_holm" in d else "delong_p_value"
    for i, (v, p) in enumerate(zip(d.delong_delta_auc, d[pcol])):
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        ax.text(v + (0.002 if v >= 0 else -0.002), i, star, va="center",
                ha="left" if v >= 0 else "right", fontsize=10, fontweight="bold")
    ax.set(xlabel="Δ ROC-AUC (champion − baseline), 95% DeLong CI",
           title=f"{champion_name} vs baselines")
    return _save(fig, outdir, name)


def plot_dashboard(runs, outdir=None, name="dashboard.png",
                   dataset_name="Biomedical dataset"):
    """The single slide to put on the projector."""
    set_style()
    cmap = color_map(runs); df = long_fold_table(runs)
    order = [r.name for r in runs]
    fig = plt.figure(figsize=(17.5, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0]); grid = np.linspace(0, 1, 200)
    for r in runs:
        mean, sd, a = _mean_roc(r, grid)
        if mean is None:
            continue
        lw, ls = _style_of(r)
        ax.plot(grid, mean, color=cmap[r.name], lw=lw * .8, ls=ls, label=f"{r.name} ({a:.3f})")
        ax.fill_between(grid, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                        color=cmap[r.name], alpha=0.10, lw=0)
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
    ax.set(title="ROC (mean ± SD)", xlabel="1 − Specificity", ylabel="Sensitivity")
    ax.legend(fontsize=7.5, loc="lower right")

    ax = fig.add_subplot(gs[0, 1])
    g = df.groupby("model")["roc_auc"]
    mean = g.mean().reindex(order).values
    ci = (1.96 * g.sem().reindex(order).fillna(0)).values
    ax.bar(range(len(order)), mean, yerr=ci, capsize=4,
           color=[cmap[k] for k in order], edgecolor="black", lw=0.7)
    ax.set_ylim(max(0, mean.min() - 0.12), min(1.03, mean.max() + 0.05))
    _rotate(ax, order, 7.5); ax.set_title("ROC-AUC (mean ± 95% CI)")

    ax = fig.add_subplot(gs[0, 2])
    data = [df[df.model == m]["roc_auc"].values for m in order]
    bp = ax.boxplot(data, positions=range(len(order)), widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.4))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(cmap[m]); patch.set_alpha(0.65)
    _rotate(ax, order, 7.5); ax.set(title="Per-fold spread", ylabel="ROC-AUC")

    ax = fig.add_subplot(gs[1, 0]); w = 0.38; x = np.arange(len(order))
    for off, m, c in ((-w / 2, "sensitivity", "#D1495B"), (w / 2, "specificity", "#3B6EA5")):
        g = df.groupby("model")[m]
        ax.bar(x + off, g.mean().reindex(order).values, w,
               yerr=(1.96 * g.sem().reindex(order).fillna(0)).values, capsize=3,
               color=c, edgecolor="black", lw=0.6, label=METRIC_LABELS[m])
    _rotate(ax, order, 7.5)
    ax.set(title="Sensitivity vs Specificity"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    eff = df.groupby(["model", "family"], as_index=False).agg(
        s=("roc_auc", "mean"), t=("train_time_s", "mean"))
    for _, row in eff.iterrows():
        ax.scatter(row.t, row.s, s=170, color=cmap[row.model],
                   marker=FAMILY_MARKER.get(row.family, "o"), edgecolor="black", zorder=3)
        ax.annotate(row.model, (row.t, row.s), textcoords="offset points",
                    xytext=(8, 5), fontsize=7.5)
    ax.set_xscale("log")
    ax.set(title="Accuracy vs cost", xlabel="Train time / fold (s)", ylabel="ROC-AUC")

    ax = fig.add_subplot(gs[1, 2])
    best = df.groupby("model")["roc_auc"].mean().idxmax()
    br = next(r for r in runs if r.name == best)
    cm = confusion_matrix(br.y_true, br.y_pred, labels=[0, 1]).astype(float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Oranges" if br.family != "classical" else "Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=10, color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks([0, 1], ["No disease", "Disease"], fontsize=8)
    ax.set_yticks([0, 1], ["No disease", "Disease"], fontsize=8)
    ax.set(title=f"Best model: {best}", xlabel="Predicted", ylabel="True"); ax.grid(False)

    fig.suptitle(f"Hybrid QML vs classical ML — {dataset_name}", fontsize=19, y=0.98)
    return _save(fig, outdir, name)


def generate_all(runs, outdir, pairwise=None, champ=None, dataset_name="Biomedical dataset"):
    figs = [plot_roc(runs, outdir), plot_pr(runs, outdir),
            plot_metric_bars(runs, outdir=outdir),
            plot_fold_distribution(runs, outdir=outdir),
            plot_confusion_grid(runs, outdir=outdir),
            plot_radar(runs, outdir=outdir),
            plot_efficiency(runs, outdir=outdir),
            plot_calibration(runs, outdir=outdir),
            plot_learning_curves(runs, outdir=outdir),
            plot_convergence(runs, outdir=outdir),
            plot_dashboard(runs, outdir=outdir, dataset_name=dataset_name)]
    if pairwise is not None:
        figs.append(plot_significance(pairwise, runs, outdir=outdir))
    if champ is not None and not champ.empty:
        figs.append(plot_delta(champ, outdir=outdir, champion_name=champ.model_a.iloc[0]))
    return [f for f in figs if isinstance(f, str)]


# =============================================================================
# 6.  EXPLAINABILITY — SHAP
#     KernelExplainer is model-agnostic: it only needs a predict function.
#     That is exactly why it works on a quantum circuit as well as on a
#     RandomForest, and it lets you put classical and quantum attributions
#     side by side on the same feature axis.
# =============================================================================

def shap_values_for(predict_fn: Callable[[np.ndarray], np.ndarray],
                    X_background: np.ndarray, X_explain: np.ndarray,
                    nsamples: int = 100, seed: int = SEED) -> Optional[np.ndarray]:
    """
    Model-agnostic SHAP values. `predict_fn` maps (n, d) -> (n,) P(disease).
    Keep X_background small (20-50 rows); KernelExplainer cost grows with it.
    """
    if not HAS_SHAP:
        return None
    np.random.seed(seed)
    try:
        expl = shap.KernelExplainer(predict_fn, X_background)
        sv = expl.shap_values(X_explain, nsamples=nsamples, silent=True)
        sv = np.asarray(sv[1] if isinstance(sv, list) and len(sv) == 2 else sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        return sv
    except Exception as e:
        print(f"[qbench] SHAP failed: {e}")
        return None


def shap_importance_frame(sv: np.ndarray, feature_names: Sequence[str],
                          model: str, family: str) -> pd.DataFrame:
    """Mean |SHAP| per feature — the standard global-importance summary."""
    imp = np.abs(sv).mean(axis=0)
    return pd.DataFrame({"model": model, "family": family,
                         "feature": list(feature_names), "mean_abs_shap": imp,
                         "mean_shap": sv.mean(axis=0)}).sort_values(
                             "mean_abs_shap", ascending=False)


def plot_shap_comparison(frames: Sequence[pd.DataFrame], outdir=None,
                         name="shap_importance.png", top_k: int = 12):
    """Grouped horizontal bars: which features does each model actually use?"""
    if not frames:
        return None
    set_style()
    df = pd.concat(frames, ignore_index=True)
    order = (df.groupby("feature")["mean_abs_shap"].mean()
             .sort_values(ascending=False).head(top_k).index.tolist())[::-1]
    models = list(dict.fromkeys(df.model))
    colors = {}
    ci = qi = 0
    for m in models:
        fam = df[df.model == m].family.iloc[0]
        if fam == "classical":
            colors[m] = CLASSICAL_COLORS[ci % len(CLASSICAL_COLORS)]; ci += 1
        else:
            colors[m] = QUANTUM_COLORS[qi % len(QUANTUM_COLORS)]; qi += 1
    h = 0.8 / len(models); y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.6, 0.46 * len(order) * len(models) + 2.4))
    for k, m in enumerate(models):
        sub = df[df.model == m].set_index("feature").reindex(order)
        ax.barh(y + (k - (len(models) - 1) / 2) * h, sub["mean_abs_shap"].values, h,
                color=colors[m], edgecolor="black", lw=0.5, label=m)
    ax.set_yticks(y, order, fontsize=9)
    ax.set(xlabel="mean |SHAP value|  (average impact on predicted disease probability)",
           title="Feature importance — classical vs quantum on identical features")
    ax.legend(fontsize=9)
    return _save(fig, outdir, name)


def plot_shap_beeswarm(sv: np.ndarray, X: np.ndarray, feature_names: Sequence[str],
                       model_name: str, outdir=None, name="shap_beeswarm.png",
                       top_k: int = 10):
    """Per-patient attributions: direction and spread, not just magnitude."""
    set_style()
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:top_k][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(order) + 2.6))
    for row, j in enumerate(order):
        v = sv[:, j]
        x = X[:, j].astype(float)
        rng = np.ptp(x)
        c = (x - x.min()) / rng if rng > 0 else np.full_like(x, 0.5)
        ax.scatter(v, row + np.random.uniform(-0.16, 0.16, len(v)),
                   c=c, cmap="coolwarm", s=20, alpha=0.85,
                   edgecolor="none", vmin=0, vmax=1)
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(range(len(order)), [feature_names[j] for j in order], fontsize=9)
    ax.set(xlabel="SHAP value  (→ pushes prediction toward DISEASE)",
           title=f"Per-patient explanations — {model_name}")
    fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax,
                 label="Feature value (low → high)", fraction=0.03)
    return _save(fig, outdir, name)


# =============================================================================
#     BENCHMARK — orchestrates stages 1-6 and writes the report
# =============================================================================

class Benchmark:
    def __init__(self, runs: Sequence[ModelRun], dataset_name="Biomedical dataset",
                 primary_metric="roc_auc"):
        if not runs:
            raise ValueError("no runs supplied")
        self.runs = list(runs)
        self.dataset_name = dataset_name
        self.primary_metric = primary_metric
        self.splits_verified = assert_same_splits(self.runs)

    @classmethod
    def from_dir(cls, directory: str, **kw) -> "Benchmark":
        return cls(load_dir(directory), **kw)

    @property
    def champion(self) -> str:
        """Best non-classical model by the primary metric."""
        df = long_fold_table(self.runs)
        q = [r.name for r in self.runs if r.family != "classical"]
        pool = df[df.model.isin(q)] if q else df
        return pool.groupby("model")[self.primary_metric].mean().idxmax()

    @property
    def best_classical(self) -> Optional[str]:
        c = [r.name for r in self.runs if r.family == "classical"]
        if not c:
            return None
        df = long_fold_table(self.runs)
        return df[df.model.isin(c)].groupby("model")[self.primary_metric].mean().idxmax()

    def run(self, outdir="report", shap_frames=None, shap_beeswarm=None) -> str:
        os.makedirs(outdir, exist_ok=True)
        figdir = os.path.join(outdir, "figures")

        summary = summary_table(self.runs)
        folds = long_fold_table(self.runs)
        pw = pairwise_comparisons(self.runs, metric=self.primary_metric)
        champ = best_vs_rest(self.runs, self.champion, metric=self.primary_metric)
        if "family_b" in champ:
            champ = champ[champ.family_b == "classical"]

        summary.to_csv(os.path.join(outdir, "summary_metrics.csv"), index=False)
        folds.to_csv(os.path.join(outdir, "per_fold_metrics.csv"), index=False)
        pw.to_csv(os.path.join(outdir, "pairwise_significance.csv"), index=False)
        champ.to_csv(os.path.join(outdir, "champion_vs_classical.csv"), index=False)
        format_headline(summary).to_csv(os.path.join(outdir, "headline_table.csv"), index=False)

        figs = generate_all(self.runs, figdir, pairwise=pw, champ=champ,
                            dataset_name=self.dataset_name)
        if shap_frames:
            pd.concat(shap_frames, ignore_index=True).to_csv(
                os.path.join(outdir, "shap_importance.csv"), index=False)
            f = plot_shap_comparison(shap_frames, outdir=figdir)
            if f:
                figs.append(f)
        if shap_beeswarm:
            f = plot_shap_beeswarm(*shap_beeswarm, outdir=figdir)
            if f:
                figs.append(f)

        path = os.path.join(outdir, "REPORT.md")
        # encoding is explicit: the report contains ±, —, Δ, which crash the
        # default cp1252 codec on Windows.
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self._markdown(summary, champ, figs, outdir))
        print(f"[qbench] wrote {len(figs)} figures + CSVs + REPORT.md to {outdir}/")
        return path

    def _verdict(self, champ: pd.DataFrame) -> str:
        if champ.empty or "delong_p_value_holm" not in champ:
            return "Not enough paired information to test significance."
        wins = champ[(champ.delong_delta_auc > 0) & (champ.delong_p_value_holm < .05)]
        losses = champ[(champ.delong_delta_auc < 0) & (champ.delong_p_value_holm < .05)]
        n = len(champ)
        if len(wins) == n:
            return (f"**{self.champion} significantly outperforms all {n} classical "
                    f"baselines** on ROC-AUC (Holm-adjusted DeLong p < 0.05).")
        if len(wins):
            return (f"**{self.champion} significantly beats {len(wins)}/{n} classical "
                    f"baselines**; the rest are statistically indistinguishable.")
        if len(losses):
            return (f"{self.champion} is significantly *worse* than {len(losses)}/{n} "
                    f"baselines. Report this honestly and analyse why: feature-map "
                    f"expressivity, encoding, kernel scaling, qubit budget.")
        return (f"No significant difference between {self.champion} and the classical "
                f"baselines — the quantum model is **competitive but not superior** at "
                f"this sample size. That is a legitimate, defensible finding.")

    def _markdown(self, summary, champ, figs, outdir) -> str:
        top = summary.iloc[0]
        cols = [c for c in ["model_b", "delong_delta_auc", "delong_p_value_holm",
                            "mcnemar_p_value", "fold_wins", "fold_losses", "fold_mean_diff"]
                if c in champ]
        try:
            head_md = format_headline(summary).to_markdown(index=False)
            champ_md = champ[cols].round(4).to_markdown(index=False) if not champ.empty else "_n/a_"
        except ImportError:                       # tabulate missing
            head_md = format_headline(summary).to_string(index=False)
            champ_md = champ[cols].round(4).to_string(index=False) if not champ.empty else "_n/a_"
        fig_md = "\n".join(
            f"### {os.path.basename(p).replace('_', ' ').replace('.png', '').title()}\n"
            f"![{os.path.basename(p)}](figures/{os.path.basename(p)})\n" for p in figs)
        splits = "verified" if self.splits_verified else "NOT verified — supply test_index"
        return "\n".join([
            "# Benchmark Report — Hybrid QML vs Classical ML", "",
            f"**Dataset:** {self.dataset_name}  ",
            f"**Models compared:** {len(self.runs)}  ",
            f"**Evaluation:** {len(self.runs[0].folds)}-fold stratified CV, "
            f"identical splits for every model ({splits})  ",
            f"**Primary metric:** {METRIC_LABELS.get(self.primary_metric, self.primary_metric)}",
            "", "## Verdict", "", self._verdict(champ), "",
            f"Best overall: **{top.model}** ({top.family}), pooled ROC-AUC "
            f"{top.roc_auc_pooled:.3f} (95% CI {top.roc_auc_ci_lo:.3f}–{top.roc_auc_ci_hi:.3f}).  ",
            f"Best classical baseline: **{self.best_classical}**.",
            "", "## Headline results", "", head_md,
            "", f"## Champion ({self.champion}) vs classical baselines", "", champ_md, "",
            "`delong_delta_auc` = champion AUC − baseline AUC (positive favours the "
            "champion). `fold_wins/losses` = folds the champion won/lost. p-values are "
            "Holm-corrected for multiple comparisons.",
            "", "## Figures", "", fig_md,
            "", "## Reproducibility", "",
            "Per-fold numbers are in `per_fold_metrics.csv`; circuit metadata in "
            "`results/runs_meta.csv`. Regenerate with:", "",
            "```bash", f"python qbench.py --results results/ --out {outdir}/ --no-demo",
            "```", "",
        ])


# =============================================================================
#     REFERENCE QUANTUM MODELS (pure NumPy, exact statevector)
#     These exist so the harness is testable before the QML teammate's
#     PennyLane/Qiskit code lands — and as a fallback if it misbehaves at 3am.
#     Both are real simulations, not mock-ups. Swap in the team's models by
#     just producing a ModelRun.
# =============================================================================

def _hadamard_matrix(n: int) -> np.ndarray:
    d = 1 << n
    idx = np.arange(d)
    pc = np.zeros((d, d), dtype=int)
    tmp = np.bitwise_and(idx[:, None], idx[None, :])
    while tmp.any():
        pc += tmp & 1
        tmp >>= 1
    return (2.0 ** (-n / 2)) * np.where(pc % 2 == 0, 1.0, -1.0)


class ZZFeatureMapKernel:
    """
    Fidelity quantum kernel, Havlicek ZZ feature map. K(x,x') = |<phi(x')|phi(x)>|^2.
    Drop-in for SVC(kernel='precomputed').

    `scale` is the single most important knob: an unscaled kernel collapses toward
    the identity matrix and the SVM just memorises the training set. Tune it.
    """

    def __init__(self, n_qubits: int, reps: int = 2, entanglement: str = "full",
                 scale: float = 1.0):
        if n_qubits > 14:
            raise ValueError("statevector simulation beyond 14 qubits is impractical")
        self.n_qubits, self.reps, self.scale = n_qubits, reps, scale
        self._H = _hadamard_matrix(n_qubits)
        d = 1 << n_qubits
        bits = (np.arange(d)[:, None] >> np.arange(n_qubits - 1, -1, -1)) & 1
        self._Z = 1.0 - 2.0 * bits                       # +1 for |0>, -1 for |1>
        self.pairs = (list(combinations(range(n_qubits), 2)) if entanglement == "full"
                      else [(i, i + 1) for i in range(n_qubits - 1)])

    @property
    def circuit_depth(self) -> int:
        return self.reps * (2 + 3 * len(self.pairs))

    @property
    def n_gates(self) -> int:
        return self.reps * (self.n_qubits * 2 + 3 * len(self.pairs))

    def _phases(self, X):
        Z = self._Z
        out = -(X @ Z.T)                                  # RZ(2 x_i) -> phase -x_i z_i
        pi_x = np.pi - X
        for (i, j) in self.pairs:
            out -= np.outer(pi_x[:, i] * pi_x[:, j], Z[:, i] * Z[:, j])
        return out

    def statevectors(self, X):
        X = np.asarray(X, float) * self.scale
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"expected {self.n_qubits} features, got {X.shape[1]}")
        d = 1 << self.n_qubits
        psi = np.full((X.shape[0], d), 1.0 / np.sqrt(d), dtype=complex)
        for r in range(self.reps):
            if r > 0:
                psi = psi @ self._H.T                     # H^{\otimes n} layer
            psi = psi * np.exp(1j * self._phases(X))
        return psi

    def __call__(self, A, B=None):
        pa = self.statevectors(A)
        pb = pa if B is None else self.statevectors(B)
        return np.abs(pa @ pb.conj().T) ** 2


def suggest_scale(X, n_qubits=4, reps=2, candidates=(0.1, 0.25, 0.5, 1.0, 2.0)):
    """Pick the scale whose kernel is neither near-identity nor near-constant."""
    diag, best, best_spread = {}, candidates[0], -np.inf
    sub = X[np.random.default_rng(0).choice(len(X), min(120, len(X)), replace=False)]
    for s in candidates:
        K = ZZFeatureMapKernel(n_qubits, reps=reps, scale=s)(sub)
        off = K[~np.eye(len(sub), dtype=bool)]
        diag[s] = {"mean_offdiag": float(off.mean()), "std_offdiag": float(off.std())}
        if off.std() > best_spread:
            best, best_spread = s, off.std()
    return best, diag


def _apply_1q(psi, U, q, n):
    B = psi.shape[0]
    v = psi.reshape(B, 1 << q, 2, 1 << (n - q - 1))
    return np.einsum("ab,ilbr->ilar", U, v).reshape(B, 1 << n)


def _cnot_perm(control, target, n):
    idx = np.arange(1 << n)
    return np.where((idx >> (n - 1 - control)) & 1, idx ^ (1 << (n - 1 - target)), idx)


class VariationalQuantumClassifier:
    """
    Angle encoding -> RY/CNOT-ring ansatz -> <Z_0> readout, trained with COBYLA
    (gradient-free, which is what you would realistically use on hardware).
    Exposes `.history_` so the benchmark can draw the convergence curve.
    """

    def __init__(self, n_qubits: int, n_layers: int = 3, maxiter: int = 160,
                 seed: int = 0, encoding_scale: float = np.pi):
        self.n_qubits, self.n_layers, self.maxiter = n_qubits, n_layers, maxiter
        self.seed, self.encoding_scale = seed, encoding_scale
        self._ring = [(i, (i + 1) % n_qubits) for i in range(n_qubits)] if n_qubits > 1 else []
        self._perms = [_cnot_perm(c, t, n_qubits) for c, t in self._ring]
        self.history_ = {"epoch": [], "train_loss": []}

    @property
    def n_params(self):     return self.n_qubits * self.n_layers
    @property
    def circuit_depth(self): return 1 + self.n_layers * (1 + len(self._ring))

    def _forward(self, X, theta):
        n, B = self.n_qubits, X.shape[0]
        psi = np.zeros((B, 1 << n), dtype=complex); psi[:, 0] = 1.0
        ang = self.encoding_scale * X                      # data-dependent -> batched RY
        for q in range(n):
            c, s = np.cos(ang[:, q] / 2), np.sin(ang[:, q] / 2)
            v = psi.reshape(B, 1 << q, 2, 1 << (n - q - 1))
            a, b = v[:, :, 0, :], v[:, :, 1, :]
            new = np.empty_like(v)
            new[:, :, 0, :] = c[:, None, None] * a - s[:, None, None] * b
            new[:, :, 1, :] = s[:, None, None] * a + c[:, None, None] * b
            psi = new.reshape(B, 1 << n)
        p = theta.reshape(self.n_layers, n)
        for l in range(self.n_layers):
            for q in range(n):
                th = p[l, q]
                c, s = np.cos(th / 2), np.sin(th / 2)
                psi = _apply_1q(psi, np.array([[c, -s], [s, c]], dtype=complex), q, n)
            for perm in self._perms:
                psi = psi[:, perm]
        probs = np.abs(psi) ** 2
        sign = np.where((np.arange(1 << n) >> (n - 1)) & 1, -1.0, 1.0)
        return probs @ sign

    def _proba(self, X, theta):
        return np.clip((1.0 + self._forward(X, theta)) / 2.0, 1e-7, 1 - 1e-7)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        rng = np.random.default_rng(self.seed)
        theta0 = rng.uniform(0, 2 * np.pi, self.n_params)
        # class weights: a missed disease costs more than a false alarm
        w = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
        self.history_ = {"epoch": [], "train_loss": []}

        def loss(t):
            p = self._proba(X, t)
            L = float(-np.mean(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))
            self.history_["epoch"].append(len(self.history_["epoch"]) + 1)
            self.history_["train_loss"].append(L)
            return L

        self.theta_ = minimize(loss, theta0, method="COBYLA",
                               options={"maxiter": self.maxiter, "rhobeg": 0.7}).x
        return self

    def predict_proba(self, X):
        p = self._proba(np.asarray(X, float), self.theta_)
        return np.column_stack([1 - p, p])

    def decision_function(self, X): return self.predict_proba(X)[:, 1]
    def predict(self, X, threshold=0.5):
        return (self.decision_function(X) >= threshold).astype(int)


# =============================================================================
#     DATASET LOADING — CSV first, sklearn built-in as fallback
# =============================================================================

DEFAULT_DATA_FILES = ("breast_cancer_wisconsin.csv", "data/breast_cancer_wisconsin.csv")

# Columns never used as features. "target" and "diagnosis" are BOTH listed because
# this CSV carries the label twice (numeric + M/B); leaving either one in X would be
# straight label leakage and would hand every model a perfect 1.000 AUC.
DEFAULT_DROP_COLS = ("sample_index", "diagnosis", "target", "label", "class",
                     "id", "ID", "Unnamed: 0", "index")


def load_dataset(path: Optional[str] = None, target_col: str = "target",
                 positive_label: Any = None,
                 drop_cols: Sequence[str] = DEFAULT_DROP_COLS):
    """
    Returns (X, y, feature_names, dataset_name) with y in {0, 1}, 1 = disease.

    Resolution order:
      1. `path` if given
      2. breast_cancer_wisconsin.csv in the working directory (or ./data/)
      3. sklearn's built-in load_breast_cancer()

    Works for any binary-outcome CSV: non-numeric and ID-like columns are
    dropped automatically, and a string target (e.g. "M"/"B") is binarised.
    Set `positive_label` to say which class means disease; otherwise the
    minority class is taken as positive, which is the usual clinical setup.
    """
    if path is None:
        for cand in DEFAULT_DATA_FILES:
            if os.path.exists(cand):
                path = cand
                break

    if path is None:                                   # fallback: sklearn built-in
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        y = (d.target == 0).astype(int)                # RULE 2: 1 = malignant
        return d.data, y, list(d.feature_names), "Wisconsin Breast Cancer (sklearn built-in)"

    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"{path}: no column named {target_col!r}. "
                         f"Columns present: {list(df.columns)[:12]} ... "
                         f"Pass --target-col to point at the label column.")

    raw = df[target_col]
    if raw.dtype == object or raw.nunique() != 2 or not set(raw.unique()) <= {0, 1}:
        vals = raw.dropna().unique()
        if len(vals) != 2:
            raise ValueError(f"{target_col!r} must be binary; found {len(vals)} values: {vals[:6]}")
        if positive_label is None:                     # minority class = disease
            positive_label = raw.value_counts().idxmin()
        y = (raw == positive_label).astype(int).to_numpy()
        print(f"[data] binarised {target_col!r}: {positive_label!r} -> 1 (disease)")
    else:
        y = raw.astype(int).to_numpy()

    drop = [c for c in list(drop_cols) + [target_col] if c in df.columns]
    leaky = [c for c in drop if c != target_col and c not in
             ("sample_index", "id", "ID", "Unnamed: 0", "index")]
    if leaky:
        print(f"[data] dropping possible duplicate-label columns (leakage guard): {leaky}")
    Xdf = df.drop(columns=drop)
    non_numeric = [c for c in Xdf.columns if not pd.api.types.is_numeric_dtype(Xdf[c])]
    if non_numeric:
        print(f"[data] dropping non-numeric columns: {non_numeric}")
        Xdf = Xdf.drop(columns=non_numeric)
    if Xdf.isna().any().any():
        n = int(Xdf.isna().sum().sum())
        print(f"[data] imputing {n} missing values with column medians")
        Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()
    return Xdf.to_numpy(dtype=float), y, list(Xdf.columns), name


# =============================================================================
#     DEMO — 4 classical baselines + 2 real quantum models, end to end
# =============================================================================

def run_demo(n_folds=5, n_qubits=5, results_dir="results", out_dir="report",
             do_learning_curve=True, do_shap=True,
             data_path=None, target_col="target", dataset_name=None):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.svm import SVC

    X, y, feature_names, auto_name = load_dataset(data_path, target_col=target_col)
    dataset_name = dataset_name or auto_name
    print(f"[demo] dataset='{dataset_name}'  X={X.shape}  "
          f"prevalence(disease)={y.mean():.3f}  ({int(y.sum())} positive / "
          f"{int((1 - y).sum())} negative)")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)  # RULE 1

    # RULE 3: every transform is fit on the training fold only
    def reduce_q(Xtr, Xte, k):
        sc = StandardScaler().fit(Xtr)
        pca = PCA(n_components=k, random_state=SEED).fit(sc.transform(Xtr))
        mm = MinMaxScaler((0, np.pi)).fit(pca.transform(sc.transform(Xtr)))
        f = lambda A: mm.transform(pca.transform(sc.transform(A)))
        return f(Xtr), np.clip(f(Xte), 0, np.pi)

    def run_classical(name, est):
        folds = []
        for k, (tr, te) in enumerate(cv.split(X, y)):
            m = Pipeline([("sc", StandardScaler()), ("clf", est)])
            t0 = time.perf_counter(); m.fit(X[tr], y[tr]); tt = time.perf_counter() - t0
            t0 = time.perf_counter(); s = m.predict_proba(X[te])[:, 1]
            ti = time.perf_counter() - t0
            folds.append(FoldResult(k, y[te], (s >= .5).astype(int), s, tt, ti, te))
        return ModelRun(name, "classical", folds,
                        meta={"n_features": X.shape[1], "estimator": type(est).__name__})

    def run_qsvm(scale=0.5, C=2.0, reps=2):
        kern = ZZFeatureMapKernel(n_qubits, reps=reps, scale=scale)
        folds = []
        for k, (tr, te) in enumerate(cv.split(X, y)):
            Qtr, Qte = reduce_q(X[tr], X[te], n_qubits)
            t0 = time.perf_counter()
            svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(kern(Qtr), y[tr])
            tt = time.perf_counter() - t0
            t0 = time.perf_counter(); d = svc.decision_function(kern(Qte, Qtr))
            ti = time.perf_counter() - t0
            s = 1.0 / (1.0 + np.exp(-d))                  # squash for calibration plot
            folds.append(FoldResult(k, y[te], (s >= .5).astype(int), s, tt, ti, te))
        return ModelRun(f"QSVM (ZZ kernel, {n_qubits}q)", "quantum", folds,
                        meta={"n_qubits": n_qubits, "reps": reps,
                              "circuit_depth": kern.circuit_depth, "n_gates": kern.n_gates,
                              "feature_map": f"ZZFeatureMap(full, reps={reps})",
                              "encoding": "PCA -> MinMax[0,pi] -> ZZ",
                              "backend": "numpy statevector", "kernel_scale": scale, "C": C})

    def run_vqc(n_layers=3, maxiter=160):
        folds, hist, depth, npar = [], None, None, None
        for k, (tr, te) in enumerate(cv.split(X, y)):
            Qtr, Qte = reduce_q(X[tr], X[te], n_qubits)
            Qtr, Qte = Qtr / np.pi, Qte / np.pi
            clf = VariationalQuantumClassifier(n_qubits, n_layers, maxiter, seed=SEED + k)
            t0 = time.perf_counter(); clf.fit(Qtr, y[tr]); tt = time.perf_counter() - t0
            t0 = time.perf_counter(); s = clf.decision_function(Qte)
            ti = time.perf_counter() - t0
            folds.append(FoldResult(k, y[te], (s >= .5).astype(int), s, tt, ti, te))
            if hist is None:
                hist = dict(clf.history_); depth, npar = clf.circuit_depth, clf.n_params
        return ModelRun(f"VQC ({n_qubits}q, {n_layers}L)", "hybrid", folds, history=hist,
                        meta={"n_qubits": n_qubits, "n_layers": n_layers, "n_params": npar,
                              "circuit_depth": depth, "ansatz": "RY + CNOT ring",
                              "optimizer": "COBYLA", "backend": "numpy statevector"})

    runs = []
    print("[demo] classical baselines ...")
    runs.append(run_classical("Logistic Regression",
                              LogisticRegression(max_iter=2000, class_weight="balanced")))
    runs.append(run_classical("SVM (RBF)",
                              SVC(probability=True, class_weight="balanced", random_state=SEED)))
    runs.append(run_classical("Random Forest",
                              RandomForestClassifier(n_estimators=300, random_state=SEED,
                                                     class_weight="balanced")))
    runs.append(run_classical("Gradient Boosting", GradientBoostingClassifier(random_state=SEED)))
    print("[demo] quantum kernel SVM ...");   runs.append(run_qsvm())
    print("[demo] variational quantum classifier ..."); runs.append(run_vqc())

    # ---- learning curves: where QML has its strongest legitimate claim ----
    if do_learning_curve:
        print("[demo] learning curves ...")
        sizes = [20, 40, 80, 160, 320]
        kern = ZZFeatureMapKernel(n_qubits, reps=2, scale=0.5)

        def lc_classical(est, n_rep=5):
            mu, sd = [], []
            for n in sizes:
                v = []
                for r in range(n_rep):
                    a, b, ya, yb = train_test_split(X, y, train_size=n, stratify=y,
                                                    random_state=SEED + r)
                    m = Pipeline([("sc", StandardScaler()), ("clf", est)]).fit(a, ya)
                    v.append(roc_auc_score(yb, m.predict_proba(b)[:, 1]))
                mu.append(float(np.mean(v))); sd.append(float(np.std(v)))
            return {"n_train": sizes, "mean": mu, "std": sd}

        def lc_qsvm(n_rep=5):
            mu, sd = [], []
            for n in sizes:
                v = []
                for r in range(n_rep):
                    a, b, ya, yb = train_test_split(X, y, train_size=n, stratify=y,
                                                    random_state=SEED + r)
                    sel = np.random.default_rng(r).choice(len(b), min(200, len(b)), replace=False)
                    b, yb = b[sel], yb[sel]
                    Qa, Qb = reduce_q(a, b, n_qubits)
                    svc = SVC(kernel="precomputed", C=2.0,
                              class_weight="balanced").fit(kern(Qa), ya)
                    v.append(roc_auc_score(yb, svc.decision_function(kern(Qb, Qa))))
                mu.append(float(np.mean(v))); sd.append(float(np.std(v)))
            return {"n_train": sizes, "mean": mu, "std": sd}

        lcc = lc_classical(LogisticRegression(max_iter=2000, class_weight="balanced"))
        lcq = lc_qsvm()
        for r in runs:
            if r.name == "Logistic Regression":
                r.learning_curve = lcc
            if r.name.startswith("QSVM"):
                r.learning_curve = lcq

    written = save_runs(runs, results_dir)
    for v in written.values():
        print(f"[demo] wrote {v}")

    # ---- SHAP on IDENTICAL features (RULE 4) so attributions are comparable ----
    shap_frames, beeswarm = None, None
    if do_shap and HAS_SHAP:
        print("[demo] SHAP explainability (classical and quantum on the same features) ...")
        tr, te = next(iter(cv.split(X, y)))
        Qtr, Qte = reduce_q(X[tr], X[te], n_qubits)
        names = [f"PC{i + 1}" for i in range(n_qubits)]
        rng = np.random.default_rng(SEED)
        bg = Qtr[rng.choice(len(Qtr), min(30, len(Qtr)), replace=False)]
        Xe = Qte[rng.choice(len(Qte), min(60, len(Qte)), replace=False)]

        rf = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                    class_weight="balanced").fit(Qtr, y[tr])
        kern = ZZFeatureMapKernel(n_qubits, reps=2, scale=0.5)
        qsvc = SVC(kernel="precomputed", C=2.0, class_weight="balanced").fit(kern(Qtr), y[tr])
        q_predict = lambda A: 1.0 / (1.0 + np.exp(-qsvc.decision_function(kern(np.atleast_2d(A), Qtr))))

        frames = []
        sv_rf = shap_values_for(lambda A: rf.predict_proba(A)[:, 1], bg, Xe, nsamples=120)
        if sv_rf is not None:
            frames.append(shap_importance_frame(sv_rf, names,
                                                "Random Forest (PCA feats)", "classical"))
        sv_q = shap_values_for(q_predict, bg, Xe, nsamples=120)
        if sv_q is not None:
            frames.append(shap_importance_frame(sv_q, names,
                                                f"QSVM ({n_qubits}q)", "quantum"))
            beeswarm = (sv_q, Xe, names, f"QSVM ({n_qubits}q, ZZ kernel)")
        shap_frames = frames or None
    elif do_shap:
        print("[demo] shap not installed — skipping explainability (pip install shap)")

    Benchmark(runs, dataset_name=f"{dataset_name} (disease = positive)").run(
        out_dir, shap_frames=shap_frames, shap_beeswarm=beeswarm)
    return runs


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Benchmark classical ML vs hybrid QML.")
    ap.add_argument("--results", default="results", help="folder for/of prediction CSVs")
    ap.add_argument("--out", default="report", help="folder for figures + report")
    ap.add_argument("--dataset", default="Biomedical dataset")
    ap.add_argument("--metric", default="roc_auc")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--qubits", type=int, default=5)
    ap.add_argument("--data", default=None,
                    help="path to the dataset CSV (default: breast_cancer_wisconsin.csv "
                         "in the working directory, else sklearn's built-in copy)")
    ap.add_argument("--target-col", default="target", help="label column in --data")
    ap.add_argument("--no-demo", action="store_true",
                    help="skip the demo; just benchmark the CSVs already in --results")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--no-learning-curve", action="store_true")
    ap.add_argument("--template", action="store_true",
                    help="write predictions_template.csv and exit")
    a = ap.parse_args()

    if a.template:
        print("[qbench] wrote", blank_template()); return
    if a.no_demo:
        Benchmark.from_dir(a.results, dataset_name=a.dataset,
                           primary_metric=a.metric).run(a.out)
    else:
        run_demo(n_folds=a.folds, n_qubits=a.qubits, results_dir=a.results,
                 out_dir=a.out, do_learning_curve=not a.no_learning_curve,
                 do_shap=not a.no_shap, data_path=a.data, target_col=a.target_col,
                 dataset_name=None if a.dataset == "Biomedical dataset" else a.dataset)


if __name__ == "__main__":
    main()
