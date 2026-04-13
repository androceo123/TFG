from __future__ import annotations

"""
waf_ml.scripts.train_multilabel_lr_harvard_thesis

Multilabel OvR Logistic Regression for Harvard / SR-BH style data, optimized to stay
as close as possible to the thesis/proposal requirements while remaining practical at scale.

What this script enforces:
- fixed outer 80/20 split with fixed seed
- tuning ONLY inside the training block
- supervised tuning via internal CV (bounded random search)
- train-only fitting of scalers / binarizers / transforms (no leakage)
- multilabel metrics: Hamming Loss, Jaccard, Exact Match Ratio, F1 micro/macro
- optional reduced-binary compatibility for datasets whose multilabel column has a single possible positive label
- per-label analysis + prevalence/cardinality reports before/after split
- coefficient-based interpretability
- automatic ablation with/without suspicious-token features
- feasibility benchmark with latency/throughput/CPU/RAM/model size and an approximate load profile
"""

import argparse
import inspect
import json
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:
    import resource
except Exception:  # pragma: no cover
    resource = None

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import accuracy_score as subset_accuracy_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

try:  # optional approximate multilabel stratification
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit  # type: ignore
except Exception:  # pragma: no cover
    MultilabelStratifiedKFold = None
    MultilabelStratifiedShuffleSplit = None

try:  # package execution
    from waf_ml.features.http_features import extract_http_features
except Exception:  # pragma: no cover
    from http_features import extract_http_features


warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\\.linear_model\\._logistic")
warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*'n_jobs' has no effect.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*Label not .* is present in all training examples.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\\.multiclass")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module=r"sklearn")

DEFAULT_FEATURES = [
    "uri_len",
    "path_depth",
    "query_len",
    "n_query_params",
    "max_param_value_len",
    "uri_pct_non_alnum_ratio",
    "encoded",
    "suspicious_tokens_count",
    "has_suspicious_tokens",
    "uncommon_method",
    "req_content_length",
    "body_len",
    "body_suspicious_tokens_count",
    "body_has_suspicious_tokens",
    "body_encoded",
    "method_GET",
    "method_POST",
    "method_HEAD",
    "method_PUT",
    "method_DELETE",
    "method_PATCH",
    "method_OPTIONS",
    "method_TRACE",
    "method_CONNECT",
    "method_OTHER",
]

RAW_METHOD_COL = "request_http_method"
RAW_URI_COL = "request_http_request"
RAW_BODY_COL = "request_body"
RAW_HEADERS_COL = "request_headers_json"

SUSPICIOUS_TOKEN_FEATURES = [
    "suspicious_tokens_count",
    "has_suspicious_tokens",
    "body_suspicious_tokens_count",
    "body_has_suspicious_tokens",
]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def _resolve_features(drop_features_csv: Optional[str]) -> List[str]:
    drop = {s.strip() for s in str(drop_features_csv or "").split(",") if s.strip()}
    feats = [f for f in DEFAULT_FEATURES if f not in drop]
    if not feats:
        raise ValueError("No features left after applying --drop-features.")
    unknown = sorted(drop - set(DEFAULT_FEATURES))
    if unknown:
        raise ValueError(f"Unknown features in --drop-features: {unknown}")
    return feats


def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_csv(path)


def _save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _safe_name(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_")
    return s or "CLASS"


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            return None
        return fv
    except Exception:
        return None


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _save_json(data: Dict[str, object], path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"Saved JSON: {path}")


def _parse_csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in str(s).split(",") if p.strip()]


def _parse_float_grid(s: Optional[str]) -> List[float]:
    return [float(t) for t in _parse_csv_list(s)]


def _parse_optional_class_weight(s: Optional[str]):
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in {"", "none", "null", "false"}:
        return None
    if t == "balanced":
        return "balanced"
    raise ValueError(f"Unsupported class_weight='{s}'. Use none|balanced.")


def _parse_optional_penalty(s: Optional[str]):
    if s is None:
        return "l2"
    t = str(s).strip().lower()
    if t in {"", "auto"}:
        return "l2"
    if t in {"none", "null"}:
        return None
    if t in {"l1", "l2", "elasticnet"}:
        return t
    raise ValueError(f"Unsupported penalty='{s}'.")


def _save_search_df(df: pd.DataFrame, path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved search results: {path}")


def _with_suffix(path: Optional[str], suffix: str) -> Optional[str]:
    if not path:
        return None
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


# -----------------------------------------------------------------------------
# Feature extraction helpers for benchmark
# -----------------------------------------------------------------------------

def _to_bytes(v) -> bytes:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return b""
    s = str(v)
    if s in {"", "nan", "None"}:
        return b""
    return s.encode("utf-8", errors="ignore")


def _json_to_headers(v) -> Dict[str, str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return {}
    s = str(v).strip()
    if not s or s in {"nan", "None"}:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _can_extract_raw(df: pd.DataFrame) -> bool:
    return RAW_METHOD_COL in df.columns and RAW_URI_COL in df.columns and RAW_BODY_COL in df.columns


def _features_from_raw_row(row: pd.Series, features: List[str]) -> pd.DataFrame:
    feat = extract_http_features(
        method=str(row.get(RAW_METHOD_COL, "")),
        uri=str(row.get(RAW_URI_COL, "")),
        headers=_json_to_headers(row.get(RAW_HEADERS_COL, "")),
        body=_to_bytes(row.get(RAW_BODY_COL, "")),
    )
    return pd.DataFrame([{c: feat.get(c, 0) for c in features}])


def _ensure_features(df: pd.DataFrame, features: List[str]) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature columns: {missing}")


# -----------------------------------------------------------------------------
# Multilabel parsing / encoding
# -----------------------------------------------------------------------------

def _parse_multilabel_cell(v) -> List[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, tuple):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
    if s == "":
        return []
    if s.startswith("[") and s.endswith("]"):
        s2 = s[1:-1].strip()
        if not s2:
            return []
        parts = [p.strip().strip("'\"") for p in s2.split(",")]
        return [p for p in parts if p]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _label_union(label_lists: Sequence[Sequence[str]]) -> List[str]:
    labs = sorted({str(l) for row in label_lists for l in row if str(l).strip()})
    return labs


def _fit_train_only_mlb(train_lists: Sequence[Sequence[str]]) -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer()
    mlb.fit(train_lists)
    return mlb


def _encode_with_known_classes(label_lists: Sequence[Sequence[str]], classes: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    cls = [str(c) for c in classes]
    pos = {c: i for i, c in enumerate(cls)}
    y = np.zeros((len(label_lists), len(cls)), dtype=np.int8)
    unseen: Dict[str, int] = {}
    for r, row in enumerate(label_lists):
        for lab in row:
            lab = str(lab)
            idx = pos.get(lab)
            if idx is None:
                unseen[lab] = unseen.get(lab, 0) + 1
            else:
                y[r, idx] = 1
    return y, unseen


def _label_sets_from_binary_matrix(y: np.ndarray, labels: Sequence[str]) -> List[List[str]]:
    labs = [str(x) for x in labels]
    arr = np.asarray(y, dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out: List[List[str]] = []
    for row in arr:
        row_arr = np.atleast_1d(np.asarray(row, dtype=int)).reshape(-1)
        idx = np.where(row_arr == 1)[0].tolist()
        out.append([labs[i] for i in idx if i < len(labs)])
    return out


def _cardinality_stats(y: np.ndarray, n_labels: int) -> Dict[str, object]:
    y_arr = np.asarray(y, dtype=int)
    if y_arr.size == 0:
        return {
            "rows": 0,
            "mean_labels_per_instance": None,
            "median_labels_per_instance": None,
            "max_labels_per_instance": None,
            "proportion_empty_instances": None,
            "proportion_multilabel_instances": None,
            "label_density": None,
        }
    per_row = y_arr.sum(axis=1)
    return {
        "rows": int(len(y_arr)),
        "mean_labels_per_instance": _safe_float(per_row.mean()),
        "median_labels_per_instance": _safe_float(np.median(per_row)),
        "max_labels_per_instance": int(per_row.max()) if len(per_row) else None,
        "proportion_empty_instances": _safe_float(np.mean(per_row == 0)),
        "proportion_multilabel_instances": _safe_float(np.mean(per_row > 1)),
        "label_density": _safe_float(per_row.mean() / float(n_labels)) if n_labels > 0 else None,
    }


def _label_distribution(y: np.ndarray, labels: Sequence[str]) -> List[Dict[str, object]]:
    y_arr = np.asarray(y, dtype=int)
    n = len(y_arr)
    out: List[Dict[str, object]] = []
    for i, lab in enumerate(labels):
        support = int(y_arr[:, i].sum()) if n else 0
        out.append({
            "label": str(lab),
            "support": support,
            "prevalence": _safe_float(support / n) if n > 0 else None,
        })
    out.sort(key=lambda d: (d["support"], d["label"]), reverse=True)
    return out


# -----------------------------------------------------------------------------
# LR / OvR construction
# -----------------------------------------------------------------------------

def _is_valid_lr_combo(solver: str, penalty, l1_ratio) -> bool:
    if solver == "lbfgs":
        return penalty in {"l2", None}
    if solver == "liblinear":
        return penalty in {"l1", "l2"}
    if solver == "newton-cg":
        return penalty in {"l2", None}
    if solver == "newton-cholesky":
        return penalty in {"l2", None}
    if solver == "sag":
        return penalty in {"l2", None}
    if solver == "saga":
        if penalty == "elasticnet":
            return l1_ratio is not None
        return penalty in {"l1", "l2", None}
    return False


def _build_logreg(*, solver: str, penalty, C: float, class_weight, max_iter: int, l1_ratio, tol: float) -> LogisticRegression:
    sig = inspect.signature(LogisticRegression.__init__)
    solver_in = str(solver or "lbfgs").strip().lower()
    penalty_in = _parse_optional_penalty(penalty)
    class_weight_in = _parse_optional_class_weight(class_weight)
    if not _is_valid_lr_combo(solver_in, penalty_in, l1_ratio):
        raise ValueError(f"Invalid LogisticRegression combo: solver={solver_in}, penalty={penalty_in}, l1_ratio={l1_ratio}")

    kwargs = {
        "solver": solver_in,
        "C": float(C),
        "max_iter": int(max_iter),
        "class_weight": class_weight_in,
        "tol": float(tol),
    }
    if "multi_class" in sig.parameters:
        kwargs["multi_class"] = "auto"
    if "penalty" in sig.parameters:
        kwargs["penalty"] = penalty_in
    if penalty_in == "elasticnet" and solver_in == "saga" and "l1_ratio" in sig.parameters:
        kwargs["l1_ratio"] = float(l1_ratio)
    return LogisticRegression(**kwargs)


def _build_multilabel_pipe(args, *, params: Optional[Dict[str, object]] = None, ovr_n_jobs: Optional[int] = None) -> Pipeline:
    params = dict(params or {})
    solver = params.get("ovr__estimator__solver", args.solver)
    penalty = params.get("ovr__estimator__penalty", args.penalty)
    C = params.get("ovr__estimator__C", args.C)
    class_weight = params.get("ovr__estimator__class_weight", args.class_weight)
    l1_ratio = params.get("ovr__estimator__l1_ratio", args.l1_ratio)

    base_lr = _build_logreg(
        solver=str(solver),
        penalty=penalty,
        C=float(C),
        class_weight=class_weight,
        max_iter=args.max_iter,
        l1_ratio=l1_ratio,
        tol=args.tol,
    )
    ovr_sig = inspect.signature(OneVsRestClassifier.__init__)
    ovr_kwargs = {}
    if "n_jobs" in ovr_sig.parameters and ovr_n_jobs is not None:
        ovr_kwargs["n_jobs"] = int(ovr_n_jobs)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ovr", OneVsRestClassifier(base_lr, **ovr_kwargs)),
    ])


# -----------------------------------------------------------------------------
# Metrics / explainability
# -----------------------------------------------------------------------------

def _coef_importance_multilabel(pipeline: Pipeline, features: List[str], labels: List[str]) -> pd.DataFrame:
    ovr: OneVsRestClassifier = pipeline.named_steps["ovr"]
    n_labels = len(labels)
    n_feats = len(features)
    coef_mat = np.zeros((n_labels, n_feats), dtype=float)

    for i, est in enumerate(getattr(ovr, "estimators_", []) or []):
        if i >= n_labels or est is None or not hasattr(est, "coef_"):
            continue
        c = np.asarray(est.coef_).reshape(-1)
        if c.shape[0] == n_feats:
            coef_mat[i, :] = c

    out = pd.DataFrame({"feature": features, "importance_abs_mean": np.mean(np.abs(coef_mat), axis=0)})
    for i, lab in enumerate(labels):
        out[f"coef_{_safe_name(lab)}"] = coef_mat[i, :]
    return out.sort_values("importance_abs_mean", ascending=False).reset_index(drop=True)


def _extract_multilabel_scores(estimator: Pipeline, X: pd.DataFrame) -> Optional[np.ndarray]:
    try:
        proba = estimator.predict_proba(X)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr
    except Exception:
        pass
    try:
        dec = estimator.decision_function(X)
        arr = np.asarray(dec, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return 1.0 / (1.0 + np.exp(-arr))
    except Exception:
        return None


def _binary_metrics_from_single_label(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, object]:
    yt = np.asarray(y_true, dtype=int).reshape(-1)
    yp = np.asarray(y_pred, dtype=int).reshape(-1)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=labels).ravel()
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else None
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else None
    fnr = (fn / (fn + tp)) if (fn + tp) > 0 else None

    out: Dict[str, object] = {
        "accuracy": _safe_float(np.mean(yt == yp)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(yt, yp)),
        "precision": _safe_float(precision_score(yt, yp, zero_division=0)),
        "recall": _safe_float(recall_score(yt, yp, zero_division=0)),
        "f1": _safe_float(f1_score(yt, yp, zero_division=0)),
        "mcc": _safe_float(matthews_corrcoef(yt, yp)),
        "specificity": _safe_float(specificity),
        "fpr": _safe_float(fpr),
        "fnr": _safe_float(fnr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "support_positive": int(np.sum(yt == 1)),
        "support_negative": int(np.sum(yt == 0)),
    }

    if y_score is not None:
        ys = np.asarray(y_score, dtype=float).reshape(-1)
        try:
            if len(np.unique(yt)) >= 2:
                out["roc_auc"] = _safe_float(roc_auc_score(yt, ys))
                out["pr_auc"] = _safe_float(average_precision_score(yt, ys))
            else:
                out["roc_auc"] = None
                out["pr_auc"] = None
        except Exception:
            out["roc_auc"] = None
            out["pr_auc"] = None

    return out


def _multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], y_score: Optional[np.ndarray] = None, reduced_binary_mode: str = "auto") -> Dict[str, object]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    labels_list = [str(x) for x in labels]

    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    effective_binary = False
    mode = str(reduced_binary_mode or "auto").strip().lower()
    if mode not in {"auto", "off", "force"}:
        raise ValueError(f"Unsupported reduced_binary_mode='{reduced_binary_mode}'. Use auto|off|force.")
    if mode == "force":
        if yt.shape[1] != 1:
            raise ValueError("--reduced-binary-mode force requires exactly one label column after train-only fitting.")
        effective_binary = True
    elif mode == "auto" and yt.shape[1] == 1:
        effective_binary = True

    metrics: Dict[str, object] = {
        "n": int(len(yt)),
        "labels": labels_list,
        "effective_problem_type": "binary_reduced" if effective_binary else "multilabel",
        "f1_micro": _safe_float(f1_score(yt, yp, average="micro", zero_division=0)),
        "f1_macro": _safe_float(f1_score(yt, yp, average="macro", zero_division=0)),
        "precision_micro": _safe_float(precision_score(yt, yp, average="micro", zero_division=0)),
        "precision_macro": _safe_float(precision_score(yt, yp, average="macro", zero_division=0)),
        "recall_micro": _safe_float(recall_score(yt, yp, average="micro", zero_division=0)),
        "recall_macro": _safe_float(recall_score(yt, yp, average="macro", zero_division=0)),
        "hamming_loss": _safe_float(hamming_loss(yt, yp)),
        "jaccard_micro": _safe_float(jaccard_score(yt, yp, average="micro", zero_division=0)),
        "jaccard_macro": _safe_float(jaccard_score(yt, yp, average="macro", zero_division=0)),
        "jaccard_samples": None if effective_binary else _safe_float(jaccard_score(yt, yp, average="samples", zero_division=0)),
        "exact_match_ratio": _safe_float(subset_accuracy_score(yt, yp)),
        "cardinality_true": _cardinality_stats(yt, len(labels_list)),
        "cardinality_pred": _cardinality_stats(yp, len(labels_list)),
    }
    if effective_binary:
        score_col = None if y_score is None else np.asarray(y_score, dtype=float).reshape(-1, 1)[:, 0]
        metrics["binary_metrics"] = _binary_metrics_from_single_label(yt[:, 0], yp[:, 0], score_col)

    labelwise: List[Dict[str, object]] = []
    for i, lab in enumerate(labels_list):
        yti = yt[:, i]
        ypi = yp[:, i]
        tp = int(np.sum((yti == 1) & (ypi == 1)))
        fp = int(np.sum((yti == 0) & (ypi == 1)))
        fn = int(np.sum((yti == 1) & (ypi == 0)))
        tn = int(np.sum((yti == 0) & (ypi == 0)))
        block: Dict[str, object] = {
            "label": lab,
            "support_true": int(yti.sum()),
            "support_pred": int(ypi.sum()),
            "prevalence_true": _safe_float(yti.mean()),
            "prevalence_pred": _safe_float(ypi.mean()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": _safe_float(precision_score(yti, ypi, zero_division=0)),
            "recall": _safe_float(recall_score(yti, ypi, zero_division=0)),
            "f1": _safe_float(f1_score(yti, ypi, zero_division=0)),
            "jaccard": _safe_float(jaccard_score(yti, ypi, zero_division=0)),
        }
        if y_score is not None:
            try:
                if len(np.unique(yti)) >= 2:
                    block["roc_auc"] = _safe_float(roc_auc_score(yti, y_score[:, i]))
                    block["pr_auc"] = _safe_float(average_precision_score(yti, y_score[:, i]))
                else:
                    block["roc_auc"] = None
                    block["pr_auc"] = None
            except Exception:
                block["roc_auc"] = None
                block["pr_auc"] = None
        labelwise.append(block)

    labelwise.sort(key=lambda d: ((d.get("support_true") or 0), d["label"]), reverse=True)
    metrics["labelwise"] = labelwise

    if y_score is not None:
        try:
            metrics["pr_auc_micro"] = _safe_float(average_precision_score(yt, y_score, average="micro"))
        except Exception:
            metrics["pr_auc_micro"] = None
        try:
            metrics["pr_auc_macro"] = _safe_float(average_precision_score(yt, y_score, average="macro"))
        except Exception:
            metrics["pr_auc_macro"] = None
        try:
            valid_cols = [i for i in range(yt.shape[1]) if len(np.unique(yt[:, i])) >= 2]
            if valid_cols:
                metrics["roc_auc_macro"] = _safe_float(roc_auc_score(yt[:, valid_cols], y_score[:, valid_cols], average="macro"))
                metrics["roc_auc_micro"] = _safe_float(roc_auc_score(yt[:, valid_cols], y_score[:, valid_cols], average="micro"))
            else:
                metrics["roc_auc_macro"] = None
                metrics["roc_auc_micro"] = None
        except Exception:
            metrics["roc_auc_macro"] = None
            metrics["roc_auc_micro"] = None

    return metrics


def _print_multilabel_metrics(metrics: Dict[str, object], *, title: str) -> None:
    print(f"=== {title} ===")
    print(f"N:                 {metrics.get('n')}")
    print(f"Problem type:      {metrics.get('effective_problem_type')}")
    print(f"F1 micro/macro:    {metrics.get('f1_micro')} / {metrics.get('f1_macro')}")
    print(f"Hamming loss:      {metrics.get('hamming_loss')}")
    print(f"Jaccard micro/mac: {metrics.get('jaccard_micro')} / {metrics.get('jaccard_macro')}")
    if metrics.get('jaccard_samples') is not None:
        print(f"Jaccard samples:   {metrics.get('jaccard_samples')}")
    print(f"Exact match:       {metrics.get('exact_match_ratio')}")
    if metrics.get('effective_problem_type') == 'binary_reduced':
        bm = metrics.get('binary_metrics') or {}
        print(f"Accuracy / BalAcc: {bm.get('accuracy')} / {bm.get('balanced_accuracy')}")
        print(f"Precision/Recall:  {bm.get('precision')} / {bm.get('recall')}")
        print(f"Specificity:       {bm.get('specificity')}")
        print(f"MCC:               {bm.get('mcc')}")


# -----------------------------------------------------------------------------
# Benchmark helpers
# -----------------------------------------------------------------------------

def _peak_rss_mb() -> Optional[float]:
    if resource is None:
        return None
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(peak) / 1024.0
    except Exception:
        return None


def _percentiles_ms(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean_ms": None, "std_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def _predict_one(estimator: Pipeline, row: pd.Series, features: List[str], use_full_pipeline: bool) -> Tuple[float, float, float, bool]:
    t0 = time.perf_counter_ns()
    t1 = t0
    try:
        if use_full_pipeline:
            x_one = _features_from_raw_row(row, features)
            t1 = time.perf_counter_ns()
        else:
            x_one = pd.DataFrame([{c: row.get(c, 0) for c in features}])
        x_one = x_one[features].fillna(0).astype(np.float32)
        _ = estimator.predict(x_one)
        try:
            _ = estimator.predict_proba(x_one)
        except Exception:
            pass
        t2 = time.perf_counter_ns()
        return (t2 - t0) / 1e6, (t1 - t0) / 1e6, (t2 - t1) / 1e6, True
    except Exception:
        return None, None, None, False


def _benchmark_load_profiles(estimator: Pipeline, df_bench: pd.DataFrame, features: List[str], *, use_full_pipeline: bool, levels: List[int], max_rows: int) -> List[Dict[str, object]]:
    if len(df_bench) == 0 or not levels:
        return []
    rows = df_bench.head(min(len(df_bench), int(max_rows))).reset_index(drop=True)
    profiles: List[Dict[str, object]] = []
    for workers in sorted({max(1, int(x)) for x in levels}):
        lat_ms: List[float] = []
        ok = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_predict_one, estimator, row, features, use_full_pipeline) for _, row in rows.iterrows()]
            for fut in as_completed(futs):
                total_ms, _, _, success = fut.result()
                if success and total_ms is not None:
                    ok += 1
                    lat_ms.append(float(total_ms))
        wall = time.perf_counter() - t0
        profiles.append({
            "concurrency": int(workers),
            "requests": int(len(rows)),
            "successful_requests": int(ok),
            "failure_rate": _safe_float((len(rows) - ok) / len(rows)) if len(rows) else None,
            "throughput_req_per_sec": _safe_float(ok / wall) if wall > 0 else None,
            "latency_total": _percentiles_ms(lat_ms),
            "wall_time_seconds": _safe_float(wall),
            "note": "Approximate threaded load profile for feasibility, not a full networked WAF stress test.",
        })
    return profiles


def _benchmark_supervised(
    estimator: Pipeline,
    df: pd.DataFrame,
    features: List[str],
    *,
    benchmark_mode: str,
    benchmark_max_rows: int,
    benchmark_warmup_rows: int,
    benchmark_repeats: int,
    benchmark_load_levels: List[int],
    benchmark_load_rows: int,
    require_resource_metrics: bool,
    seed: int,
    model_path: Optional[str],
    train_time_seconds: Optional[float],
) -> Dict[str, object]:
    if df is None or len(df) == 0:
        return {"enabled": False, "reason": "empty_eval_df"}

    mode = (benchmark_mode or "auto").strip().lower()
    raw_available = _can_extract_raw(df)
    if mode == "none":
        return {"enabled": False, "reason": "benchmark_disabled"}
    if mode == "full" and not raw_available:
        return {"enabled": False, "reason": "raw_http_columns_missing_for_full_pipeline"}

    use_full_pipeline = raw_available if mode == "auto" else (mode == "full")

    if benchmark_max_rows and benchmark_max_rows > 0 and len(df) > benchmark_max_rows:
        df_bench = df.sample(benchmark_max_rows, random_state=seed).reset_index(drop=True)
    else:
        df_bench = df.reset_index(drop=True)

    warmup_n = min(int(max(0, benchmark_warmup_rows)), len(df_bench))
    total_lat_ms: List[float] = []
    extract_lat_ms: List[float] = []
    infer_lat_ms: List[float] = []
    failures = 0

    if require_resource_metrics and psutil is None:
        return {"enabled": False, "reason": "psutil_not_available"}
    if require_resource_metrics and resource is None:
        return {"enabled": False, "reason": "resource_not_available"}

    proc = psutil.Process(os.getpid()) if psutil is not None else None
    cpu_before = proc.cpu_times() if proc is not None else None
    rss_before = proc.memory_info().rss / (1024 ** 2) if proc is not None else None
    peak_before = _peak_rss_mb()

    def _run_one(row: pd.Series) -> None:
        nonlocal failures
        total_ms, ext_ms, inf_ms, ok = _predict_one(estimator, row, features, use_full_pipeline)
        if ok:
            total_lat_ms.append(float(total_ms))
            extract_lat_ms.append(float(ext_ms))
            infer_lat_ms.append(float(inf_ms))
        else:
            failures += 1

    if warmup_n > 0:
        for _, row in df_bench.head(warmup_n).iterrows():
            _run_one(row)
        total_lat_ms = []
        extract_lat_ms = []
        infer_lat_ms = []
        failures = 0

    wall_start = time.perf_counter()
    for _ in range(max(1, int(benchmark_repeats))):
        for _, row in df_bench.iterrows():
            _run_one(row)
    wall_elapsed = time.perf_counter() - wall_start

    cpu_after = proc.cpu_times() if proc is not None else None
    rss_after = proc.memory_info().rss / (1024 ** 2) if proc is not None else None
    peak_after = _peak_rss_mb()

    cpu_util = None
    if cpu_before is not None and cpu_after is not None and wall_elapsed > 0:
        cpu_seconds = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
        cpu_util = float(100.0 * cpu_seconds / wall_elapsed)

    total_requests = len(df_bench) * max(1, int(benchmark_repeats))
    successful = max(0, total_requests - failures)
    throughput_rps = float(successful / wall_elapsed) if wall_elapsed > 0 else None
    failure_rate = float(failures / total_requests) if total_requests > 0 else None

    model_size_bytes = None
    if model_path and os.path.exists(model_path):
        try:
            model_size_bytes = int(os.path.getsize(model_path))
        except Exception:
            model_size_bytes = None

    peak_rss = _safe_float(peak_after if peak_after is not None else peak_before)
    rss_before_safe = _safe_float(rss_before)
    rss_after_safe = _safe_float(rss_after)
    if require_resource_metrics and (cpu_util is None or rss_before_safe is None or rss_after_safe is None or peak_rss is None):
        return {
            "enabled": False,
            "reason": "required_resource_metrics_unavailable",
            "cpu_utilization_pct_approx": _safe_float(cpu_util),
            "rss_mb_before": rss_before_safe,
            "rss_mb_after": rss_after_safe,
            "peak_rss_mb_approx": peak_rss,
        }

    load_profiles = _benchmark_load_profiles(
        estimator,
        df_bench,
        features,
        use_full_pipeline=use_full_pipeline,
        levels=benchmark_load_levels,
        max_rows=benchmark_load_rows,
    )

    return {
        "enabled": True,
        "mode": "feature_extraction_plus_inference" if use_full_pipeline else "inference_only",
        "n_rows_sampled": int(len(df_bench)),
        "repeats": int(max(1, int(benchmark_repeats))),
        "total_requests_measured": int(total_requests),
        "successful_requests": int(successful),
        "failures": int(failures),
        "failure_rate": failure_rate,
        "throughput_req_per_sec": throughput_rps,
        "latency_total": _percentiles_ms(total_lat_ms),
        "latency_feature_extraction": _percentiles_ms(extract_lat_ms),
        "latency_inference": _percentiles_ms(infer_lat_ms),
        "cpu_utilization_pct_approx": cpu_util,
        "rss_mb_before": rss_before_safe,
        "rss_mb_after": rss_after_safe,
        "peak_rss_mb_approx": peak_rss,
        "model_size_bytes": model_size_bytes,
        "train_time_seconds": _safe_float(train_time_seconds),
        "wall_time_seconds": _safe_float(wall_elapsed),
        "load_profiles": load_profiles,
    }


def _print_benchmark(bench: Dict[str, object], *, title: str) -> None:
    if not bench.get("enabled"):
        print(f"=== {title} ===")
        print(f"Benchmark not available: {bench.get('reason')}")
        return
    print(f"=== {title} ===")
    print(f"Mode:                 {bench['mode']}")
    print(f"Rows sampled:         {bench['n_rows_sampled']}")
    print(f"Throughput req/s:     {bench['throughput_req_per_sec']}")
    print(f"Failure rate:         {bench['failure_rate']}")
    print(f"CPU util. approx %:   {bench['cpu_utilization_pct_approx']}")
    print(f"RSS before / after:   {bench['rss_mb_before']} / {bench['rss_mb_after']} MB")
    print(f"Peak RSS approx:      {bench['peak_rss_mb_approx']} MB")
    print(f"Model size bytes:     {bench['model_size_bytes']}")
    print(f"Train time seconds:   {bench['train_time_seconds']}")
    block = bench.get("latency_total", {}) or {}
    print(f"latency_total: mean={block.get('mean_ms')} p50={block.get('p50_ms')} p95={block.get('p95_ms')} p99={block.get('p99_ms')}")
    if bench.get("load_profiles"):
        print("load_profiles:")
        for lp in bench["load_profiles"]:
            lpb = lp.get("latency_total", {}) or {}
            print(f"  concurrency={lp.get('concurrency')} throughput={lp.get('throughput_req_per_sec')} p95={lpb.get('p95_ms')} fail={lp.get('failure_rate')}")


# -----------------------------------------------------------------------------
# Split / CV helpers
# -----------------------------------------------------------------------------

def _make_outer_split(n_rows: int, y_full_global: np.ndarray, test_size: float, seed: int, prefer_stratified: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    indices = np.arange(n_rows)
    if prefer_stratified and MultilabelStratifiedShuffleSplit is not None:
        try:
            splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
            train_idx, test_idx = next(splitter.split(indices.reshape(-1, 1), y_full_global))
            return train_idx, test_idx, {
                "split_strategy": "approx_multilabel_stratified",
                "split_note": "Approximate multilabel stratification via iterative stratification.",
            }
        except Exception as e:
            return _make_outer_split(n_rows, y_full_global, test_size, seed, prefer_stratified=False)
    train_idx, test_idx = train_test_split(indices, test_size=float(test_size), random_state=int(seed), shuffle=True)
    return np.asarray(train_idx), np.asarray(test_idx), {
        "split_strategy": "fixed_non_stratified",
        "split_note": "Approximate multilabel stratification unavailable or not viable; fixed non-stratified split with explicit prevalence reporting.",
    }


def _build_cv(args, y_train: np.ndarray):
    if args.prefer_stratified_cv and MultilabelStratifiedKFold is not None:
        try:
            return MultilabelStratifiedKFold(n_splits=max(2, int(args.cv)), shuffle=True, random_state=int(args.seed)), "approx_multilabel_stratified_kfold"
        except Exception:
            pass
    return KFold(n_splits=max(2, int(args.cv)), shuffle=True, random_state=int(args.seed)), "kfold"


# -----------------------------------------------------------------------------
# Tuning
# -----------------------------------------------------------------------------

def _expand_multilabel_candidates(args) -> List[Dict[str, object]]:
    c_vals = _parse_float_grid(args.tune_c_grid)
    if not c_vals:
        raise ValueError("--tune-c-grid cannot be empty.")
    solvers = _parse_csv_list(args.tune_solvers) or [str(args.solver)]
    penalties = [_parse_optional_penalty(x) for x in (_parse_csv_list(args.tune_penalties) or [str(args.penalty)])]
    class_weights = [_parse_optional_class_weight(x) for x in (_parse_csv_list(args.tune_class_weights) or [str(args.class_weight)])]
    l1_ratios = _parse_float_grid(args.tune_l1_ratios) or [float(args.l1_ratio)]

    combos: List[Dict[str, object]] = []
    for solver in solvers:
        for penalty in penalties:
            for class_weight in class_weights:
                for C in c_vals:
                    solver = str(solver).strip().lower()
                    if penalty == "elasticnet":
                        for l1_ratio in l1_ratios:
                            if _is_valid_lr_combo(solver, penalty, l1_ratio):
                                combos.append({
                                    "ovr__estimator__solver": solver,
                                    "ovr__estimator__penalty": penalty,
                                    "ovr__estimator__class_weight": class_weight,
                                    "ovr__estimator__C": float(C),
                                    "ovr__estimator__l1_ratio": float(l1_ratio),
                                })
                    elif _is_valid_lr_combo(solver, penalty, None):
                        combos.append({
                            "ovr__estimator__solver": solver,
                            "ovr__estimator__penalty": penalty,
                            "ovr__estimator__class_weight": class_weight,
                            "ovr__estimator__C": float(C),
                        })
    unique: List[Dict[str, object]] = []
    seen = set()
    for c in combos:
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    if not unique:
        raise ValueError("No valid multilabel tuning candidate pool.")
    return unique


def _maybe_cap_tune_sample(x, y, sample_n: int, seed: int):
    if sample_n and sample_n > 0 and len(x) > sample_n:
        idx = np.random.default_rng(seed).choice(len(x), size=int(sample_n), replace=False)
        x2 = x.iloc[idx] if hasattr(x, "iloc") else x[idx]
        y2 = y[idx]
        print(f"[TUNE] Using tune sample n={len(x2)}")
        return x2.reset_index(drop=True), y2
    if hasattr(x, "reset_index"):
        x = x.reset_index(drop=True)
    return x, y


def _run_cv_tuning(args, x_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[Dict[str, object], Dict[str, object], pd.DataFrame]:
    x_tune, y_tune = _maybe_cap_tune_sample(x_train, y_train, args.tune_sample_n, args.seed)
    scorer = make_scorer(f1_score, average="micro", zero_division=0)
    cv_obj, cv_name = _build_cv(args, y_tune)
    base_est = _build_multilabel_pipe(args, params=None, ovr_n_jobs=1)
    search_space = [{k: [v] for k, v in c.items()} for c in _expand_multilabel_candidates(args)]
    n_iter = len(search_space) if args.tune == "grid" else min(int(args.tune_n_iter), len(search_space))
    print(f"[TUNE] protocol=cv cv={cv_name} folds={int(args.cv)} candidates={n_iter}")
    search = RandomizedSearchCV(
        estimator=base_est,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv_obj,
        n_jobs=int(args.tune_n_jobs),
        random_state=int(args.seed),
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    search.fit(x_tune, y_tune)
    cv_search_seconds = time.perf_counter() - t0
    print(f"[TUNE] cv_search_seconds={cv_search_seconds:.2f}")
    best_params = dict(search.best_params_)
    tune_summary = {
        "enabled": True,
        "requested_mode": args.tune,
        "effective_mode": "cv_random" if args.tune != "grid" else "cv_grid",
        "selection_metric": "f1_micro",
        "protocol": "cv",
        "cv": int(args.cv),
        "cv_splitter": cv_name,
        "tune_sample_n": int(args.tune_sample_n),
        "n_candidates_evaluated": int(n_iter),
        "candidate_pool_size": int(len(search_space)),
        "best_params": best_params,
        "best_score": _safe_float(search.best_score_),
        "subtrain_rows": int(len(x_tune)),
        "tuning_time_seconds": float(cv_search_seconds),
    }
    return best_params, tune_summary, pd.DataFrame(search.cv_results_)


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def _build_global_summary(label_lists_all: Sequence[Sequence[str]], train_idx: np.ndarray, test_idx: np.ndarray, split_meta: Dict[str, object]) -> Dict[str, object]:
    global_labels = _label_union(label_lists_all)
    all_y, _ = _encode_with_known_classes(label_lists_all, global_labels)
    train_lists = [label_lists_all[i] for i in train_idx]
    test_lists = [label_lists_all[i] for i in test_idx]
    train_y, _ = _encode_with_known_classes(train_lists, global_labels)
    test_y, _ = _encode_with_known_classes(test_lists, global_labels)
    return {
        **split_meta,
        "all": {
            "rows": int(len(label_lists_all)),
            "label_distribution": _label_distribution(all_y, global_labels),
            "cardinality": _cardinality_stats(all_y, len(global_labels)),
        },
        "train": {
            "rows": int(len(train_idx)),
            "label_distribution": _label_distribution(train_y, global_labels),
            "cardinality": _cardinality_stats(train_y, len(global_labels)),
        },
        "test": {
            "rows": int(len(test_idx)),
            "label_distribution": _label_distribution(test_y, global_labels),
            "cardinality": _cardinality_stats(test_y, len(global_labels)),
        },
    }


def _run_single_experiment(
    args,
    df: pd.DataFrame,
    all_label_lists: Sequence[Sequence[str]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_summary: Dict[str, object],
    features: List[str],
    *,
    tag: str,
    out_path: str,
    metrics_out: Optional[str],
    benchmark_out: Optional[str],
    tune_results_out: Optional[str],
    pred_out: Optional[str],
) -> Dict[str, object]:
    print(f"\n===== EXPERIMENT: {tag} =====")
    x_train = df.iloc[train_idx][features].fillna(0).astype(np.float32).reset_index(drop=True)
    x_test = df.iloc[test_idx][features].fillna(0).astype(np.float32).reset_index(drop=True)
    train_lists = [all_label_lists[i] for i in train_idx]
    test_lists = [all_label_lists[i] for i in test_idx]

    mlb = _fit_train_only_mlb(train_lists)
    model_labels = [str(c) for c in mlb.classes_]
    if len(model_labels) == 0:
        raise ValueError("No multilabel classes found in training split.")
    y_train, unseen_train = _encode_with_known_classes(train_lists, model_labels)
    y_test, unseen_test = _encode_with_known_classes(test_lists, model_labels)

    if unseen_train:
        raise RuntimeError(f"Unexpected unseen labels inside train encoding: {unseen_train}")

    estimator = _build_multilabel_pipe(args, params=None, ovr_n_jobs=args.n_jobs)
    tune_summary: Dict[str, object] = {
        "enabled": args.tune != "none",
        "requested_mode": args.tune,
        "effective_mode": args.tune,
        "selection_metric": "f1_micro",
        "protocol": "cv" if args.tune != "none" else None,
        "tune_sample_n": int(args.tune_sample_n),
        "tuning_time_seconds": None,
    }

    if args.tune != "none":
        best_params, tune_summary, search_df = _run_cv_tuning(args, x_train, y_train)
        estimator = _build_multilabel_pipe(args, params=best_params, ovr_n_jobs=args.n_jobs)
        _save_search_df(search_df, tune_results_out)

    t0 = time.perf_counter()
    estimator.fit(x_train, y_train)
    train_time_seconds = time.perf_counter() - t0

    y_pred = estimator.predict(x_test)
    y_score = _extract_multilabel_scores(estimator, x_test)
    metrics = _multilabel_metrics(y_test, y_pred, model_labels, y_score=y_score, reduced_binary_mode=args.reduced_binary_mode)
    _print_multilabel_metrics(metrics, title=f"Test evaluation ({tag})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "task": "multilabel",
        "tag": tag,
        "model": estimator,
        "mlb": mlb,
        "features": features,
        "labels": model_labels,
        "config": {
            "solver": args.solver,
            "penalty": args.penalty,
            "C": float(args.C),
            "class_weight": args.class_weight,
            "l1_ratio": float(args.l1_ratio),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "n_jobs": int(args.n_jobs),
        },
    }
    joblib.dump(payload, out_path)
    print(f"Saved model+mlb: {out_path}")

    metrics_payload: Dict[str, object] = {
        "task": "multilabel",
        "experiment_tag": tag,
        "dataset_name": str(df.get("dataset_name").iloc[0]) if "dataset_name" in df.columns and len(df) > 0 else None,
        "data": str(args.data),
        "label_col": str(args.label_col),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features_used": features,
        "features_dropped": [f for f in DEFAULT_FEATURES if f not in features],
        "suspicious_token_ablation": sorted([f for f in SUSPICIOUS_TOKEN_FEATURES if f not in features]),
        "tuning": tune_summary,
        "split_summary": split_summary,
        "model_label_vocabulary": model_labels,
        "reduced_binary_mode": str(args.reduced_binary_mode),
        "excluded_unseen_test_labels": unseen_test,
        "tuning_time_seconds": _safe_float(tune_summary.get("tuning_time_seconds")),
        "train_time_seconds": float(train_time_seconds),
        "evaluation": metrics,
    }

    if pred_out:
        y_test_arr = np.asarray(y_test, dtype=int)
        y_pred_arr = np.asarray(y_pred, dtype=int)
        if y_test_arr.ndim == 1:
            y_test_arr = y_test_arr.reshape(-1, 1)
        if y_pred_arr.ndim == 1:
            y_pred_arr = y_pred_arr.reshape(-1, 1)
        y_score_arr = None if y_score is None else np.asarray(y_score)
        if y_score_arr is not None and y_score_arr.ndim == 1:
            y_score_arr = y_score_arr.reshape(-1, 1)

        true_sets = _label_sets_from_binary_matrix(y_test_arr, model_labels)
        pred_sets = _label_sets_from_binary_matrix(y_pred_arr, model_labels)
        pred_df = pd.DataFrame({
            "row_index": test_idx,
            "n_labels_true": y_test_arr.sum(axis=1),
            "n_labels_pred": y_pred_arr.sum(axis=1),
            "true_labels": [json.dumps(v, ensure_ascii=False) for v in true_sets],
            "pred_labels": [json.dumps(v, ensure_ascii=False) for v in pred_sets],
        })
        for i, lab in enumerate(model_labels):
            tag_lab = _safe_name(lab)
            pred_df[f"true_{tag_lab}"] = y_test_arr[:, i].astype(int)
            pred_df[f"pred_{tag_lab}"] = y_pred_arr[:, i].astype(int)
            if y_score_arr is not None and y_score_arr.shape[1] > i:
                pred_df[f"score_{tag_lab}"] = y_score_arr[:, i]
        _save_table(pred_df, pred_out)
        print(f"Saved predictions: {pred_out}")

    if args.fi_kind in {"coef", "both"}:
        df_coef = _coef_importance_multilabel(estimator, features, model_labels)
        coef_path = f"{out_path}.feature_importance.coef.csv"
        df_coef.to_csv(coef_path, index=False)
        print(f"Saved coef importance: {coef_path}")

    df_eval_bench = df.iloc[test_idx].copy().reset_index(drop=True)
    if args.max_eval and args.max_eval > 0 and len(df_eval_bench) > args.max_eval:
        df_eval_bench = df_eval_bench.sample(args.max_eval, random_state=args.seed).reset_index(drop=True)
    bench_payload = _benchmark_supervised(
        estimator,
        df_eval_bench,
        features,
        benchmark_mode=args.benchmark_mode,
        benchmark_max_rows=args.benchmark_max_rows,
        benchmark_warmup_rows=args.benchmark_warmup_rows,
        benchmark_repeats=args.benchmark_repeats,
        benchmark_load_levels=[int(x) for x in _parse_csv_list(args.benchmark_load_levels)] if args.benchmark_load_levels else [],
        benchmark_load_rows=args.benchmark_load_rows,
        require_resource_metrics=args.require_resource_metrics,
        seed=args.seed,
        model_path=out_path,
        train_time_seconds=train_time_seconds,
    )
    if args.require_resource_metrics and not bench_payload.get("enabled"):
        raise RuntimeError(f"Required benchmark resource metrics are missing: {bench_payload.get('reason')}")
    _print_benchmark(bench_payload, title=f"Benchmark ({tag})")

    _save_json(metrics_payload, metrics_out)
    _save_json(bench_payload, benchmark_out)

    return {
        "tag": tag,
        "model_out": out_path,
        "metrics_out": metrics_out,
        "benchmark_out": benchmark_out,
        "pred_out": pred_out,
        "features_used": features,
        "metrics": metrics,
        "train_time_seconds": float(train_time_seconds),
        "tuning": tune_summary,
        "excluded_unseen_test_labels": unseen_test,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train thesis-aligned multilabel OvR Logistic Regression on Harvard/SR-BH WAF features with optional reduced-binary compatibility")
    ap.add_argument("--data", required=True)
    ap.add_argument("--label-col", default="label_multilabel")
    ap.add_argument("--reduced-binary-mode", default="auto", choices=["auto", "off", "force"], help="Compatibility mode for datasets whose multilabel column reduces to a single positive label (e.g., CSIC). Harvard/SR-BH remains unchanged under auto unless only one train-time label exists.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-features", default=None, help="Comma-separated feature names to exclude (for baseline ablations)")

    ap.add_argument("--solver", default="lbfgs")
    ap.add_argument("--penalty", default="l2")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--l1-ratio", type=float, default=0.5)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--n-jobs", type=int, default=-1)

    ap.add_argument("--tune", default="random", choices=["none", "grid", "random", "halving"])
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--tune-n-iter", "--tune-budget", dest="tune_n_iter", type=int, default=4)
    ap.add_argument("--tune-n-jobs", type=int, default=1)
    ap.add_argument("--tune-sample-n", type=int, default=40000, help="Optional cap rows for tuning only (0 = full train)")
    ap.add_argument("--tune-results-out", default=None)
    ap.add_argument("--tune-c-grid", default="0.1,1,10")
    ap.add_argument("--tune-solvers", default="lbfgs")
    ap.add_argument("--tune-penalties", default="l2")
    ap.add_argument("--tune-class-weights", default="balanced,none")
    ap.add_argument("--tune-l1-ratios", default="0.5")
    ap.add_argument("--prefer-stratified-split", action="store_true")
    ap.add_argument("--prefer-stratified-cv", action="store_true")

    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--pred-out", default=None)
    ap.add_argument("--benchmark-out", default=None)

    ap.add_argument("--fi-kind", default="coef", choices=["coef", "none"])
    ap.add_argument("--max-eval", type=int, default=0)

    ap.add_argument("--benchmark-mode", default="auto", choices=["auto", "full", "inference", "none"])
    ap.add_argument("--benchmark-max-rows", type=int, default=3000)
    ap.add_argument("--benchmark-warmup-rows", type=int, default=200)
    ap.add_argument("--benchmark-repeats", type=int, default=1)
    ap.add_argument("--benchmark-load-levels", default="1,2,4")
    ap.add_argument("--benchmark-load-rows", type=int, default=400)
    ap.add_argument("--require-resource-metrics", action="store_true", help="Fail the run if CPU/RAM benchmark metrics cannot be measured.")

    ap.add_argument("--run-suspicious-ablation", action="store_true")
    ap.add_argument("--ablation-retune", action="store_true")
    ap.add_argument("--ablation-comparison-out", default=None)

    args = ap.parse_args()

    if args.tune == "halving":
        print("[TUNE] Halving multilabel is not reliable in this setup; using bounded RandomizedSearchCV with CV.")
        args.tune = "random"

    df = _read_table(args.data)
    base_features = _resolve_features(args.drop_features)
    _ensure_features(df, base_features)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing multilabel column '{args.label_col}'. Available columns sample: {list(df.columns)[:50]}")

    all_label_lists = df[args.label_col].apply(_parse_multilabel_cell).tolist()
    global_labels = _label_union(all_label_lists)
    if not global_labels:
        raise ValueError("No multilabel classes were found after parsing the label column.")
    y_full_global, _ = _encode_with_known_classes(all_label_lists, global_labels)

    train_idx, test_idx, split_meta = _make_outer_split(len(df), y_full_global, args.test_size, args.seed, args.prefer_stratified_split)
    split_summary = _build_global_summary(all_label_lists, train_idx, test_idx, split_meta)

    baseline_result = _run_single_experiment(
        args,
        df,
        all_label_lists,
        train_idx=train_idx,
        test_idx=test_idx,
        split_summary=split_summary,
        features=base_features,
        tag="baseline",
        out_path=args.out,
        metrics_out=args.metrics_out,
        benchmark_out=args.benchmark_out,
        tune_results_out=args.tune_results_out,
        pred_out=args.pred_out,
    )

    results = {"baseline": baseline_result}

    if args.run_suspicious_ablation:
        ablation_features = [f for f in base_features if f not in SUSPICIOUS_TOKEN_FEATURES]
        if not ablation_features:
            raise ValueError("Ablation removed all features; cannot continue.")
        ab_args = argparse.Namespace(**vars(args))
        if not args.ablation_retune:
            ab_args.tune = "none"
        ab_result = _run_single_experiment(
            ab_args,
            df,
            all_label_lists,
            train_idx=train_idx,
            test_idx=test_idx,
            split_summary=split_summary,
            features=ablation_features,
            tag="no_suspicious_tokens",
            out_path=_with_suffix(args.out, ".no_suspicious"),
            metrics_out=_with_suffix(args.metrics_out, ".no_suspicious"),
            benchmark_out=_with_suffix(args.benchmark_out, ".no_suspicious"),
            tune_results_out=_with_suffix(args.tune_results_out, ".no_suspicious"),
            pred_out=_with_suffix(args.pred_out, ".no_suspicious"),
        )
        results["no_suspicious_tokens"] = ab_result

        comparison = {
            "baseline_metrics": {
                "f1_micro": baseline_result["metrics"].get("f1_micro"),
                "f1_macro": baseline_result["metrics"].get("f1_macro"),
                "hamming_loss": baseline_result["metrics"].get("hamming_loss"),
                "jaccard_micro": baseline_result["metrics"].get("jaccard_micro"),
                "exact_match_ratio": baseline_result["metrics"].get("exact_match_ratio"),
                "train_time_seconds": baseline_result.get("train_time_seconds"),
            },
            "no_suspicious_tokens_metrics": {
                "f1_micro": ab_result["metrics"].get("f1_micro"),
                "f1_macro": ab_result["metrics"].get("f1_macro"),
                "hamming_loss": ab_result["metrics"].get("hamming_loss"),
                "jaccard_micro": ab_result["metrics"].get("jaccard_micro"),
                "exact_match_ratio": ab_result["metrics"].get("exact_match_ratio"),
                "train_time_seconds": ab_result.get("train_time_seconds"),
            },
            "delta_baseline_minus_ablation": {
                "f1_micro": _safe_float((baseline_result["metrics"].get("f1_micro") or 0.0) - (ab_result["metrics"].get("f1_micro") or 0.0)),
                "f1_macro": _safe_float((baseline_result["metrics"].get("f1_macro") or 0.0) - (ab_result["metrics"].get("f1_macro") or 0.0)),
                "hamming_loss": _safe_float((baseline_result["metrics"].get("hamming_loss") or 0.0) - (ab_result["metrics"].get("hamming_loss") or 0.0)),
                "jaccard_micro": _safe_float((baseline_result["metrics"].get("jaccard_micro") or 0.0) - (ab_result["metrics"].get("jaccard_micro") or 0.0)),
                "exact_match_ratio": _safe_float((baseline_result["metrics"].get("exact_match_ratio") or 0.0) - (ab_result["metrics"].get("exact_match_ratio") or 0.0)),
                "train_time_seconds": _safe_float((baseline_result.get("train_time_seconds") or 0.0) - (ab_result.get("train_time_seconds") or 0.0)),
            },
            "baseline_features_used": baseline_result.get("features_used"),
            "ablation_features_used": ab_result.get("features_used"),
            "ablation_removed_features": sorted(set(baseline_result.get("features_used", [])) - set(ab_result.get("features_used", []))),
        }
        _save_json(comparison, args.ablation_comparison_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
