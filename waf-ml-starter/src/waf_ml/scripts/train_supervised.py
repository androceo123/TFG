from __future__ import annotations

"""
waf_ml.scripts.train_supervised

Supervised baselines for WAF-ML features:
  - Multiclass Logistic Regression
  - Multilabel One-vs-Rest Logistic Regression

Updated to align with the thesis/proposal methodology:
- fixed outer 80/20 split for supervised evaluation
- tuning only inside the training partition
- bounded tuning with grid / random / halving
- scalable linear models and compatible solver/penalty search spaces
- effectiveness metrics saved to JSON
- optional benchmark for feasibility in WAF-like real-time settings
- coefficient-based explainability + drop-features support for ablations
"""

import argparse
import inspect
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
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
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score as subset_accuracy_score

try:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
except Exception:  # pragma: no cover
    HalvingGridSearchCV = None
    HalvingRandomSearchCV = None

try:
    from sklearn.inspection import permutation_importance
except Exception:  # pragma: no cover
    permutation_importance = None

try:  # package execution
    from waf_ml.features.http_features import extract_http_features
except Exception:  # pragma: no cover - local fallback for standalone testing
    from http_features import extract_http_features


DEFAULT_FEATURES = [
    "uri_len", "path_depth", "query_len", "n_query_params", "max_param_value_len",
    "uri_pct_non_alnum_ratio", "encoded", "suspicious_tokens_count", "has_suspicious_tokens",
    "uncommon_method", "req_content_length", "body_len", "body_suspicious_tokens_count", "body_has_suspicious_tokens", "body_encoded",
    "method_GET", "method_POST", "method_HEAD", "method_PUT", "method_DELETE", "method_PATCH", "method_OPTIONS",
    "method_TRACE", "method_CONNECT", "method_OTHER",
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
    s = str(s)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")
    return s or "CLASS"


def _derive_fi_path(model_out: str, tag: str) -> str:
    return f"{model_out}.feature_importance.{tag}.csv"


def _print_top(df_imp: pd.DataFrame, col: str, k: int) -> None:
    if df_imp is None or df_imp.empty:
        return
    k = int(k) if k else 0
    if k <= 0:
        return
    view = df_imp.sort_values(col, ascending=False).head(k)
    print(f"\n=== Top {k} features by {col} ===")
    for _, r in view.iterrows():
        print(f"{r['feature']}: {r[col]:.6g}")


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


def _safe_train_test_split_multiclass(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    seed: int,
    try_stratify: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not try_stratify:
        return train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)

    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y, shuffle=True)
    except ValueError as e:
        vc = y.value_counts(dropna=False)
        rare = vc[vc < 2]
        print("\n[WARN] Stratified split failed; falling back to non-stratified split.")
        print(f"[WARN] Reason: {e}")
        if len(rare) > 0:
            print("[WARN] Classes with <2 samples:")
            for cls, cnt in rare.items():
                print(f"  - {cls}: {cnt}")
        return train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)


def _make_cv_for_multiclass(y: pd.Series, n_splits: int, seed: int):
    n_splits = int(n_splits)
    if n_splits < 2:
        n_splits = 2
    vc = y.value_counts(dropna=False)
    min_count = int(vc.min()) if len(vc) else 0
    if min_count >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    print(f"[WARN] CV: min class count={min_count} < cv={n_splits}. Using non-stratified KFold for tuning.")
    return KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))


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


def _build_logreg(
    *,
    solver: str,
    penalty,
    C: float,
    class_weight,
    max_iter: int,
    l1_ratio,
    n_jobs: Optional[int] = None,
) -> LogisticRegression:
    sig = inspect.signature(LogisticRegression.__init__)

    solver_in = str(solver or "auto").strip().lower()
    penalty_in = _parse_optional_penalty(penalty)
    class_weight_in = _parse_optional_class_weight(class_weight)

    if solver_in in {"", "auto"}:
        if penalty_in == "elasticnet":
            solver_order = ["saga"]
        elif penalty_in == "l1":
            solver_order = ["saga", "liblinear"]
        else:
            solver_order = ["saga", "lbfgs", "liblinear"]
    else:
        solver_order = [solver_in]

    last_err: Exception | None = None
    for solver_name in solver_order:
        if not _is_valid_lr_combo(solver_name, penalty_in, l1_ratio):
            continue

        kwargs = {
            "solver": solver_name,
            "C": float(C),
            "max_iter": int(max_iter),
            "class_weight": class_weight_in,
        }
        if "multi_class" in sig.parameters:
            kwargs["multi_class"] = "auto"
        if "penalty" in sig.parameters:
            kwargs["penalty"] = penalty_in
        if penalty_in == "elasticnet" and solver_name == "saga" and "l1_ratio" in sig.parameters:
            kwargs["l1_ratio"] = float(l1_ratio)
        if n_jobs is not None and "n_jobs" in sig.parameters:
            kwargs["n_jobs"] = int(n_jobs)

        try:
            return LogisticRegression(**kwargs)
        except Exception as e:
            last_err = e
            continue

    if last_err is not None:
        raise last_err
    raise ValueError(
        f"No compatible LogisticRegression configuration for solver={solver_in}, penalty={penalty_in}, l1_ratio={l1_ratio}."
    )


def _build_multiclass_pipe(args) -> Pipeline:
    lr = _build_logreg(
        solver=args.solver,
        penalty=args.penalty,
        C=args.C,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
        l1_ratio=args.l1_ratio,
        n_jobs=args.n_jobs,
    )
    return Pipeline([("scaler", StandardScaler()), ("lr", lr)])


def _build_multilabel_pipe(args) -> Pipeline:
    base_lr = _build_logreg(
        solver=args.solver,
        penalty=args.penalty,
        C=args.C,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
        l1_ratio=args.l1_ratio,
        n_jobs=args.n_jobs,
    )
    return Pipeline([("scaler", StandardScaler()), ("ovr", OneVsRestClassifier(base_lr))])


def _coef_importance_multiclass(pipeline: Pipeline, features: List[str]) -> pd.DataFrame:
    lr: LogisticRegression = pipeline.named_steps["lr"]
    coefs = np.asarray(lr.coef_)
    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)

    classes = getattr(lr, "classes_", [f"class_{i}" for i in range(coefs.shape[0])])
    classes = [str(c) for c in classes]

    # Caso binario en sklearn: coef_ tiene shape (1, n_features) aunque haya 2 clases
    if coefs.shape[0] == 1 and len(classes) == 2:
        pos = coefs[0, :]
        coefs_full = np.vstack([-pos, pos])  # clase 0 y clase 1
        class_names = classes
    else:
        coefs_full = coefs
        class_names = classes[:coefs.shape[0]]

    abs_mean = np.mean(np.abs(coefs_full), axis=0)
    out = pd.DataFrame({"feature": features, "importance_abs_mean": abs_mean})

    for i, c in enumerate(class_names):
        out[f"coef_{_safe_name(c)}"] = coefs_full[i, :]

    return out.sort_values("importance_abs_mean", ascending=False).reset_index(drop=True)


def _coef_importance_multilabel(pipeline: Pipeline, features: List[str], labels: List[str]) -> pd.DataFrame:
    ovr: OneVsRestClassifier = pipeline.named_steps["ovr"]

    n_labels = len(labels)
    n_feats = len(features)
    coef_mat = np.zeros((n_labels, n_feats), dtype=float)

    ests = getattr(ovr, "estimators_", None) or []
    for i in range(min(n_labels, len(ests))):
        est = ests[i]
        if est is None or not hasattr(est, "coef_"):
            continue
        c = np.asarray(est.coef_).reshape(-1)
        if c.shape[0] == n_feats:
            coef_mat[i, :] = c

    abs_mean = np.mean(np.abs(coef_mat), axis=0)
    out = pd.DataFrame({"feature": features, "importance_abs_mean": abs_mean})

    for i, lab in enumerate(labels):
        out[f"coef_{_safe_name(lab)}"] = coef_mat[i, :]

    return out.sort_values("importance_abs_mean", ascending=False).reset_index(drop=True)


def _perm_importance(
    estimator: Pipeline,
    X: pd.DataFrame,
    y,
    features: List[str],
    *,
    scoring,
    n_repeats: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    if permutation_importance is None:
        raise RuntimeError(
            "permutation_importance is not available in your scikit-learn version. "
            "Upgrade scikit-learn or run with --fi-kind coef/none."
        )

    res = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=int(n_repeats),
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )

    return (
        pd.DataFrame({
            "feature": features,
            "perm_importance_mean": res.importances_mean,
            "perm_importance_std": res.importances_std,
        })
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _parse_prob_columns(y_prob: np.ndarray, classes: Sequence[str], prefix: str = "proba") -> pd.DataFrame:
    cols = {}
    for i, cls in enumerate(classes):
        cols[f"{prefix}_{_safe_name(cls)}"] = y_prob[:, i]
    return pd.DataFrame(cols)


def _multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str], y_prob=None) -> Dict[str, object]:
    y_true_arr = np.asarray([str(x) for x in y_true])
    y_pred_arr = np.asarray([str(x) for x in y_pred])
    classes_list = [str(c) for c in classes]

    metrics: Dict[str, object] = {
        "n": int(len(y_true_arr)),
        "labels": classes_list,
        "accuracy": _safe_float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "f1_micro": _safe_float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
        "f1_macro": _safe_float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "f1_weighted": _safe_float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "precision_micro": _safe_float(precision_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
        "precision_macro": _safe_float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "precision_weighted": _safe_float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "recall_micro": _safe_float(recall_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
        "recall_macro": _safe_float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "recall_weighted": _safe_float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "mcc": _safe_float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=classes_list).tolist(),
        "classification_report": classification_report(
            y_true_arr, y_pred_arr, labels=classes_list, digits=6, zero_division=0, output_dict=True
        ),
    }

    unique_present = sorted(set(y_true_arr.tolist()))
    metrics["labels_present_in_test"] = unique_present

    if y_prob is not None and len(classes_list) >= 2:
        try:
            y_bin = label_binarize(y_true_arr, classes=classes_list)
            if len(classes_list) == 2:
                pos_scores = np.asarray(y_prob)[:, 1]
                pos_true = y_bin[:, 0] if y_bin.shape[1] == 1 else y_bin[:, 1]
                tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=classes_list[:2]).ravel()
                metrics.update({
                    "roc_auc": _safe_float(roc_auc_score(pos_true, pos_scores)),
                    "pr_auc": _safe_float(average_precision_score(pos_true, pos_scores)),
                    "specificity": _safe_float(tn / (tn + fp)) if (tn + fp) > 0 else None,
                    "fpr": _safe_float(fp / (fp + tn)) if (fp + tn) > 0 else None,
                    "fnr": _safe_float(fn / (fn + tp)) if (fn + tp) > 0 else None,
                    "tpr": _safe_float(tp / (tp + fn)) if (tp + fn) > 0 else None,
                })
            else:
                metrics.update({
                    "roc_auc_ovr_macro": _safe_float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")),
                    "roc_auc_ovr_weighted": _safe_float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")),
                    "pr_auc_macro": _safe_float(average_precision_score(y_bin, y_prob, average="macro")),
                    "pr_auc_weighted": _safe_float(average_precision_score(y_bin, y_prob, average="weighted")),
                })
        except Exception:
            metrics.setdefault("roc_auc", None)
            metrics.setdefault("pr_auc", None)

    return metrics


def _print_multiclass_metrics(metrics: Dict[str, object], *, title: str) -> None:
    print(f"=== {title} ===")
    print(f"N:                 {metrics.get('n')}")
    print(f"Accuracy:          {metrics.get('accuracy')}")
    print(f"Balanced acc.:     {metrics.get('balanced_accuracy')}")
    print(f"F1 micro/macro/w: {metrics.get('f1_micro')} / {metrics.get('f1_macro')} / {metrics.get('f1_weighted')}")
    print(f"Precision macro:   {metrics.get('precision_macro')}")
    print(f"Recall macro:      {metrics.get('recall_macro')}")
    print(f"MCC:               {metrics.get('mcc')}")
    if "roc_auc" in metrics or "roc_auc_ovr_macro" in metrics:
        print(f"ROC-AUC:           {metrics.get('roc_auc', metrics.get('roc_auc_ovr_macro'))}")
    if "pr_auc" in metrics or "pr_auc_macro" in metrics:
        print(f"PR-AUC:            {metrics.get('pr_auc', metrics.get('pr_auc_macro'))}")


def _multilabel_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, labels: Sequence[str]) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "n": int(len(Y_true)),
        "labels": [str(x) for x in labels],
        "f1_micro": _safe_float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "f1_macro": _safe_float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "hamming_loss": _safe_float(hamming_loss(Y_true, Y_pred)),
        "jaccard_micro": _safe_float(jaccard_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "jaccard_macro": _safe_float(jaccard_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "exact_match_ratio": _safe_float(subset_accuracy_score(Y_true, Y_pred)),
    }
    return metrics


def _print_multilabel_metrics(metrics: Dict[str, object], *, title: str) -> None:
    print(f"=== {title} ===")
    print(f"N:                 {metrics.get('n')}")
    print(f"F1 micro/macro:    {metrics.get('f1_micro')} / {metrics.get('f1_macro')}")
    print(f"Hamming loss:      {metrics.get('hamming_loss')}")
    print(f"Jaccard micro/mac: {metrics.get('jaccard_micro')} / {metrics.get('jaccard_macro')}")
    print(f"Exact match:       {metrics.get('exact_match_ratio')}")


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


def _predict_one_benchmark_supervised(estimator: Pipeline, row: pd.Series, features: List[str], use_full_pipeline: bool) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
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


def _benchmark_load_profiles_supervised(estimator: Pipeline, df_bench: pd.DataFrame, features: List[str], *, use_full_pipeline: bool, levels: List[int], max_rows: int) -> List[Dict[str, object]]:
    if len(df_bench) == 0 or not levels:
        return []
    rows = df_bench.head(min(len(df_bench), int(max_rows))).reset_index(drop=True)
    profiles: List[Dict[str, object]] = []
    for workers in sorted({max(1, int(x)) for x in levels}):
        lat_ms: List[float] = []
        ok = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_predict_one_benchmark_supervised, estimator, row, features, use_full_pipeline) for _, row in rows.iterrows()]
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
        total_ms, extract_ms, infer_ms, success = _predict_one_benchmark_supervised(estimator, row, features, use_full_pipeline)
        if success and total_ms is not None and extract_ms is not None and infer_ms is not None:
            total_lat_ms.append(float(total_ms))
            extract_lat_ms.append(float(extract_ms))
            infer_lat_ms.append(float(infer_ms))
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

    load_profiles = _benchmark_load_profiles_supervised(
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
    print(f"Repeats:              {bench['repeats']}")
    print(f"Throughput req/s:     {bench['throughput_req_per_sec']}")
    print(f"Failure rate:         {bench['failure_rate']}")
    print(f"CPU util. approx %:   {bench['cpu_utilization_pct_approx']}")
    print(f"RSS before / after:   {bench['rss_mb_before']} / {bench['rss_mb_after']} MB")
    print(f"Peak RSS approx:      {bench['peak_rss_mb_approx']} MB")
    print(f"Model size bytes:     {bench['model_size_bytes']}")
    print(f"Train time seconds:   {bench['train_time_seconds']}")
    for key in ["latency_total", "latency_feature_extraction", "latency_inference"]:
        block = bench.get(key, {}) or {}
        print(
            f"{key}: mean={block.get('mean_ms')} std={block.get('std_ms')} "
            f"p50={block.get('p50_ms')} p95={block.get('p95_ms')} p99={block.get('p99_ms')}"
        )


def _build_multiclass_search_space(args) -> List[dict]:
    c_vals = _parse_float_grid(args.tune_c_grid)
    if not c_vals:
        raise ValueError("--tune-c-grid cannot be empty.")

    solvers = _parse_csv_list(args.tune_solvers) or ["saga", "lbfgs"]
    penalties_raw = _parse_csv_list(args.tune_penalties) or ["l2", "l1", "elasticnet", "none"]
    penalties = [_parse_optional_penalty(x) for x in penalties_raw]
    class_weights = [_parse_optional_class_weight(x) for x in (_parse_csv_list(args.tune_class_weights) or ["balanced", "none"])]
    l1_ratios = _parse_float_grid(args.tune_l1_ratios) or [0.15, 0.5, 0.85]

    spaces: List[dict] = []
    for solver in solvers:
        solver = solver.strip().lower()
        for penalty in penalties:
            if penalty == "elasticnet":
                if _is_valid_lr_combo(solver, penalty, 0.5):
                    spaces.append({
                        "lr__solver": [solver],
                        "lr__penalty": [penalty],
                        "lr__class_weight": class_weights,
                        "lr__C": c_vals,
                        "lr__l1_ratio": l1_ratios,
                    })
                continue

            if _is_valid_lr_combo(solver, penalty, None):
                spaces.append({
                    "lr__solver": [solver],
                    "lr__penalty": [penalty],
                    "lr__class_weight": class_weights,
                    "lr__C": c_vals,
                })

    if not spaces:
        raise ValueError("No valid multiclass tuning search space. Check --tune-solvers / --tune-penalties.")
    return spaces


def _build_multilabel_search_space(args) -> List[dict]:
    c_vals = _parse_float_grid(args.tune_c_grid)
    if not c_vals:
        raise ValueError("--tune-c-grid cannot be empty.")

    solvers = _parse_csv_list(args.tune_solvers) or ["saga", "lbfgs"]
    penalties_raw = _parse_csv_list(args.tune_penalties) or ["l2", "l1", "elasticnet", "none"]
    penalties = [_parse_optional_penalty(x) for x in penalties_raw]
    class_weights = [_parse_optional_class_weight(x) for x in (_parse_csv_list(args.tune_class_weights) or ["balanced", "none"])]
    l1_ratios = _parse_float_grid(args.tune_l1_ratios) or [0.15, 0.5, 0.85]

    spaces: List[dict] = []
    for solver in solvers:
        solver = solver.strip().lower()
        for penalty in penalties:
            if penalty == "elasticnet":
                if _is_valid_lr_combo(solver, penalty, 0.5):
                    spaces.append({
                        "ovr__estimator__solver": [solver],
                        "ovr__estimator__penalty": [penalty],
                        "ovr__estimator__class_weight": class_weights,
                        "ovr__estimator__C": c_vals,
                        "ovr__estimator__l1_ratio": l1_ratios,
                    })
                continue

            if _is_valid_lr_combo(solver, penalty, None):
                spaces.append({
                    "ovr__estimator__solver": [solver],
                    "ovr__estimator__penalty": [penalty],
                    "ovr__estimator__class_weight": class_weights,
                    "ovr__estimator__C": c_vals,
                })

    if not spaces:
        raise ValueError("No valid multilabel tuning search space. Check --tune-solvers / --tune-penalties.")
    return spaces


def _make_search(
    *,
    estimator,
    mode: str,
    search_space,
    scoring,
    cv,
    n_iter: int,
    n_jobs: int,
    seed: int,
    factor: int,
):
    mode = (mode or "none").strip().lower()
    if mode == "grid":
        return GridSearchCV(
            estimator,
            param_grid=search_space,
            scoring=scoring,
            cv=cv,
            n_jobs=int(n_jobs),
            refit=True,
            verbose=1,
            return_train_score=True,
        )
    if mode == "random":
        return RandomizedSearchCV(
            estimator,
            param_distributions=search_space,
            n_iter=int(n_iter),
            scoring=scoring,
            cv=cv,
            n_jobs=int(n_jobs),
            random_state=int(seed),
            refit=True,
            verbose=1,
            return_train_score=True,
        )
    if mode == "halving":
        if HalvingRandomSearchCV is None:
            raise RuntimeError("Halving search is not available in this scikit-learn installation.")
        return HalvingRandomSearchCV(
            estimator,
            param_distributions=search_space,
            n_candidates=int(n_iter),
            factor=max(2, int(factor)),
            scoring=scoring,
            cv=cv,
            n_jobs=int(n_jobs),
            random_state=int(seed),
            refit=True,
            verbose=1,
            return_train_score=True,
        )
    raise ValueError(f"Unsupported tune mode: {mode}")


def _maybe_cap_tune_sample(X, y, sample_n: int, seed: int):
    if sample_n and sample_n > 0 and len(X) > sample_n:
        idx = np.random.default_rng(seed).choice(len(X), size=int(sample_n), replace=False)
        if hasattr(X, "iloc"):
            X2 = X.iloc[idx]
        else:
            X2 = X[idx]
        if hasattr(y, "iloc"):
            y2 = y.iloc[idx]
        else:
            y2 = y[idx]
        print(f"[TUNE] Using tune sample n={len(X2)}")
        return X2, y2
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Train supervised baselines (multiclass/multilabel) on WAF features")
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", required=True, choices=["multiclass", "multilabel"])
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-features", default=None, help="Comma-separated feature names to exclude (for ablations)")

    # Baseline logistic-regression params
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--penalty", default="auto")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--l1-ratio", type=float, default=0.5)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=-1)

    # Rare-class handling (multiclass only)
    ap.add_argument("--rare-class-policy", default="keep", choices=["keep", "merge", "drop"])
    ap.add_argument("--rare-class-min-count", type=int, default=2)

    # Tuning
    ap.add_argument("--tune", default="none", choices=["none", "grid", "random", "halving"])
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--tune-n-iter", "--tune-budget", dest="tune_n_iter", type=int, default=24)
    ap.add_argument("--tune-n-jobs", type=int, default=-1)
    ap.add_argument("--tune-sample-n", type=int, default=0, help="Optional cap rows for tuning only (0 = full train)")
    ap.add_argument("--tune-results-out", default=None)
    ap.add_argument("--tune-factor", type=int, default=3)
    ap.add_argument("--tune-c-grid", default="0.001,0.01,0.1,1,10,100")
    ap.add_argument("--tune-solvers", default="saga,lbfgs")
    ap.add_argument("--tune-penalties", default="l2,l1,elasticnet,none")
    ap.add_argument("--tune-class-weights", default="balanced,none")
    ap.add_argument("--tune-l1-ratios", default="0.15,0.5,0.85")

    # Outputs / reproducibility
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--pred-out", default=None)

    # Feature importance
    ap.add_argument("--fi-kind", default="coef", choices=["coef", "permutation", "both", "none"])
    ap.add_argument("--fi-topk", type=int, default=25)
    ap.add_argument("--fi-n-repeats", type=int, default=5)
    ap.add_argument("--fi-n-jobs", type=int, default=-1)
    ap.add_argument("--fi-scoring", default=None)
    ap.add_argument("--max-eval", type=int, default=0, help="Optional cap rows for permutation FI and benchmark (0 = full test split)")

    # Real-time / efficiency metrics
    ap.add_argument("--benchmark-mode", default="auto", choices=["auto", "full", "inference", "none"])
    ap.add_argument("--benchmark-max-rows", type=int, default=2000)
    ap.add_argument("--benchmark-warmup-rows", type=int, default=100)
    ap.add_argument("--benchmark-repeats", type=int, default=1)
    ap.add_argument("--benchmark-load-levels", default="1,2,4")
    ap.add_argument("--benchmark-load-rows", type=int, default=300)
    ap.add_argument("--require-resource-metrics", action="store_true", help="Fail the run if CPU/RAM benchmark metrics cannot be measured.")
    ap.add_argument("--benchmark-out", default=None)

    args = ap.parse_args()

    df = _read_table(args.data)
    features = _resolve_features(args.drop_features)
    _ensure_features(df, features)

    fi_kind = (args.fi_kind or "none").strip().lower()
    do_coef = fi_kind in {"coef", "both"}
    do_perm = fi_kind in {"permutation", "both"}

    X_all = df[features].fillna(0)

    if args.task == "multiclass":
        y_all = df[args.label_col].astype(str)

        vc = y_all.value_counts(dropna=False)
        rare_classes = set(vc[vc < int(args.rare_class_min_count)].index.tolist())
        if rare_classes:
            if args.rare_class_policy == "drop":
                mask = ~y_all.isin(rare_classes)
                X_all = X_all.loc[mask].reset_index(drop=True)
                y_all = y_all.loc[mask].reset_index(drop=True)
                df = df.loc[mask].reset_index(drop=True)
                print(f"[INFO] Dropped {int((~mask).sum())} rows from rare classes (<{args.rare_class_min_count}).")
            elif args.rare_class_policy == "merge":
                y_all = y_all.where(~y_all.isin(rare_classes), other="OTHER_RARE")
                print(f"[INFO] Merged {len(rare_classes)} rare classes into OTHER_RARE.")
            else:
                print(f"[INFO] Keeping {len(rare_classes)} rare classes; stratify may fail.")

        X_train, X_test, y_train, y_test = _safe_train_test_split_multiclass(
            X_all, y_all, test_size=args.test_size, seed=args.seed, try_stratify=True
        )

        base_pipe = _build_multiclass_pipe(args)
        estimator: Pipeline = base_pipe
        tune_summary: Dict[str, object] = {
            "enabled": args.tune != "none",
            "mode": args.tune,
            "selection_metric": "f1_weighted",
            "tuning_time_seconds": None,
        }

        if args.tune != "none":
            X_tune, y_tune = _maybe_cap_tune_sample(X_train, y_train, args.tune_sample_n, args.seed)
            cv_obj = _make_cv_for_multiclass(y_tune, args.cv, args.seed)
            scoring = "f1_weighted"
            search_space = _build_multiclass_search_space(args)
            search = _make_search(
                estimator=estimator,
                mode=args.tune,
                search_space=search_space,
                scoring=scoring,
                cv=cv_obj,
                n_iter=args.tune_n_iter,
                n_jobs=args.tune_n_jobs,
                seed=args.seed,
                factor=args.tune_factor,
            )
            tune_t0 = time.perf_counter()
            search.fit(X_tune, y_tune)
            tuning_time_seconds = time.perf_counter() - tune_t0
            best_params = dict(search.best_params_)
            print(f"[TUNE] best_params={best_params}")
            print(f"[TUNE] best_score={search.best_score_}")
            print(f"[TUNE] tuning_time_seconds={tuning_time_seconds:.6f}")
            tune_summary.update({
                "best_params": best_params,
                "best_score": _safe_float(search.best_score_),
                "cv": int(args.cv),
                "tuning_time_seconds": float(tuning_time_seconds),
            })
            if args.tune_results_out:
                os.makedirs(os.path.dirname(args.tune_results_out) or ".", exist_ok=True)
                pd.DataFrame(search.cv_results_).to_csv(args.tune_results_out, index=False)
                print(f"[TUNE] Saved results: {args.tune_results_out}")
            estimator = base_pipe.set_params(**best_params)

        t0 = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_time_seconds = time.perf_counter() - t0

        y_pred = estimator.predict(X_test)
        y_prob = None
        try:
            y_prob = estimator.predict_proba(X_test)
        except Exception:
            y_prob = None

        classes = list(getattr(estimator.named_steps["lr"], "classes_", sorted(set(y_train.astype(str)))))
        metrics = _multiclass_metrics(y_test, y_pred, classes, y_prob=y_prob)
        _print_multiclass_metrics(metrics, title="Test evaluation (multiclass)")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {
            "task": "multiclass",
            "model": estimator,
            "features": features,
            "classes": classes,
            "config": {
                "solver": args.solver,
                "penalty": args.penalty,
                "C": float(args.C),
                "class_weight": args.class_weight,
                "l1_ratio": float(args.l1_ratio),
                "max_iter": int(args.max_iter),
            },
        }
        joblib.dump(payload, args.out)
        print(f"Saved model: {args.out}")

        metrics_payload: Dict[str, object] = {
            "task": "multiclass",
            "data": str(args.data),
            "label_col": str(args.label_col),
            "test_size": float(args.test_size),
            "seed": int(args.seed),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features_used": features,
            "features_dropped": [f for f in DEFAULT_FEATURES if f not in features],
            "suspicious_token_ablation": sorted([f for f in SUSPICIOUS_TOKEN_FEATURES if f not in features]),
            "tuning": tune_summary,
            "tuning_time_seconds": _safe_float(tune_summary.get("tuning_time_seconds")),
            "train_time_seconds": float(train_time_seconds),
            "evaluation": metrics,
        }

        if args.pred_out:
            pred_df = pd.DataFrame({"y_true": y_test.astype(str).to_numpy(), "y_pred": np.asarray(y_pred).astype(str)})
            if y_prob is not None:
                pred_df["pred_prob_max"] = np.asarray(y_prob).max(axis=1)
                pred_df = pd.concat([pred_df, _parse_prob_columns(np.asarray(y_prob), classes)], axis=1)
            _save_table(pred_df, args.pred_out)
            print(f"Saved predictions: {args.pred_out}")

        if do_coef:
            df_coef = _coef_importance_multiclass(estimator, features)
            out_path = _derive_fi_path(args.out, "coef")
            df_coef.to_csv(out_path, index=False)
            print(f"Saved coef importance: {out_path}")
            _print_top(df_coef, "importance_abs_mean", args.fi_topk)

        if do_perm:
            X_fi, y_fi = X_test, y_test
            if args.max_eval and args.max_eval > 0 and len(X_fi) > args.max_eval:
                idx = np.random.default_rng(args.seed).choice(len(X_fi), size=int(args.max_eval), replace=False)
                X_fi = X_fi.iloc[idx]
                y_fi = y_fi.iloc[idx]
            scoring = args.fi_scoring or "f1_weighted"
            df_perm = _perm_importance(
                estimator,
                X_fi,
                y_fi,
                features,
                scoring=scoring,
                n_repeats=int(args.fi_n_repeats),
                seed=int(args.seed),
                n_jobs=int(args.fi_n_jobs),
            )
            out_path = _derive_fi_path(args.out, "perm")
            df_perm.to_csv(out_path, index=False)
            print(f"Saved permutation importance (scoring={scoring}): {out_path}")
            _print_top(df_perm, "perm_importance_mean", args.fi_topk)

        df_eval_bench = df.loc[X_test.index].copy()
        if args.max_eval and args.max_eval > 0 and len(df_eval_bench) > args.max_eval:
            df_eval_bench = df_eval_bench.sample(args.max_eval, random_state=args.seed).reset_index(drop=True)
        else:
            df_eval_bench = df_eval_bench.reset_index(drop=True)

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
            model_path=args.out,
            train_time_seconds=train_time_seconds,
        )
        _print_benchmark(bench_payload, title="Benchmark")

        _save_json(metrics_payload, args.metrics_out)
        _save_json(bench_payload, args.benchmark_out)
        return

    # -------------------------
    # Multilabel (OvR)
    # -------------------------
    y_raw = df[args.label_col].apply(_parse_multilabel_cell)
    mlb = MultiLabelBinarizer()
    Y_all = mlb.fit_transform(y_raw)
    labels = [str(c) for c in mlb.classes_]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_all, Y_all, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    base_pipe = _build_multilabel_pipe(args)
    estimator: Pipeline = base_pipe
    tune_summary: Dict[str, object] = {
        "enabled": args.tune != "none",
        "mode": args.tune,
        "selection_metric": "f1_micro",
        "tuning_time_seconds": None,
    }

    if args.tune != "none":
        X_tune, Y_tune = _maybe_cap_tune_sample(X_train, Y_train, args.tune_sample_n, args.seed)
        scorer = make_scorer(f1_score, average="micro", zero_division=0)
        cv_obj = KFold(n_splits=max(2, int(args.cv)), shuffle=True, random_state=int(args.seed))
        search_space = _build_multilabel_search_space(args)
        search = _make_search(
            estimator=estimator,
            mode=args.tune,
            search_space=search_space,
            scoring=scorer,
            cv=cv_obj,
            n_iter=args.tune_n_iter,
            n_jobs=args.tune_n_jobs,
            seed=args.seed,
            factor=args.tune_factor,
        )
        tune_t0 = time.perf_counter()
        search.fit(X_tune, Y_tune)
        tuning_time_seconds = time.perf_counter() - tune_t0
        best_params = dict(search.best_params_)
        print(f"[TUNE] best_params={best_params}")
        print(f"[TUNE] best_score={search.best_score_}")
        print(f"[TUNE] tuning_time_seconds={tuning_time_seconds:.6f}")
        tune_summary.update({
            "best_params": best_params,
            "best_score": _safe_float(search.best_score_),
            "cv": int(args.cv),
            "tuning_time_seconds": float(tuning_time_seconds),
        })
        if args.tune_results_out:
            os.makedirs(os.path.dirname(args.tune_results_out) or ".", exist_ok=True)
            pd.DataFrame(search.cv_results_).to_csv(args.tune_results_out, index=False)
            print(f"[TUNE] Saved results: {args.tune_results_out}")
        estimator = base_pipe.set_params(**best_params)

    t0 = time.perf_counter()
    estimator.fit(X_train, Y_train)
    train_time_seconds = time.perf_counter() - t0

    Y_pred = estimator.predict(X_test)
    metrics = _multilabel_metrics(Y_test, Y_pred, labels)
    _print_multilabel_metrics(metrics, title="Test evaluation (multilabel)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "task": "multilabel",
        "model": estimator,
        "mlb": mlb,
        "features": features,
        "labels": labels,
        "config": {
            "solver": args.solver,
            "penalty": args.penalty,
            "C": float(args.C),
            "class_weight": args.class_weight,
            "l1_ratio": float(args.l1_ratio),
            "max_iter": int(args.max_iter),
        },
    }
    joblib.dump(payload, args.out)
    print(f"Saved model+mlb: {args.out}")

    metrics_payload = {
        "task": "multilabel",
        "data": str(args.data),
        "label_col": str(args.label_col),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features_used": features,
        "features_dropped": [f for f in DEFAULT_FEATURES if f not in features],
        "suspicious_token_ablation": sorted([f for f in SUSPICIOUS_TOKEN_FEATURES if f not in features]),
        "tuning": tune_summary,
        "tuning_time_seconds": _safe_float(tune_summary.get("tuning_time_seconds")),
        "train_time_seconds": float(train_time_seconds),
        "evaluation": metrics,
    }

    if args.pred_out:
        pred_df = pd.DataFrame({"n_labels_true": Y_test.sum(axis=1), "n_labels_pred": Y_pred.sum(axis=1)})
        _save_table(pred_df, args.pred_out)
        print(f"Saved predictions: {args.pred_out}")

    if do_coef:
        df_coef = _coef_importance_multilabel(estimator, features, labels)
        out_path = _derive_fi_path(args.out, "coef")
        df_coef.to_csv(out_path, index=False)
        print(f"Saved coef importance: {out_path}")
        _print_top(df_coef, "importance_abs_mean", args.fi_topk)

    if do_perm:
        X_fi, Y_fi = X_test, Y_test
        if args.max_eval and args.max_eval > 0 and len(X_fi) > args.max_eval:
            idx = np.random.default_rng(args.seed).choice(len(X_fi), size=int(args.max_eval), replace=False)
            X_fi = X_fi.iloc[idx]
            Y_fi = Y_fi[idx]
        scorer = make_scorer(f1_score, average="micro", zero_division=0) if args.fi_scoring is None else args.fi_scoring
        df_perm = _perm_importance(
            estimator,
            X_fi,
            Y_fi,
            features,
            scoring=scorer,
            n_repeats=int(args.fi_n_repeats),
            seed=int(args.seed),
            n_jobs=int(args.fi_n_jobs),
        )
        out_path = _derive_fi_path(args.out, "perm")
        df_perm.to_csv(out_path, index=False)
        print(f"Saved permutation importance: {out_path}")
        _print_top(df_perm, "perm_importance_mean", args.fi_topk)

    df_eval_bench = df.iloc[X_test.index].copy().reset_index(drop=True)
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
        seed=args.seed,
        model_path=args.out,
        train_time_seconds=train_time_seconds,
    )
    _print_benchmark(bench_payload, title="Benchmark")

    _save_json(metrics_payload, args.metrics_out)
    _save_json(bench_payload, args.benchmark_out)


if __name__ == "__main__":
    main()
