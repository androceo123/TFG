from __future__ import annotations

"""waf_ml.scripts.train_oneclass

Scalable one-class detector aligned with the thesis methodology, but configured
for exhaustive optimization when requested:

- Backend principal: StandardScaler + Nystroem + SGDOneClassSVM.
- Fits the one-class model ONLY with normal traffic.
- Default final training uses only the training-normal partition, preserving a
  final holdout for evaluation.
- Hyperparameter selection can use K-fold CV over normal traffic and attack
  validation folds when labels are available.
- Reports binary effectiveness metrics and optional real-time feasibility metrics.
- Computes direct per-feature permutation importance over the evaluation set.

Example:
  python -m waf_ml.scripts.train_ocsvm_updated \
      --backend sgd_ocsvm --kernel-approx nystroem \
      --mode train --data dataset.parquet --label-col label_binary \
      --test-size 0.2 --tune grid --cv 5 --final-train-on train_normals \
      --out model.joblib --metrics-out metrics.json
"""

import argparse
import json
import os
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

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def _load_many(paths: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = _read_table(p)
        df["_source_file"] = Path(p).name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _ensure_features(df: pd.DataFrame, features: List[str]) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} feature columns: {missing}\n"
            f"Available columns (sample): {list(df.columns)[:50]}"
        )


def _save_table(df: pd.DataFrame, path: str) -> None:
    p = path.lower()
    if p.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


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


def _get_model(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _predict(model: Pipeline, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df[features].fillna(0)
    pred_raw = model.predict(X)  # +1 normal, -1 anomaly
    out = df.copy()
    out["pred_raw"] = pred_raw
    out["pred_anomaly"] = (pred_raw == -1).astype(int)

    try:
        score_normal = np.asarray(model.decision_function(X), dtype=float)
        out["score_normal"] = score_normal  # higher = more normal
        out["score_anomaly"] = -score_normal  # higher = more anomalous
    except Exception:
        pass

    return out


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        if isinstance(v, (np.floating, float, int, np.integer)):
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                return None
            return fv
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            return None
        return fv
    except Exception:
        return None


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    labels_present = sorted(np.unique(np.concatenate([y_true, y_pred]))) if len(y_true) else [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out: Dict[str, object] = {
        "n": int(len(y_true)),
        "labels_present": labels_present,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(y_true, y_pred)),
        "precision": _safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _safe_float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": _safe_float(matthews_corrcoef(y_true, y_pred)),
        "specificity": _safe_float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        "fpr": _safe_float(fp / (fp + tn)) if (fp + tn) > 0 else None,
        "fnr": _safe_float(fn / (fn + tp)) if (fn + tp) > 0 else None,
        "tpr": _safe_float(tp / (tp + fn)) if (tp + fn) > 0 else None,
        "classification_report": classification_report(y_true, y_pred, digits=6, zero_division=0, output_dict=True),
    }

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true)) >= 2:
            try:
                out["roc_auc"] = _safe_float(roc_auc_score(y_true, y_score))
            except Exception:
                out["roc_auc"] = None
            try:
                out["pr_auc"] = _safe_float(average_precision_score(y_true, y_score))
            except Exception:
                out["pr_auc"] = None
        else:
            out["roc_auc"] = None
            out["pr_auc"] = None
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None

    return out


def _print_binary_metrics(metrics: Dict[str, object], *, title: str) -> None:
    print(f"=== {title} ===")
    cm = metrics.get("confusion_matrix", {}) or {}
    print(f"N:                    {metrics.get('n')}")
    print(f"TN / FP / FN / TP:    {cm.get('tn')} / {cm.get('fp')} / {cm.get('fn')} / {cm.get('tp')}")
    print(f"Accuracy:             {metrics.get('accuracy')}")
    print(f"Balanced accuracy:    {metrics.get('balanced_accuracy')}")
    print(f"Precision:            {metrics.get('precision')}")
    print(f"Recall (TPR):         {metrics.get('recall')}")
    print(f"Specificity (TNR):    {metrics.get('specificity')}")
    print(f"FPR / FNR:            {metrics.get('fpr')} / {metrics.get('fnr')}")
    print(f"F1:                   {metrics.get('f1')}")
    print(f"MCC:                  {metrics.get('mcc')}")
    print(f"ROC-AUC / PR-AUC:     {metrics.get('roc_auc')} / {metrics.get('pr_auc')}")


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


def _parse_int_grid(s: Optional[str]) -> List[int]:
    return [int(float(t)) for t in _parse_csv_list(s)]


def _parse_gamma_grid(s: Optional[str]) -> List[float]:
    return [float(t) for t in _parse_csv_list(s)]


def _build_oneclass_pipeline(
    *,
    backend: str,
    kernel_approx: str,
    nu: float,
    gamma: float,
    n_components: int,
    max_iter: int,
    seed: int,
) -> Pipeline:
    backend = (backend or "sgd_ocsvm").strip().lower()
    kernel_approx = (kernel_approx or "nystroem").strip().lower()
    if backend != "sgd_ocsvm":
        raise ValueError("Only --backend sgd_ocsvm is supported in this scalable script.")
    if kernel_approx != "nystroem":
        raise ValueError("Only --kernel-approx nystroem is supported in this scalable script.")

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "kernel_map",
            Nystroem(kernel="rbf", gamma=float(gamma), n_components=int(n_components), random_state=int(seed)),
        ),
        (
            "ocsvm",
            SGDOneClassSVM(
                nu=float(nu),
                max_iter=int(max_iter),
                shuffle=True,
                tol=1e-3,
                random_state=int(seed),
            ),
        ),
    ])


def _score_candidate(model: Pipeline, X_eval: pd.DataFrame, metric: str) -> Tuple[float, Dict[str, float]]:
    metric = (metric or "auto").strip().lower()

    pred = model.predict(X_eval)
    pred_anom = (pred == -1).astype(int)
    false_reject_rate = float(np.mean(pred_anom)) if len(pred_anom) else 0.0
    normal_acceptance = float(1.0 - false_reject_rate)

    try:
        mean_score = float(np.mean(model.decision_function(X_eval)))
    except Exception:
        mean_score = normal_acceptance

    details = {
        "normal_acceptance": normal_acceptance,
        "false_reject_rate": false_reject_rate,
        "mean_score": mean_score,
    }

    if metric in {"", "auto", "normal_acceptance", "acceptance"}:
        return normal_acceptance, details
    if metric in {"false_reject_rate", "frr"}:
        return -false_reject_rate, details
    if metric in {"mean_score"}:
        return mean_score, details

    raise ValueError(
        f"Unknown tuning metric '{metric}'. Use normal_acceptance | false_reject_rate | mean_score | auto"
    )


def _iter_candidates(
    tune: str,
    nu_grid: List[float],
    gamma_grid: List[float],
    n_components_grid: List[int],
    max_iter_grid: List[int],
    n_iter: int,
    seed: int,
) -> Iterable[Tuple[float, float, int, int]]:
    tune = (tune or "none").strip().lower()

    if tune == "grid":
        for nu in nu_grid:
            for gamma in gamma_grid:
                for n_components in n_components_grid:
                    for max_iter in max_iter_grid:
                        yield float(nu), float(gamma), int(n_components), int(max_iter)
        return

    if tune == "random":
        rng = np.random.default_rng(int(seed))
        combos = [
            (float(nu), float(gamma), int(n_components), int(max_iter))
            for nu in nu_grid
            for gamma in gamma_grid
            for n_components in n_components_grid
            for max_iter in max_iter_grid
        ]
        rng.shuffle(combos)
        for row in combos[: int(n_iter)]:
            yield row
        return

    return


def _percentiles_ms(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean_ms": None,
            "std_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def _peak_rss_mb() -> Optional[float]:
    if resource is None:
        return None
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(peak) / 1024.0
    except Exception:
        return None


def _predict_one_benchmark_oneclass(model: Pipeline, row: pd.Series, features: List[str], use_full_pipeline: bool) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
    t0 = time.perf_counter_ns()
    t1 = t0
    try:
        if use_full_pipeline:
            x_one = _features_from_raw_row(row, features)
            t1 = time.perf_counter_ns()
        else:
            x_one = pd.DataFrame([{c: row.get(c, 0) for c in features}])
        x_one = x_one[features].fillna(0).astype(np.float32)
        _ = model.predict(x_one)
        try:
            _ = model.decision_function(x_one)
        except Exception:
            pass
        t2 = time.perf_counter_ns()
        return (t2 - t0) / 1e6, (t1 - t0) / 1e6, (t2 - t1) / 1e6, True
    except Exception:
        return None, None, None, False


def _benchmark_load_profiles_oneclass(model: Pipeline, df_bench: pd.DataFrame, features: List[str], *, use_full_pipeline: bool, levels: List[int], max_rows: int) -> List[Dict[str, object]]:
    if len(df_bench) == 0 or not levels:
        return []
    rows = df_bench.head(min(len(df_bench), int(max_rows))).reset_index(drop=True)
    profiles: List[Dict[str, object]] = []
    for workers in sorted({max(1, int(x)) for x in levels}):
        lat_ms: List[float] = []
        ok = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_predict_one_benchmark_oneclass, model, row, features, use_full_pipeline) for _, row in rows.iterrows()]
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


def _benchmark_oneclass(
    model: Pipeline,
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
        total_ms, extract_ms, infer_ms, success = _predict_one_benchmark_oneclass(model, row, features, use_full_pipeline)
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

    load_profiles = _benchmark_load_profiles_oneclass(
        model,
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



# -----------------------------------------------------------------------------
# Exhaustive CV tuning + direct feature importance
# -----------------------------------------------------------------------------

def _build_oneclass_pipeline_v2(
    *,
    backend: str,
    kernel_approx: str,
    nu: float,
    gamma: float,
    n_components: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> Pipeline:
    backend = (backend or "sgd_ocsvm").strip().lower()
    kernel_approx = (kernel_approx or "nystroem").strip().lower()
    if backend != "sgd_ocsvm":
        raise ValueError("Only --backend sgd_ocsvm is supported in this scalable script.")
    if kernel_approx != "nystroem":
        raise ValueError("Only --kernel-approx nystroem is supported in this scalable script.")

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "kernel_map",
            Nystroem(kernel="rbf", gamma=float(gamma), n_components=int(n_components), random_state=int(seed)),
        ),
        (
            "ocsvm",
            SGDOneClassSVM(
                nu=float(nu),
                max_iter=int(max_iter),
                shuffle=True,
                tol=float(tol),
                random_state=int(seed),
            ),
        ),
    ])


def _normal_acceptance_from_pred(y_pred: np.ndarray) -> float:
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_pred) == 0:
        return 0.0
    return float(np.mean(y_pred == 0))


def _metric_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    metric: str,
) -> Tuple[float, str]:
    """Return selection score and effective metric name. Higher is always better."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metric_in = (metric or "auto").strip().lower()
    has_both_classes = len(np.unique(y_true)) >= 2 if len(y_true) else False

    # If no attacks are available in a tuning fold, supervised metrics are not
    # meaningful. In that case optimize low false rejection of normal traffic.
    supervised_metrics = {
        "f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc",
        "recall", "tpr", "precision", "accuracy",
    }
    if metric_in in {"", "auto"}:
        metric_eff = "f1" if has_both_classes else "normal_acceptance"
    elif (not has_both_classes) and metric_in in supervised_metrics:
        metric_eff = "normal_acceptance"
    else:
        metric_eff = metric_in

    try:
        if metric_eff in {"normal_acceptance", "acceptance"}:
            return _normal_acceptance_from_pred(y_pred), "normal_acceptance"
        if metric_eff in {"false_reject_rate", "frr"}:
            return -float(np.mean(y_pred == 1)) if len(y_pred) else 0.0, "false_reject_rate"
        if metric_eff == "accuracy":
            return float(accuracy_score(y_true, y_pred)), "accuracy"
        if metric_eff == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred)), "balanced_accuracy"
        if metric_eff in {"recall", "tpr"}:
            return float(recall_score(y_true, y_pred, zero_division=0)), "recall"
        if metric_eff == "precision":
            return float(precision_score(y_true, y_pred, zero_division=0)), "precision"
        if metric_eff == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0)), "f1"
        if metric_eff == "mcc":
            return float(matthews_corrcoef(y_true, y_pred)), "mcc"
        if metric_eff == "roc_auc":
            if y_score is None or not has_both_classes:
                return -np.inf, "roc_auc"
            return float(roc_auc_score(y_true, y_score)), "roc_auc"
        if metric_eff == "pr_auc":
            if y_score is None or not has_both_classes:
                return -np.inf, "pr_auc"
            return float(average_precision_score(y_true, y_score)), "pr_auc"
        if metric_eff == "mean_score":
            if y_score is None or len(y_score) == 0:
                return 0.0, "mean_score"
            # y_score is anomaly-oriented here; lower mean anomaly score on normal-only folds is better.
            if has_both_classes:
                return float(np.mean(y_score)), "mean_score"
            return -float(np.mean(y_score)), "mean_score"
    except Exception:
        return -np.inf, metric_eff

    raise ValueError(
        f"Unknown metric '{metric}'. Use auto|f1|balanced_accuracy|mcc|roc_auc|pr_auc|recall|precision|accuracy|normal_acceptance|false_reject_rate|mean_score."
    )


def _evaluate_oneclass_labeled(model: Pipeline, X_eval: pd.DataFrame, y_true: np.ndarray, metric: str) -> Tuple[float, str, Dict[str, object]]:
    pred_raw = model.predict(X_eval)
    y_pred = (np.asarray(pred_raw) == -1).astype(int)
    try:
        y_score = -np.asarray(model.decision_function(X_eval), dtype=float)
    except Exception:
        y_score = None
    score, effective_metric = _metric_from_arrays(y_true, y_pred, y_score, metric)
    metrics = _binary_metrics(y_true, y_pred, y_score)
    return score, effective_metric, metrics


def _iter_candidates_v2(
    tune: str,
    nu_grid: List[float],
    gamma_grid: List[float],
    n_components_grid: List[int],
    max_iter_grid: List[int],
    tol_grid: List[float],
    n_iter: int,
    seed: int,
) -> List[Dict[str, object]]:
    combos = [
        {
            "nu": float(nu),
            "gamma": float(gamma),
            "n_components": int(n_components),
            "max_iter": int(max_iter),
            "tol": float(tol),
        }
        for nu in nu_grid
        for gamma in gamma_grid
        for n_components in n_components_grid
        for max_iter in max_iter_grid
        for tol in tol_grid
    ]
    tune = (tune or "none").strip().lower()
    if tune == "random":
        rng = np.random.default_rng(int(seed))
        rng.shuffle(combos)
        combos = combos[: max(1, int(n_iter))]
    elif tune == "grid":
        pass
    elif tune == "none":
        combos = []
    else:
        raise ValueError(f"Unsupported tune mode: {tune}")
    return combos


def _make_cv_splits(df_norm: pd.DataFrame, df_att: pd.DataFrame, cv: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_norm = len(df_norm)
    if n_norm < 2:
        raise ValueError("CV tuning requires at least 2 normal rows.")
    folds = max(2, min(int(cv), n_norm))

    norm_indices = np.arange(n_norm)
    norm_kf = KFold(n_splits=folds, shuffle=True, random_state=int(seed))
    norm_splits = list(norm_kf.split(norm_indices))

    n_att = len(df_att)
    if n_att == 0:
        att_val_splits = [np.asarray([], dtype=int) for _ in range(folds)]
    elif n_att >= folds:
        att_indices = np.arange(n_att)
        att_kf = KFold(n_splits=folds, shuffle=True, random_state=int(seed) + 17)
        att_val_splits = [val_idx for _, val_idx in att_kf.split(att_indices)]
    else:
        # With very few attacks, use them in each validation fold for tuning only.
        att_val_splits = [np.arange(n_att, dtype=int) for _ in range(folds)]

    return [(train_idx, val_idx, att_val_splits[i]) for i, (train_idx, val_idx) in enumerate(norm_splits)]


def _mean_metric_from_folds(fold_metrics: List[Dict[str, object]], key: str) -> Optional[float]:
    vals = [_safe_float(m.get(key)) for m in fold_metrics]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _evaluate_candidate_cv(
    candidate: Dict[str, object],
    df_norm: pd.DataFrame,
    df_att: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    features: List[str],
    label_col: str,
    metric: str,
    backend: str,
    kernel_approx: str,
    seed: int,
    tune_max_eval: int,
) -> Dict[str, object]:
    row: Dict[str, object] = dict(candidate)
    fold_scores: List[float] = []
    fold_metrics: List[Dict[str, object]] = []
    effective_metrics: List[str] = []
    t0 = time.perf_counter()
    try:
        for fold_id, (norm_train_idx, norm_val_idx, att_val_idx) in enumerate(splits, start=1):
            df_fit = df_norm.iloc[norm_train_idx]
            val_parts = [df_norm.iloc[norm_val_idx]]
            if len(att_val_idx) > 0:
                val_parts.append(df_att.iloc[att_val_idx])
            df_val = pd.concat(val_parts, ignore_index=True)
            if tune_max_eval and tune_max_eval > 0 and len(df_val) > tune_max_eval:
                df_val = df_val.sample(int(tune_max_eval), random_state=int(seed) + fold_id).reset_index(drop=True)

            X_fit = df_fit[features].fillna(0)
            X_val = df_val[features].fillna(0)
            y_val = df_val[label_col].astype(int).to_numpy()

            m = _build_oneclass_pipeline_v2(
                backend=backend,
                kernel_approx=kernel_approx,
                nu=float(candidate["nu"]),
                gamma=float(candidate["gamma"]),
                n_components=int(candidate["n_components"]),
                max_iter=int(candidate["max_iter"]),
                tol=float(candidate["tol"]),
                seed=int(seed) + fold_id,
            )
            m.fit(X_fit)
            score, effective_metric, metrics = _evaluate_oneclass_labeled(m, X_val, y_val, metric)
            fold_scores.append(float(score))
            effective_metrics.append(effective_metric)
            fold_metrics.append(metrics)

        row.update({
            "score": float(np.mean(fold_scores)) if fold_scores else -np.inf,
            "score_std": float(np.std(fold_scores, ddof=0)) if fold_scores else None,
            "effective_metric": effective_metrics[0] if effective_metrics else str(metric),
            "folds": int(len(fold_scores)),
            "candidate_time_seconds": float(time.perf_counter() - t0),
            "accuracy_mean": _mean_metric_from_folds(fold_metrics, "accuracy"),
            "balanced_accuracy_mean": _mean_metric_from_folds(fold_metrics, "balanced_accuracy"),
            "precision_mean": _mean_metric_from_folds(fold_metrics, "precision"),
            "recall_mean": _mean_metric_from_folds(fold_metrics, "recall"),
            "f1_mean": _mean_metric_from_folds(fold_metrics, "f1"),
            "mcc_mean": _mean_metric_from_folds(fold_metrics, "mcc"),
            "roc_auc_mean": _mean_metric_from_folds(fold_metrics, "roc_auc"),
            "pr_auc_mean": _mean_metric_from_folds(fold_metrics, "pr_auc"),
            "fpr_mean": _mean_metric_from_folds(fold_metrics, "fpr"),
            "fnr_mean": _mean_metric_from_folds(fold_metrics, "fnr"),
            "specificity_mean": _mean_metric_from_folds(fold_metrics, "specificity"),
            "error": "",
        })
        return row
    except Exception as e:
        row.update({
            "score": -np.inf,
            "score_std": None,
            "effective_metric": str(metric),
            "folds": 0,
            "candidate_time_seconds": float(time.perf_counter() - t0),
            "error": repr(e),
        })
        return row


def _derive_fi_path(model_out: str, tag: str) -> str:
    return f"{model_out}.feature_importance.{tag}.csv"


def _print_top_importance(df_imp: pd.DataFrame, col: str, k: int) -> None:
    if df_imp is None or df_imp.empty or int(k) <= 0:
        return
    print(f"\n=== Top {int(k)} features by {col} ===")
    for _, r in df_imp.sort_values(col, ascending=False).head(int(k)).iterrows():
        print(f"{r['feature']}: {r[col]:.6g}")


def _permutation_feature_importance_oneclass(
    model: Pipeline,
    X: pd.DataFrame,
    y_true: np.ndarray,
    features: List[str],
    *,
    metric: str,
    n_repeats: int,
    seed: int,
    n_jobs: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    X = X[features].fillna(0).reset_index(drop=True)
    y_true = np.asarray(y_true).astype(int)

    pred_raw = model.predict(X)
    y_pred = (np.asarray(pred_raw) == -1).astype(int)
    try:
        y_score = -np.asarray(model.decision_function(X), dtype=float)
    except Exception:
        y_score = None
    baseline_score, effective_metric = _metric_from_arrays(y_true, y_pred, y_score, metric)

    repeats = max(1, int(n_repeats))
    n_jobs_eff = int(n_jobs) if int(n_jobs) != 0 else 1

    def _one_feature(j: int, feat: str) -> Dict[str, object]:
        rng = np.random.default_rng(int(seed) + 1009 * (j + 1))
        scores: List[float] = []
        values_original = X[feat].to_numpy(copy=True)
        for _ in range(repeats):
            Xp = X.copy()
            values = values_original.copy()
            rng.shuffle(values)
            Xp[feat] = values
            pred_raw_p = model.predict(Xp)
            y_pred_p = (np.asarray(pred_raw_p) == -1).astype(int)
            try:
                y_score_p = -np.asarray(model.decision_function(Xp), dtype=float)
            except Exception:
                y_score_p = None
            s, _ = _metric_from_arrays(y_true, y_pred_p, y_score_p, effective_metric)
            scores.append(float(s))
        arr = np.asarray(scores, dtype=float)
        perm_mean = float(arr.mean()) if len(arr) else None
        perm_std = float(arr.std(ddof=0)) if len(arr) else None
        return {
            "feature": feat,
            "baseline_score": float(baseline_score),
            "permuted_score_mean": perm_mean,
            "permuted_score_std": perm_std,
            "importance_mean_delta": float(baseline_score - perm_mean) if perm_mean is not None else None,
            "importance_abs_delta": abs(float(baseline_score - perm_mean)) if perm_mean is not None else None,
            "metric": effective_metric,
            "n_repeats": int(repeats),
            "n_rows": int(len(X)),
        }

    if n_jobs_eff == 1:
        rows = [_one_feature(j, feat) for j, feat in enumerate(features)]
    else:
        rows = joblib.Parallel(n_jobs=n_jobs_eff, prefer="threads")(
            joblib.delayed(_one_feature)(j, feat) for j, feat in enumerate(features)
        )

    df_imp = pd.DataFrame(rows).sort_values("importance_mean_delta", ascending=False).reset_index(drop=True)
    summary = {
        "kind": "permutation",
        "metric": effective_metric,
        "baseline_score": _safe_float(baseline_score),
        "n_rows": int(len(X)),
        "n_repeats": int(repeats),
    }
    return df_imp, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/Test optimized scalable one-class detector on WAF features.")
    ap.add_argument("--backend", default="sgd_ocsvm", choices=["sgd_ocsvm"])
    ap.add_argument("--kernel-approx", default="nystroem", choices=["nystroem"])
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    ap.add_argument("--data", nargs="+", required=True, help="One or more CSV/Parquet files with feature columns")
    ap.add_argument("--label-col", default="label_binary", help="0 normal / 1 attack (if present)")
    ap.add_argument("--out", help="(train) Path to save joblib pipeline")
    ap.add_argument("--model", help="(test) Path to load joblib pipeline")
    ap.add_argument("--drop-features", default=None, help="Comma-separated feature names to exclude manually. Default uses all features.")

    # Baseline params (used when --tune none)
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--n-components", type=int, default=512)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=1e-4)

    ap.add_argument("--test-size", type=float, default=0.2, help="Held-out normal/attack fraction for final evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--final-train-on",
        default="train_normals",
        choices=["train_normals", "all_normals"],
        help="Use train_normals for clean holdout metrics; use all_normals only for final deployment training.",
    )

    ap.add_argument("--max-train", type=int, default=0, help="Optional cap for FINAL training normal rows (0 = no cap)")
    ap.add_argument("--max-eval", type=int, default=0, help="Optional cap for final evaluation rows/predictions/FI (0 = use all).")
    ap.add_argument("--pred-out", default=None, help="Optional: save predictions for evaluation/test data")
    ap.add_argument("--eval-data", nargs="*", default=None, help="(train) Extra CSV/Parquet file(s) to include in final evaluation")

    # Exhaustive tuning options
    ap.add_argument("--tune", default="grid", choices=["none", "grid", "random"], help="Hyperparameter search. Default grid is intentionally exhaustive.")
    ap.add_argument("--cv", type=int, default=5, help="K-fold CV over normal traffic during tuning.")
    ap.add_argument(
        "--tune-metric",
        default="auto",
        choices=[
            "auto", "f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc",
            "recall", "tpr", "precision", "accuracy", "normal_acceptance",
            "false_reject_rate", "mean_score",
        ],
        help="Metric optimized during CV. auto uses f1 when attacks are available; otherwise normal_acceptance.",
    )
    ap.add_argument("--tune-nu-grid", default="0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2")
    ap.add_argument("--tune-gamma-grid", default="0.001,0.003,0.01,0.03,0.05,0.1,0.2,0.3,0.5,1.0")
    ap.add_argument("--tune-n-components-grid", default="128,256,512,1024,2048")
    ap.add_argument("--tune-max-iter-grid", default="1000,2000,3000,5000,8000")
    ap.add_argument("--tune-tol-grid", default="0.001,0.0005,0.0001")
    ap.add_argument("--tune-n-iter", "--tune-budget", dest="tune_n_iter", type=int, default=128, help="Only used with --tune random")
    ap.add_argument("--tune-n-jobs", type=int, default=-1, help="Parallel candidate evaluation. Use 1 if RAM is tight.")
    ap.add_argument("--tune-max-train", type=int, default=0, help="Optional cap for tuning normal rows (0 = no cap)")
    ap.add_argument("--tune-max-eval", type=int, default=0, help="Optional cap for validation rows per CV fold (0 = no cap)")
    ap.add_argument("--tune-results-out", default=None, help="Optional CSV to save per-candidate CV scores")

    # Effectiveness outputs
    ap.add_argument("--metrics-out", default=None, help="Optional JSON path for binary effectiveness metrics")

    # Direct feature importance: no automatic 'without suspicious tokens' variant.
    ap.add_argument("--fi-kind", default="permutation", choices=["permutation", "none"], help="Direct per-feature importance on the final evaluation set.")
    ap.add_argument("--fi-out", default=None, help="Optional CSV path for feature importance. Default: <out>.feature_importance.permutation.csv")
    ap.add_argument("--fi-topk", type=int, default=25)
    ap.add_argument("--fi-n-repeats", type=int, default=10)
    ap.add_argument("--fi-n-jobs", type=int, default=-1)
    ap.add_argument("--fi-scoring", default=None, choices=["auto", "f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc", "recall", "tpr", "precision", "accuracy", "normal_acceptance", "false_reject_rate", "mean_score"])
    ap.add_argument("--fi-max-rows", type=int, default=0, help="Optional cap only for permutation FI rows (0 = full eval set).")

    # Real-time / efficiency metrics
    ap.add_argument("--benchmark-mode", default="auto", choices=["auto", "full", "inference", "none"])
    ap.add_argument("--benchmark-max-rows", type=int, default=2000)
    ap.add_argument("--benchmark-warmup-rows", type=int, default=100)
    ap.add_argument("--benchmark-repeats", type=int, default=1)
    ap.add_argument("--benchmark-load-levels", default="1,2,4")
    ap.add_argument("--benchmark-load-rows", type=int, default=300)
    ap.add_argument("--require-resource-metrics", action="store_true", help="Fail the run if CPU/RAM benchmark metrics cannot be measured.")
    ap.add_argument("--benchmark-out", default=None, help="Optional JSON path for latency/throughput/CPU/RAM metrics")

    args = ap.parse_args()

    if args.mode == "train" and not args.out:
        ap.error("--out is required in --mode train")
    if args.mode == "test" and not args.model:
        ap.error("--model is required in --mode test")

    df = _load_many(args.data)
    features = _resolve_features(args.drop_features)
    _ensure_features(df, features)

    if args.mode == "train":
        label_col = args.label_col
        if label_col not in df.columns:
            raise ValueError(f"Training requires '{label_col}' to identify normal traffic (0).")

        df_norm = df[df[label_col] == 0].copy().reset_index(drop=True)
        df_att = df[df[label_col] == 1].copy().reset_index(drop=True)
        if len(df_norm) == 0:
            raise ValueError(f"No normal rows found where {label_col} == 0. Cannot train one-class detector.")

        if args.test_size and 0.0 < args.test_size < 1.0 and len(df_norm) >= 2:
            df_norm_train, df_norm_hold = train_test_split(
                df_norm,
                test_size=float(args.test_size),
                random_state=int(args.seed),
                shuffle=True,
            )
            df_norm_train = df_norm_train.reset_index(drop=True)
            df_norm_hold = df_norm_hold.reset_index(drop=True)
        else:
            df_norm_train, df_norm_hold = df_norm, df_norm.iloc[0:0].copy()

        if args.test_size and 0.0 < args.test_size < 1.0 and len(df_att) >= 2:
            df_att_tune, df_att_hold = train_test_split(
                df_att,
                test_size=float(args.test_size),
                random_state=int(args.seed),
                shuffle=True,
            )
            df_att_tune = df_att_tune.reset_index(drop=True)
            df_att_hold = df_att_hold.reset_index(drop=True)
        else:
            # If there are too few attacks to split safely, keep them only for final evaluation.
            df_att_tune = df_att.iloc[0:0].copy()
            df_att_hold = df_att.copy()

        print(
            "[INFO] one-class split summary: "
            f"total_normals={len(df_norm)} train_normals={len(df_norm_train)} holdout_normals={len(df_norm_hold)} "
            f"total_attacks={len(df_att)} tune_attacks={len(df_att_tune)} holdout_attacks={len(df_att_hold)}"
        )

        # Build final evaluation set first. This set is not used for fitting.
        eval_parts: List[pd.DataFrame] = []
        if len(df_norm_hold) > 0:
            eval_parts.append(df_norm_hold)
        if len(df_att_hold) > 0:
            eval_parts.append(df_att_hold)
        if args.eval_data:
            df_extra = _load_many(args.eval_data)
            _ensure_features(df_extra, features)
            eval_parts.append(df_extra)
        df_eval_full = pd.concat(eval_parts, ignore_index=True) if eval_parts else None
        if df_eval_full is not None and label_col in df_eval_full.columns:
            print(
                "[INFO] Final evaluation set summary: "
                f"rows={len(df_eval_full)} normals={(df_eval_full[label_col] == 0).sum()} "
                f"anomalies={(df_eval_full[label_col] == 1).sum()}"
            )

        best_params = {
            "nu": float(args.nu),
            "gamma": float(args.gamma),
            "n_components": int(args.n_components),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
        }
        tune_mode = (args.tune or "none").strip().lower()
        tune_summary: Dict[str, object] = {
            "enabled": tune_mode != "none",
            "mode": tune_mode,
            "protocol": "kfold_cv",
            "requested_metric": str(args.tune_metric),
            "cv": int(args.cv),
            "tuning_time_seconds": None,
            "n_candidates_evaluated": 0,
        }

        if tune_mode != "none":
            df_norm_tune = df_norm_train
            if args.tune_max_train and args.tune_max_train > 0 and len(df_norm_tune) > args.tune_max_train:
                df_norm_tune = df_norm_tune.sample(int(args.tune_max_train), random_state=int(args.seed)).reset_index(drop=True)

            if len(df_norm_tune) < 2:
                raise ValueError("--tune requires at least 2 normal rows inside the normal training block.")

            nu_grid = _parse_float_grid(args.tune_nu_grid)
            gamma_grid = _parse_gamma_grid(args.tune_gamma_grid)
            n_components_grid = _parse_int_grid(args.tune_n_components_grid)
            max_iter_grid = _parse_int_grid(args.tune_max_iter_grid)
            tol_grid = _parse_float_grid(args.tune_tol_grid)
            if not nu_grid or not gamma_grid or not n_components_grid or not max_iter_grid or not tol_grid:
                raise ValueError("Empty tuning grids. Check the --tune-* grid arguments.")

            candidates = _iter_candidates_v2(
                tune_mode,
                nu_grid,
                gamma_grid,
                n_components_grid,
                max_iter_grid,
                tol_grid,
                args.tune_n_iter,
                args.seed,
            )
            splits = _make_cv_splits(df_norm_tune, df_att_tune, args.cv, args.seed)
            print(
                f"[TUNE] protocol=kfold_cv mode={tune_mode} folds={len(splits)} candidates={len(candidates)} "
                f"normals_for_tuning={len(df_norm_tune)} attacks_for_validation={len(df_att_tune)} metric={args.tune_metric}"
            )

            tune_t0 = time.perf_counter()
            n_jobs_eff = int(args.tune_n_jobs) if int(args.tune_n_jobs) != 0 else 1
            if n_jobs_eff == 1:
                rows = []
                for i, cand in enumerate(candidates, start=1):
                    row = _evaluate_candidate_cv(
                        cand,
                        df_norm_tune,
                        df_att_tune,
                        splits,
                        features,
                        label_col,
                        args.tune_metric,
                        args.backend,
                        args.kernel_approx,
                        args.seed,
                        args.tune_max_eval,
                    )
                    rows.append(row)
                    if i % 10 == 0 or i == len(candidates):
                        best_so_far = max(rows, key=lambda r: float(r.get("score", -np.inf)))
                        print(
                            f"[TUNE] tried {i}/{len(candidates)} candidates; "
                            f"best_score={best_so_far.get('score')} params={{nu={best_so_far.get('nu')}, gamma={best_so_far.get('gamma')}, "
                            f"n_components={best_so_far.get('n_components')}, max_iter={best_so_far.get('max_iter')}, tol={best_so_far.get('tol')}}}"
                        )
            else:
                rows = joblib.Parallel(n_jobs=n_jobs_eff, prefer="threads")(
                    joblib.delayed(_evaluate_candidate_cv)(
                        cand,
                        df_norm_tune,
                        df_att_tune,
                        splits,
                        features,
                        label_col,
                        args.tune_metric,
                        args.backend,
                        args.kernel_approx,
                        args.seed,
                        args.tune_max_eval,
                    )
                    for cand in candidates
                )

            tuning_time_seconds = time.perf_counter() - tune_t0
            result_df = pd.DataFrame(rows)
            ok_df = result_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
            if ok_df.empty:
                raise RuntimeError("All tuning candidates failed. Check tune_results_out for errors.")
            best_row = ok_df.sort_values(["score", "score_std"], ascending=[False, True]).iloc[0].to_dict()
            best_params = {
                "nu": float(best_row["nu"]),
                "gamma": float(best_row["gamma"]),
                "n_components": int(best_row["n_components"]),
                "max_iter": int(best_row["max_iter"]),
                "tol": float(best_row["tol"]),
            }
            print(f"[TUNE] BEST score={best_row.get('score')} effective_metric={best_row.get('effective_metric')} params={best_params}")
            print(f"[TUNE] tuning_time_seconds={tuning_time_seconds:.6f}")

            tune_summary.update({
                "best_score": _safe_float(best_row.get("score")),
                "best_score_std": _safe_float(best_row.get("score_std")),
                "effective_metric": str(best_row.get("effective_metric")),
                "best_params": best_params,
                "normal_rows_for_tuning": int(len(df_norm_tune)),
                "attack_rows_for_validation": int(len(df_att_tune)),
                "tuning_time_seconds": float(tuning_time_seconds),
                "n_candidates_evaluated": int(len(rows)),
                "failed_candidates": int((result_df.get("error", "") != "").sum()) if "error" in result_df.columns else 0,
            })
            if args.tune_results_out:
                os.makedirs(os.path.dirname(args.tune_results_out) or ".", exist_ok=True)
                result_df.sort_values("score", ascending=False).to_csv(args.tune_results_out, index=False)
                print(f"[TUNE] Saved results: {args.tune_results_out}")

        if args.final_train_on == "all_normals":
            df_final_train = df_norm
            train_source = "all_normals"
        else:
            df_final_train = df_norm_train
            train_source = "train_normals"

        if args.max_train and args.max_train > 0 and len(df_final_train) > args.max_train:
            df_final_train = df_final_train.sample(int(args.max_train), random_state=int(args.seed)).reset_index(drop=True)

        X_train = df_final_train[features].fillna(0)
        model = _build_oneclass_pipeline_v2(
            backend=args.backend,
            kernel_approx=args.kernel_approx,
            nu=float(best_params["nu"]),
            gamma=float(best_params["gamma"]),
            n_components=int(best_params["n_components"]),
            max_iter=int(best_params["max_iter"]),
            tol=float(best_params["tol"]),
            seed=int(args.seed),
        )
        t_train0 = time.perf_counter()
        model.fit(X_train)
        train_time_seconds = time.perf_counter() - t_train0

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload_to_save = {
            "model": model,
            "backend": args.backend,
            "kernel_approx": args.kernel_approx,
            "features": features,
            "params": best_params,
            "tuning": tune_summary,
        }
        joblib.dump(payload_to_save, args.out)
        print(f"Saved model: {args.out}")

        metrics_payload: Dict[str, object] = {
            "mode": "train_eval",
            "backend": args.backend,
            "kernel_approx": args.kernel_approx,
            "label_col": label_col,
            "test_size": float(args.test_size),
            "seed": int(args.seed),
            "final_train_on": train_source,
            "best_params": best_params,
            "train_rows_used": int(len(df_final_train)),
            "train_time_seconds": float(train_time_seconds),
            "features_used": features,
            "features_dropped_manual": [f for f in DEFAULT_FEATURES if f not in features],
            "tuning": tune_summary,
            "tuning_time_seconds": _safe_float(tune_summary.get("tuning_time_seconds")),
            "split_summary": {
                "total_rows": int(len(df)),
                "total_normals": int(len(df_norm)),
                "total_attacks": int(len(df_att)),
                "train_normals": int(len(df_norm_train)),
                "holdout_normals": int(len(df_norm_hold)),
                "tune_attacks": int(len(df_att_tune)),
                "holdout_attacks": int(len(df_att_hold)),
            },
        }

        bench_payload: Dict[str, object] = {
            "enabled": False,
            "reason": "no_eval_data",
            "train_time_seconds": float(train_time_seconds),
        }

        if df_eval_full is not None and len(df_eval_full) > 0:
            df_eval = df_eval_full
            if args.max_eval and args.max_eval > 0 and len(df_eval) > args.max_eval:
                df_eval = df_eval.sample(int(args.max_eval), random_state=int(args.seed)).reset_index(drop=True)

            df_pred = _predict(model, df_eval, features)
            if label_col in df_pred.columns:
                y_true = df_pred[label_col].astype(int).to_numpy()
                y_hat = df_pred["pred_anomaly"].astype(int).to_numpy()
                y_score = df_pred["score_anomaly"].to_numpy() if "score_anomaly" in df_pred.columns else None
                binary = _binary_metrics(y_true, y_hat, y_score)
                metrics_payload["evaluation"] = binary
                _print_binary_metrics(binary, title="Final holdout metrics")
            else:
                anomaly_rate = float(df_pred["pred_anomaly"].mean()) if len(df_pred) else 0.0
                metrics_payload["evaluation"] = {"predicted_anomaly_rate": anomaly_rate}
                print(f"No '{label_col}' column found; anomaly rate = {anomaly_rate:.6f}")

            if args.pred_out:
                os.makedirs(os.path.dirname(args.pred_out) or ".", exist_ok=True)
                _save_table(df_pred, args.pred_out)
                print(f"Saved predictions: {args.pred_out}")

            if (args.fi_kind or "none").strip().lower() == "permutation" and label_col in df_eval.columns:
                df_fi = df_eval
                if args.fi_max_rows and args.fi_max_rows > 0 and len(df_fi) > args.fi_max_rows:
                    df_fi = df_fi.sample(int(args.fi_max_rows), random_state=int(args.seed)).reset_index(drop=True)
                X_fi = df_fi[features].fillna(0)
                y_fi = df_fi[label_col].astype(int).to_numpy()
                fi_metric = args.fi_scoring or args.tune_metric or "auto"
                fi_df, fi_summary = _permutation_feature_importance_oneclass(
                    model,
                    X_fi,
                    y_fi,
                    features,
                    metric=fi_metric,
                    n_repeats=args.fi_n_repeats,
                    seed=args.seed,
                    n_jobs=args.fi_n_jobs,
                )
                fi_out = args.fi_out or _derive_fi_path(args.out, "permutation")
                os.makedirs(os.path.dirname(fi_out) or ".", exist_ok=True)
                fi_df.to_csv(fi_out, index=False)
                print(f"Saved feature importance: {fi_out}")
                _print_top_importance(fi_df, "importance_mean_delta", args.fi_topk)
                fi_summary["path"] = fi_out
                metrics_payload["feature_importance"] = fi_summary
            else:
                metrics_payload["feature_importance"] = {"enabled": False, "reason": "disabled_or_missing_labels"}

            bench_payload = _benchmark_oneclass(
                model,
                df_eval,
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

    # Test mode
    loaded = joblib.load(args.model)
    model = _get_model(loaded)
    trained_features = loaded.get("features") if isinstance(loaded, dict) else None
    use_features = trained_features or features
    _ensure_features(df, use_features)

    df_pred = _predict(model, df, use_features)
    metrics_to_save: Optional[Dict[str, object]] = None
    if args.label_col in df_pred.columns:
        y_true = df_pred[args.label_col].astype(int).to_numpy()
        y_hat = df_pred["pred_anomaly"].astype(int).to_numpy()
        y_score = df_pred["score_anomaly"].to_numpy() if "score_anomaly" in df_pred.columns else None
        binary = _binary_metrics(y_true, y_hat, y_score)
        _print_binary_metrics(binary, title="Test metrics")
        metrics_to_save = {
            "mode": "test",
            "backend": loaded.get("backend") if isinstance(loaded, dict) else None,
            "kernel_approx": loaded.get("kernel_approx") if isinstance(loaded, dict) else None,
            "features_used": use_features,
            "evaluation": binary,
        }

        if (args.fi_kind or "none").strip().lower() == "permutation":
            df_fi = df
            if args.fi_max_rows and args.fi_max_rows > 0 and len(df_fi) > args.fi_max_rows:
                df_fi = df_fi.sample(int(args.fi_max_rows), random_state=int(args.seed)).reset_index(drop=True)
            X_fi = df_fi[use_features].fillna(0)
            y_fi = df_fi[args.label_col].astype(int).to_numpy()
            fi_metric = args.fi_scoring or args.tune_metric or "auto"
            fi_df, fi_summary = _permutation_feature_importance_oneclass(
                model,
                X_fi,
                y_fi,
                use_features,
                metric=fi_metric,
                n_repeats=args.fi_n_repeats,
                seed=args.seed,
                n_jobs=args.fi_n_jobs,
            )
            fi_out = args.fi_out or _derive_fi_path(args.model, "permutation.test")
            os.makedirs(os.path.dirname(fi_out) or ".", exist_ok=True)
            fi_df.to_csv(fi_out, index=False)
            print(f"Saved feature importance: {fi_out}")
            _print_top_importance(fi_df, "importance_mean_delta", args.fi_topk)
            fi_summary["path"] = fi_out
            metrics_to_save["feature_importance"] = fi_summary
    else:
        anomaly_rate = float(df_pred["pred_anomaly"].mean()) if len(df_pred) else 0.0
        print(f"No '{args.label_col}' column found; anomaly rate = {anomaly_rate:.6f}")
        metrics_to_save = {
            "mode": "test",
            "backend": loaded.get("backend") if isinstance(loaded, dict) else None,
            "kernel_approx": loaded.get("kernel_approx") if isinstance(loaded, dict) else None,
            "features_used": use_features,
            "evaluation": {"predicted_anomaly_rate": anomaly_rate},
        }

    if args.metrics_out and metrics_to_save is not None:
        _save_json(metrics_to_save, args.metrics_out)

    if args.pred_out:
        os.makedirs(os.path.dirname(args.pred_out) or ".", exist_ok=True)
        _save_table(df_pred, args.pred_out)
        print(f"Saved predictions: {args.pred_out}")

    bench_payload = _benchmark_oneclass(
        model,
        df,
        use_features,
        benchmark_mode=args.benchmark_mode,
        benchmark_max_rows=args.benchmark_max_rows,
        benchmark_warmup_rows=args.benchmark_warmup_rows,
        benchmark_repeats=args.benchmark_repeats,
        benchmark_load_levels=[int(x) for x in _parse_csv_list(args.benchmark_load_levels)] if args.benchmark_load_levels else [],
        benchmark_load_rows=args.benchmark_load_rows,
        require_resource_metrics=args.require_resource_metrics,
        seed=args.seed,
        model_path=args.model,
        train_time_seconds=None,
    )
    _print_benchmark(bench_payload, title="Benchmark")
    _save_json(bench_payload, args.benchmark_out)


if __name__ == "__main__":
    main()
