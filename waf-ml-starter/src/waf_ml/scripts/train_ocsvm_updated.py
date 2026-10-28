from __future__ import annotations

"""waf_ml.scripts.train_oneclass

Scalable one-class detector aligned with the thesis methodology:

- Backend principal: StandardScaler + Nystroem + SGDOneClassSVM
- Train ONLY with normal traffic.
- Default final training uses ONLY the 80% normal train split.
- Evaluation uses the 20% holdout of normals + attacks/anomalies.
- Reports binary effectiveness metrics and optional real-time feasibility metrics.
- Supports bounded tuning under a normal-only internal holdout.
- Supports efficient ablations through --drop-features.

Example:
  python -m waf_ml.scripts.train_oneclass \
      --backend sgd_ocsvm --kernel-approx nystroem \
      --mode train --data dataset.parquet --label-col label_binary \
      --test-size 0.2 --final-train-on train_normals \
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
from sklearn.model_selection import train_test_split
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/Test scalable one-class detector on WAF features.")
    ap.add_argument("--backend", default="sgd_ocsvm", choices=["sgd_ocsvm"])
    ap.add_argument("--kernel-approx", default="nystroem", choices=["nystroem"])
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    ap.add_argument("--data", nargs="+", required=True, help="One or more CSV/Parquet files with feature columns")
    ap.add_argument("--label-col", default="label_binary", help="0 normal / 1 attack (if present)")
    ap.add_argument("--out", help="(train) Path to save joblib pipeline")
    ap.add_argument("--model", help="(test) Path to load joblib pipeline")
    ap.add_argument("--drop-features", default=None, help="Comma-separated feature names to exclude (useful for ablations)")

    # Baseline params (used when --tune none)
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--n-components", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=2000)

    ap.add_argument("--test-size", type=float, default=0.2, help="Held-out NORMAL fraction for evaluation/tuning (thesis default: 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--final-train-on",
        default="train_normals",
        choices=["train_normals", "all_normals"],
        help="Final fit source. 'train_normals' matches the thesis holdout methodology.",
    )

    ap.add_argument("--max-train", type=int, default=0, help="Optional cap for FINAL training normal rows (0 = no cap)")
    ap.add_argument("--max-eval", type=int, default=0, help="Optional cap for evaluation rows used for metrics/predictions (0 = use all).")
    ap.add_argument("--pred-out", default=None, help="Optional: save predictions for evaluation/test data")
    ap.add_argument("--eval-data", nargs="*", default=None, help="(train) Extra CSV/Parquet file(s) to include in evaluation")

    # Tuning options
    ap.add_argument("--tune", default="none", choices=["none", "grid", "random"], help="Bounded search on scalable one-class hyperparameters")
    ap.add_argument(
        "--tune-metric",
        default="auto",
        choices=["auto", "normal_acceptance", "false_reject_rate", "mean_score"],
        help="Internal one-class tuning metric computed ONLY on a normal holdout split inside the normal training block.",
    )
    ap.add_argument("--tune-nu-grid", default="0.01,0.03,0.05,0.1")
    ap.add_argument("--tune-gamma-grid", default="0.01,0.05,0.1,0.2")
    ap.add_argument("--tune-n-components-grid", default="128,256,512")
    ap.add_argument("--tune-max-iter-grid", default="1000,2000,3000")
    ap.add_argument("--tune-n-iter", "--tune-budget", dest="tune_n_iter", type=int, default=16)
    ap.add_argument("--tune-holdout-size", type=float, default=0.25, help="Internal normal-only holdout fraction inside train_normals for tuning")
    ap.add_argument("--tune-max-train", type=int, default=0, help="Optional cap for tuning train normals (0 = no cap)")
    ap.add_argument("--tune-max-eval", type=int, default=0, help="Optional cap for tuning HOLDOUT normals rows (0 = no cap)")
    ap.add_argument("--tune-results-out", default=None, help="Optional CSV to save per-candidate tuning scores")

    # Effectiveness outputs
    ap.add_argument("--metrics-out", default=None, help="Optional JSON path for binary effectiveness metrics")

    # Efficient explainability path for one-class: use ablation runs, not permutation.
    ap.add_argument("--fi-kind", default="none", choices=["none"], help="One-class explainability is handled via ablation runs with --drop-features.")

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

        df_norm = df[df[label_col] == 0].copy()
        if len(df_norm) == 0:
            raise ValueError(f"No normal rows found where {label_col} == 0. Cannot train one-class detector.")

        if args.test_size and 0.0 < args.test_size < 1.0 and len(df_norm) >= 2:
            df_norm_train, df_norm_hold = train_test_split(
                df_norm,
                test_size=args.test_size,
                random_state=args.seed,
                shuffle=True,
            )
        else:
            df_norm_train, df_norm_hold = df_norm, df_norm.iloc[0:0].copy()

        print(
            "[INFO] one-class split summary: "
            f"total_normals={len(df_norm)} train_normals={len(df_norm_train)} holdout_normals={len(df_norm_hold)}"
        )

        df_norm_train_tune = df_norm_train
        if args.tune_max_train and args.tune_max_train > 0 and len(df_norm_train_tune) > args.tune_max_train:
            df_norm_train_tune = df_norm_train_tune.sample(args.tune_max_train, random_state=args.seed).reset_index(drop=True)

        df_tune_fit = df_norm_train_tune
        df_tune_hold = df_norm_train_tune.iloc[0:0].copy()
        tune_mode = (args.tune or "none").strip().lower()
        if tune_mode != "none":
            if len(df_norm_train_tune) < 2:
                raise ValueError("--tune requires at least 2 normal rows inside the normal training block.")
            if not (0.0 < float(args.tune_holdout_size) < 1.0):
                raise ValueError("--tune-holdout-size must be in (0, 1).")

            df_tune_fit, df_tune_hold = train_test_split(
                df_norm_train_tune,
                test_size=float(args.tune_holdout_size),
                random_state=args.seed,
                shuffle=True,
            )

            if len(df_tune_fit) == 0 or len(df_tune_hold) == 0:
                raise ValueError(
                    "Internal tuning split produced an empty fit/holdout block. Adjust --tune-holdout-size or provide more normal rows."
                )

            if args.tune_max_eval and args.tune_max_eval > 0 and len(df_tune_hold) > args.tune_max_eval:
                df_tune_hold = df_tune_hold.sample(args.tune_max_eval, random_state=args.seed).reset_index(drop=True)

            print(
                "[TUNE] internal normal-only split: "
                f"fit_normals={len(df_tune_fit)} holdout_normals={len(df_tune_hold)} metric={args.tune_metric}"
            )

        eval_parts: List[pd.DataFrame] = []
        if len(df_norm_hold) > 0:
            eval_parts.append(df_norm_hold)
        df_att_in_data = df[df[label_col] == 1].copy()
        if len(df_att_in_data) > 0:
            eval_parts.append(df_att_in_data)
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

        best_nu = float(args.nu)
        best_gamma = float(args.gamma)
        best_n_components = int(args.n_components)
        best_max_iter = int(args.max_iter)
        best_score = -np.inf
        tune_summary: Dict[str, object] = {
            "enabled": tune_mode != "none",
            "metric": str(args.tune_metric),
            "tune_holdout_size": float(args.tune_holdout_size),
            "tuning_time_seconds": None,
            "n_candidates_evaluated": 0,
        }

        if tune_mode != "none":
            X_train_tune = df_tune_fit[features].fillna(0)
            X_eval_tune = df_tune_hold[features].fillna(0)

            nu_grid = _parse_float_grid(args.tune_nu_grid)
            gamma_grid = _parse_gamma_grid(args.tune_gamma_grid)
            n_components_grid = _parse_int_grid(args.tune_n_components_grid)
            max_iter_grid = _parse_int_grid(args.tune_max_iter_grid)
            if not nu_grid or not gamma_grid or not n_components_grid or not max_iter_grid:
                raise ValueError("Empty tuning grids. Check the --tune-* grid arguments.")

            rows = []
            tune_t0 = time.perf_counter()
            print(
                f"[TUNE] mode={tune_mode} candidates: "
                f"nu={len(nu_grid)} gamma={len(gamma_grid)} n_components={len(n_components_grid)} max_iter={len(max_iter_grid)}"
            )
            for i, (nu, gamma, n_components, max_iter) in enumerate(
                _iter_candidates(tune_mode, nu_grid, gamma_grid, n_components_grid, max_iter_grid, args.tune_n_iter, args.seed),
                start=1,
            ):
                m = _build_oneclass_pipeline(
                    backend=args.backend,
                    kernel_approx=args.kernel_approx,
                    nu=nu,
                    gamma=gamma,
                    n_components=n_components,
                    max_iter=max_iter,
                    seed=args.seed,
                )
                candidate_t0 = time.perf_counter()
                m.fit(X_train_tune)
                score, diag = _score_candidate(m, X_eval_tune, args.tune_metric)
                candidate_time_seconds = time.perf_counter() - candidate_t0
                row = {
                    "nu": float(nu),
                    "gamma": float(gamma),
                    "n_components": int(n_components),
                    "max_iter": int(max_iter),
                    "score": float(score),
                    "normal_acceptance": float(diag["normal_acceptance"]),
                    "false_reject_rate": float(diag["false_reject_rate"]),
                    "mean_score": float(diag["mean_score"]),
                    "candidate_time_seconds": float(candidate_time_seconds),
                }
                rows.append(row)
                if score > best_score:
                    best_score = score
                    best_nu = float(nu)
                    best_gamma = float(gamma)
                    best_n_components = int(n_components)
                    best_max_iter = int(max_iter)
                    tune_summary["best_diag"] = diag
                if i % 10 == 0:
                    print(
                        f"[TUNE] tried {i} candidates; current best score={best_score:.6g} "
                        f"(nu={best_nu}, gamma={best_gamma}, n_components={best_n_components}, max_iter={best_max_iter})"
                    )

            tuning_time_seconds = time.perf_counter() - tune_t0
            print(
                f"[TUNE] BEST score={best_score:.6g} "
                f"(nu={best_nu}, gamma={best_gamma}, n_components={best_n_components}, max_iter={best_max_iter})"
            )
            print(f"[TUNE] tuning_time_seconds={tuning_time_seconds:.6f}")
            tune_summary.update({
                "best_score": float(best_score),
                "best_params": {
                    "nu": float(best_nu),
                    "gamma": float(best_gamma),
                    "n_components": int(best_n_components),
                    "max_iter": int(best_max_iter),
                },
                "fit_normals": int(len(df_tune_fit)),
                "holdout_normals": int(len(df_tune_hold)),
                "tuning_time_seconds": float(tuning_time_seconds),
                "n_candidates_evaluated": int(len(rows)),
            })
            if args.tune_results_out:
                os.makedirs(os.path.dirname(args.tune_results_out) or ".", exist_ok=True)
                pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(args.tune_results_out, index=False)
                print(f"[TUNE] Saved results: {args.tune_results_out}")

        if args.final_train_on == "all_normals":
            df_final_train = df_norm
            train_source = "all_normals"
        else:
            df_final_train = df_norm_train
            train_source = "train_normals"

        if args.max_train and args.max_train > 0 and len(df_final_train) > args.max_train:
            df_final_train = df_final_train.sample(args.max_train, random_state=args.seed).reset_index(drop=True)

        X_train = df_final_train[features].fillna(0)
        model = _build_oneclass_pipeline(
            backend=args.backend,
            kernel_approx=args.kernel_approx,
            nu=best_nu,
            gamma=best_gamma,
            n_components=best_n_components,
            max_iter=best_max_iter,
            seed=args.seed,
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
            "params": {
                "nu": best_nu,
                "gamma": best_gamma,
                "n_components": best_n_components,
                "max_iter": best_max_iter,
            },
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
            "best_params": {
                "nu": float(best_nu),
                "gamma": float(best_gamma),
                "n_components": int(best_n_components),
                "max_iter": int(best_max_iter),
            },
            "train_rows_used": int(len(df_final_train)),
            "train_time_seconds": float(train_time_seconds),
            "features_used": features,
            "features_dropped": [f for f in DEFAULT_FEATURES if f not in features],
            "suspicious_token_ablation": sorted([f for f in SUSPICIOUS_TOKEN_FEATURES if f not in features]),
            "tuning": tune_summary,
            "tuning_time_seconds": _safe_float(tune_summary.get("tuning_time_seconds")),
            "split_summary": {
                "total_rows": int(len(df)),
                "total_normals": int(len(df_norm)),
                "train_normals": int(len(df_norm_train)),
                "holdout_normals": int(len(df_norm_hold)),
                "attacks_in_main_data": int(len(df_att_in_data)),
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
                df_eval = df_eval.sample(args.max_eval, random_state=args.seed).reset_index(drop=True)

            df_pred = _predict(model, df_eval, features)
            if label_col in df_pred.columns:
                y_true = df_pred[label_col].astype(int).to_numpy()
                y_hat = df_pred["pred_anomaly"].astype(int).to_numpy()
                y_score = df_pred["score_anomaly"].to_numpy() if "score_anomaly" in df_pred.columns else None
                binary = _binary_metrics(y_true, y_hat, y_score)
                metrics_payload["evaluation"] = binary
                _print_binary_metrics(binary, title="Train/Eval metrics (holdout normals + attacks)")
            else:
                anomaly_rate = float(df_pred["pred_anomaly"].mean()) if len(df_pred) else 0.0
                metrics_payload["evaluation"] = {"predicted_anomaly_rate": anomaly_rate}
                print(f"No '{label_col}' column found; anomaly rate = {anomaly_rate:.6f}")

            if args.pred_out:
                os.makedirs(os.path.dirname(args.pred_out) or ".", exist_ok=True)
                _save_table(df_pred, args.pred_out)
                print(f"Saved predictions: {args.pred_out}")

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

    loaded = joblib.load(args.model)
    model = _get_model(loaded)
    trained_features = loaded.get("features") if isinstance(loaded, dict) else None
    use_features = trained_features or features
    _ensure_features(df, use_features)

    df_pred = _predict(model, df, use_features)
    if args.label_col in df_pred.columns:
        y_true = df_pred[args.label_col].astype(int).to_numpy()
        y_hat = df_pred["pred_anomaly"].astype(int).to_numpy()
        y_score = df_pred["score_anomaly"].to_numpy() if "score_anomaly" in df_pred.columns else None
        binary = _binary_metrics(y_true, y_hat, y_score)
        _print_binary_metrics(binary, title="Test metrics")
        if args.metrics_out:
            _save_json({
                "mode": "test",
                "backend": loaded.get("backend") if isinstance(loaded, dict) else None,
                "kernel_approx": loaded.get("kernel_approx") if isinstance(loaded, dict) else None,
                "features_used": use_features,
                "evaluation": binary,
            }, args.metrics_out)
    else:
        anomaly_rate = float(df_pred["pred_anomaly"].mean()) if len(df_pred) else 0.0
        print(f"No '{args.label_col}' column found; anomaly rate = {anomaly_rate:.6f}")
        if args.metrics_out:
            _save_json({
                "mode": "test",
                "backend": loaded.get("backend") if isinstance(loaded, dict) else None,
                "kernel_approx": loaded.get("kernel_approx") if isinstance(loaded, dict) else None,
                "features_used": use_features,
                "evaluation": {"predicted_anomaly_rate": anomaly_rate},
            }, args.metrics_out)

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
