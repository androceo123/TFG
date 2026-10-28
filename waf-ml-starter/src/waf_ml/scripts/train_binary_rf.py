#!/usr/bin/env python3
"""
train_binary_rf.py — RandomForest binario application-independent.

Replica el pipeline CSIC (rf_supervised) en cualquier dataset que tenga
las columnas:
    request_http_method, request_http_request, request_body, label_binary

Parametros identicos al experimento CSIC baseline:
    - 80/20 split estratificado, seed=42
    - RandomForestClassifier n_estimators=500, class_weight=balanced
    - Threshold sweep 0.05-0.95, optimo por F1
    - Mismas 57 features (o subconjunto: 25 / 34)

Uso:
    PYTHONPATH=src python src/waf_ml/scripts/train_binary_rf.py \
        --parquet data/processed/torpeda/torpeda_features.parquet \
        --dataset torpeda \
        --out-dir results/torpeda/binary \
        --feat-set 57 \
        --rebuild-features
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# =============================================================================
# Feature sets — identicos al pipeline CSIC
# =============================================================================

FEAT_25 = [
    # 15 estructurales
    "uri_len", "path_depth", "query_len", "n_query_params", "max_param_value_len",
    "uri_pct_non_alnum_ratio", "encoded", "suspicious_tokens_count", "has_suspicious_tokens",
    "body_suspicious_tokens_count", "body_has_suspicious_tokens", "body_encoded",
    "uncommon_method", "req_content_length", "body_len",
    # 10 metodo one-hot
    "method_GET", "method_POST", "method_HEAD", "method_PUT", "method_DELETE",
    "method_PATCH", "method_OPTIONS", "method_TRACE", "method_CONNECT", "method_OTHER",
]

FEAT_ENTROPY_V1 = [
    "uri_entropy", "query_entropy", "max_param_value_entropy",
    "query_pct_digit", "query_pct_alpha",
    "body_entropy", "body_pct_digit", "body_pct_alpha",
]

FEAT_CHAR_DIST_V2 = [
    "query_char_dist_i0", "query_char_dist_i1", "query_char_dist_i2",
    "query_char_dist_i3", "query_char_dist_i4",
    "max_param_char_dist_i0", "max_param_char_dist_i1", "max_param_char_dist_i2",
    "max_param_char_dist_i3", "max_param_char_dist_i4",
    "body_char_dist_i0", "body_char_dist_i1", "body_char_dist_i2",
    "body_char_dist_i3", "body_char_dist_i4",
    "max_body_param_char_dist_i0", "max_body_param_char_dist_i1", "max_body_param_char_dist_i2",
    "max_body_param_char_dist_i3", "max_body_param_char_dist_i4",
    "mean_param_value_entropy", "std_param_value_entropy",
    "max_body_param_value_entropy", "mean_body_param_value_entropy",
]

FEAT_34 = FEAT_25 + FEAT_ENTROPY_V1         # 25 + 8 = 33... se llaman 34 en el TFG
FEAT_57 = FEAT_25 + FEAT_ENTROPY_V1 + FEAT_CHAR_DIST_V2   # 25 + 8 + 24 = 57

FEATURE_SETS = {25: FEAT_25, 34: FEAT_34, 57: FEAT_57}


# =============================================================================
# Extracción de features — paralela con multiprocessing
# =============================================================================

def _extract_batch(args: tuple) -> list[dict]:
    """Worker: extrae features de un lote de requests."""
    methods, uris, bodies = args
    # Import dentro del worker para evitar problemas de pickling
    from waf_ml.features.http_features import extract_http_features  # noqa: PLC0415
    results = []
    for m, u, b in zip(methods, uris, bodies):
        body_bytes = b.encode("utf-8", errors="ignore") if b else None
        feat = extract_http_features(
            method=str(m or ""),
            uri=str(u or ""),
            body=body_bytes,
        )
        results.append(feat)
    return results


def rebuild_features(df: pd.DataFrame, n_jobs: int = 8) -> pd.DataFrame:
    """Re-extrae las 57 features HTTP en paralelo desde las columnas crudas."""
    methods = df["request_http_method"].fillna("").tolist()
    uris = df["request_http_request"].fillna("").tolist()
    bodies = df["request_body"].fillna("").tolist()

    n = len(df)
    batch_size = max(1, (n + n_jobs - 1) // n_jobs)
    batches = [
        (
            methods[i * batch_size: (i + 1) * batch_size],
            uris[i * batch_size: (i + 1) * batch_size],
            bodies[i * batch_size: (i + 1) * batch_size],
        )
        for i in range(n_jobs)
        if i * batch_size < n
    ]

    print(f"[INFO] Feature extraction: {n} filas en {len(batches)} lotes "
          f"(~{batch_size} filas/lote, {n_jobs} workers)...", flush=True)
    t0 = time.time()
    with mp.Pool(n_jobs) as pool:
        raw = pool.map(_extract_batch, batches)
    elapsed = time.time() - t0

    feat_rows = [row for batch in raw for row in batch]
    print(f"[INFO] Extraccion completada: {elapsed:.1f}s ({n / elapsed:.0f} filas/s)", flush=True)
    return pd.DataFrame(feat_rows)


# =============================================================================
# Metricas
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcula todas las metricas del pipeline CSIC + extras."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision":         float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1":                float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc":           float(roc_auc_score(y_true, y_prob)),
        "pr_auc":            float(average_precision_score(y_true, y_prob)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
        "fpr":               float(fpr),
        "fnr":               float(fnr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    """Barre umbrales 0.05-0.95 (paso 0.01) y devuelve tabla + fila optima (max F1)."""
    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1  = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        pre = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ba  = balanced_accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rows.append({
            "threshold":        round(float(thr), 2),
            "f1":               f1,
            "recall":           rec,
            "precision":        pre,
            "fpr":              fpr,
            "balanced_accuracy": ba,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })
    df = pd.DataFrame(rows)
    best = df.iloc[df["f1"].idxmax()]
    return df, best


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RF binario — pipeline identico al CSIC")
    parser.add_argument("--parquet",    required=True,  help="Parquet de entrada")
    parser.add_argument("--dataset",    required=True,  help="Nombre del dataset (torpeda/harvard/csic)")
    parser.add_argument("--out-dir",    required=True,  help="Directorio de salida (ej: results/torpeda/binary)")
    parser.add_argument("--feat-set",   type=int, choices=[25, 34, 57], default=57,
                        help="Conjunto de features: 25 / 34 / 57 (default: 57)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--test-size",  type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs",     type=int,   default=8)
    parser.add_argument("--rebuild-features", action="store_true",
                        help="Forzar re-extraccion de features desde columnas HTTP crudas")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr_dir = out_dir.parent / (out_dir.name + "_threshold")
    thr_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar parquet ────────────────────────────────────────────────────────
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        sys.exit(f"[ERROR] No se encuentra: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"\n[{args.dataset.upper()}] {len(df)} filas × {len(df.columns)} columnas", flush=True)
    label_dist = df["label_binary"].value_counts().sort_index()
    print(f"[INFO] label_binary: {label_dist.to_dict()}")
    print(f"[INFO] Ratio normal/ataque: {label_dist[0] / label_dist.get(1, 1):.2f}x", flush=True)

    # ── Re-extraer features si faltan o se fuerza ─────────────────────────────
    feat_cols = FEATURE_SETS[args.feat_set]
    missing   = [f for f in feat_cols if f not in df.columns]

    if args.rebuild_features or missing:
        if missing:
            print(f"[INFO] {len(missing)} features faltantes en el parquet — se re-extraen")
        else:
            print("[INFO] --rebuild-features activo — re-extrayendo todos los features")

        feat_df = rebuild_features(df, n_jobs=args.n_jobs)
        for col in feat_df.columns:
            df[col] = feat_df[col].values

        # Guardar parquet enriquecido en data/tmp/
        tmp_dir = Path("data/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{args.dataset}_binary_v2.parquet"
        df.to_parquet(tmp_path, index=False)
        print(f"[INFO] Parquet enriquecido guardado: {tmp_path}", flush=True)

    # Verificar que todos los features existen
    missing_final = [f for f in feat_cols if f not in df.columns]
    if missing_final:
        sys.exit(f"[ERROR] Features todavia faltantes tras extraccion: {missing_final}")

    # ── Split 80/20 estratificado — identico al pipeline CSIC ─────────────────
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df["label_binary"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    train_n0 = int((y_train == 0).sum())
    train_n1 = int((y_train == 1).sum())
    test_n0  = int((y_test  == 0).sum())
    test_n1  = int((y_test  == 1).sum())

    print(f"[INFO] Train: {len(X_train):>7}  (normal={train_n0}, ataque={train_n1})")
    print(f"[INFO] Test:  {len(X_test):>7}  (normal={test_n0},  ataque={test_n1})")
    print(f"[INFO] Features: {len(feat_cols)}", flush=True)

    # ── Entrenar RF — identico al pipeline CSIC ───────────────────────────────
    print(f"\n[TRAIN] RandomForestClassifier "
          f"n_estimators={args.n_estimators}, class_weight=balanced, seed={args.seed}...",
          flush=True)
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        oob_score=True,
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"[INFO] Entrenamiento: {train_time:.1f}s | OOB: {rf.oob_score_:.4f}", flush=True)

    # ── Prediccion ────────────────────────────────────────────────────────────
    t0 = time.time()
    y_pred_05 = rf.predict(X_test)
    y_prob    = rf.predict_proba(X_test)[:, 1]
    pred_time = time.time() - t0

    # ── Metricas thr=0.50 ─────────────────────────────────────────────────────
    m05 = compute_metrics(y_test, y_pred_05, y_prob)
    print(f"\n{'=' * 60}")
    print(f" {args.dataset.upper()} — thr=0.50 — {args.feat_set} features")
    print(f"{'=' * 60}")
    print(f"Accuracy:       {m05['accuracy']:.4f}")
    print(f"Balanced acc.:  {m05['balanced_accuracy']:.4f}")
    print(f"ROC-AUC:        {m05['roc_auc']:.4f}")
    print(f"PR-AUC:         {m05['pr_auc']:.4f}")
    print(f"MCC:            {m05['mcc']:.4f}")
    print(f"\nClase ATAQUE (1):")
    print(f"  F1={m05['f1']:.4f}  Recall={m05['recall']:.4f}  "
          f"Precision={m05['precision']:.4f}  FPR={m05['fpr']:.4f}  FNR={m05['fnr']:.4f}")
    print(f"\n{classification_report(y_test, y_pred_05)}", flush=True)

    # Feature importance
    fi = pd.DataFrame({
        "feature":    feat_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"Top 15 features (Gini importancia):")
    print(fi.head(15).to_string(index=False), flush=True)

    # ── Guardar artefactos ────────────────────────────────────────────────────
    joblib.dump(rf, out_dir / "model.joblib")
    fi.to_csv(out_dir / "feature_importance_rf.csv", index=False)
    pd.DataFrame({"feature": feat_cols}).to_csv(out_dir / "features_used.csv", index=False)
    pd.DataFrame({
        "y_true":  y_test,
        "y_pred":  y_pred_05,
        "proba_0": 1 - y_prob,
        "proba_1": y_prob,
    }).to_csv(out_dir / "pred.csv", index=False)

    # Confusion matrix
    cm_labels = ["normal(0)", "ataque(1)"]
    cm = confusion_matrix(y_test, y_pred_05, labels=[0, 1])
    pd.DataFrame(cm, index=cm_labels, columns=cm_labels).to_csv(
        out_dir / "confusion_matrix.csv"
    )

    # Classification report
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred_05))

    # Model config
    model_cfg = {
        "model":        "RandomForestClassifier",
        "n_estimators": args.n_estimators,
        "max_depth":    None,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "n_jobs":       args.n_jobs,
        "random_state": args.seed,
        "oob_score":    True,
    }
    with open(out_dir / "model_config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)

    # metrics.json thr=0.50
    metrics_05 = {
        "dataset":        args.dataset,
        "model":          f"RandomForest_{args.feat_set}feat_balanced",
        "n_features":     args.feat_set,
        "feature_set":    f"feat_{args.feat_set}",
        "n_estimators":   args.n_estimators,
        "seed":           args.seed,
        "test_size":      args.test_size,
        "oob_score":      float(rf.oob_score_),
        "train_time_s":   round(train_time, 2),
        "predict_time_s": round(pred_time, 4),
        "train_normal":   train_n0,
        "train_attack":   train_n1,
        "test_normal":    test_n0,
        "test_attack":    test_n1,
        **m05,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_05, f, indent=2)
    print(f"\n[GUARDADO] {out_dir}/metrics.json", flush=True)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print(f"\n[THRESHOLD SWEEP] barriendo 0.05-0.95 (paso 0.01)...", flush=True)
    sweep_df, best = threshold_sweep(y_test, y_prob)
    sweep_df.to_csv(thr_dir / "threshold_sweep.csv", index=False)

    best_pred = (y_prob >= best["threshold"]).astype(int)
    best_m    = compute_metrics(y_test, best_pred, y_prob)

    print(f"\n{'=' * 60}")
    print(f" {args.dataset.upper()} — thr=OPTIMO ({best['threshold']:.2f}) — {args.feat_set} features")
    print(f"{'=' * 60}")
    print(f"F1:             {best_m['f1']:.4f}")
    print(f"Recall:         {best_m['recall']:.4f}")
    print(f"Precision:      {best_m['precision']:.4f}")
    print(f"Balanced acc.:  {best_m['balanced_accuracy']:.4f}")
    print(f"ROC-AUC:        {best_m['roc_auc']:.4f}")
    print(f"PR-AUC:         {best_m['pr_auc']:.4f}")
    print(f"MCC:            {best_m['mcc']:.4f}")
    print(f"FPR:            {best_m['fpr']:.4f}")
    print(f"FNR:            {best_m['fnr']:.4f}")

    # Confusion matrix para umbral optimo
    cm_opt = confusion_matrix(y_test, best_pred, labels=[0, 1])
    pd.DataFrame(cm_opt, index=cm_labels, columns=cm_labels).to_csv(
        thr_dir / "confusion_matrix.csv"
    )
    with open(thr_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, best_pred))

    # metrics.json thr=optimo (formato completo para tabla comparativa)
    metrics_opt = {
        "dataset":        args.dataset,
        "model":          f"RandomForest_{args.feat_set}feat_balanced_threshold",
        "n_features":     args.feat_set,
        "feature_set":    f"feat_{args.feat_set}",
        "n_estimators":   args.n_estimators,
        "threshold":      float(best["threshold"]),
        "seed":           args.seed,
        "test_size":      args.test_size,
        "oob_score":      float(rf.oob_score_),
        "train_time_s":   round(train_time, 2),
        "predict_time_s": round(pred_time, 4),
        "train_normal":   train_n0,
        "train_attack":   train_n1,
        "test_normal":    test_n0,
        "test_attack":    test_n1,
        **best_m,
    }
    with open(thr_dir / "metrics.json", "w") as f:
        json.dump(metrics_opt, f, indent=2)
    print(f"[GUARDADO] {thr_dir}/metrics.json", flush=True)

    # Top umbrales
    print(f"\nTop 10 umbrales por F1:")
    print(f"{'Thr':>6} {'F1':>8} {'Recall':>8} {'Prec.':>8} {'FPR':>8} {'BAcc':>8}")
    print("-" * 54)
    for _, row in sweep_df.nlargest(10, "f1").iterrows():
        print(f"{row['threshold']:>6.2f} {row['f1']:>8.4f} {row['recall']:>8.4f} "
              f"{row['precision']:>8.4f} {row['fpr']:>8.4f} {row['balanced_accuracy']:>8.4f}")

    print(f"\n[DONE] {args.dataset} — {args.feat_set} features completado.", flush=True)


if __name__ == "__main__":
    main()
