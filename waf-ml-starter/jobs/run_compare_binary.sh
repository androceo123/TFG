#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:10:00
#SBATCH --job-name=compare_binary
#SBATCH --output=slurm-compare-binary-%j.out

# =============================================================================
# Tabla comparativa final: CSIC 2010 vs TorpEda vs Harvard/SR-BH
#
# Prerequisitos (ejecutar antes en orden):
#   1. sbatch jobs/run_csic_random_forest.sh      -> CSIC resultados
#   2. sbatch jobs/run_torpeda_binary_rf.sh        -> TorpEda resultados
#   3. sbatch jobs/run_harvard_binary_rf.sh        -> Harvard resultados
#
# Este job carga pred.csv de cada dataset, recomputa TODAS las metricas
# desde cero (garantizando consistencia), y genera la tabla final.
# Para CSIC recupera el threshold optimo de rf_supervised_threshold/.
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

echo "======================================================"
echo " Tabla comparativa binaria: CSIC / TorpEda / Harvard"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    confusion_matrix,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_all(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision":         precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall":            recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1":                f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc":           roc_auc_score(y_true, y_prob),
        "pr_auc":            average_precision_score(y_true, y_prob),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "fpr":               fpr,
        "fnr":               fnr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def load_pred(pred_path):
    df = pd.read_csv(pred_path)
    return df["y_true"].values, df["y_pred"].values, df["proba_1"].values


def best_threshold_from_sweep(sweep_path):
    """Lee el threshold optimo del CSV de sweep."""
    p = Path(sweep_path)
    if not p.exists():
        return 0.50
    df = pd.read_csv(p)
    return float(df.iloc[df["f1"].idxmax()]["threshold"])


def load_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)

# ── Dataset configurations ────────────────────────────────────────────────────

datasets = {
    "CSIC 2010": {
        "pred_path":   "resultsOptimo/csic_sin_registro/rf_supervised/pred.csv",
        "sweep_path":  "resultsOptimo/csic_sin_registro/rf_supervised_threshold/threshold_sweep.csv",
        "meta_path":   "resultsOptimo/csic_sin_registro/rf_supervised_threshold/metrics.json",
        "parquet":     "data/tmp/csic_sin_registro_v2.parquet",
        "n_features":  57,
    },
    "TorpEda": {
        "pred_path":   "results/torpeda/binary/pred.csv",
        "sweep_path":  "results/torpeda/binary_threshold/threshold_sweep.csv",
        "meta_path":   "results/torpeda/binary_threshold/metrics.json",
        "parquet":     None,
        "n_features":  57,
    },
    "Harvard SR-BH": {
        "pred_path":   "results/harvard/binary/pred.csv",
        "sweep_path":  "results/harvard/binary_threshold/threshold_sweep.csv",
        "meta_path":   "results/harvard/binary_threshold/metrics.json",
        "parquet":     None,
        "n_features":  57,
    },
}

# ── Construir tabla ───────────────────────────────────────────────────────────

rows = []
for ds_name, cfg in datasets.items():
    pred_p = Path(cfg["pred_path"])
    if not pred_p.exists():
        print(f"[WARN] {ds_name}: pred.csv no encontrado en {pred_p} — saltando")
        continue

    y_true, _, y_prob = load_pred(pred_p)
    thr = best_threshold_from_sweep(cfg["sweep_path"])
    y_pred_opt = (y_prob >= thr).astype(int)
    m = compute_all(y_true, y_pred_opt, y_prob)

    # Conteos train/test
    meta = load_json(cfg["meta_path"])
    train_n0 = meta.get("train_normal", "N/A")
    train_n1 = meta.get("train_attack", "N/A")
    test_n0  = meta.get("test_normal",  int((y_true == 0).sum()))
    test_n1  = meta.get("test_attack",  int((y_true == 1).sum()))
    oob      = meta.get("oob_score",    None)
    train_t  = meta.get("train_time_s", None)

    rows.append({
        "Dataset":         ds_name,
        "Modelo":          "RandomForest",
        "N feat.":         cfg["n_features"],
        "Threshold":       f"{thr:.2f}",
        "Train normal":    train_n0,
        "Train ataque":    train_n1,
        "Test normal":     test_n0,
        "Test ataque":     test_n1,
        "OOB score":       f"{oob:.4f}" if oob else "N/A",
        "Train time (s)":  f"{train_t:.1f}" if train_t else "N/A",
        "Accuracy":        f"{m['accuracy']:.4f}",
        "Balanced acc.":   f"{m['balanced_accuracy']:.4f}",
        "Precision":       f"{m['precision']:.4f}",
        "Recall":          f"{m['recall']:.4f}",
        "F1":              f"{m['f1']:.4f}",
        "ROC-AUC":         f"{m['roc_auc']:.4f}",
        "PR-AUC":          f"{m['pr_auc']:.4f}",
        "MCC":             f"{m['mcc']:.4f}",
        "FPR":             f"{m['fpr']:.4f}",
        "FNR":             f"{m['fnr']:.4f}",
        "TN":              m["tn"], "FP": m["fp"],
        "FN":              m["fn"], "TP": m["tp"],
    })

if not rows:
    print("[ERROR] No se encontraron resultados. Ejecutar los jobs previos.")
    sys.exit(1)

table = pd.DataFrame(rows)

# ── Imprimir tabla completa ────────────────────────────────────────────────────
print(f"\n{'=' * 130}")
print(" TABLA COMPARATIVA FINAL — RF BINARIO (thr=OPTIMO)")
print(f" Mismo pipeline: 500 arboles, balanced, seed=42, 80/20 split")
print(f"{'=' * 130}")

main_cols = [
    "Dataset", "N feat.", "Threshold",
    "Train normal", "Train ataque", "Test normal", "Test ataque",
    "F1", "Recall", "Precision", "Balanced acc.", "ROC-AUC", "PR-AUC",
    "MCC", "FPR", "FNR", "Accuracy",
]
available = [c for c in main_cols if c in table.columns]
print(table[available].to_string(index=False))

# ── Tabla de confusion por dataset ────────────────────────────────────────────
print(f"\n{'─' * 80}")
print(" MATRICES DE CONFUSION (thr=OPTIMO):")
for _, row in table.iterrows():
    print(f"\n  {row['Dataset']} (thr={row['Threshold']}):")
    print(f"               Pred Normal  Pred Ataque")
    print(f"  Real Normal    {row['TN']:>9}   {row['FP']:>9}   <- FPR={row['FPR']}")
    print(f"  Real Ataque    {row['FN']:>9}   {row['TP']:>9}   <- Recall={row['Recall']}")

# ── Guardar CSV de la tabla ────────────────────────────────────────────────────
out_path = Path("results/comparison_binary_rf.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(out_path, index=False)
print(f"\n[GUARDADO] {out_path}")

# ── Interpretar diferencias ───────────────────────────────────────────────────
print(f"\n{'=' * 130}")
print(" INTERPRETACION:")
if len(rows) >= 2:
    f1_vals = {r["Dataset"]: float(r["F1"]) for r in rows}
    best_ds  = max(f1_vals, key=f1_vals.get)
    worst_ds = min(f1_vals, key=f1_vals.get)
    print(f"  Mejor  F1: {best_ds}  ({f1_vals[best_ds]:.4f})")
    print(f"  Peor   F1: {worst_ds} ({f1_vals[worst_ds]:.4f})")
    delta = f1_vals[best_ds] - f1_vals[worst_ds]
    print(f"  Rango F1 entre datasets: {delta:.4f} puntos")
    print(f"  {'Pipeline generaliza bien' if delta < 0.05 else 'Pipeline muestra variacion entre datasets'}")
print(f"  Referencia Nico/Ralf (OCSVM per-group, app-dependent): F1=0.950, FPR=0.030")
PY

echo "[COMPARACION BINARIA COMPLETA]"
