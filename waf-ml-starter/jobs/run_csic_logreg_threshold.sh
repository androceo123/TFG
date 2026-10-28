#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:15:00
#SBATCH --job-name=logreg_threshold
#SBATCH --output=slurm-logreg-threshold-%j.out

# =============================================================================
# PASO 1: Threshold tuning sobre el modelo LogReg balanceado existente
#
# Motivacion:
#   LogReg con class_weight=balanced da F1(ataque)=0.639, Recall=0.705,
#   Precision=0.585, FPR=0.158 -> sobre-predice ataques.
#   Sin embargo, ROC-AUC(LogReg)=0.861 > ROC-AUC(OCSVM)=0.839, lo que
#   indica que con un umbral optimo el supervisado SUPERA al one-class.
#
#   Este script NO reentrena. Solo carga pred.csv (ya tiene proba_1) y
#   barre umbrales 0.05..0.95 para encontrar el que maximiza F1(ataque=1).
#
# Input:  resultsOptimo/csic_sin_registro/supervised_binary/pred.csv
# Output: resultsOptimo/csic_sin_registro/supervised_binary_threshold/
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p resultsOptimo/csic_sin_registro/supervised_binary_threshold

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score,
    confusion_matrix,
)

pred_path = Path("resultsOptimo/csic_sin_registro/supervised_binary/pred.csv")
if not pred_path.exists():
    raise FileNotFoundError(
        f"No encontrado: {pred_path}\n"
        "Ejecutar primero: sbatch jobs/run_csic_supervised_binary.sh"
    )

df = pd.read_csv(pred_path)
print(f"[INFO] pred.csv cargado: {len(df)} filas, columnas: {list(df.columns)}")

if "proba_1" not in df.columns:
    raise KeyError("Columna 'proba_1' no encontrada en pred.csv")
if "y_true" not in df.columns:
    raise KeyError("Columna 'y_true' no encontrada en pred.csv")

y_true = df["y_true"].values
proba_1 = df["proba_1"].values

# ROC-AUC del modelo (umbral-independiente)
roc_auc = roc_auc_score(y_true, proba_1)
print(f"[INFO] ROC-AUC (balanced LogReg): {roc_auc:.4f}")

# === Barrido de umbrales ===
thresholds = np.arange(0.05, 0.96, 0.01)
results = []

for thr in thresholds:
    y_pred = (proba_1 >= thr).astype(int)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    pre = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    ba = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    results.append({
        "threshold": round(float(thr), 2),
        "f1": f1,
        "recall": rec,
        "precision": pre,
        "fpr": fpr,
        "balanced_accuracy": ba,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })

res_df = pd.DataFrame(results)

# Umbral optimo (max F1 ataque)
best_idx = res_df["f1"].idxmax()
best = res_df.iloc[best_idx]
print(f"\n[OPTIMO] Umbral que maximiza F1(ataque=1): {best['threshold']:.2f}")
print(f"  F1={best['f1']:.4f}  Recall={best['recall']:.4f}  "
      f"Precision={best['precision']:.4f}  FPR={best['fpr']:.4f}  "
      f"BAcc={best['balanced_accuracy']:.4f}")

# Comparacion threshold=0.5 vs optimo
default = res_df[res_df["threshold"] == 0.50].iloc[0]
print(f"\n[DEFAULT thr=0.50] F1={default['f1']:.4f}  Recall={default['recall']:.4f}  "
      f"Precision={default['precision']:.4f}  FPR={default['fpr']:.4f}")

# Guardar CSV completo del barrido
out_csv = Path("resultsOptimo/csic_sin_registro/supervised_binary_threshold/threshold_sweep.csv")
res_df.to_csv(out_csv, index=False)
print(f"\n[GUARDADO] Barrido completo: {out_csv}")

# Guardar metricas del optimo como JSON (mismo formato que metrics.json)
best_metrics = {
    "model": "LogisticRegression_balanced_threshold",
    "threshold": float(best["threshold"]),
    "roc_auc": float(roc_auc),
    "evaluation": {
        "f1": float(best["f1"]),
        "recall": float(best["recall"]),
        "precision": float(best["precision"]),
        "fpr": float(best["fpr"]),
        "balanced_accuracy": float(best["balanced_accuracy"]),
    }
}
out_json = Path("resultsOptimo/csic_sin_registro/supervised_binary_threshold/metrics.json")
with open(out_json, "w") as f:
    json.dump(best_metrics, f, indent=2)
print(f"[GUARDADO] Metricas optimas: {out_json}")

# === Tabla top 10 umbrales por F1 ===
print("\nTop 10 umbrales por F1(ataque=1):")
print(f"{'Thr':>6} {'F1':>8} {'Recall':>8} {'Prec.':>8} {'FPR':>8} {'BAcc':>8}")
print("-" * 54)
for _, row in res_df.nlargest(10, "f1").iterrows():
    print(f"{row['threshold']:>6.2f} {row['f1']:>8.4f} {row['recall']:>8.4f} "
          f"{row['precision']:>8.4f} {row['fpr']:>8.4f} {row['balanced_accuracy']:>8.4f}")
PY

echo ""
echo "======================================================"
echo " Comparacion final: OCSVM vs LogReg (balanced) vs LogReg (threshold optimo)"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

def get_m(path, key):
    p = Path(path)
    if not p.exists():
        return "N/A"
    with open(p) as f:
        d = json.load(f)
    # Buscar en sub-dicts estandar
    for sub in ("evaluation", "eval", "metrics", "test"):
        if sub in d and isinstance(d[sub], dict) and key in d[sub]:
            v = d[sub][key]
            if isinstance(v, (int, float)):
                return f"{v:.4f}"
            if isinstance(v, dict):
                # metricas por clase: buscar clase "1" (ataque)
                for cls in ("1", "attack", "anomaly"):
                    if cls in v:
                        return f"{v[cls]:.4f}"
    # Nivel raiz
    v = d.get(key)
    if v is not None and not isinstance(v, dict):
        return f"{float(v):.4f}"
    return "N/A"

runs = [
    ("OCSVM one-class  nu=0.001 auto-tuning (26 feat)", "resultsOptimo/csic_sin_registro/oneclass/metrics.json"),
    ("OCSVM one-class  nu=0.05, gamma=0.1  (26 feat)",  "resultsOptimo/csic_sin_registro/nicoRalf_nu05/metrics.json"),
    ("OCSVM one-class  nu=0.05, gamma=0.1  (57 feat)",  "resultsOptimo/csic_sin_registro/nu05_v2/metrics.json"),
    ("LogReg balanced  thr=0.50  (57 feat)",            "resultsOptimo/csic_sin_registro/supervised_binary/metrics.json"),
    ("LogReg balanced  thr=OPTIMO (57 feat)",           "resultsOptimo/csic_sin_registro/supervised_binary_threshold/metrics.json"),
]

print(f"\n{'Experimento':<52} {'F1':>8} {'Recall':>8} {'Prec.':>8} {'FPR':>8} {'BAcc':>8}")
print("-" * 100)
for name, path in runs:
    f1  = get_m(path, "f1")
    rec = get_m(path, "recall")
    pre = get_m(path, "precision")
    fpr = get_m(path, "fpr")
    ba  = get_m(path, "balanced_accuracy")
    print(f"{name:<52} {f1:>8} {rec:>8} {pre:>8} {fpr:>8} {ba:>8}")

print()
print("Referencia Nico/Ralf (OCSVM per-group, app-dependent):")
print("  F1=0.95  Recall=0.93  FPR=0.03")
print()
print("INTERPRETACION:")
print("  LogReg-optimo >> OCSVM  =>  el paradigma supervisado supera al one-class")
print("  LogReg-optimo ~= OCSVM  =>  el umbral no rescata el modelo; ir al PASO 2 (sin balanced)")
PY

echo "[PASO 1 COMPLETO]"
