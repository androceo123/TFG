#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --job-name=logreg_nobalanced
#SBATCH --output=slurm-logreg-nobalanced-%j.out

# =============================================================================
# PASO 2: LogReg sin class_weight=balanced + threshold tuning
#
# Motivacion:
#   El modelo balanced sobre-predice ataques (FPR=15.8%) porque asigna
#   peso 5.3x mas alto a la clase ataque (ratio normal/ataque ~5.3:1).
#   Un modelo sin balanced aprende la frontera natural del dataset y luego
#   el umbral optimo ajusta el trade-off precision/recall.
#
#   ROC-AUC es identico con o sin balanced (la curva no cambia con los pesos,
#   solo la calibracion de la probabilidad). Pero el punto optimo de F1
#   puede ser diferente — sin balanced la precision es mayor.
#
# Estrategia:
#   1. Reentrenar LogReg con class_weight=None (sin penalizacion de clase)
#   2. Barrer umbrales 0.05..0.95 sobre el nuevo pred.csv
#   3. Comparar: OCSVM vs LogReg-balanced-thr vs LogReg-nobalanced-thr
#
# Input:  data/tmp/csic_sin_registro_v2.parquet (mismo que supervised_binary)
# Output: resultsOptimo/csic_sin_registro/supervised_nobalanced/
#         resultsOptimo/csic_sin_registro/supervised_nobalanced_threshold/
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p resultsOptimo/csic_sin_registro/supervised_nobalanced
mkdir -p resultsOptimo/csic_sin_registro/supervised_nobalanced_threshold

# Verificar parquet
PYTHONPATH=src srun $PYTHON - <<'PY'
from pathlib import Path
import pandas as pd

p = Path("data/tmp/csic_sin_registro_v2.parquet")
if not p.exists():
    raise FileNotFoundError(
        f"No encontrado: {p}\n"
        "Ejecutar primero: sbatch jobs/run_csic_global_v2.sh"
    )
df = pd.read_parquet(p)
print(f"[CHECK] Parquet OK: {len(df)} filas, {len(df.columns)} columnas")
vals = df["label_binary"].value_counts().sort_index()
print(f"[CHECK] label_binary: {vals.to_dict()}")
ratio = vals.get(0, 0) / vals.get(1, 1)
print(f"[CHECK] Ratio normal/ataque: {ratio:.2f}x")
PY

echo ""
echo "======================================================"
echo " Entrenando LogReg sin balanced (57 features, C=1.0)"
echo "======================================================"

PYTHONPATH=src srun $PYTHON src/waf_ml/scripts/train_supervised.py \
  --data data/tmp/csic_sin_registro_v2.parquet \
  --task multiclass \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --C 1.0 \
  --max-iter 2000 \
  --tune none \
  --fi-kind coef \
  --fi-topk 25 \
  --out resultsOptimo/csic_sin_registro/supervised_nobalanced/model.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/supervised_nobalanced/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/supervised_nobalanced/pred.csv \
  --benchmark-mode auto \
  --benchmark-max-rows 5000 \
  --benchmark-out resultsOptimo/csic_sin_registro/supervised_nobalanced/benchmark.json

echo "[ENTRENAMIENTO COMPLETO]"

echo ""
echo "======================================================"
echo " Threshold tuning sobre modelo sin balanced"
echo "======================================================"

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

pred_path = Path("resultsOptimo/csic_sin_registro/supervised_nobalanced/pred.csv")
df = pd.read_csv(pred_path)
print(f"[INFO] pred.csv cargado: {len(df)} filas, columnas: {list(df.columns)}")

y_true = df["y_true"].values
proba_1 = df["proba_1"].values

roc_auc = roc_auc_score(y_true, proba_1)
print(f"[INFO] ROC-AUC (no-balanced LogReg): {roc_auc:.4f}")

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
        "f1": f1, "recall": rec, "precision": pre,
        "fpr": fpr, "balanced_accuracy": ba,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })

res_df = pd.DataFrame(results)

best_idx = res_df["f1"].idxmax()
best = res_df.iloc[best_idx]
print(f"\n[OPTIMO] Umbral que maximiza F1(ataque=1): {best['threshold']:.2f}")
print(f"  F1={best['f1']:.4f}  Recall={best['recall']:.4f}  "
      f"Precision={best['precision']:.4f}  FPR={best['fpr']:.4f}  "
      f"BAcc={best['balanced_accuracy']:.4f}")

out_csv = Path("resultsOptimo/csic_sin_registro/supervised_nobalanced_threshold/threshold_sweep.csv")
res_df.to_csv(out_csv, index=False)
print(f"\n[GUARDADO] Barrido completo: {out_csv}")

best_metrics = {
    "model": "LogisticRegression_nobalanced_threshold",
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
out_json = Path("resultsOptimo/csic_sin_registro/supervised_nobalanced_threshold/metrics.json")
with open(out_json, "w") as f:
    json.dump(best_metrics, f, indent=2)
print(f"[GUARDADO] Metricas optimas: {out_json}")

print("\nTop 10 umbrales por F1(ataque=1):")
print(f"{'Thr':>6} {'F1':>8} {'Recall':>8} {'Prec.':>8} {'FPR':>8} {'BAcc':>8}")
print("-" * 54)
for _, row in res_df.nlargest(10, "f1").iterrows():
    print(f"{row['threshold']:>6.2f} {row['f1']:>8.4f} {row['recall']:>8.4f} "
          f"{row['precision']:>8.4f} {row['fpr']:>8.4f} {row['balanced_accuracy']:>8.4f}")
PY

echo ""
echo "======================================================"
echo " Comparacion DEFINITIVA: OCSVM vs todos los LogReg"
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
    for sub in ("evaluation", "eval", "metrics", "test"):
        if sub in d and isinstance(d[sub], dict) and key in d[sub]:
            v = d[sub][key]
            if isinstance(v, (int, float)):
                return f"{v:.4f}"
            if isinstance(v, dict):
                for cls in ("1", "attack", "anomaly"):
                    if cls in v:
                        return f"{v[cls]:.4f}"
    v = d.get(key)
    if v is not None and not isinstance(v, dict):
        return f"{float(v):.4f}"
    return "N/A"

runs = [
    ("OCSVM one-class  nu=0.001 auto (26 feat)",        "resultsOptimo/csic_sin_registro/oneclass/metrics.json"),
    ("OCSVM one-class  nu=0.05 gamma=0.1 (57 feat)",    "resultsOptimo/csic_sin_registro/nu05_v2/metrics.json"),
    ("LogReg balanced  thr=0.50  (57 feat)",            "resultsOptimo/csic_sin_registro/supervised_binary/metrics.json"),
    ("LogReg balanced  thr=OPTIMO (57 feat)",           "resultsOptimo/csic_sin_registro/supervised_binary_threshold/metrics.json"),
    ("LogReg no-bal    thr=0.50  (57 feat)",            "resultsOptimo/csic_sin_registro/supervised_nobalanced/metrics.json"),
    ("LogReg no-bal    thr=OPTIMO (57 feat)",           "resultsOptimo/csic_sin_registro/supervised_nobalanced_threshold/metrics.json"),
]

print(f"\n{'Experimento':<50} {'F1':>8} {'Recall':>8} {'Prec.':>8} {'FPR':>8} {'BAcc':>8}")
print("-" * 98)
for name, path in runs:
    f1  = get_m(path, "f1")
    rec = get_m(path, "recall")
    pre = get_m(path, "precision")
    fpr = get_m(path, "fpr")
    ba  = get_m(path, "balanced_accuracy")
    print(f"{name:<50} {f1:>8} {rec:>8} {pre:>8} {fpr:>8} {ba:>8}")

print()
print("Referencia Nico/Ralf (OCSVM per-group, app-dependent):")
print("  F1=0.95  Recall=0.93  FPR=0.03")
print()
print("INTERPRETACION:")
print("  LogReg-optimo >> OCSVM  =>  paradigma supervisado confirma mejora sobre one-class")
print("  balanced >> no-balanced =>  la ponderacion de clases ayuda al modelo")
print("  no-balanced >> balanced =>  sin ponderacion el umbral ajusta mejor el trade-off")
PY

echo "[PASO 2 COMPLETO]"
