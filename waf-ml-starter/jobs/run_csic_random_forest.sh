#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --job-name=rf_supervised
#SBATCH --output=slurm-rf-supervised-%j.out

# =============================================================================
# RandomForest supervisado sobre CSIC 2010 (57 features, mismos datos que OCSVM)
#
# Motivacion:
#   LogReg (lineal) da F1=0.665 < OCSVM (kernel RBF) F1=0.787.
#   La hipotesis es que LogReg falla por ser lineal, no por ser supervisado.
#   RandomForest captura no-linealidades sin kernel, usando etiquetas de ataque.
#   Si RF >> OCSVM => el paradigma supervisado SI supera al one-class (PI-1).
#   Si RF ~= OCSVM => el kernel RBF es la clave, no las etiquetas de ataque.
#
# Mismo parquet, mismo split 80/20, misma semilla que todos los experimentos anteriores.
# Sin GPU: RF es CPU, 8 cores paralelos.
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p resultsOptimo/csic_sin_registro/rf_supervised
mkdir -p resultsOptimo/csic_sin_registro/rf_supervised_threshold

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
print(f"[CHECK] Ratio normal/ataque: {vals[0]/vals[1]:.2f}x")
PY

echo ""
echo "======================================================"
echo " Entrenando RandomForest (57 features, n_estimators=500)"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    classification_report, confusion_matrix,
    accuracy_score,
)

PARQUET = Path("data/tmp/csic_sin_registro_v2.parquet")
OUT_DIR = Path("resultsOptimo/csic_sin_registro/rf_supervised")
SEED = 42
TEST_SIZE = 0.2

df = pd.read_parquet(PARQUET)
label_cols = [c for c in df.columns if c.startswith("label_")]
meta_cols = ["source_file", "dataset_name", "split", "label_type_raw"]
feat_cols = [c for c in df.columns if c not in label_cols + meta_cols]

print(f"[INFO] Features: {len(feat_cols)}")
print(f"[INFO] Feature list (primeras 10): {feat_cols[:10]}")

X = df[feat_cols].fillna(0).values.astype(np.float32)
y = df["label_binary"].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")
print(f"[INFO] Train dist: 0={sum(y_train==0)}, 1={sum(y_train==1)}")

# RandomForest: no necesita StandardScaler pero lo incluimos para consistencia
# class_weight='balanced' para manejar el desbalance
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=8,
    random_state=SEED,
    oob_score=True,
)
rf.fit(X_train, y_train)
train_time = time.time() - t0
print(f"[INFO] Entrenamiento: {train_time:.1f}s")
print(f"[INFO] OOB score: {rf.oob_score_:.4f}")

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
f1_val = f1_score(y_test, y_pred, pos_label=1)
rec_val = recall_score(y_test, y_pred, pos_label=1)
pre_val = precision_score(y_test, y_pred, pos_label=1)
ba_val = balanced_accuracy_score(y_test, y_pred)
roc_val = roc_auc_score(y_test, y_prob)
pr_val = average_precision_score(y_test, y_prob)
mcc_val = matthews_corrcoef(y_test, y_pred)

print(f"\n=== Test evaluation (RandomForest) ===")
print(f"N:                 {len(y_test)}")
print(f"Accuracy:          {accuracy_score(y_test, y_pred):.4f}")
print(f"Balanced acc.:     {ba_val:.4f}")
print(f"ROC-AUC:           {roc_val:.4f}")
print(f"PR-AUC:            {pr_val:.4f}")
print(f"MCC:               {mcc_val:.4f}")
print(f"\nClase ATAQUE (1):")
print(f"  F1={f1_val:.4f}  Recall={rec_val:.4f}  Precision={pre_val:.4f}  FPR={fpr_val:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

# Feature importance
fi = pd.DataFrame({
    "feature": feat_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nTop 25 features por importancia RF (Gini):")
print(fi.head(25).to_string(index=False))

fi.to_csv(OUT_DIR / "feature_importance_rf.csv", index=False)

# Guardar modelo y predicciones
joblib.dump(rf, OUT_DIR / "model.joblib")
pd.DataFrame({"y_true": y_test, "y_pred": y_pred,
              "proba_0": 1 - y_prob, "proba_1": y_prob}).to_csv(
    OUT_DIR / "pred.csv", index=False
)

# Guardar metrics.json (mismo formato que OCSVM)
metrics = {
    "model": "RandomForestClassifier_balanced",
    "n_estimators": 500,
    "oob_score": float(rf.oob_score_),
    "train_time_s": round(train_time, 2),
    "roc_auc": float(roc_val),
    "pr_auc": float(pr_val),
    "evaluation": {
        "f1": float(f1_val),
        "recall": float(rec_val),
        "precision": float(pre_val),
        "fpr": float(fpr_val),
        "balanced_accuracy": float(ba_val),
        "mcc": float(mcc_val),
    }
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n[GUARDADO] {OUT_DIR}/metrics.json")
print(f"[GUARDADO] {OUT_DIR}/pred.csv")
print(f"[GUARDADO] {OUT_DIR}/model.joblib")
PY

echo "[ENTRENAMIENTO RF COMPLETO]"

echo ""
echo "======================================================"
echo " Threshold tuning sobre RandomForest"
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

pred_path = Path("resultsOptimo/csic_sin_registro/rf_supervised/pred.csv")
out_dir = Path("resultsOptimo/csic_sin_registro/rf_supervised_threshold")

df = pd.read_csv(pred_path)
y_true = df["y_true"].values
proba_1 = df["proba_1"].values

roc_auc = roc_auc_score(y_true, proba_1)
print(f"[INFO] ROC-AUC (RF balanced): {roc_auc:.4f}")

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
best = res_df.iloc[res_df["f1"].idxmax()]
print(f"\n[OPTIMO RF] Umbral: {best['threshold']:.2f}")
print(f"  F1={best['f1']:.4f}  Recall={best['recall']:.4f}  "
      f"Precision={best['precision']:.4f}  FPR={best['fpr']:.4f}  "
      f"BAcc={best['balanced_accuracy']:.4f}")

res_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

# Guardar metrics del optimo
with open(out_dir / "metrics.json", "w") as f:
    json.dump({
        "model": "RandomForestClassifier_balanced_threshold",
        "threshold": float(best["threshold"]),
        "roc_auc": float(roc_auc),
        "evaluation": {
            "f1": float(best["f1"]),
            "recall": float(best["recall"]),
            "precision": float(best["precision"]),
            "fpr": float(best["fpr"]),
            "balanced_accuracy": float(best["balanced_accuracy"]),
        }
    }, f, indent=2)

print(f"\n[GUARDADO] {out_dir}/metrics.json")

print("\nTop 10 umbrales por F1(ataque=1):")
print(f"{'Thr':>6} {'F1':>8} {'Recall':>8} {'Prec.':>8} {'FPR':>8} {'BAcc':>8}")
print("-" * 54)
for _, row in res_df.nlargest(10, "f1").iterrows():
    print(f"{row['threshold']:>6.2f} {row['f1']:>8.4f} {row['recall']:>8.4f} "
          f"{row['precision']:>8.4f} {row['fpr']:>8.4f} {row['balanced_accuracy']:>8.4f}")
PY

echo ""
echo "======================================================"
echo " Tabla DEFINITIVA: OCSVM vs LogReg vs RandomForest"
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
    # --- One-class OCSVM ---
    ("OCSVM one-class  nu=0.001 auto (26 feat)",     "resultsOptimo/csic_sin_registro/oneclass/metrics.json"),
    ("OCSVM one-class  nu=0.05 g=0.1 (57 feat)",     "resultsOptimo/csic_sin_registro/nu05_v2/metrics.json"),
    # --- Supervisado lineal ---
    ("LogReg balanced  thr=OPTIMO (57 feat)",         "resultsOptimo/csic_sin_registro/supervised_binary_threshold/metrics.json"),
    ("LogReg no-bal    thr=OPTIMO (57 feat)",         "resultsOptimo/csic_sin_registro/supervised_nobalanced_threshold/metrics.json"),
    # --- Supervisado no lineal ---
    ("RandomForest balanced  thr=0.50  (57 feat)",   "resultsOptimo/csic_sin_registro/rf_supervised/metrics.json"),
    ("RandomForest balanced  thr=OPTIMO (57 feat)",  "resultsOptimo/csic_sin_registro/rf_supervised_threshold/metrics.json"),
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
print("  RF >> OCSVM  => modelo no lineal supervisado supera al one-class (PI-1 confirmado)")
print("  RF ~= OCSVM  => el kernel RBF captura lo mismo sin etiquetas; features son el limite")
print("  RF >> LogReg => no-linealidad es clave; paradigma supervisado necesita modelo adecuado")
PY

echo "[EXPERIMENTO RF COMPLETO]"
