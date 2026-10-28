#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-00:20:00
#SBATCH --job-name=rf_top10
#SBATCH --output=slurm-rf-top10-%j.out

# =============================================================================
# RF CSIC — Top-10 features por importancia Gini (feature selection)
#
# Motivacion (petición del profesor):
#   "Demasiados features hace que sea más lento todo y además es difícil de
#    explicar porque se está detectando."
#
#   Lee el ranking de feature importance del RF con 57 features y entrena
#   un nuevo RF con solo los 10 más importantes.
#
#   PREREQUISITO: ejecutar primero run_csic_random_forest.sh para generar
#   resultsOptimo/csic_sin_registro/rf_supervised/feature_importance_rf.csv
#
# Usa: data/tmp/csic_sin_registro_v2.parquet (ya construido)
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p resultsOptimo/csic_sin_registro/rf_top10
mkdir -p resultsOptimo/csic_sin_registro/rf_top10_threshold

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix,
)

PARQUET    = Path("data/tmp/csic_sin_registro_v2.parquet")
FI_CSV     = Path("resultsOptimo/csic_sin_registro/rf_supervised/feature_importance_rf.csv")
OUT_DIR    = Path("resultsOptimo/csic_sin_registro/rf_top10")
SEED       = 42
TEST_SIZE  = 0.2
TOP_K      = 10

# ── Cargar ranking de importancia del RF de 57 features ────────────────────
if not FI_CSV.exists():
    raise FileNotFoundError(
        f"No encontrado: {FI_CSV}\n"
        "Ejecutar primero: sbatch jobs/run_csic_random_forest.sh"
    )
fi_df = pd.read_csv(FI_CSV)
top_features = fi_df.nlargest(TOP_K, "importance")["feature"].tolist()

print(f"\n[INFO] Top-{TOP_K} features seleccionados del ranking RF-57:")
for i, (feat, imp) in enumerate(
    zip(top_features, fi_df.nlargest(TOP_K, "importance")["importance"].tolist()), 1
):
    print(f"  {i:2d}. {feat:<35} importance={imp:.6f}")

# ── Entrenar RF con solo top-K features ───────────────────────────────────
df = pd.read_parquet(PARQUET)
print(f"\n[INFO] Parquet OK: {len(df)} filas")

missing = [f for f in top_features if f not in df.columns]
if missing:
    raise ValueError(f"Features no encontrados en parquet: {missing}")

feat_cols = top_features
X = df[feat_cols].fillna(0).values.astype(np.float32)
y = df["label_binary"].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

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
print(f"[INFO] Entrenamiento: {train_time:.1f}s  |  OOB: {rf.oob_score_:.4f}")

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
f1_val  = f1_score(y_test, y_pred, pos_label=1)
rec_val = recall_score(y_test, y_pred, pos_label=1)
pre_val = precision_score(y_test, y_pred, pos_label=1)
ba_val  = balanced_accuracy_score(y_test, y_pred)
roc_val = roc_auc_score(y_test, y_prob)

print(f"\n=== Test evaluation (RF top-{TOP_K} features) ===")
print(f"ROC-AUC:       {roc_val:.4f}")
print(f"Balanced acc.: {ba_val:.4f}")
print(f"\nClase ATAQUE (1):")
print(f"  F1={f1_val:.4f}  Recall={rec_val:.4f}  Precision={pre_val:.4f}  FPR={fpr_val:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

fi_new = pd.DataFrame({
    "feature": feat_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)
print(f"\nImportancia RF (Gini) — top-{TOP_K} feat:")
print(fi_new.to_string(index=False))
fi_new.to_csv(OUT_DIR / "feature_importance_rf.csv", index=False)

joblib.dump(rf, OUT_DIR / "model.joblib")
pd.DataFrame({"y_true": y_test, "y_pred": y_pred,
              "proba_0": 1 - y_prob, "proba_1": y_prob}).to_csv(
    OUT_DIR / "pred.csv", index=False
)
pd.DataFrame({"rank": range(1, TOP_K + 1), "feature": feat_cols}).to_csv(
    OUT_DIR / "selected_features.csv", index=False
)

metrics = {
    "model": f"RandomForest_top{TOP_K}feat_balanced",
    "n_features": TOP_K,
    "feature_set": f"top{TOP_K}_from_rf57_gini",
    "selected_features": feat_cols,
    "n_estimators": 500,
    "oob_score": float(rf.oob_score_),
    "train_time_s": round(train_time, 2),
    "roc_auc": float(roc_val),
    "evaluation": {
        "f1": float(f1_val),
        "recall": float(rec_val),
        "precision": float(pre_val),
        "fpr": float(fpr_val),
        "balanced_accuracy": float(ba_val),
    }
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n[GUARDADO] {OUT_DIR}/metrics.json")
print(f"[GUARDADO] {OUT_DIR}/selected_features.csv")
PY

echo ""
echo "======================================================"
echo " Threshold tuning RF top-10"
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

pred_path = Path("resultsOptimo/csic_sin_registro/rf_top10/pred.csv")
out_dir   = Path("resultsOptimo/csic_sin_registro/rf_top10_threshold")

df = pd.read_csv(pred_path)
y_true  = df["y_true"].values
proba_1 = df["proba_1"].values

roc_auc = roc_auc_score(y_true, proba_1)
print(f"[INFO] ROC-AUC (RF top-10): {roc_auc:.4f}")

thresholds = np.arange(0.05, 0.96, 0.01)
results = []
for thr in thresholds:
    y_pred = (proba_1 >= thr).astype(int)
    f1  = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    pre = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    ba  = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    results.append({"threshold": round(float(thr), 2),
                    "f1": f1, "recall": rec, "precision": pre,
                    "fpr": fpr, "balanced_accuracy": ba})

res_df = pd.DataFrame(results)
best   = res_df.iloc[res_df["f1"].idxmax()]
print(f"\n[OPTIMO RF-TOP10] Umbral: {best['threshold']:.2f}")
print(f"  F1={best['f1']:.4f}  Recall={best['recall']:.4f}  "
      f"Precision={best['precision']:.4f}  FPR={best['fpr']:.4f}  "
      f"BAcc={best['balanced_accuracy']:.4f}")

res_df.to_csv(out_dir / "threshold_sweep.csv", index=False)
with open(out_dir / "metrics.json", "w") as f:
    json.dump({
        "model": "RandomForest_top10feat_balanced_threshold",
        "n_features": 10,
        "feature_set": "top10_from_rf57_gini",
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
print(f"[GUARDADO] {out_dir}/metrics.json")
PY

echo ""
echo "======================================================"
echo " Tabla comparativa ablación de features (CSIC)"
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
    for sub in ("evaluation", "eval", "metrics"):
        if sub in d and isinstance(d[sub], dict) and key in d[sub]:
            v = d[sub][key]
            if isinstance(v, (int, float)):
                return f"{v:.4f}"
    v = d.get(key)
    if v is not None and not isinstance(v, (dict, list)):
        return f"{float(v):.4f}"
    return "N/A"

runs = [
    ("RF 57 feat (baseline)  thr=OPTIMO",
     "resultsOptimo/csic_sin_registro/rf_supervised_threshold/metrics.json"),
    ("RF 34 feat (25+entropy v1)  thr=OPTIMO",
     "resultsOptimo/csic_sin_registro/rf_34feat_threshold/metrics.json"),
    ("RF 25 feat (originales)  thr=OPTIMO",
     "resultsOptimo/csic_sin_registro/rf_25feat_threshold/metrics.json"),
    ("RF top-10 feat (feature selection)  thr=OPTIMO",
     "resultsOptimo/csic_sin_registro/rf_top10_threshold/metrics.json"),
]

print(f"\n{'Experimento':<52} {'F1':>8} {'Recall':>8} {'Prec.':>8} {'FPR':>8} {'BAcc':>8} {'ROC-AUC':>9}")
print("-" * 110)
for name, path in runs:
    f1  = get_m(path, "f1")
    rec = get_m(path, "recall")
    pre = get_m(path, "precision")
    fpr = get_m(path, "fpr")
    ba  = get_m(path, "balanced_accuracy")
    roc = get_m(path, "roc_auc")
    print(f"{name:<52} {f1:>8} {rec:>8} {pre:>8} {fpr:>8} {ba:>8} {roc:>9}")

print()
print("Referencia Nico/Ralf: F1=0.95  FPR=0.03")
print()
print("INTERPRETACION:")
print("  RF 25 feat vs RF 57: cuánto aportan entropía + char dist")
print("  RF 34 feat vs RF 57: cuánto aportan los char dist de Nico/Ralf")
print("  RF top-10  vs RF 57: versión mínima manteniendo F1 alto")
PY

echo "[ABLACION FEATURES COMPLETA]"
