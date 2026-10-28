#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-00:20:00
#SBATCH --job-name=rf_34feat
#SBATCH --output=slurm-rf-34feat-%j.out

# =============================================================================
# RF CSIC — 34 features: 25 originales + 8 entropía v1 (sin char dist v2)
#
# Motivacion:
#   El profe observó que uri_entropy es el feature #1 en importancia Gini.
#   Este job aísla la contribución de los features de entropía respecto a
#   los features de distribución de caracteres (char_dist_i*).
#
#   FEATURESET: 34 features
#     - 25 originales estructurales
#     - 8 features de entropía v1 (uri_entropy, query_entropy, body_entropy, ...)
#   Sin: query_char_dist_*, max_param_char_dist_*, body_char_dist_*,
#        max_body_param_char_dist_*, mean/std param entropy stats
#
# Usa: data/tmp/csic_sin_registro_v2.parquet (ya construido)
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p resultsOptimo/csic_sin_registro/rf_34feat
mkdir -p resultsOptimo/csic_sin_registro/rf_34feat_threshold

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

PARQUET = Path("data/tmp/csic_sin_registro_v2.parquet")
OUT_DIR = Path("resultsOptimo/csic_sin_registro/rf_34feat")
SEED = 42
TEST_SIZE = 0.2

# ── 25 features originales ──────────────────────────────────────────────────
FEAT_25 = [
    "uri_len", "path_depth", "query_len", "n_query_params", "max_param_value_len",
    "uri_pct_non_alnum_ratio", "encoded", "suspicious_tokens_count", "has_suspicious_tokens",
    "body_suspicious_tokens_count", "body_has_suspicious_tokens", "body_encoded",
    "uncommon_method", "req_content_length", "body_len",
    "method_GET", "method_POST", "method_HEAD", "method_PUT", "method_DELETE",
    "method_PATCH", "method_OPTIONS", "method_TRACE", "method_CONNECT", "method_OTHER",
]

# ── 8 features entropía v1 ──────────────────────────────────────────────────
FEAT_ENTROPY_V1 = [
    "uri_entropy",              # entropia de toda la URI
    "query_entropy",            # entropia del query string
    "max_param_value_entropy",  # entropia maxima entre parametros query
    "query_pct_digit",          # % digitos en query
    "query_pct_alpha",          # % alfa en query
    "body_entropy",             # entropia del body
    "body_pct_digit",           # % digitos en body
    "body_pct_alpha",           # % alfa en body
]

FEAT_34 = FEAT_25 + FEAT_ENTROPY_V1

df = pd.read_parquet(PARQUET)
print(f"[INFO] Parquet OK: {len(df)} filas, {len(df.columns)} columnas totales")

missing = [f for f in FEAT_34 if f not in df.columns]
if missing:
    raise ValueError(f"Features no encontrados: {missing}")

feat_cols = FEAT_34
print(f"[INFO] Features usados: {len(feat_cols)}")
print(f"[INFO]   - 25 estructurales + 8 entropía v1 = 33... verificando: {len(feat_cols)}")

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

print(f"\n=== Test evaluation (RF 34 features: estructurales + entropía) ===")
print(f"ROC-AUC:       {roc_val:.4f}")
print(f"Balanced acc.: {ba_val:.4f}")
print(f"\nClase ATAQUE (1):")
print(f"  F1={f1_val:.4f}  Recall={rec_val:.4f}  Precision={pre_val:.4f}  FPR={fpr_val:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

fi = pd.DataFrame({
    "feature": feat_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nTop 15 features por importancia RF (Gini) — 34 feat:")
print(fi.head(15).to_string(index=False))
fi.to_csv(OUT_DIR / "feature_importance_rf.csv", index=False)

joblib.dump(rf, OUT_DIR / "model.joblib")
pd.DataFrame({"y_true": y_test, "y_pred": y_pred,
              "proba_0": 1 - y_prob, "proba_1": y_prob}).to_csv(
    OUT_DIR / "pred.csv", index=False
)

metrics = {
    "model": "RandomForest_34feat_balanced",
    "n_features": len(feat_cols),
    "feature_set": "structural_25_plus_entropy_v1_8",
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
PY

echo ""
echo "======================================================"
echo " Threshold tuning RF 34 features"
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

pred_path = Path("resultsOptimo/csic_sin_registro/rf_34feat/pred.csv")
out_dir   = Path("resultsOptimo/csic_sin_registro/rf_34feat_threshold")

df = pd.read_csv(pred_path)
y_true  = df["y_true"].values
proba_1 = df["proba_1"].values

roc_auc = roc_auc_score(y_true, proba_1)
print(f"[INFO] ROC-AUC (RF 34 feat): {roc_auc:.4f}")

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
print(f"\n[OPTIMO RF-34] Umbral: {best['threshold']:.2f}")
print(f"  F1={best['f1']:.4f}  Recall={best['recall']:.4f}  "
      f"Precision={best['precision']:.4f}  FPR={best['fpr']:.4f}  "
      f"BAcc={best['balanced_accuracy']:.4f}")

res_df.to_csv(out_dir / "threshold_sweep.csv", index=False)
with open(out_dir / "metrics.json", "w") as f:
    json.dump({
        "model": "RandomForest_34feat_balanced_threshold",
        "n_features": 34,
        "feature_set": "structural_25_plus_entropy_v1_8",
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

echo "[RF 34 FEAT COMPLETO]"
