#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --job-name=torpeda_rf_mc
#SBATCH --output=slurm-torpeda-rf-mc-%j.out

# =============================================================================
# TorpEda — RandomForest MULTICLASE (clasificación de tipo de ataque)
#
# Motivacion:
#   Con CSIC 2010 solo había 2 clases (normal / ataque) → clasificación binaria.
#   TorpEda tiene etiquetas de tipo de ataque: SQLi, XSS, SSI, BufferOverflow,
#   CRLFi, XPath, LDAPi, FormatString, ANOMALOUS + NORMAL.
#   El profe pide: "Probar con TorpEda, a ver qué tal la clasificación
#   multiclase con RF".
#
# FEATURESET:
#   - Paso 1: Reconstruir parquet TorpEda con 57 features completos (si no existe)
#     El parquet antiguo solo tiene 25 features (estructurales, sin entropía).
#   - Paso 2: RF multiclase con las 3 configuraciones de features:
#       * 57 feat (todos)
#       * 34 feat (25 orig + 8 entropía v1)
#       * 25 feat (solo estructurales originales)
#
# Dataset TorpEda:
#   - 74.133 muestras
#   - 8.363 NORMAL (11.3%), 65.770 ataques (88.7%)
#   - Clases: SQLi(43k), ANOMALOUS(16k), NORMAL(8k), XSS(4.8k), SSI(451),
#             BufferOverflow(412), CRLFi(327), XPath(175), LDAPi(74), FormatString(41)
#
# PREREQUISITO: data/raw/torpeda/*.xml en el HPC
#   (o data/processed/torpeda/torpeda_features.parquet si ya está procesado sin v2)
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p data/tmp
mkdir -p resultsOptimo/torpeda/rf_multiclass_57feat
mkdir -p resultsOptimo/torpeda/rf_multiclass_34feat
mkdir -p resultsOptimo/torpeda/rf_multiclass_25feat

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: Construir parquet TorpEda con features v2 (57 features)
# ─────────────────────────────────────────────────────────────────────────────
echo "======================================================"
echo " PASO 1: Build TorpEda features v2 (57 features)"
echo "======================================================"

TORPEDA_V2="data/tmp/torpeda_v2.parquet"
TORPEDA_RAW="data/raw/torpeda"

if [ -f "$TORPEDA_V2" ]; then
    echo "[SKIP] Ya existe: $TORPEDA_V2"
else
    echo "[BUILD] Construyendo $TORPEDA_V2 desde $TORPEDA_RAW ..."
    PYTHONPATH=src srun $PYTHON -m waf_ml.scripts.build_features_torpeda \
        --inputs "$TORPEDA_RAW" \
        --output "$TORPEDA_V2" \
        --label-prefix "TORPEDA" \
        --out-format parquet
    echo "[BUILD] Parquet construido: $TORPEDA_V2"
fi

# Verificar
PYTHONPATH=src srun $PYTHON - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("data/tmp/torpeda_v2.parquet")
if not p.exists():
    raise FileNotFoundError(
        f"No encontrado: {p}\n"
        "¿Están los archivos XML en data/raw/torpeda/ ?"
    )
df = pd.read_parquet(p)
print(f"[CHECK] Parquet TorpEda v2: {len(df)} filas, {len(df.columns)} columnas")
print(f"[CHECK] label_multiclass:\n{df['label_multiclass'].value_counts().to_string()}")
print(f"[CHECK] label_binary: {df['label_binary'].value_counts().to_dict()}")

label_cols = [c for c in df.columns if c.startswith("label_")]
meta_cols = ["source_file", "dataset_name", "sample_id", "http_version",
             "dataset_author", "request_http_method", "request_http_request",
             "request_body", "request_headers_json", "label_type_raw",
             "label_attack_raw", "label_multilabel"]
feat_cols = [c for c in df.columns if c not in label_cols + meta_cols]
print(f"[CHECK] Features: {len(feat_cols)}")
PY

# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: RF multiclase con diferentes conjuntos de features
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "======================================================"
echo " PASO 2: RF Multiclase TorpEda"
echo "======================================================"

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
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

PARQUET   = Path("data/tmp/torpeda_v2.parquet")
SEED      = 42
TEST_SIZE = 0.2

# ── Definición de feature sets ─────────────────────────────────────────────
FEAT_25 = [
    "uri_len", "path_depth", "query_len", "n_query_params", "max_param_value_len",
    "uri_pct_non_alnum_ratio", "encoded", "suspicious_tokens_count", "has_suspicious_tokens",
    "body_suspicious_tokens_count", "body_has_suspicious_tokens", "body_encoded",
    "uncommon_method", "req_content_length", "body_len",
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
    "max_body_param_char_dist_i0", "max_body_param_char_dist_i1",
    "max_body_param_char_dist_i2", "max_body_param_char_dist_i3",
    "max_body_param_char_dist_i4",
    "mean_param_value_entropy", "std_param_value_entropy",
    "max_body_param_value_entropy", "mean_body_param_value_entropy",
]

FEAT_34 = FEAT_25 + FEAT_ENTROPY_V1

feature_sets = {
    "25feat": (FEAT_25,
               Path("resultsOptimo/torpeda/rf_multiclass_25feat")),
    "34feat": (FEAT_34,
               Path("resultsOptimo/torpeda/rf_multiclass_34feat")),
}

df = pd.read_parquet(PARQUET)
print(f"[INFO] Dataset TorpEda v2: {len(df)} filas")
print(f"[INFO] Distribución clases:\n{df['label_multiclass'].value_counts().to_string()}")

# Usar label_multiclass como target
y_str = df["label_multiclass"].fillna("UNKNOWN").values
le = LabelEncoder()
y   = le.fit_transform(y_str)
classes = list(le.classes_)
print(f"\n[INFO] Clases ({len(classes)}): {classes}")

# Verificar disponibilidad de features v2
has_v2 = all(f in df.columns for f in FEAT_CHAR_DIST_V2[:5])
if has_v2:
    feature_sets["57feat"] = (
        FEAT_25 + FEAT_ENTROPY_V1 + FEAT_CHAR_DIST_V2,
        Path("resultsOptimo/torpeda/rf_multiclass_57feat"),
    )
    print("[INFO] Features v2 disponibles — se entrenará también RF con 57 feat")
else:
    print("[WARN] Features v2 no disponibles en parquet — se omite RF-57feat")

# ── Entrenar un RF por cada feature set ────────────────────────────────────
for fs_name, (feat_cols, out_dir) in feature_sets.items():
    print(f"\n{'='*60}")
    print(f" RF Multiclase TorpEda — {fs_name} ({len(feat_cols)} features)")
    print(f"{'='*60}")

    avail = [f for f in feat_cols if f in df.columns]
    if len(avail) < len(feat_cols):
        missing = [f for f in feat_cols if f not in df.columns]
        print(f"[WARN] Features faltantes: {missing}")
        feat_cols = avail

    X = df[feat_cols].fillna(0).values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # Distribución de clases en train
    unique, counts = np.unique(y_train, return_counts=True)
    print("[INFO] Distribución train:")
    for cls_idx, cnt in zip(unique, counts):
        print(f"       {classes[cls_idx]:<35} {cnt:>6}")

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",  # crucial para clases muy desbalanceadas
        n_jobs=8,
        random_state=SEED,
        oob_score=True,
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"[INFO] Entrenamiento: {train_time:.1f}s  |  OOB: {rf.oob_score_:.4f}")

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    ba   = balanced_accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n=== Resultados RF {fs_name} ===")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Balanced Accuracy:   {ba:.4f}")
    print(f"F1 (weighted):       {f1_w:.4f}")
    print(f"F1 (macro):          {f1_m:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=classes, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix (filas=real, cols=predicho):")
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())

    # Feature importance
    fi = pd.DataFrame({
        "feature": feat_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\nTop 15 features (Gini) — {fs_name}:")
    print(fi.head(15).to_string(index=False))
    fi.to_csv(out_dir / "feature_importance_rf.csv", index=False)

    joblib.dump(rf, out_dir / "model.joblib")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
        out_dir / "pred.csv", index=False
    )

    # Per-class metrics
    per_class = {}
    unique_test = np.unique(y_test)
    for cls_idx in unique_test:
        cls_name = classes[cls_idx]
        y_t_bin = (y_test == cls_idx).astype(int)
        y_p_bin = (y_pred == cls_idx).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t_bin, y_p_bin, labels=[0,1]).ravel()
        per_class[cls_name] = {
            "f1":        float(f1_score(y_t_bin, y_p_bin, zero_division=0)),
            "recall":    float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "fpr":       float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "support":   int(sum(y_test == cls_idx)),
        }

    metrics = {
        "model": f"RandomForest_{fs_name}_multiclass_balanced",
        "task": "multiclass",
        "dataset": "TorpEda",
        "n_features": len(feat_cols),
        "feature_set": fs_name,
        "n_classes": len(classes),
        "classes": classes,
        "n_estimators": 500,
        "oob_score": float(rf.oob_score_),
        "train_time_s": round(train_time, 2),
        "evaluation": {
            "accuracy": float(acc),
            "balanced_accuracy": float(ba),
            "f1_weighted": float(f1_w),
            "f1_macro": float(f1_m),
        },
        "per_class": per_class,
    }
    with open(out_dir / "metrics.json", "w") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"\n[GUARDADO] {out_dir}/metrics.json")

print("\n[TODOS LOS RF MULTICLASE COMPLETADOS]")
PY

echo ""
echo "======================================================"
echo " Resumen comparativo TorpEda multiclase"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

runs = [
    ("RF 57 feat", "resultsOptimo/torpeda/rf_multiclass_57feat/metrics.json"),
    ("RF 34 feat", "resultsOptimo/torpeda/rf_multiclass_34feat/metrics.json"),
    ("RF 25 feat", "resultsOptimo/torpeda/rf_multiclass_25feat/metrics.json"),
]

for name, path in runs:
    p = Path(path)
    if not p.exists():
        print(f"{name}: NO DISPONIBLE")
        continue
    with open(p) as f:
        m = json.load(f)
    ev = m["evaluation"]
    print(f"\n{'='*50}")
    print(f" {name} ({m['n_features']} features)")
    print(f"{'='*50}")
    print(f"  Accuracy:          {ev['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {ev['balanced_accuracy']:.4f}")
    print(f"  F1 (weighted):     {ev['f1_weighted']:.4f}")
    print(f"  F1 (macro):        {ev['f1_macro']:.4f}")
    print(f"\n  Por clase:")
    print(f"  {'Clase':<35} {'F1':>8} {'Recall':>8} {'Prec.':>8} {'FPR':>8} {'Support':>9}")
    print(f"  {'-'*80}")
    for cls, cm in sorted(m["per_class"].items(), key=lambda x: -x[1]["support"]):
        print(f"  {cls:<35} {cm['f1']:>8.4f} {cm['recall']:>8.4f} "
              f"{cm['precision']:>8.4f} {cm['fpr']:>8.4f} {cm['support']:>9}")
PY

echo "[TORPEDA RF MULTICLASE COMPLETO]"
