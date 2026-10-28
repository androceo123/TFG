#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --job-name=torpeda_binary_rf
#SBATCH --output=slurm-torpeda-binary-rf-%j.out

# =============================================================================
# RandomForest binario sobre TorpEda — mismo pipeline que CSIC
#
# Motivacion (PI-2 / comparacion cruzada de datasets):
#   Verificar si el mismo RF (57 features, parametros identicos) que obtuvo
#   F1=0.947 en CSIC 2010 mantiene buenas metricas en TorpEda.
#   TorpEda tiene mayor proporcion de ataques (88.7%) y 10 tipos distintos.
#   Este experimento responde: ¿generaliza el pipeline application-independent?
#
# Pipeline identico al CSIC baseline:
#   - 80/20 split estratificado, seed=42
#   - RandomForestClassifier: 500 arboles, class_weight=balanced, n_jobs=8
#   - Threshold sweep 0.05-0.95, seleccion por max F1
#   - Mismas 57 features (re-extraidas desde columnas HTTP crudas)
#
# Dataset:
#   data/processed/torpeda/torpeda_features.parquet
#   74.133 filas | normal=8.363 (11.3%) | ataque=65.770 (88.7%)
#
# Salidas:
#   results/torpeda/binary/             (thr=0.50)
#   results/torpeda/binary_threshold/   (thr=optimo)
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"
PARQUET="data/processed/torpeda/torpeda_features.parquet"
OUT_DIR="results/torpeda/binary"

# Verificar parquet de entrada
PYTHONPATH=src srun $PYTHON - <<'PY'
from pathlib import Path
import pandas as pd

p = Path("data/processed/torpeda/torpeda_features.parquet")
if not p.exists():
    raise FileNotFoundError(
        f"No encontrado: {p}\n"
        "El parquet de TorpEda debe existir en data/processed/torpeda/"
    )
df = pd.read_parquet(p)
print(f"[CHECK] TorpEda parquet OK: {len(df)} filas, {len(df.columns)} columnas")
vals = df["label_binary"].value_counts().sort_index()
print(f"[CHECK] label_binary: {vals.to_dict()}")
feat_25 = [
    "uri_len", "path_depth", "query_len", "n_query_params", "max_param_value_len",
    "uri_pct_non_alnum_ratio", "encoded", "suspicious_tokens_count", "has_suspicious_tokens",
    "body_suspicious_tokens_count", "body_has_suspicious_tokens", "body_encoded",
    "uncommon_method", "req_content_length", "body_len",
    "method_GET", "method_POST", "method_HEAD", "method_PUT", "method_DELETE",
    "method_PATCH", "method_OPTIONS", "method_TRACE", "method_CONNECT", "method_OTHER",
]
feat_extra = [
    "uri_entropy", "query_entropy", "body_entropy",
    "query_char_dist_i0", "body_char_dist_i0",
]
missing = [f for f in feat_extra if f not in df.columns]
if missing:
    print(f"[CHECK] Features v1/v2 faltantes ({len(missing)}) -> se re-extraeran")
else:
    print(f"[CHECK] Todos los features presentes en el parquet")
PY

echo ""
echo "======================================================"
echo " TorpEda — RF binario (57 features, pipeline CSIC)"
echo "======================================================"

# --rebuild-features fuerza re-extraccion completa desde columnas HTTP
# (garantiza misma logica de extract_http_features que en CSIC)
PYTHONPATH=src srun $PYTHON \
    src/waf_ml/scripts/train_binary_rf.py \
    --parquet   "$PARQUET" \
    --dataset   torpeda \
    --out-dir   "$OUT_DIR" \
    --feat-set  57 \
    --seed      42 \
    --test-size 0.2 \
    --n-estimators 500 \
    --n-jobs    8 \
    --rebuild-features

echo ""
echo "======================================================"
echo " TorpEda — Tabla comparativa CSIC vs TorpEda"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

def load_thr_metrics(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

csic  = load_thr_metrics("resultsOptimo/csic_sin_registro/rf_supervised_threshold/metrics.json")
torpe = load_thr_metrics("results/torpeda/binary_threshold/metrics.json")

def safe(d, key, subkey=None):
    if d is None: return "N/A"
    if subkey:
        sub = d.get(subkey, {})
        v = sub.get(key) if isinstance(sub, dict) else None
    else:
        v = d.get(key)
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)

def ev(d, key):
    if d is None: return "N/A"
    # buscar en evaluation (formato antiguo CSIC) o directo (nuevo formato)
    v = d.get(key)
    if v is None and "evaluation" in d:
        v = d["evaluation"].get(key)
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)

cols = ["dataset", "n_features", "threshold", "train_normal", "train_attack",
        "test_normal", "test_attack", "f1", "recall", "precision",
        "balanced_accuracy", "roc_auc", "pr_auc", "mcc", "fpr", "fnr"]

print(f"\n{'=' * 120}")
print(f" TABLA COMPARATIVA: CSIC vs TorpEda (RF binario, thr=OPTIMO)")
print(f"{'=' * 120}")

# CSIC (formato antiguo, campos en evaluation + directo)
csic_row = {
    "dataset":      "CSIC 2010",
    "n_features":   safe(csic, "n_features") if csic and "n_features" in csic else "57",
    "threshold":    safe(csic, "threshold"),
    "train_normal": "N/A",  # no en formato antiguo
    "train_attack": "N/A",
    "test_normal":  "N/A",
    "test_attack":  "N/A",
    "f1":           ev(csic, "f1"),
    "recall":       ev(csic, "recall"),
    "precision":    ev(csic, "precision"),
    "balanced_accuracy": ev(csic, "balanced_accuracy"),
    "roc_auc":      safe(csic, "roc_auc"),
    "pr_auc":       "N/A",  # no en formato antiguo
    "mcc":          "N/A",
    "fpr":          ev(csic, "fpr"),
    "fnr":          "N/A",
}

# TorpEda (nuevo formato completo)
torpe_row = {k: safe(torpe, k) if k not in ["f1","recall","precision","balanced_accuracy","roc_auc","pr_auc","mcc","fpr","fnr"] else ev(torpe, k)
             for k in cols}
torpe_row["dataset"] = "TorpEda"

for row_name, row in [("CSIC 2010", csic_row), ("TorpEda", torpe_row)]:
    print(f"\nDataset:          {row['dataset']}")
    print(f"  N features:     {row['n_features']}")
    print(f"  Threshold:      {row['threshold']}")
    print(f"  Train normal:   {row['train_normal']}  |  Train ataque: {row['train_attack']}")
    print(f"  Test  normal:   {row['test_normal']}   |  Test  ataque: {row['test_attack']}")
    print(f"  F1:             {row['f1']}")
    print(f"  Recall:         {row['recall']}")
    print(f"  Precision:      {row['precision']}")
    print(f"  Balanced acc.:  {row['balanced_accuracy']}")
    print(f"  ROC-AUC:        {row['roc_auc']}")
    print(f"  PR-AUC:         {row['pr_auc']}")
    print(f"  MCC:            {row['mcc']}")
    print(f"  FPR:            {row['fpr']}")
    print(f"  FNR:            {row['fnr']}")

print(f"\n{'=' * 120}")
print("INTERPRETACION:")
print("  Si TorpEda F1 ~= CSIC F1  => pipeline generaliza bien entre datasets")
print("  Si TorpEda F1 >> CSIC F1  => TorpEda mas facil (mas ataques, patron claro)")
print("  Si TorpEda F1 << CSIC F1  => TorpEda requiere ajuste (distribucion diferente)")
PY

echo "[EXPERIMENTO TORPEDA RF COMPLETO]"
