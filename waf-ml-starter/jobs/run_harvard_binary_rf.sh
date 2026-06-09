#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:30:00
#SBATCH --job-name=harvard_binary_rf
#SBATCH --output=slurm-harvard-binary-rf-%j.out

# =============================================================================
# RandomForest binario sobre Harvard/SR-BH — mismo pipeline que CSIC
#
# Motivacion (PI-2 / comparacion cruzada de datasets):
#   Verificar si el mismo RF que obtuvo F1=0.947 en CSIC 2010 mantiene
#   buenas metricas sobre Harvard SR-BH 2020 (dataset real de honeypot).
#   Harvard tiene 907.815 requests, 12 categorias CAPEC, multi-label real.
#   Para este experimento se usa solo la dimension BINARIA (normal vs ataque).
#
# Pipeline identico al CSIC baseline:
#   - 80/20 split estratificado, seed=42
#   - RandomForestClassifier: 500 arboles, class_weight=balanced, n_jobs=8
#   - Threshold sweep 0.05-0.95, seleccion por max F1
#   - Mismas 57 features (re-extraidas desde columnas HTTP crudas)
#
# Nota de memoria: Harvard tiene 907K filas. La re-extraccion de features
# con 8 workers tarda ~10-15 min. El entrenamiento RF ~10-15 min adicionales.
# Total estimado: 30-60 min. Se asignan 1.5h como margen.
# RAM ampliada a 32G por el tamanio del dataset.
#
# Dataset:
#   data/processed/harvard/harvard.parquet
#   907.815 filas | normal=525.195 (57.8%) | ataque=382.620 (42.2%)
#
# Etiquetas binarias:
#   label_binary=0 -> NORMAL (000 - Normal en CAPEC)
#   label_binary=1 -> ATAQUE (cualquier categoria CAPEC)
#   La granularidad multilabel se ignora en este experimento.
#
# Salidas:
#   results/harvard/binary/             (thr=0.50)
#   results/harvard/binary_threshold/   (thr=optimo)
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"
PARQUET="data/processed/harvard/harvard.parquet"
OUT_DIR="results/harvard/binary"

# Verificar parquet de entrada
PYTHONPATH=src srun $PYTHON - <<'PY'
from pathlib import Path
import pandas as pd

p = Path("data/processed/harvard/harvard.parquet")
if not p.exists():
    raise FileNotFoundError(
        f"No encontrado: {p}\n"
        "El parquet de Harvard debe existir en data/processed/harvard/"
    )
df = pd.read_parquet(p)
print(f"[CHECK] Harvard parquet OK: {len(df)} filas, {len(df.columns)} columnas")
vals = df["label_binary"].value_counts().sort_index()
print(f"[CHECK] label_binary: {vals.to_dict()}")
print(f"[CHECK] Ratio normal/ataque: {vals[0] / vals.get(1, 1):.2f}x")

# Verificar columnas HTTP necesarias para re-extraccion
required = ["request_http_method", "request_http_request", "request_body"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Columnas HTTP faltantes en Harvard: {missing}")
print(f"[CHECK] Columnas HTTP OK: {required}")

feat_extra = [
    "uri_entropy", "query_entropy", "body_entropy",
    "query_char_dist_i0", "body_char_dist_i0",
]
missing_feats = [f for f in feat_extra if f not in df.columns]
if missing_feats:
    print(f"[CHECK] Features v1/v2 faltantes ({len(missing_feats)}) -> se re-extraeran")
else:
    print(f"[CHECK] Todos los features ya presentes en el parquet")
PY

echo ""
echo "======================================================"
echo " Harvard/SR-BH — RF binario (57 features, pipeline CSIC)"
echo "======================================================"
echo " NOTA: 907K filas — feature extraction ~10-15 min con 8 workers"
echo "======================================================"

# --rebuild-features: re-extrae las 57 features desde columnas HTTP crudas.
# Necesario porque Harvard solo tiene 25 features estructurales en su parquet.
PYTHONPATH=src srun $PYTHON \
    src/waf_ml/scripts/train_binary_rf.py \
    --parquet   "$PARQUET" \
    --dataset   harvard \
    --out-dir   "$OUT_DIR" \
    --feat-set  57 \
    --seed      42 \
    --test-size 0.2 \
    --n-estimators 500 \
    --n-jobs    8 \
    --rebuild-features

echo ""
echo "======================================================"
echo " Harvard — Tabla comparativa CSIC vs Harvard"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

def load_metrics(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def ev(d, key):
    """Extrae metrica soportando formato antiguo (evaluation:{}) y nuevo (directo)."""
    if d is None: return "N/A"
    v = d.get(key)
    if v is None and "evaluation" in d:
        v = d["evaluation"].get(key)
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)

def safe(d, key):
    if d is None: return "N/A"
    v = d.get(key)
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)

csic    = load_metrics("resultsOptimo/csic_sin_registro/rf_supervised_threshold/metrics.json")
harvard = load_metrics("results/harvard/binary_threshold/metrics.json")

rows = [
    ("CSIC 2010",       csic,    "57",    "N/A", "N/A", "N/A", "N/A"),
    ("Harvard SR-BH",   harvard, None,    None,  None,  None,  None),
]

print(f"\n{'=' * 120}")
print(f" TABLA COMPARATIVA: CSIC vs Harvard SR-BH (RF binario, thr=OPTIMO)")
print(f"{'=' * 120}")

for name, d, nfeat, tn0, ta0, tn1, ta1 in rows:
    nf  = nfeat if nfeat else safe(d, "n_features")
    thr = safe(d, "threshold")
    trn = tn0 if tn0 else safe(d, "train_normal")
    tra = ta0 if ta0 else safe(d, "train_attack")
    ten = tn1 if tn1 else safe(d, "test_normal")
    tea = ta1 if ta1 else safe(d, "test_attack")

    print(f"\nDataset:          {name}")
    print(f"  N features:     {nf}")
    print(f"  Threshold:      {thr}")
    print(f"  Train normal:   {trn}  |  Train ataque: {tra}")
    print(f"  Test  normal:   {ten}  |  Test  ataque: {tea}")
    print(f"  F1:             {ev(d, 'f1')}")
    print(f"  Recall:         {ev(d, 'recall')}")
    print(f"  Precision:      {ev(d, 'precision')}")
    print(f"  Balanced acc.:  {ev(d, 'balanced_accuracy')}")
    print(f"  ROC-AUC:        {safe(d, 'roc_auc')}")
    print(f"  PR-AUC:         {safe(d, 'pr_auc')}")
    print(f"  MCC:            {safe(d, 'mcc')}")
    print(f"  FPR:            {ev(d, 'fpr')}")
    print(f"  FNR:            {safe(d, 'fnr')}")

print(f"\n{'=' * 120}")
print("INTERPRETACION:")
print("  Harvard es un dataset de honeypot real (WordPress, julio 2020).")
print("  Las diferencias vs CSIC pueden deberse a:")
print("    - Distribucion de ataques (Harvard: SQLi dominante 65%, CSIC: mas variado)")
print("    - Requests HTTP reales vs sinteticas (CSIC es generado)")
print("    - Distribucion de classes: Harvard ~58/42, CSIC ~76/24")
PY

echo "[EXPERIMENTO HARVARD RF COMPLETO]"
