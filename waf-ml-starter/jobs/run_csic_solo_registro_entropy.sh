#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=ocsvm_registro_entropy
#SBATCH --output=slurm-registro-entropy-%j.out
#SBATCH --gres=gpu:1

# =============================================================================
# Experimento: OCSVM solo registro.jsp con features de entropia
#
# Motivacion:
#   El experimento anterior (run_csic_solo_registro.sh) mostro que entrenar
#   solo con datos de registro.jsp NO mejora los resultados (F1=0.457 vs 0.787).
#   Esto confirma que el problema es de FEATURES, no de heterogeneidad de datos.
#
#   Nico/Ralf (OCS-WAF 2017) usaban features de entropia y distribucion de
#   caracteres por parametro. Este experimento agrega 8 features nuevos al
#   extractor global:
#     - uri_entropy, query_entropy, max_param_value_entropy
#     - query_pct_digit, query_pct_alpha
#     - body_entropy, body_pct_digit, body_pct_alpha
#
#   Estrategia: re-extraer features desde los campos raw del parquet existente
#   (no necesita los archivos .txt originales).
#
# Comparacion esperada:
#   Si F1 sube significativamente vs run_csic_solo_registro => features importan
#   Si F1 sigue igual => el gap es por arquitectura (per-group) no por features
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p data/tmp
mkdir -p resultsOptimo/csic_solo_registro/nu05_entropy

# Verificar GPU
srun $PYTHON -c "import sys; print('Python:', sys.executable); import cuml.accel; cuml.accel.install(); print('GPU cuml.accel OK')"
nvidia-smi || true

# =============================================================================
# PASO 1: Re-extraer features con nuevo http_features.py (34 features)
#         Filtrar solo filas de registro.jsp
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 1: Re-extraer features con entropia (34 features)"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
import math
import pandas as pd
from pathlib import Path
from waf_ml.features.http_features import extract_http_features

def _to_bytes(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return b""
    s = str(v)
    if s in {"", "nan", "None"}:
        return b""
    return s.encode("utf-8", errors="ignore")

def _json_to_headers(s):
    if not s:
        return {}
    try:
        return json.loads(str(s))
    except Exception:
        return {}

src = Path("data/processed/csic/csic_features.parquet")
out = Path("data/tmp/csic_solo_registro_entropy.parquet")

print(f"[STEP1] Leyendo parquet completo CSIC: {src}")
df = pd.read_parquet(src)
print(f"[STEP1] Total filas: {len(df)}")

# Filtrar solo registro.jsp
mask = df["request_http_request"].str.contains("registro", case=False, na=False)
df_reg = df[mask].reset_index(drop=True)
print(f"[STEP1] Filas de registro.jsp: {len(df_reg)}")

if "label_binary" in df_reg.columns:
    dist = df_reg["label_binary"].value_counts().sort_index()
    print(f"[STEP1] Distribucion label_binary:\n{dist.to_string()}")

# Re-extraer features usando el http_features.py actualizado (34 features)
print("[STEP1] Re-extrayendo features con entropia...")
new_feats = df_reg.apply(
    lambda r: extract_http_features(
        method=str(r.get("request_http_method", "")),
        uri=str(r.get("request_http_request", "")),
        headers=_json_to_headers(r.get("request_headers_json", "")),
        body=_to_bytes(r.get("request_body", None)),
    ),
    axis=1,
)
feat_df = pd.DataFrame(list(new_feats))
print(f"[STEP1] Features extraidos: {len(feat_df.columns)}")
print(f"[STEP1] Nuevos features: uri_entropy, query_entropy, max_param_value_entropy, "
      f"query_pct_digit, query_pct_alpha, body_entropy, body_pct_digit, body_pct_alpha")

# Conservar columnas de etiqueta y metadata
label_cols = [c for c in df_reg.columns if c.startswith("label_")]
meta_cols = [c for c in ["source_file", "dataset_name", "split", "label_type_raw"] if c in df_reg.columns and c not in label_cols]
keep_cols = label_cols + meta_cols

result = pd.concat([df_reg[keep_cols].reset_index(drop=True), feat_df], axis=1)
result.to_parquet(out, index=False)
print(f"[STEP1] Guardado: {out}  filas={len(result)}  columnas={len(result.columns)}")

# Mostrar estadisticas de los nuevos features
print("\n[STEP1] Estadisticas de los 8 nuevos features:")
new_feat_cols = ["uri_entropy","query_entropy","max_param_value_entropy",
                 "query_pct_digit","query_pct_alpha","body_entropy",
                 "body_pct_digit","body_pct_alpha"]
print(result[new_feat_cols].describe().round(4).to_string())
PY

echo "[PASO 1 COMPLETO]"

# =============================================================================
# PASO 2: OCSVM con 34 features, solo registro.jsp
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 2: OCSVM nu=0.05, gamma=0.1 con 34 features"
echo "======================================================"

PYTHONPATH=src srun $PYTHON src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data data/tmp/csic_solo_registro_entropy.parquet \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --final-train-on train_normals \
  --tune none \
  --nu 0.05 \
  --gamma 0.1 \
  --require-gpu \
  --out resultsOptimo/csic_solo_registro/nu05_entropy/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_solo_registro/nu05_entropy/metrics.json \
  --pred-out resultsOptimo/csic_solo_registro/nu05_entropy/pred.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_solo_registro/nu05_entropy/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 0 \
  --benchmark-out resultsOptimo/csic_solo_registro/nu05_entropy/benchmark.json

echo "[PASO 2 COMPLETO] Resultados en resultsOptimo/csic_solo_registro/nu05_entropy/"

# =============================================================================
# PASO 3: Comparacion completa de los 3 escenarios del registro.jsp
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 3: Comparacion de escenarios registro.jsp"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

runs = [
    ("sin registro nu=0.05 (26 feat, MEJOR previo)",    "resultsOptimo/csic_sin_registro/nicoRalf_nu05/metrics.json"),
    ("SOLO registro nu=0.05 (26 feat)",                 "resultsOptimo/csic_solo_registro/nu05/metrics.json"),
    ("SOLO registro nu=0.05 (34 feat + entropia)",      "resultsOptimo/csic_solo_registro/nu05_entropy/metrics.json"),
]

def get_m(path, key):
    p = Path(path)
    if not p.exists():
        return "N/A"
    with open(p) as f:
        d = json.load(f)
    for sub in ("evaluation", "eval", "metrics", "test"):
        if sub in d and isinstance(d[sub], dict) and key in d[sub]:
            return f"{d[sub][key]:.4f}"
    v = d.get(key)
    if v is not None and not isinstance(v, dict):
        return f"{v:.4f}"
    return "N/A"

print(f"\n{'Experimento':<48} {'F1':>8} {'Recall':>8} {'Precision':>10} {'FPR':>8} {'BAcc':>8}")
print("-" * 96)
for name, path in runs:
    f1  = get_m(path, "f1")
    rec = get_m(path, "recall")
    pre = get_m(path, "precision")
    fpr = get_m(path, "fpr")
    ba  = get_m(path, "balanced_accuracy")
    print(f"{name:<48} {f1:>8} {rec:>8} {pre:>10} {fpr:>8} {ba:>8}")

print()
print("Referencia Nico/Ralf (OCS-WAF 2017) grupos c12/c13/t01:")
print("  F1=0.95   Recall(TPR)=0.93   FPR=0.03")
print()
print("INTERPRETACION:")
print("  34 feat >> 26 feat en registro  =>  los features de entropia explican el gap")
print("  34 feat ~= 26 feat en registro  =>  el gap es por arquitectura per-group")
PY
