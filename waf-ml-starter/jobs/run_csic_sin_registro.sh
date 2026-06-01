#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ocsvm_csic_sin_registro_gpu
#SBATCH --output=slurm-csic-sin-registro-%j.out
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

mkdir -p resultsOptimo/csic_sin_registro/oneclass

# Verificar GPU
srun python3.11 -c "import sys; print('Python:', sys.executable); import cuml.accel; cuml.accel.install(); print('GPU cuml.accel OK')"
nvidia-smi || true

# Generar dataset filtrado (sin aplicacion registro) si no existe
if [ ! -f data/processed/csic_sin_registro/csic_features_sin_registro.parquet ]; then
    echo "[INFO] Generando dataset CSIC sin aplicacion registro..."
    PYTHONPATH=src srun python3.11 - <<'PY'
from pathlib import Path
import pandas as pd

src = Path("data/processed/csic/csic_features.parquet")
out = Path("data/processed/csic_sin_registro/csic_features_sin_registro.parquet")

df = pd.read_parquet(src)
mask_registro = df['request_http_request'].str.contains('registro', case=False, na=False)
df_filtrado = df[~mask_registro].reset_index(drop=True)

out.parent.mkdir(parents=True, exist_ok=True)
df_filtrado.to_parquet(out, index=False)

print(f"[FILTRO] Original: {len(df)} filas | Eliminadas (registro): {mask_registro.sum()} | Resultado: {len(df_filtrado)} filas")
if 'label_type_raw' in df_filtrado.columns:
    print("[FILTRO] Distribucion labels:")
    print(df_filtrado['label_type_raw'].value_counts().sort_index().to_string())
PY
else
    echo "[INFO] Dataset filtrado ya existe, usando el existente."
fi

PYTHONPATH=src srun python3.11 src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data data/processed/csic_sin_registro/csic_features_sin_registro.parquet \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --final-train-on train_normals \
  --tune fast \
  --cv 3 \
  --tune-metric auto \
  --tune-n-iter 32 \
  --tune-n-jobs 4 \
  --require-gpu \
  --out resultsOptimo/csic_sin_registro/oneclass/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/oneclass/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/oneclass/pred.csv \
  --tune-results-out resultsOptimo/csic_sin_registro/oneclass/search.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_sin_registro/oneclass/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 0 \
  --benchmark-out resultsOptimo/csic_sin_registro/oneclass/benchmark.json
