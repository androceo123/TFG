#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=ocsvm_csic_sin_registro_1000_gpu
#SBATCH --output=slurm-csic-sin-registro-1000-%j.out
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/waf-ml-starter" || exit 1

source .venv/bin/activate

mkdir -p resultsOptimo/csic_sin_registro/oneclass_test1000 data/tmp

# Paso 1: Generar dataset CSIC filtrado (sin aplicacion registro) y tomar muestra de 1000 filas.
# Si label_binary existe, el muestreo es estratificado para preservar la proporcion normal/anomalo.
PYTHONPATH=src srun .venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

src = Path("data/processed/csic/csic_features.parquet")
out = Path("data/tmp/csic_features_sin_registro_1000.parquet")
label_col = "label_binary"
seed = 42
n = 1000

df = pd.read_parquet(src)

# Filtrar aplicacion registro
mask_registro = df['request_http_request'].str.contains('registro', case=False, na=False)
df = df[~mask_registro].reset_index(drop=True)
print(f"[FILTRO] Eliminadas {mask_registro.sum()} filas de registro. Dataset resultante: {len(df)} filas")

# Muestreo estratificado
if len(df) <= n:
    sample = df.copy().reset_index(drop=True)
else:
    stratify = None
    if label_col in df.columns and df[label_col].nunique(dropna=True) > 1:
        counts = df[label_col].value_counts(dropna=True)
        if int(counts.min()) >= 2:
            stratify = df[label_col]
    sample, _ = train_test_split(df, train_size=n, random_state=seed, shuffle=True, stratify=stratify)
    sample = sample.reset_index(drop=True)

out.parent.mkdir(parents=True, exist_ok=True)
sample.to_parquet(out, index=False)
print(f"[SAMPLE] saved={out} rows={len(sample)}")
if label_col in sample.columns:
    print("[SAMPLE] label_binary counts:")
    print(sample[label_col].value_counts().sort_index().to_string())
PY

# Paso 2: Verificar GPU
PYTHONPATH=src srun .venv/bin/python -c "import sys; print('Python:', sys.executable); import cuml.accel; cuml.accel.install(); print('GPU cuml.accel OK')"
nvidia-smi || true

# Paso 3: Entrenar OCSVM sobre dataset filtrado (muestra de 1000)
PYTHONPATH=src srun .venv/bin/python src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data data/tmp/csic_features_sin_registro_1000.parquet \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --final-train-on train_normals \
  --tune fast \
  --cv 3 \
  --tune-metric auto \
  --tune-n-iter 8 \
  --tune-n-jobs 4 \
  --require-gpu \
  --out resultsOptimo/csic_sin_registro/oneclass_test1000/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/oneclass_test1000/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/oneclass_test1000/pred.csv \
  --tune-results-out resultsOptimo/csic_sin_registro/oneclass_test1000/search.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_sin_registro/oneclass_test1000/feature_importance_shap.csv \
  --fi-max-rows 200 \
  --shap-background-rows 50 \
  --benchmark-mode auto \
  --benchmark-max-rows 500 \
  --benchmark-out resultsOptimo/csic_sin_registro/oneclass_test1000/benchmark.json
