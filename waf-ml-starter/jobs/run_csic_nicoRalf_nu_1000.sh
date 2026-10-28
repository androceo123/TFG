#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --job-name=ocsvm_nicoRalf_1000
#SBATCH --output=slurm-nicoRalf-nu-1000-%j.out
#SBATCH --gres=gpu:1

# =============================================================================
# TEST RAPIDO (1000 filas): Recalibracion nu/gamma rango Nico & Ralf
# Ejecutar este job ANTES del completo para verificar que funciona correctamente.
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu05
mkdir -p resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu10
mkdir -p data/tmp

# Generar muestra de 1000 filas estratificada
PYTHONPATH=src srun $PYTHON - <<'PY'
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

src = Path("data/processed/csic_sin_registro/csic_features_sin_registro.parquet")
out = Path("data/tmp/csic_nicoRalf_1000.parquet")
label_col = "label_binary"
seed = 42
n = 1000

df = pd.read_parquet(src)
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
    print(sample[label_col].value_counts().sort_index().to_string())
PY

# Verificar GPU
srun $PYTHON -c "import cuml.accel; cuml.accel.install(); print('GPU OK')"

DATA="data/tmp/csic_nicoRalf_1000.parquet"

echo ""
echo "=== TEST nu=0.05, gamma=0.1 ==="
PYTHONPATH=src srun $PYTHON src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data $DATA \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --final-train-on train_normals \
  --tune none \
  --nu 0.05 \
  --gamma 0.1 \
  --require-gpu \
  --out resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu05/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu05/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu05/pred.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu05/feature_importance_shap.csv \
  --fi-max-rows 200 \
  --shap-background-rows 50 \
  --benchmark-mode auto \
  --benchmark-max-rows 500 \
  --benchmark-out resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu05/benchmark.json

echo ""
echo "=== TEST nu=0.1, gamma=0.1 ==="
PYTHONPATH=src srun $PYTHON src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data $DATA \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --final-train-on train_normals \
  --tune none \
  --nu 0.1 \
  --gamma 0.1 \
  --require-gpu \
  --out resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu10/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu10/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu10/pred.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu10/feature_importance_shap.csv \
  --fi-max-rows 200 \
  --shap-background-rows 50 \
  --benchmark-mode auto \
  --benchmark-max-rows 500 \
  --benchmark-out resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu10/benchmark.json

# Resumen comparativo
PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

runs = [
    ("BASE 1000 (auto-tuning normal_accept)", "resultsOptimo/csic_sin_registro/oneclass_test1000/metrics.json"),
    ("TEST nu=0.05, gamma=0.1",               "resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu05/metrics.json"),
    ("TEST nu=0.10, gamma=0.1",               "resultsOptimo/csic_sin_registro/test1000_nicoRalf_nu10/metrics.json"),
]

def get_m(path, key):
    p = Path(path)
    if not p.exists():
        return "N/A"
    with open(p) as f:
        d = json.load(f)
    for sub in ("eval", "metrics", "test"):
        if sub in d and isinstance(d[sub], dict) and key in d[sub]:
            return f"{d[sub][key]:.4f}"
    return f"{d.get(key, 'N/A')}" if not isinstance(d.get(key), dict) else "N/A"

print(f"\n{'Experimento':<42} {'F1':>8} {'Recall':>8} {'Precision':>10} {'FPR':>8}")
print("-" * 80)
for name, path in runs:
    f1 = get_m(path, "f1")
    rec = get_m(path, "recall")
    pre = get_m(path, "precision")
    fpr = get_m(path, "fpr")
    print(f"{name:<42} {f1:>8} {rec:>8} {pre:>10} {fpr:>8}")
print()
print("Referencia Nico/Ralf (OCS-WAF 2017): F1=0.95, Recall=0.93, FPR=0.03")
PY
