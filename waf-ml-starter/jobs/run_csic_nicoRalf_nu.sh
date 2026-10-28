#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-03:00:00
#SBATCH --job-name=ocsvm_nicoRalf_nu
#SBATCH --output=slurm-nicoRalf-nu-%j.out
#SBATCH --gres=gpu:1

# =============================================================================
# Experimento: Recalibracion de nu/gamma al rango de Nico & Ralf (OCS-WAF 2017)
#
# Problema identificado:
#   - El tuner con --tune-metric auto/normal_acceptance siempre prefiere nu=0.001
#     porque maximiza aceptacion de normales SIN ver ataques durante el tuning.
#   - Resultado: Recall=0.35, F1=0.52 (modelo demasiado conservador).
#
# Solucion:
#   - Nico/Ralf buscaron en nu=[0.0001, 0.1] y gamma=[0.0001, 0.1].
#   - Ejecutamos 3 variantes con nu fijo (sin auto-tuning): 0.05, 0.1 y 0.01
#     para ver cual da mejor balance TPR/FPR.
#   - Tambien un run con --tune-metric mean_score (menos sesgado).
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p resultsOptimo/csic_sin_registro/nicoRalf_nu05
mkdir -p resultsOptimo/csic_sin_registro/nicoRalf_nu10
mkdir -p resultsOptimo/csic_sin_registro/nicoRalf_nu01
mkdir -p resultsOptimo/csic_sin_registro/nicoRalf_tune_meanscore

# Verificar GPU
srun $PYTHON -c "import sys; print('Python:', sys.executable); import cuml.accel; cuml.accel.install(); print('GPU cuml.accel OK')"
nvidia-smi || true

DATA="data/processed/csic_sin_registro/csic_features_sin_registro.parquet"

# =============================================================================
# RUN 1: nu=0.05, gamma=0.1 (valores tipicos rango Nico/Ralf, sin auto-tuning)
# Significado: 5% de normales puede estar fuera del hiperplano (mas flexible)
# =============================================================================
echo ""
echo "======================================================"
echo " RUN 1: nu=0.05, gamma=0.1 (sin auto-tuning)"
echo "======================================================"
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
  --out resultsOptimo/csic_sin_registro/nicoRalf_nu05/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/nicoRalf_nu05/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/nicoRalf_nu05/pred.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_sin_registro/nicoRalf_nu05/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 0 \
  --benchmark-out resultsOptimo/csic_sin_registro/nicoRalf_nu05/benchmark.json

echo "[RUN 1 COMPLETO] Resultados en resultsOptimo/csic_sin_registro/nicoRalf_nu05/"

# =============================================================================
# RUN 2: nu=0.1, gamma=0.1 (valor maximo del rango de Nico/Ralf)
# Significado: 10% de normales fuera del hiperplano (mas agresivo, mas recall)
# =============================================================================
echo ""
echo "======================================================"
echo " RUN 2: nu=0.1, gamma=0.1 (sin auto-tuning)"
echo "======================================================"
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
  --out resultsOptimo/csic_sin_registro/nicoRalf_nu10/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/nicoRalf_nu10/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/nicoRalf_nu10/pred.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_sin_registro/nicoRalf_nu10/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 0 \
  --benchmark-out resultsOptimo/csic_sin_registro/nicoRalf_nu10/benchmark.json

echo "[RUN 2 COMPLETO] Resultados en resultsOptimo/csic_sin_registro/nicoRalf_nu10/"

# =============================================================================
# RUN 3: nu=0.01, gamma=0.1 (valor intermedio bajo)
# =============================================================================
echo ""
echo "======================================================"
echo " RUN 3: nu=0.01, gamma=0.1 (sin auto-tuning)"
echo "======================================================"
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
  --nu 0.01 \
  --gamma 0.1 \
  --require-gpu \
  --out resultsOptimo/csic_sin_registro/nicoRalf_nu01/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/nicoRalf_nu01/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/nicoRalf_nu01/pred.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_sin_registro/nicoRalf_nu01/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 0 \
  --benchmark-out resultsOptimo/csic_sin_registro/nicoRalf_nu01/benchmark.json

echo "[RUN 3 COMPLETO] Resultados en resultsOptimo/csic_sin_registro/nicoRalf_nu01/"

# =============================================================================
# RUN 4: Auto-tuning con mean_score (menos sesgado que normal_acceptance)
# Permite al tuner explorar valores mas altos de nu sin penalizarlos
# =============================================================================
echo ""
echo "======================================================"
echo " RUN 4: Auto-tuning con --tune-metric mean_score"
echo "======================================================"
PYTHONPATH=src srun $PYTHON src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data $DATA \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --final-train-on train_normals \
  --tune fast \
  --cv 3 \
  --tune-metric mean_score \
  --tune-n-iter 32 \
  --tune-n-jobs 4 \
  --tune-nu-grid "0.01,0.02,0.05,0.1" \
  --tune-gamma-grid "0.01,0.03,0.1,0.3" \
  --require-gpu \
  --out resultsOptimo/csic_sin_registro/nicoRalf_tune_meanscore/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/nicoRalf_tune_meanscore/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/nicoRalf_tune_meanscore/pred.csv \
  --tune-results-out resultsOptimo/csic_sin_registro/nicoRalf_tune_meanscore/search.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_sin_registro/nicoRalf_tune_meanscore/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 0 \
  --benchmark-out resultsOptimo/csic_sin_registro/nicoRalf_tune_meanscore/benchmark.json

echo "[RUN 4 COMPLETO] Resultados en resultsOptimo/csic_sin_registro/nicoRalf_tune_meanscore/"

# =============================================================================
# Resumen final de los 4 experimentos
# =============================================================================
echo ""
echo "======================================================"
echo " RESUMEN DE METRICAS - Comparacion de 4 variantes"
echo "======================================================"
PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

runs = [
    ("BASE (nu=0.001 auto-tuning)", "resultsOptimo/csic_sin_registro/oneclass/metrics.json"),
    ("RUN1: nu=0.05, gamma=0.1",    "resultsOptimo/csic_sin_registro/nicoRalf_nu05/metrics.json"),
    ("RUN2: nu=0.10, gamma=0.1",    "resultsOptimo/csic_sin_registro/nicoRalf_nu10/metrics.json"),
    ("RUN3: nu=0.01, gamma=0.1",    "resultsOptimo/csic_sin_registro/nicoRalf_nu01/metrics.json"),
    ("RUN4: mean_score auto-tune",  "resultsOptimo/csic_sin_registro/nicoRalf_tune_meanscore/metrics.json"),
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

print(f"{'Experimento':<35} {'F1':>8} {'Recall':>8} {'Precision':>10} {'FPR':>8} {'BAcc':>8}")
print("-" * 80)
for name, path in runs:
    f1 = get_m(path, "f1")
    rec = get_m(path, "recall")
    pre = get_m(path, "precision")
    fpr = get_m(path, "fpr")
    ba  = get_m(path, "balanced_accuracy")
    print(f"{name:<35} {f1:>8} {rec:>8} {pre:>10} {fpr:>8} {ba:>8}")
print()
print("Referencia Nico/Ralf (OCS-WAF 2017): F1=0.95, Recall=0.93, FPR=0.03")
PY
