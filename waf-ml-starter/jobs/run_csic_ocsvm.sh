#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ocsvm_csic_gpu
#SBATCH --output=slurm-csic-%j.out
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/waf-ml-starter" || exit 1

source .venv/bin/activate

mkdir -p resultsOptimo/csic/oneclass

python -c "import sys; print('Python:', sys.executable); import cuml.accel; cuml.accel.install(); print('GPU cuml.accel OK')"
nvidia-smi || true

PYTHONPATH=src srun .venv/bin/python src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data data/processed/csic/csic_features.parquet \
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
  --out resultsOptimo/csic/oneclass/ocsvm.joblib \
  --metrics-out resultsOptimo/csic/oneclass/metrics.json \
  --pred-out resultsOptimo/csic/oneclass/pred.csv \
  --tune-results-out resultsOptimo/csic/oneclass/search.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic/oneclass/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 0 \
  --benchmark-out resultsOptimo/csic/oneclass/benchmark.json
