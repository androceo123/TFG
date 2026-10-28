#!/bin/bash
#SBATCH --job-name=csic-oneclass-gpu
#SBATCH --output=slurm-csic-oneclass-gpu-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

# Uso recomendado:
#   cp csic_oneclass_gpu_optimo.py /home_data/aroman/TFG/waf-ml-starter/
#   cp run_csic_oneclass_gpu_optimo.sh /home_data/aroman/TFG/waf-ml-starter/
#   cd /home_data/aroman/TFG/waf-ml-starter
#   chmod +x run_csic_oneclass_gpu_optimo.sh
#   sbatch run_csic_oneclass_gpu_optimo.sh
#
# Si tu Slurm usa una particion GPU concreta, agrega/adapta arriba, por ejemplo:
#   #SBATCH --partition=gpu
#
# Variables utiles sin editar el archivo:
#   GPU_MODE=force sbatch run_csic_oneclass_gpu_optimo.sh
#   TUNE_TRIALS=30 FI_MAX_ROWS=25000 sbatch run_csic_oneclass_gpu_optimo.sh
#   ENABLE_AUTOENCODER=1 sbatch run_csic_oneclass_gpu_optimo.sh

PROJECT_DIR="${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}"
RAW_DIR="${RAW_DIR:-$PROJECT_DIR/data/raw/csic}"
PROCESSED_DIR="${PROCESSED_DIR:-$PROJECT_DIR/data/processed/csic_oneclass_gpu_optimo}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/resultsOptimo/csic/oneclass_gpu_optimo}"
PY_SCRIPT="${PY_SCRIPT:-$PROJECT_DIR/csic_oneclass_gpu_optimo.py}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

cd "$PROJECT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[ERROR] No encuentro el Python del venv en: $VENV_DIR/bin/python"
  echo "        Revisa VENV_DIR o crea/activa el .venv dentro de $PROJECT_DIR"
  exit 1
fi

# Activar el .venv ANTES de cualquier diagnostico/corrida.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PYTHON_EXE="$VENV_DIR/bin/python"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] No encuentro el script Python: $PY_SCRIPT"
  echo "        Copia csic_oneclass_gpu_optimo.py a $PROJECT_DIR/ o exporta PY_SCRIPT=/ruta/script.py"
  exit 1
fi

for f in \
  "$RAW_DIR/normalTrafficTraining.txt" \
  "$RAW_DIR/normalTrafficTest.txt" \
  "$RAW_DIR/anomalousTrafficTest.txt"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Falta archivo raw CSIC: $f"
    exit 1
  fi
done

mkdir -p "$RESULTS_DIR"

# Limitar threads internos para no pasarse de lo pedido por Slurm.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Diagnostico rapido de entorno.
echo "[RUN] PROJECT_DIR=$PROJECT_DIR"
echo "[RUN] RAW_DIR=$RAW_DIR"
echo "[RUN] PROCESSED_DIR=$PROCESSED_DIR"
echo "[RUN] RESULTS_DIR=$RESULTS_DIR"
echo "[RUN] PY_SCRIPT=$PY_SCRIPT"
echo "[RUN] PYTHON_EXE=$PYTHON_EXE"
echo "[RUN] CPUs=${SLURM_CPUS_PER_TASK:-8}"
echo "[RUN] GPU_MODE=${GPU_MODE:-auto}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU] nvidia-smi:"
  nvidia-smi || true
else
  echo "[GPU][WARN] nvidia-smi no esta en PATH. El .py intentara detectar cuML/PyTorch igualmente."
fi

"$PYTHON_EXE" - <<'PY'
import sys
print('[PYTHON] executable=', sys.executable, flush=True)
print('[PYTHON] version=', sys.version.replace('\n', ' '), flush=True)
required = ['numpy', 'pandas', 'sklearn', 'joblib']
missing = []
for mod in required:
    try:
        __import__(mod)
        print(f'[PYTHON] {mod}=OK', flush=True)
    except Exception as exc:
        missing.append(f'{mod} ({exc!r})')
try:
    import pyarrow  # noqa: F401
    print('[PYTHON] pyarrow=OK; se guardara parquet', flush=True)
except Exception as exc:
    print(f'[PYTHON][WARN] pyarrow no disponible ({exc!r}); fallback CSV', flush=True)
try:
    import cuml  # noqa: F401
    import cupy  # noqa: F401
    print('[GPU] RAPIDS cuml/cupy=OK; se puede usar IsolationForest GPU', flush=True)
except Exception as exc:
    print(f'[GPU][WARN] RAPIDS cuml/cupy no disponible: {exc!r}', flush=True)
try:
    import torch
    print(f'[GPU] torch={torch.__version__} cuda_available={torch.cuda.is_available()}', flush=True)
    if torch.cuda.is_available():
        print(f'[GPU] torch device={torch.cuda.get_device_name(0)}', flush=True)
except Exception as exc:
    print(f'[GPU][WARN] torch no disponible: {exc!r}', flush=True)
if missing:
    print('[ERROR] Faltan dependencias obligatorias:', '; '.join(missing), flush=True)
    print('Instala en el venv: pip install numpy pandas scikit-learn joblib pyarrow matplotlib', flush=True)
    raise SystemExit(1)
PY

EXTRA_ARGS=()
if [[ "${ENABLE_AUTOENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-autoencoder)
fi
if [[ "${COMPARE_CPU:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--compare-cpu)
fi
if [[ "${SKIP_FEATURE_IMPORTANCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-feature-importance)
fi
if [[ "${DEBUG:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--debug)
fi

# El flag --force-reprocess borra/recrea $PROCESSED_DIR, NO toca el raw.
"$PYTHON_EXE" "$PY_SCRIPT" \
  --raw-dir "$RAW_DIR" \
  --train-file "$RAW_DIR/normalTrafficTraining.txt" \
  --normal-test-file "$RAW_DIR/normalTrafficTest.txt" \
  --anomalous-test-file "$RAW_DIR/anomalousTrafficTest.txt" \
  --processed-dir "$PROCESSED_DIR" \
  --results-dir "$RESULTS_DIR" \
  --force-reprocess \
  --gpu "${GPU_MODE:-auto}" \
  --model "${MODEL:-auto}" \
  --seed "${SEED:-42}" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --tune-trials "${TUNE_TRIALS:-18}" \
  --tune-max-train-rows "${TUNE_MAX_TRAIN_ROWS:-80000}" \
  --final-max-train-rows "${FINAL_MAX_TRAIN_ROWS:-0}" \
  --n-estimators-grid "${N_ESTIMATORS_GRID:-128,256,512}" \
  --max-samples-grid "${MAX_SAMPLES_GRID:-256,512,1024,2048}" \
  --max-features-grid "${MAX_FEATURES_GRID:-0.7,0.9,1.0}" \
  --threshold-metric "${THRESHOLD_METRIC:-mcc}" \
  --final-test-size "${FINAL_TEST_SIZE:-0.50}" \
  --score-batch-size "${SCORE_BATCH_SIZE:-262144}" \
  --fi-max-rows "${FI_MAX_ROWS:-15000}" \
  --fi-metric "${FI_METRIC:-mcc}" \
  --permutation-repeats "${PERMUTATION_REPEATS:-2}" \
  --fi-top-k "${FI_TOP_K:-35}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "[DONE] Resultados en: $RESULTS_DIR"
echo "       summary.md"
echo "       metrics_test_holdout.json"
echo "       metrics_official_full.json"
echo "       search.csv"
echo "       feature_importance_permutation.csv"
echo "       feature_importance_top.png"
echo "       predictions_test_holdout.csv"
echo "       predictions_official_full.csv"
echo "       attack_family_report_official_full.csv"
echo "       model.joblib / model_meta.json"
