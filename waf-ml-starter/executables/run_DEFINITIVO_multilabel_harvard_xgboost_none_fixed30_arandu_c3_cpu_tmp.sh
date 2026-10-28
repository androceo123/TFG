#!/usr/bin/env bash
#SBATCH --job-name=def_ml_xgb_gpu3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
# #SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/home_data/aroman/TFG/waf-ml-starter
#SBATCH --output=/home_data/aroman/TFG/waf-ml-starter/DEFINITIVO_multilabel_xgboost_gpu_nodo3_%j.out
#SBATCH --error=/home_data/aroman/TFG/waf-ml-starter/DEFINITIVO_multilabel_xgboost_gpu_nodo3_%j.err

# Pipeline DEFINITIVO multietiqueta Harvard/SR-BH con XGBoost/none fixed30, GPU y auditoría de confusión.
# Este .sh pide 1 GPU y nodo completo. La partición y el nodo se pueden fijar por comando sbatch.
# Submit recomendado para nodo 3:
#   sbatch -p normal -w nodo3 --gres=gpu:1 --exclusive --cpus-per-task=32 --mem=0 DEFINITIVO/multietiqueta/run_DEFINITIVO_multilabel_harvard_xgboost_none_fixed30_arandu_nodo3_gpu_metrics.sh
#
# Colocar este .sh y el .py compañero dentro de:
#   /home_data/aroman/TFG/waf-ml-starter/DEFINITIVO/multietiqueta/
#
# Salida principal:
#   /home_data/aroman/TFG/waf-ml-starter/DEFINITIVO/multietiqueta/multiquetaSalidaXG/

set -Eeuo pipefail

on_error() {
  local code=$?
  echo ""
  echo "[ERROR] Job falló con exit_code=${code}"
  echo "[ERROR] Línea: ${BASH_LINENO[0]:-unknown}"
  echo "[ERROR] Comando: ${BASH_COMMAND}"
  echo "[ERROR] Revisá estado Slurm con:"
  echo "  sacct -j ${SLURM_JOB_ID:-JOBID} --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS,ReqMem,Reason%80"
  echo "[ERROR] Log interno esperado: ${LOG_PREFIX:-unset}.err"
  exit "${code}"
}
trap on_error ERR

PROJECT_DIR="${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
DEFINITIVO_DIR="${DEFINITIVO_DIR:-${PROJECT_DIR}/DEFINITIVO}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFINITIVO_DIR}/multietiqueta/multiquetaSalidaXG}"
HARVARD_RAW_DIR="${HARVARD_RAW_DIR:-${PROJECT_DIR}/data/raw/harvard}"

TRAIN_PY="${TRAIN_PY:-${DEFINITIVO_DIR}/multietiqueta/train_waf_multilabel_DEFINITIVO_harvard_xgboost_none_fixed30_gpu_metrics.py}"
if [[ ! -f "${TRAIN_PY}" && -f "${PROJECT_DIR}/train_waf_multilabel_DEFINITIVO_harvard_xgboost_none_fixed30_gpu_metrics.py" ]]; then
  TRAIN_PY="${PROJECT_DIR}/train_waf_multilabel_DEFINITIVO_harvard_xgboost_none_fixed30_gpu_metrics.py"
fi

# Esquema Harvard/SR-BH.
HARVARD_SEP="${HARVARD_SEP:-auto}"
HARVARD_METHOD_COL="${HARVARD_METHOD_COL:-request_http_method}"
HARVARD_URI_COL="${HARVARD_URI_COL:-request_http_request}"
HARVARD_BODY_COL="${HARVARD_BODY_COL:-request_body}"
HARVARD_NORMAL_COL="${HARVARD_NORMAL_COL:-000 - Normal}"
HARVARD_LABEL_COLS="${HARVARD_LABEL_COLS:-}"

# Multietiqueta Harvard: native conserva etiquetas CAPEC.
LABEL_MODE="${LABEL_MODE:-native}"
MIN_POSITIVE_COUNT="${MIN_POSITIVE_COUNT:-20}"
MAX_NEGATIVE_POSITIVE_RATIO="${MAX_NEGATIVE_POSITIVE_RATIO:-0}"
MAX_NORMAL_ROWS="${MAX_NORMAL_ROWS:-0}"
KEEP_LABELS_REGEX="${KEEP_LABELS_REGEX:-}"
DROP_LABELS_REGEX="${DROP_LABELS_REGEX:-}"

# Modelo definitivo elegido: XGBoost/none.
MODEL_BACKEND="xgboost"
CANDIDATE_WEIGHT_MODES="none"
SEARCH_LEVEL="none"
SELECTION_METRIC="${SELECTION_METRIC:-f1_macro}"
USE_GPU="${USE_GPU:-on}"
REQUIRE_XGBOOST="${REQUIRE_XGBOOST:-1}"
XGB_PRESET="${XGB_PRESET:-max}"
XGB_EARLY_STOPPING_ROUNDS="${XGB_EARLY_STOPPING_ROUNDS:-80}"
SCALE_POS_WEIGHT_CAP="${SCALE_POS_WEIGHT_CAP:-100}"
REFIT_FULL_AFTER_THRESHOLDS="${REFIT_FULL_AFTER_THRESHOLDS:-1}"

# Split y umbrales.
TEST_SIZE="${TEST_SIZE:-0.20}"
VALID_SIZE="${VALID_SIZE:-0.20}"
SEED="${SEED:-42}"
THRESHOLD_OBJECTIVE="${THRESHOLD_OBJECTIVE:-f1}"
THRESHOLD_GRID_SIZE="${THRESHOLD_GRID_SIZE:-199}"
THRESHOLD_MIN="${THRESHOLD_MIN:-0.01}"
THRESHOLD_MAX="${THRESHOLD_MAX:-0.99}"
THRESHOLD_MIN_RECALL="${THRESHOLD_MIN_RECALL:-0.0}"
MIN_ATTACK_SCORE="${MIN_ATTACK_SCORE:-0.0}"

# Métricas no funcionales / feature importance.
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-32}}"
FEATURE_WORKERS="${FEATURE_WORKERS:-${N_JOBS}}"
FEATURE_CHUNKSIZE="${FEATURE_CHUNKSIZE:-512}"
FI_N_REPEATS="${FI_N_REPEATS:-3}"
FI_MAX_ROWS="${FI_MAX_ROWS:-10000}"
FI_SCORING="${FI_SCORING:-f1_macro}"
OPERATIONAL_MAX_ROWS="${OPERATIONAL_MAX_ROWS:-0}"   # 0 = medir todo el test
SAMPLE_N="${SAMPLE_N:-0}"

# Reproducibilidad/resume.
# FORCE_REPROCESS=1 hace pipeline completo desde dataset crudo la primera vez.
# Si un job cae y querés reanudar sin reprocesar features, relanzá con FORCE_REPROCESS=0.
RESUME="${RESUME:-1}"
RESUME_SKIP_COMPLETE="${RESUME_SKIP_COMPLETE:-1}"
FORCE_REPROCESS="${FORCE_REPROCESS:-1}"
CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
CHECKPOINT_COMPRESS="${CHECKPOINT_COMPRESS:-3}"
CHECK_ONLY="${CHECK_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${PROJECT_DIR}"
mkdir -p "${DEFINITIVO_DIR}/multietiqueta" "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs" "${DEFINITIVO_DIR}/multietiqueta/logs"

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] No existe TRAIN_PY=${TRAIN_PY}" >&2
  echo "Copiá train_waf_multilabel_DEFINITIVO_harvard_xgboost_none_fixed30_gpu_metrics.py en ${DEFINITIVO_DIR}/multietiqueta/" >&2
  exit 2
fi
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[ERROR] No existe ${VENV_DIR}/bin/activate" >&2
  exit 2
fi
if [[ ! -d "${HARVARD_RAW_DIR}" && ! -f "${HARVARD_RAW_DIR}" ]]; then
  echo "[ERROR] No existe HARVARD_RAW_DIR=${HARVARD_RAW_DIR}" >&2
  exit 2
fi

HARVARD_TABLE_COUNT="$(find "${HARVARD_RAW_DIR}" -type f \( -iname '*.csv' -o -iname '*.tsv' -o -iname '*.csv.gz' -o -iname '*.tsv.gz' \) | wc -l | tr -d ' ')"
echo "[INFO] HARVARD_TABLE_COUNT=${HARVARD_TABLE_COUNT} en ${HARVARD_RAW_DIR}"
if [[ "${HARVARD_TABLE_COUNT}" == "0" ]]; then
  echo "[ERROR] No encontré CSV/TSV/GZ de Harvard en ${HARVARD_RAW_DIR}" >&2
  exit 2
fi

LOG_PREFIX="${DEFINITIVO_DIR}/multietiqueta/logs/definitivo_multilabel_harvard_xgboost_gpu_nodo3_${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_PREFIX}.out") 2> >(tee -a "${LOG_PREFIX}.err" >&2)

source "${VENV_DIR}/bin/activate"
hash -r
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
ulimit -s unlimited || true
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${N_JOBS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"

cat <<EOF
================================================================
[START] $(date '+%F %T')
[INFO] Host=$(hostname) JOB=${SLURM_JOB_ID:-manual} PARTITION=${SLURM_JOB_PARTITION:-manual}
[INFO] PROJECT_DIR=${PROJECT_DIR}
[INFO] TRAIN_PY=${TRAIN_PY}
[INFO] DEFINITIVO_DIR=${DEFINITIVO_DIR}
[INFO] OUTPUT_DIR=${OUTPUT_DIR}
[INFO] Modelo fijo: XGBoost One-vs-Rest, weight_mode=none, preset=${XGB_PRESET}, GPU=${USE_GPU}
[INFO] LABEL_MODE=${LABEL_MODE} TEST_SIZE=${TEST_SIZE} VALID_SIZE=${VALID_SIZE} SEED=${SEED}
[INFO] USE_GPU=${USE_GPU}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}; SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}
[INFO] CPU threads: N_JOBS=${N_JOBS} OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}
[INFO] Métricas operativas: OPERATIONAL_MAX_ROWS=${OPERATIONAL_MAX_ROWS}; FI_N_REPEATS=${FI_N_REPEATS}; FI_MAX_ROWS=${FI_MAX_ROWS}
================================================================
EOF

"${PYTHON_BIN}" -u - <<'PYCODE'
import os, sys, importlib.util
print(f"[INFO] sys.executable={sys.executable}", flush=True)
print(f"[INFO] sys.prefix={sys.prefix}", flush=True)
expected = os.environ.get("VIRTUAL_ENV")
if expected and os.path.realpath(sys.prefix) != os.path.realpath(expected):
    raise SystemExit(f"[ERROR] Python no usa el .venv esperado: {sys.prefix} != {expected}")
required = ["numpy", "pandas", "sklearn", "joblib", "tqdm", "xgboost"]
missing = [p for p in required if importlib.util.find_spec(p) is None]
if missing:
    raise SystemExit("[ERROR] Faltan dependencias: " + ", ".join(missing))
parquet = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
print(f"[INFO] Dependency check: xgboost=True parquet={parquet}", flush=True)
if not parquet:
    print("[WARN] Falta pyarrow/fastparquet: processed se guardará como CSV fallback.", flush=True)
PYCODE

"${PYTHON_BIN}" -u -m py_compile "${TRAIN_PY}"

if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "${SLURM_JOB_ID}" | egrep "JobState|Partition|ReqTRES|AllocTRES|TresPerNode|NodeList|NumCPUs|MinMemory" || true
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "[WARN] nvidia-smi no está disponible; XGBoost usará CPU si no hay GPU visible."
fi

ARGS=(
  --project-dir "${PROJECT_DIR}"
  --definitivo-dir "${DEFINITIVO_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --harvard-inputs "${HARVARD_RAW_DIR}"
  --sep "${HARVARD_SEP}"
  --method-col "${HARVARD_METHOD_COL}"
  --uri-col "${HARVARD_URI_COL}"
  --body-col "${HARVARD_BODY_COL}"
  --normal-col "${HARVARD_NORMAL_COL}"
  --label-cols "${HARVARD_LABEL_COLS}"
  --label-mode "${LABEL_MODE}"
  --min-positive-count "${MIN_POSITIVE_COUNT}"
  --max-negative-positive-ratio "${MAX_NEGATIVE_POSITIVE_RATIO}"
  --max-normal-rows "${MAX_NORMAL_ROWS}"
  --keep-labels-regex "${KEEP_LABELS_REGEX}"
  --drop-labels-regex "${DROP_LABELS_REGEX}"
  --test-size "${TEST_SIZE}"
  --valid-size "${VALID_SIZE}"
  --seed "${SEED}"
  --selection-metric "${SELECTION_METRIC}"
  --search-level "${SEARCH_LEVEL}"
  --candidate-weight-modes "${CANDIDATE_WEIGHT_MODES}"
  --model-backend "${MODEL_BACKEND}"
  --use-gpu "${USE_GPU}"
  --xgb-preset "${XGB_PRESET}"
  --xgb-early-stopping-rounds "${XGB_EARLY_STOPPING_ROUNDS}"
  --scale-pos-weight-cap "${SCALE_POS_WEIGHT_CAP}"
  --threshold-objective "${THRESHOLD_OBJECTIVE}"
  --threshold-grid-size "${THRESHOLD_GRID_SIZE}"
  --threshold-min "${THRESHOLD_MIN}"
  --threshold-max "${THRESHOLD_MAX}"
  --threshold-min-recall "${THRESHOLD_MIN_RECALL}"
  --min-attack-score "${MIN_ATTACK_SCORE}"
  --n-jobs "${N_JOBS}"
  --feature-workers "${FEATURE_WORKERS}"
  --feature-chunksize "${FEATURE_CHUNKSIZE}"
  --operational-max-rows "${OPERATIONAL_MAX_ROWS}"
  --fi-n-repeats "${FI_N_REPEATS}"
  --fi-max-rows "${FI_MAX_ROWS}"
  --fi-scoring "${FI_SCORING}"
  --sample-n "${SAMPLE_N}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --checkpoint-compress "${CHECKPOINT_COMPRESS}"
)

if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi
if [[ "${RESUME_SKIP_COMPLETE}" == "1" ]]; then
  ARGS+=(--resume-skip-complete)
else
  ARGS+=(--no-resume-skip-complete)
fi
if [[ "${FORCE_REPROCESS}" == "1" ]]; then
  ARGS+=(--force-reprocess)
fi
if [[ "${CLEAN_OUTPUT_DIR}" == "1" ]]; then
  ARGS+=(--clean-output-dir)
fi
if [[ "${REFIT_FULL_AFTER_THRESHOLDS}" == "1" ]]; then
  ARGS+=(--refit-full-after-thresholds)
else
  ARGS+=(--no-refit-full-after-thresholds)
fi
if [[ "${CHECK_ONLY}" == "1" ]]; then
  ARGS+=(--check-only)
fi

# EXTRA_ARGS se deja para pruebas manuales. Ejemplo: EXTRA_ARGS='--operational-max-rows 10000'
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=( ${EXTRA_ARGS} )
  ARGS+=("${EXTRA_ARRAY[@]}")
fi

echo "[INFO] Ejecutando Python..."
"${PYTHON_BIN}" -u "${TRAIN_PY}" "${ARGS[@]}"

cat <<EOF
================================================================
[OK] $(date '+%F %T') DEFINITIVO multietiqueta Harvard XGBoost/none finalizado.
[OK] Resultados principales:
  ${OUTPUT_DIR}/metrics_multilabel_harvard_xgboost_none_fixed30.json
  ${OUTPUT_DIR}/metrics_flat.csv
  ${OUTPUT_DIR}/model_multilabel_harvard_xgboost_none_fixed30.joblib
  ${OUTPUT_DIR}/test_metrics_multilabel.json
  ${OUTPUT_DIR}/per_label_metrics.csv
  ${OUTPUT_DIR}/binary_normal_vs_attack_metrics.json
  ${OUTPUT_DIR}/thresholds_validation.csv
  ${OUTPUT_DIR}/test_predictions.csv
  ${OUTPUT_DIR}/request_level_multilabel_match.csv
  ${OUTPUT_DIR}/request_match_by_true_and_correct_count.csv
  ${OUTPUT_DIR}/request_match_summary.json
  ${OUTPUT_DIR}/confusion_matrix_by_label_wide.csv
  ${OUTPUT_DIR}/confusion_matrix_by_label_long.csv
  ${OUTPUT_DIR}/confusion_matrix_labelset.csv
  ${OUTPUT_DIR}/confusion_matrix_binary_normal_vs_attack.csv
  ${OUTPUT_DIR}/feature_importance.csv
  ${OUTPUT_DIR}/feature_importance_xgb_gain_aggregated.csv
  ${OUTPUT_DIR}/operational_metrics.json
  ${OUTPUT_DIR}/book_metric_mapping.csv
  ${DEFINITIVO_DIR}/multietiqueta/model_comparison_multilabel_definitivo.csv
  ${DEFINITIVO_DIR}/manifest_definitivo_multilabel.json
================================================================
EOF
