#!/usr/bin/env bash
#SBATCH --job-name=multietq_c2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=c3
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=/home_data/aroman/TFG/waf-ml-starter/multietiquetaCorrida2/logs/slurm-%x-%j.out
#SBATCH --error=/home_data/aroman/TFG/waf-ml-starter/multietiquetaCorrida2/logs/slurm-%x-%j.err

# Harvard/SR-BH native multilabel: mismo modelo XGBoost OVR, 90 features
# request-only, tuning independiente por etiqueta y umbrales exactos.
#
# Este launcher interpreta "c3" como hostname Slurm. Si en tu clúster c3 es
# una PARTICIÓN, reemplazá la línea --nodelist=c3 por:
#   #SBATCH --partition=c3
#
# Uso normal:
#   cd /home_data/aroman/TFG/waf-ml-starter
#   mkdir -p multietiquetaCorrida2/logs
#   sbatch multietiquetaCorrida2.sh
#
# Prueba de entorno sin entrenar:
#   sbatch --export=ALL,CHECK_ONLY=1 multietiquetaCorrida2.sh
#
# Reanudar después de timeout: volver a ejecutar exactamente el mismo sbatch.

set -Eeuo pipefail
umask 027

on_error() {
  local code=$?
  echo "[ERROR] exit_code=${code} line=${BASH_LINENO[0]:-unknown} command=${BASH_COMMAND}" >&2
  if command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,Reason%60 || true
  fi
  exit "${code}"
}
trap on_error ERR

PROJECT_DIR="${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}"
TRAIN_PY="${TRAIN_PY:-${PROJECT_DIR}/multietiquetaCorrida2.py}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
HARVARD_RAW_DIR="${HARVARD_RAW_DIR:-${PROJECT_DIR}/data/raw/harvard}"

# Directorio exclusivo de la segunda corrida.
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/multietiquetaCorrida2}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/harvard}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
if (( N_JOBS > 8 )); then
  DEFAULT_FEATURE_WORKERS=8
else
  DEFAULT_FEATURE_WORKERS="${N_JOBS}"
fi
FEATURE_WORKERS="${FEATURE_WORKERS:-${DEFAULT_FEATURE_WORKERS}}"
FEATURE_CHUNKSIZE="${FEATURE_CHUNKSIZE:-256}"

SEED="${SEED:-42}"
TEST_SIZE="${TEST_SIZE:-0.20}"
VALID_SIZE="${VALID_SIZE:-0.20}"
CALIBRATION_SHARE="${CALIBRATION_SHARE:-0.50}"
MIN_CALIBRATION_POSITIVES="${MIN_CALIBRATION_POSITIVES:-3}"
LABEL_MODE="${LABEL_MODE:-native}"
MIN_POSITIVE_COUNT="${MIN_POSITIVE_COUNT:-20}"
DEDUPLICATE_REQUESTS="${DEDUPLICATE_REQUESTS:-1}"
GROUP_SPLIT_BY_REQUEST="${GROUP_SPLIT_BY_REQUEST:-1}"

# Tuning por etiqueta. CAPEC-153/272/274 reciben más presupuesto.
XGB_PRESET="${XGB_PRESET:-max}"
XGB_EARLY_STOPPING_ROUNDS="${XGB_EARLY_STOPPING_ROUNDS:-100}"
XGB_TUNING_TRIALS="${XGB_TUNING_TRIALS:-12}"
XGB_WEAK_LABEL_TRIALS="${XGB_WEAK_LABEL_TRIALS:-28}"
XGB_WEAK_LABEL_REGEX="${XGB_WEAK_LABEL_REGEX:-CAPEC-(153|272|274)\\b}"
XGB_LABEL_SELECTION_METRIC="${XGB_LABEL_SELECTION_METRIC:-blend}"
SCALE_POS_WEIGHT_CAP="${SCALE_POS_WEIGHT_CAP:-100}"

THRESHOLD_OBJECTIVE="${THRESHOLD_OBJECTIVE:-f1}"
THRESHOLD_MIN="${THRESHOLD_MIN:-0.001}"
THRESHOLD_MAX="${THRESHOLD_MAX:-0.999}"
THRESHOLD_MIN_RECALL="${THRESHOLD_MIN_RECALL:-0.0}"
MIN_ATTACK_SCORE="${MIN_ATTACK_SCORE:-0.0}"
JOINT_THRESHOLD_OPTIMIZATION="${JOINT_THRESHOLD_OPTIMIZATION:-1}"
TARGET_LABEL_F1="${TARGET_LABEL_F1:-0.85}"
TARGET_NORMAL="${TARGET_NORMAL:-0.99}"
TARGET_NORMAL_METRIC="${TARGET_NORMAL_METRIC:-recall}"
JOINT_THRESHOLD_MAX_PASSES="${JOINT_THRESHOLD_MAX_PASSES:-6}"

FI_N_REPEATS="${FI_N_REPEATS:-2}"
FI_MAX_ROWS="${FI_MAX_ROWS:-10000}"
OPERATIONAL_MAX_ROWS="${OPERATIONAL_MAX_ROWS:-10000}"
CHECKPOINT_COMPRESS="${CHECKPOINT_COMPRESS:-3}"
SAMPLE_N="${SAMPLE_N:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
RESUME="${RESUME:-1}"
FORCE_REPROCESS="${FORCE_REPROCESS:-0}"

[[ -d "${PROJECT_DIR}" ]] || { echo "[ERROR] No existe PROJECT_DIR=${PROJECT_DIR}" >&2; exit 2; }
[[ -f "${TRAIN_PY}" ]] || { echo "[ERROR] No existe TRAIN_PY=${TRAIN_PY}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] No existe Python ejecutable ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${HARVARD_RAW_DIR}" || -f "${HARVARD_RAW_DIR}" ]] || {
  echo "[ERROR] No existe HARVARD_RAW_DIR=${HARVARD_RAW_DIR}" >&2
  exit 2
}

mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}" "${LOG_DIR}"
cd "${PROJECT_DIR}"

LOG_PREFIX="${LOG_DIR}/multietiquetaCorrida2_${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_PREFIX}.out") 2> >(tee -a "${LOG_PREFIX}.err" >&2)

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED}"
export OMP_NUM_THREADS="${N_JOBS}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "================================================================"
echo "[START] $(date '+%F %T %z')"
echo "[HOST] $(hostname) job=${SLURM_JOB_ID:-manual} partition=${SLURM_JOB_PARTITION:-manual}"
echo "[PROJECT_DIR] ${PROJECT_DIR}"
echo "[TRAIN_PY] ${TRAIN_PY}"
echo "[HARVARD_RAW_DIR] ${HARVARD_RAW_DIR}"
echo "[OUTPUT_DIR] ${OUTPUT_DIR}"
echo "[CHECKPOINT_DIR] ${CHECKPOINT_DIR}"
echo "[CPU] n_jobs=${N_JOBS} feature_workers=${FEATURE_WORKERS}"
echo "[MODEL] XGBoost OVR; preset=${XGB_PRESET}; trials=${XGB_TUNING_TRIALS}; weak_trials=${XGB_WEAK_LABEL_TRIALS}"
echo "[FEATURES] request90 (30 base + 60 HTTP request-only)"
echo "[SPLIT] test=${TEST_SIZE}; holdout tune+cal=${VALID_SIZE}; calibration_share=${CALIBRATION_SHARE} (defaults efectivos 64/8/8/20)"
echo "[DEDUP] enabled=${DEDUPLICATE_REQUESTS} group_split=${GROUP_SPLIT_BY_REQUEST} fingerprint=canonical-request-only"
echo "[TARGETS] label_f1>=${TARGET_LABEL_F1}; NORMAL ${TARGET_NORMAL_METRIC}>=${TARGET_NORMAL}; joint=${JOINT_THRESHOLD_OPTIMIZATION}"
echo "================================================================"

if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "${SLURM_JOB_ID}" | grep -E 'JobState=|Partition=|NodeList=|NumCPUs=|MinMemory' || true
fi

TABLE_COUNT="$(find "${HARVARD_RAW_DIR}" -type f \( -iname '*.csv' -o -iname '*.tsv' -o -iname '*.csv.gz' -o -iname '*.tsv.gz' \) | wc -l | tr -d ' ')"
echo "[DATA] archivos tabulares detectados=${TABLE_COUNT}"
find "${HARVARD_RAW_DIR}" -type f \( -iname '*.csv' -o -iname '*.tsv' -o -iname '*.csv.gz' -o -iname '*.tsv.gz' \) -print | sort
if [[ "${TABLE_COUNT}" == "0" ]]; then
  echo "[ERROR] No hay CSV/TSV/GZ en ${HARVARD_RAW_DIR}" >&2
  exit 2
fi
if (( TABLE_COUNT > 1 )); then
  echo "[WARN] El cargador concatenará TODOS estos archivos. Verificá que no haya una copia duplicada del dataset ni resultados dentro de raw."
fi

"${PYTHON_BIN}" -u - <<'PYCODE'
import importlib.util
import sys

required = ["numpy", "pandas", "sklearn", "joblib", "tqdm", "xgboost"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("[ERROR] Dependencias faltantes: " + ", ".join(missing))
try:
    import xgboost
except Exception as exc:
    raise SystemExit(f"[ERROR] xgboost está instalado pero no se puede importar: {type(exc).__name__}: {exc}")
parquet = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
print(f"[PYTHON] executable={sys.executable} version={sys.version.split()[0]} xgboost={xgboost.__version__} parquet={parquet}", flush=True)
if sys.version_info < (3, 9):
    raise SystemExit("[ERROR] Se requiere Python >= 3.9")
if not parquet:
    print("[WARN] Instalá pyarrow: sin él el processed request90 caerá a un CSV muy grande y lento.", flush=True)
PYCODE

"${PYTHON_BIN}" -m py_compile "${TRAIN_PY}"

ARGS=(
  --project-dir "${PROJECT_DIR}"
  --definitivo-dir "${RUN_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --harvard-inputs "${HARVARD_RAW_DIR}"
  --sep auto
  --method-col request_http_method
  --uri-col request_http_request
  --protocol-col request_http_protocol
  --body-col request_body
  --normal-col "000 - Normal"
  --label-mode "${LABEL_MODE}"
  --min-positive-count "${MIN_POSITIVE_COUNT}"
  --drop-invalid-rows
  --test-size "${TEST_SIZE}"
  --valid-size "${VALID_SIZE}"
  --calibration-share "${CALIBRATION_SHARE}"
  --min-calibration-positives "${MIN_CALIBRATION_POSITIVES}"
  --seed "${SEED}"
  --selection-metric f1_macro
  --search-level none
  --candidate-weight-modes none
  --model-backend xgboost
  --use-gpu off
  --xgb-preset "${XGB_PRESET}"
  --xgb-early-stopping-rounds "${XGB_EARLY_STOPPING_ROUNDS}"
  --xgb-per-label-tuning
  --xgb-tuning-trials "${XGB_TUNING_TRIALS}"
  --xgb-weak-label-trials "${XGB_WEAK_LABEL_TRIALS}"
  --xgb-weak-label-regex "${XGB_WEAK_LABEL_REGEX}"
  --xgb-label-selection-metric "${XGB_LABEL_SELECTION_METRIC}"
  --scale-pos-weight-cap "${SCALE_POS_WEIGHT_CAP}"
  --n-jobs "${N_JOBS}"
  --threshold-objective "${THRESHOLD_OBJECTIVE}"
  --threshold-min "${THRESHOLD_MIN}"
  --threshold-max "${THRESHOLD_MAX}"
  --threshold-min-recall "${THRESHOLD_MIN_RECALL}"
  --target-label-f1 "${TARGET_LABEL_F1}"
  --target-normal "${TARGET_NORMAL}"
  --target-normal-metric "${TARGET_NORMAL_METRIC}"
  --joint-threshold-max-passes "${JOINT_THRESHOLD_MAX_PASSES}"
  --min-attack-score "${MIN_ATTACK_SCORE}"
  --feature-workers "${FEATURE_WORKERS}"
  --feature-chunksize "${FEATURE_CHUNKSIZE}"
  --fi-n-repeats "${FI_N_REPEATS}"
  --fi-max-rows "${FI_MAX_ROWS}"
  --operational-max-rows "${OPERATIONAL_MAX_ROWS}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --checkpoint-compress "${CHECKPOINT_COMPRESS}"
  --no-refit-full-after-thresholds
)

if [[ "${JOINT_THRESHOLD_OPTIMIZATION}" == "1" ]]; then
  ARGS+=(--joint-threshold-optimization)
else
  ARGS+=(--no-joint-threshold-optimization)
fi

if [[ "${DEDUPLICATE_REQUESTS}" == "1" ]]; then
  ARGS+=(--deduplicate-requests)
else
  ARGS+=(--no-deduplicate-requests)
fi
if [[ "${GROUP_SPLIT_BY_REQUEST}" == "1" ]]; then
  ARGS+=(--group-split-by-request)
else
  ARGS+=(--no-group-split-by-request)
fi

if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume --resume-skip-complete)
else
  ARGS+=(--no-resume --no-resume-skip-complete)
fi
if [[ "${FORCE_REPROCESS}" == "1" ]]; then
  ARGS+=(--force-reprocess)
fi
if [[ "${SAMPLE_N}" != "0" ]]; then
  ARGS+=(--sample-n "${SAMPLE_N}")
fi
if [[ "${CHECK_ONLY}" == "1" ]]; then
  ARGS+=(--check-only)
fi

echo "[RUN] ${PYTHON_BIN} -u ${TRAIN_PY} ${ARGS[*]}"
"${PYTHON_BIN}" -u "${TRAIN_PY}" "${ARGS[@]}"

echo "================================================================"
echo "[OK] $(date '+%F %T %z')"
echo "[METRICS] ${OUTPUT_DIR}/metrics_multilabel_xgb_request90.json"
echo "[MODEL] ${OUTPUT_DIR}/model_multilabel_xgb_request90.joblib"
echo "[PER_LABEL] ${OUTPUT_DIR}/per_label_metrics.csv"
echo "[TARGETS] ${OUTPUT_DIR}/target_attainment.csv"
echo "[TUNING] ${OUTPUT_DIR}/xgb_best_params_by_label.csv"
echo "[TRIALS] ${OUTPUT_DIR}/xgb_tuning_trials_by_label.csv"
echo "[TUNE] ${OUTPUT_DIR}/tuning_metrics_multilabel.json"
echo "[CALIBRATION] ${OUTPUT_DIR}/calibration_metrics_multilabel.json"
echo "[THRESHOLDS] ${OUTPUT_DIR}/thresholds_calibration.csv"
echo "[JOINT_REPORT] ${OUTPUT_DIR}/threshold_joint_optimization_calibration.json"
echo "================================================================"
