#!/usr/bin/env bash
#SBATCH --job-name=torpeda_mc30_opt
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home_data/aroman/TFG/waf-ml-starter
#SBATCH --output=/home_data/aroman/TFG/waf-ml-starter/torpeda_mc30_opt_%j.out
#SBATCH --error=/home_data/aroman/TFG/waf-ml-starter/torpeda_mc30_opt_%j.err

set -Eeuo pipefail

on_error() {
  local code=$?
  echo ""
  echo "[ERROR] Job falló con exit_code=${code}"
  echo "[ERROR] Línea: ${BASH_LINENO[0]:-unknown}"
  echo "[ERROR] Comando: ${BASH_COMMAND}"
  echo "[ERROR] Revisá:"
  echo "  sacct -j ${SLURM_JOB_ID:-JOBID} --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Reason%80"
  echo "  cat ${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}/torpeda_mc30_opt_${SLURM_JOB_ID:-JOBID}.err"
  exit "${code}"
}
trap on_error ERR

DATASET="torpeda"
PROJECT_DIR="${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
RAW_DIR="${RAW_DIR:-${TORPEDA_RAW_DIR:-${PROJECT_DIR}/data/raw/torpeda}}"
TRAIN_PY="${TRAIN_PY:-${PROJECT_DIR}/train_waf_multiclass_fixed30_resultsOptimo_sbatch_v10.py}"
LABEL_PREFIX="${LABEL_PREFIX:-TORPEDA}"
PROCESSED_OUT="${PROCESSED_OUT:-${PROJECT_DIR}/data/processed/torpeda/torpeda_features.parquet}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/resultsOptimo/torpeda/multiclass}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"

ONLY_COMMON_LABELS="${ONLY_COMMON_LABELS:-1}"
KEEP_ABSOLUTE_URI="${KEEP_ABSOLUTE_URI:-0}"
PRESET="${PRESET:-strong}"
USE_GPU="${USE_GPU:-auto}"
SEED="${SEED:-42}"
TEST_SIZE="${TEST_SIZE:-0.20}"
VAL_SIZE="${VAL_SIZE:-0.15}"
METRIC_SELECT="${METRIC_SELECT:-f1_macro}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
SAMPLE_N="${SAMPLE_N:-0}"
MIN_CLASS_COUNT="${MIN_CLASS_COUNT:-0}"
CANDIDATE_WEIGHT_MODES="${CANDIDATE_WEIGHT_MODES:-balanced,sqrt_balanced,none}"
SAMPLE_WEIGHT_CLIP="${SAMPLE_WEIGHT_CLIP:-20}"
NORMAL_THRESHOLD_GRID="${NORMAL_THRESHOLD_GRID:-}"     # TorpEda ya iba bien; vacío = no postprocesa NORMAL
ADD_TWO_STAGE="${ADD_TWO_STAGE:-1}"
SKIP_LOGREG="${SKIP_LOGREG:-0}"
XGB_CPU_FALLBACK="${XGB_CPU_FALLBACK:-1}"
FORCE_REQUIRE_XGBOOST="${FORCE_REQUIRE_XGBOOST:-1}"
CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

FI_KIND="${FI_KIND:-both}"
FI_N_REPEATS="${FI_N_REPEATS:-5}"
FI_MAX_ROWS="${FI_MAX_ROWS:-10000}"
FI_SCORING="${FI_SCORING:-${METRIC_SELECT}}"
FI_N_JOBS="${FI_N_JOBS:-1}"

export REQUIRE_XGBOOST="${FORCE_REQUIRE_XGBOOST}"

cd "${PROJECT_DIR}"
if [[ ! -f "${TRAIN_PY}" ]]; then echo "[ERROR] No existe TRAIN_PY=${TRAIN_PY}" >&2; exit 2; fi
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then echo "[ERROR] No existe ${VENV_DIR}/bin/activate" >&2; exit 2; fi
if [[ ! -d "${RAW_DIR}" && ! -f "${RAW_DIR}" ]]; then echo "[ERROR] No existe RAW_DIR=${RAW_DIR}" >&2; exit 2; fi

if [[ "${CLEAN_OUTPUT_DIR}" == "1" && "${CHECK_ONLY}" != "1" ]]; then
  case "${OUT_DIR}" in
    "${PROJECT_DIR}/resultsOptimo/torpeda/multiclass"|"${PROJECT_DIR}/resultsOptimo/torpeda/multiclass/"*) rm -rf "${OUT_DIR}" ;;
    *) echo "[ERROR] CLEAN_OUTPUT_DIR=1 pero OUT_DIR no es la carpeta TorpEda esperada: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

mkdir -p "${LOG_DIR}" "$(dirname "${PROCESSED_OUT}")"
LOG_PREFIX="${LOG_DIR}/torpeda_multiclass_fixed30_${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_PREFIX}.out") 2> >(tee -a "${LOG_PREFIX}.err" >&2)

echo "================================================================"
echo "[START] $(date '+%F %T')"
echo "[INFO] Host=$(hostname) DATASET=${DATASET} JOB=${SLURM_JOB_ID:-manual} PARTITION=${SLURM_JOB_PARTITION:-manual}"
echo "[INFO] GPU Slurm: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================================================"

source "${VENV_DIR}/bin/activate"
hash -r
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-${N_JOBS}}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"

"${PYTHON_BIN}" -u - <<'PYCODE'
import os, sys, importlib.util
print(f"[INFO] sys.executable={sys.executable}", flush=True)
expected = os.environ.get("VIRTUAL_ENV")
if expected and os.path.realpath(sys.prefix) != os.path.realpath(expected):
    raise SystemExit(f"[ERROR] Python no usa el .venv esperado: {sys.prefix} != {expected}")
required = ["numpy", "pandas", "sklearn", "joblib", "tqdm"]
missing = [p for p in required if importlib.util.find_spec(p) is None]
has_parquet = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
has_xgb = importlib.util.find_spec("xgboost") is not None
print(f"[INFO] Dependency check: parquet={has_parquet} xgboost={has_xgb}", flush=True)
if has_xgb:
    import xgboost
    print(f"[INFO] xgboost version={xgboost.__version__}", flush=True)
if missing: raise SystemExit("[ERROR] Faltan dependencias base: " + ", ".join(missing))
if not has_parquet: raise SystemExit("[ERROR] Falta pyarrow o fastparquet para reemplazar parquet procesado.")
if os.environ.get("REQUIRE_XGBOOST", "0") == "1" and not has_xgb: raise SystemExit("[ERROR] REQUIRE_XGBOOST=1 pero xgboost no está instalado.")
PYCODE
"${PYTHON_BIN}" -u -m py_compile "${TRAIN_PY}"

INPUT_COUNT="$(find "${RAW_DIR}" -type f -iname '*.xml' | wc -l | tr -d ' ')"
echo "[INFO] XML_COUNT=${INPUT_COUNT} en ${RAW_DIR}"
if [[ "${INPUT_COUNT}" == "0" ]]; then echo "[ERROR] No encontré XML en ${RAW_DIR}" >&2; exit 2; fi
if [[ "${CHECK_ONLY}" == "1" ]]; then echo "[OK] CHECK_ONLY=1: entorno/rutas/dependencias OK. No entreno."; exit 0; fi

ARGS=(
  --dataset "${DATASET}"
  --project-dir "${PROJECT_DIR}"
  --inputs "${RAW_DIR}"
  --processed-out "${PROCESSED_OUT}"
  --output-dir "${OUT_DIR}"
  --label-prefix "${LABEL_PREFIX}"
  --preset "${PRESET}"
  --use-gpu "${USE_GPU}"
  --test-size "${TEST_SIZE}"
  --val-size "${VAL_SIZE}"
  --seed "${SEED}"
  --metric-select "${METRIC_SELECT}"
  --n-jobs "${N_JOBS}"
  --min-class-count "${MIN_CLASS_COUNT}"
  --candidate-weight-modes "${CANDIDATE_WEIGHT_MODES}"
  --sample-weight-clip "${SAMPLE_WEIGHT_CLIP}"
  --normal-threshold-grid "${NORMAL_THRESHOLD_GRID}"
  --fi-kind "${FI_KIND}"
  --fi-n-repeats "${FI_N_REPEATS}"
  --fi-max-rows "${FI_MAX_ROWS}"
  --fi-scoring "${FI_SCORING}"
  --fi-n-jobs "${FI_N_JOBS}"
  --clean-output-dir
)
if [[ "${ONLY_COMMON_LABELS}" == "1" ]]; then ARGS+=(--only-common-labels); fi
if [[ "${KEEP_ABSOLUTE_URI}" == "1" ]]; then ARGS+=(--keep-absolute-uri); fi
if [[ "${SAMPLE_N}" != "0" ]]; then ARGS+=(--sample-n "${SAMPLE_N}"); fi
if [[ "${XGB_CPU_FALLBACK}" == "1" ]]; then ARGS+=(--xgb-cpu-fallback); fi
if [[ "${ADD_TWO_STAGE}" == "1" ]]; then ARGS+=(--add-two-stage); fi
if [[ "${SKIP_LOGREG}" == "1" ]]; then ARGS+=(--skip-logreg); fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=( ${EXTRA_ARGS} )
  ARGS+=("${EXTRA_ARRAY[@]}")
fi

echo "[RUN] ${PYTHON_BIN} -u ${TRAIN_PY} ${ARGS[*]}"
"${PYTHON_BIN}" -u "${TRAIN_PY}" "${ARGS[@]}"

echo "[OK] $(date '+%F %T') TorpEda finalizado. Resultados en ${OUT_DIR}"
