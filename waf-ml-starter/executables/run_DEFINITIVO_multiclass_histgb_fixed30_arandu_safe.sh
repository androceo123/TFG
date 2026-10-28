#!/usr/bin/env bash
#SBATCH --job-name=def_mc_histgb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home_data/aroman/TFG/waf-ml-starter
#SBATCH --output=/home_data/aroman/TFG/waf-ml-starter/DEFINITIVO_multiclass_histgb_%j.out
#SBATCH --error=/home_data/aroman/TFG/waf-ml-starter/DEFINITIVO_multiclass_histgb_%j.err

# Launcher compatible con Arandu: no fija --partition ni --gres en el archivo.
# Submit base:
#   sbatch run_DEFINITIVO_multiclass_histgb_fixed30_arandu_safe.sh
# Si tu cuenta requiere partición explícita:
#   sbatch -p normal run_DEFINITIVO_multiclass_histgb_fixed30_arandu_safe.sh
# Si Arandu sólo acepta la cola normal con GRES GPU, podés pedirla desde CLI,
# aunque HistGradientBoostingClassifier no la use:
#   sbatch -p normal --gres=gpu:1 run_DEFINITIVO_multiclass_histgb_fixed30_arandu_safe.sh

# Nota: no se pide --gres=gpu porque el modelo definitivo ganador es
# sklearn.ensemble.HistGradientBoostingClassifier, que no tiene backend CUDA.
# El script Python registra si hay GPU visible, pero este modelo se acelera con CPU/OpenMP.

set -Eeuo pipefail

on_error() {
  local code=$?
  echo ""
  echo "[ERROR] Job falló con exit_code=${code}"
  echo "[ERROR] Línea: ${BASH_LINENO[0]:-unknown}"
  echo "[ERROR] Comando: ${BASH_COMMAND}"
  echo "[ERROR] Revisá:"
  echo "  sacct -j ${SLURM_JOB_ID:-JOBID} --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Reason%80"
  echo "  cat ${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}/DEFINITIVO_multiclass_histgb_${SLURM_JOB_ID:-JOBID}.err"
  exit "${code}"
}
trap on_error ERR

PROJECT_DIR="${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
TRAIN_PY="${TRAIN_PY:-${PROJECT_DIR}/train_waf_multiclass_DEFINITIVO_histgb_fixed30.py}"
DEFINITIVO_DIR="${DEFINITIVO_DIR:-${PROJECT_DIR}/DEFINITIVO}"
RUN_DATASET="${RUN_DATASET:-both}"                 # both | torpeda | harvard

TORPEDA_RAW_DIR="${TORPEDA_RAW_DIR:-${PROJECT_DIR}/data/raw/torpeda}"
HARVARD_RAW_DIR="${HARVARD_RAW_DIR:-${PROJECT_DIR}/data/raw/harvard}"

# Preprocesamiento/labels: mismas 30 features para ambos; labels según dataset.
TORPEDA_ONLY_COMMON_LABELS="${TORPEDA_ONLY_COMMON_LABELS:-1}"
TORPEDA_MIN_CLASS_COUNT="${TORPEDA_MIN_CLASS_COUNT:-0}"
HARVARD_LABEL_MODE="${HARVARD_LABEL_MODE:-optimized-family}"
HARVARD_MULTICLASS_STRATEGY="${HARVARD_MULTICLASS_STRATEGY:-severity}"
HARVARD_MIN_CLASS_COUNT="${HARVARD_MIN_CLASS_COUNT:-20}"
HARVARD_SEP="${HARVARD_SEP:-auto}"
HARVARD_METHOD_COL="${HARVARD_METHOD_COL:-request_http_method}"
HARVARD_URI_COL="${HARVARD_URI_COL:-request_http_request}"
HARVARD_BODY_COL="${HARVARD_BODY_COL:-request_body}"
HARVARD_NORMAL_COL="${HARVARD_NORMAL_COL:-000 - Normal}"
HARVARD_LABEL_COLS="${HARVARD_LABEL_COLS:-}"
MAX_NORMAL_ROWS="${MAX_NORMAL_ROWS:-0}"
KEEP_LABELS_REGEX="${KEEP_LABELS_REGEX:-}"
DROP_LABELS_REGEX="${DROP_LABELS_REGEX:-}"

# Modelo definitivo.
HISTGB_PRESET="${HISTGB_PRESET:-strong}"           # fast | strong | max
SAMPLE_WEIGHT_MODE="${SAMPLE_WEIGHT_MODE:-}"       # vacío = torpeda:none, harvard:sqrt_balanced
SAMPLE_WEIGHT_CLIP="${SAMPLE_WEIGHT_CLIP:-20}"
TEST_SIZE="${TEST_SIZE:-0.20}"
SEED="${SEED:-42}"
USE_GPU="${USE_GPU:-auto}"                         # se registra; HistGB no usa CUDA

# Performance / feature importance.
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
FEATURE_WORKERS="${FEATURE_WORKERS:-${N_JOBS}}"
FEATURE_CHUNKSIZE="${FEATURE_CHUNKSIZE:-256}"
FI_N_REPEATS="${FI_N_REPEATS:-5}"
FI_MAX_ROWS="${FI_MAX_ROWS:-10000}"
FI_SCORING="${FI_SCORING:-f1_macro}"
FI_N_JOBS="${FI_N_JOBS:-1}"
OPERATIONAL_MAX_ROWS="${OPERATIONAL_MAX_ROWS:-0}"  # 0 = medir todo el test
SAMPLE_N="${SAMPLE_N:-0}"
CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${PROJECT_DIR}"

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] No existe TRAIN_PY=${TRAIN_PY}" >&2
  echo "Copiá train_waf_multiclass_DEFINITIVO_histgb_fixed30.py a PROJECT_DIR o exportá TRAIN_PY=/ruta/al/script.py" >&2
  exit 2
fi
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[ERROR] No existe ${VENV_DIR}/bin/activate" >&2
  exit 2
fi
case "${RUN_DATASET}" in
  both|torpeda|harvard) ;;
  *) echo "[ERROR] RUN_DATASET debe ser both, torpeda o harvard; recibido: ${RUN_DATASET}" >&2; exit 2 ;;
esac

mkdir -p "${DEFINITIVO_DIR}/multiclase/torpeda/logs" "${DEFINITIVO_DIR}/multiclase/harvard/logs"
LOG_DIR="${DEFINITIVO_DIR}/multiclase/logs"
mkdir -p "${LOG_DIR}"
LOG_PREFIX="${LOG_DIR}/definitivo_multiclass_histgb_${RUN_DATASET}_${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_PREFIX}.out") 2> >(tee -a "${LOG_PREFIX}.err" >&2)

source "${VENV_DIR}/bin/activate"
hash -r
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${N_JOBS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"

check_dir_for_dataset() {
  local ds="$1"
  local dir="$2"
  local count="0"
  if [[ ! -d "${dir}" && ! -f "${dir}" ]]; then
    echo "[ERROR] No existe RAW_DIR para ${ds}: ${dir}" >&2
    exit 2
  fi
  if [[ "${ds}" == "torpeda" ]]; then
    count="$(find "${dir}" -type f -iname '*.xml' | wc -l | tr -d ' ')"
    echo "[INFO] TORPEDA_XML_COUNT=${count} en ${dir}"
  else
    count="$(find "${dir}" -type f \( -iname '*.csv' -o -iname '*.tsv' -o -iname '*.csv.gz' -o -iname '*.tsv.gz' \) | wc -l | tr -d ' ')"
    echo "[INFO] HARVARD_TABLE_COUNT=${count} en ${dir}"
  fi
  if [[ "${count}" == "0" ]]; then
    echo "[ERROR] No encontré archivos de entrada para ${ds} en ${dir}" >&2
    exit 2
  fi
}

if [[ "${RUN_DATASET}" == "both" || "${RUN_DATASET}" == "torpeda" ]]; then
  check_dir_for_dataset "torpeda" "${TORPEDA_RAW_DIR}"
fi
if [[ "${RUN_DATASET}" == "both" || "${RUN_DATASET}" == "harvard" ]]; then
  check_dir_for_dataset "harvard" "${HARVARD_RAW_DIR}"
fi

cat <<EOF
================================================================
[START] $(date '+%F %T')
[INFO] Host=$(hostname) JOB=${SLURM_JOB_ID:-manual} PARTITION=${SLURM_JOB_PARTITION:-manual}
[INFO] RUN_DATASET=${RUN_DATASET}
[INFO] PROJECT_DIR=${PROJECT_DIR}
[INFO] TRAIN_PY=${TRAIN_PY}
[INFO] DEFINITIVO_DIR=${DEFINITIVO_DIR}
[INFO] Modelo fijo: HistGradientBoostingClassifier
[INFO] Weight mode: vacío => TorpEda none, Harvard sqrt_balanced; SAMPLE_WEIGHT_MODE='${SAMPLE_WEIGHT_MODE}'
[INFO] Features: fixed30 intra-request, iguales para ambos datasets
[INFO] GPU visible: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}; HistGB no usa CUDA
[INFO] CPU threads: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}
================================================================
EOF

"${PYTHON_BIN}" -u - <<'PYCODE'
import os, sys, importlib.util
print(f"[INFO] sys.executable={sys.executable}", flush=True)
print(f"[INFO] sys.prefix={sys.prefix}", flush=True)
expected = os.environ.get("VIRTUAL_ENV")
if expected and os.path.realpath(sys.prefix) != os.path.realpath(expected):
    raise SystemExit(f"[ERROR] Python no usa el .venv esperado: {sys.prefix} != {expected}")
required = ["numpy", "pandas", "sklearn", "joblib", "tqdm"]
missing = [p for p in required if importlib.util.find_spec(p) is None]
has_parquet = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
print(f"[INFO] Dependency check: parquet={has_parquet}", flush=True)
if missing:
    raise SystemExit("[ERROR] Faltan dependencias base: " + ", ".join(missing))
if not has_parquet:
    print("[WARN] Falta pyarrow/fastparquet: el .py guardará processed como CSV fallback.", flush=True)
PYCODE

"${PYTHON_BIN}" -u -m py_compile "${TRAIN_PY}"

if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "${SLURM_JOB_ID}" | egrep "JobState|Partition|ReqTRES|AllocTRES|TresPerNode|NodeList|NumCPUs|MinMemory" || true
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "[INFO] nvidia-smi no está disponible; esperado si no pedimos GPU."
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[OK] CHECK_ONLY=1: entorno/rutas/dependencias OK. No entreno."
  exit 0
fi

ARGS=(
  --dataset "${RUN_DATASET}"
  --project-dir "${PROJECT_DIR}"
  --definitivo-dir "${DEFINITIVO_DIR}"
  --torpeda-inputs "${TORPEDA_RAW_DIR}"
  --harvard-inputs "${HARVARD_RAW_DIR}"
  --harvard-label-mode "${HARVARD_LABEL_MODE}"
  --harvard-multiclass-strategy "${HARVARD_MULTICLASS_STRATEGY}"
  --harvard-min-class-count "${HARVARD_MIN_CLASS_COUNT}"
  --torpeda-min-class-count "${TORPEDA_MIN_CLASS_COUNT}"
  --sep "${HARVARD_SEP}"
  --method-col "${HARVARD_METHOD_COL}"
  --uri-col "${HARVARD_URI_COL}"
  --body-col "${HARVARD_BODY_COL}"
  --normal-col "${HARVARD_NORMAL_COL}"
  --max-normal-rows "${MAX_NORMAL_ROWS}"
  --keep-labels-regex "${KEEP_LABELS_REGEX}"
  --drop-labels-regex "${DROP_LABELS_REGEX}"
  --test-size "${TEST_SIZE}"
  --seed "${SEED}"
  --histgb-preset "${HISTGB_PRESET}"
  --sample-weight-clip "${SAMPLE_WEIGHT_CLIP}"
  --use-gpu "${USE_GPU}"
  --feature-workers "${FEATURE_WORKERS}"
  --feature-chunksize "${FEATURE_CHUNKSIZE}"
  --fi-n-repeats "${FI_N_REPEATS}"
  --fi-max-rows "${FI_MAX_ROWS}"
  --fi-scoring "${FI_SCORING}"
  --fi-n-jobs "${FI_N_JOBS}"
  --operational-max-rows "${OPERATIONAL_MAX_ROWS}"
)

if [[ "${TORPEDA_ONLY_COMMON_LABELS}" == "1" ]]; then
  ARGS+=(--torpeda-only-common-labels)
else
  ARGS+=(--no-torpeda-only-common-labels)
fi
if [[ -n "${HARVARD_LABEL_COLS}" ]]; then
  ARGS+=(--label-cols "${HARVARD_LABEL_COLS}")
fi
if [[ "${SAMPLE_N}" != "0" ]]; then
  ARGS+=(--sample-n "${SAMPLE_N}")
fi
if [[ -n "${SAMPLE_WEIGHT_MODE}" ]]; then
  ARGS+=(--sample-weight-mode "${SAMPLE_WEIGHT_MODE}")
fi
if [[ "${CLEAN_OUTPUT_DIR}" == "1" ]]; then
  ARGS+=(--clean-output-dir)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=( ${EXTRA_ARGS} )
  ARGS+=("${EXTRA_ARRAY[@]}")
fi

echo "[RUN] ${PYTHON_BIN} -u ${TRAIN_PY} ${ARGS[*]}"
"${PYTHON_BIN}" -u "${TRAIN_PY}" "${ARGS[@]}"

cat <<EOF

================================================================
[OK] $(date '+%F %T') DEFINITIVO multiclase finalizado.
[OUT_ROOT] ${DEFINITIVO_DIR}
[ARTIFACTS PRINCIPALES]
  comparison_csv: ${DEFINITIVO_DIR}/multiclase/model_comparison_multiclass_definitivo.csv
  comparison_json: ${DEFINITIVO_DIR}/multiclase/model_comparison_multiclass_definitivo.json
  manifest: ${DEFINITIVO_DIR}/manifest_definitivo.json
  torpeda_metrics: ${DEFINITIVO_DIR}/multiclase/torpeda/metrics_multiclass_histgb_fixed30.json
  harvard_metrics: ${DEFINITIVO_DIR}/multiclase/harvard/metrics_multiclass_histgb_fixed30.json
================================================================
EOF

find "${DEFINITIVO_DIR}" -maxdepth 3 -type f \( -name 'metrics_multiclass_histgb_fixed30.json' -o -name 'metrics_flat.csv' -o -name 'feature_importance.csv' -o -name 'operational_metrics.json' -o -name 'model_comparison_multiclass_definitivo.csv' -o -name 'selected_model_summary.txt' \) -print | sort
