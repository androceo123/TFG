#!/usr/bin/env bash
#SBATCH --job-name=3datasetsMC2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=c2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home_data/aroman/TFG/waf-ml-starter
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Launcher compatible con Arandu: no fija --partition ni --gres en el archivo.
# Submit base:
#   sbatch 3datasetsMulticlase2.sh
# Si tu cuenta requiere partición explícita:
#   sbatch -p normal 3datasetsMulticlase2.sh
# Si Arandu sólo acepta la cola normal con GRES GPU, podés pedirla desde CLI,
# aunque HistGradientBoostingClassifier no la use:
#   sbatch -p normal --gres=gpu:1 3datasetsMulticlase2.sh

# Nota: no se pide --gres=gpu porque el modelo fijado es
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
  echo "  cat ${LOG_PREFIX:-${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}/3datasetsMulticlase2/multiclase/logs/3datasetsMulticlase2_${RUN_DATASET:-all}_${SLURM_JOB_ID:-JOBID}}.err"
  exit "${code}"
}
trap on_error ERR

PROJECT_DIR="${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
TRAIN_PY="${TRAIN_PY:-${PROJECT_DIR}/3datasetsMulticlase2.py}"
RESULT_DIR="${RESULT_DIR:-${DEFINITIVO_DIR:-${PROJECT_DIR}/3datasetsMulticlase2}}"
DEFINITIVO_DIR="${RESULT_DIR}"  # compatibilidad con ejecuciones antiguas que exportaban DEFINITIVO_DIR
RUN_DATASET="${RUN_DATASET:-all}"                 # all | three | 3datasets | both | torpeda | harvard | egipcios

TORPEDA_RAW_DIR="${TORPEDA_RAW_DIR:-${PROJECT_DIR}/data/raw/torpeda}"
HARVARD_RAW_DIR="${HARVARD_RAW_DIR:-${PROJECT_DIR}/data/raw/harvard}"
EGIPCIOS_RAW_DIR="${EGIPCIOS_RAW_DIR:-${PROJECT_DIR}/egipcios/DS_Augmented_v2_csv/combined_data.csv}"

# Preprocesamiento/labels: 58 features validadas + 14 manuales nuevas; labels según dataset.
# Harvard queda en modo top10-request-groups por defecto: conserva el top 14,
# agrupa CAPEC-272/CAPEC-274/CAPEC-272+274 y CAPEC-153/CAPEC-194
# como HARVARD-RequestManipulation, y descarta el resto antes de extraer features.
# Egipcios lee CSV con col1=request HTTP completa y col2=label textual, como en la captura.
TORPEDA_ONLY_COMMON_LABELS="${TORPEDA_ONLY_COMMON_LABELS:-1}"
TORPEDA_MIN_CLASS_COUNT="${TORPEDA_MIN_CLASS_COUNT:-0}"
HARVARD_LABEL_MODE="${HARVARD_LABEL_MODE:-top10-request-groups}"
HARVARD_MULTICLASS_STRATEGY="${HARVARD_MULTICLASS_STRATEGY:-severity}"
HARVARD_MIN_CLASS_COUNT="${HARVARD_MIN_CLASS_COUNT:-20}"
HARVARD_SEP="${HARVARD_SEP:-auto}"
HARVARD_METHOD_COL="${HARVARD_METHOD_COL:-request_http_method}"
HARVARD_URI_COL="${HARVARD_URI_COL:-request_http_request}"
HARVARD_BODY_COL="${HARVARD_BODY_COL:-request_body}"
HARVARD_NORMAL_COL="${HARVARD_NORMAL_COL:-000 - Normal}"
HARVARD_LABEL_COLS="${HARVARD_LABEL_COLS:-}"
EGIPCIOS_SEP="${EGIPCIOS_SEP:-auto}"
EGIPCIOS_REQUEST_COL="${EGIPCIOS_REQUEST_COL:-col1}"
EGIPCIOS_LABEL_COL="${EGIPCIOS_LABEL_COL:-col2}"
EGIPCIOS_NORMAL_REGEX="${EGIPCIOS_NORMAL_REGEX:-^\s*0+\s*-\s*normal\s*$|\bnormal\b}"
EGIPCIOS_MIN_CLASS_COUNT="${EGIPCIOS_MIN_CLASS_COUNT:-2}"
MAX_NORMAL_ROWS="${MAX_NORMAL_ROWS:-0}"
KEEP_LABELS_REGEX="${KEEP_LABELS_REGEX:-}"
DROP_LABELS_REGEX="${DROP_LABELS_REGEX:-}"

# Modelo multiclase fijado con preset más expresivo por defecto.
HISTGB_PRESET="${HISTGB_PRESET:-max}"               # fast | strong | max
SAMPLE_WEIGHT_MODE="${SAMPLE_WEIGHT_MODE:-}"       # vacío = torpeda:none, harvard/egipcios:sqrt_balanced
SAMPLE_WEIGHT_CLIP="${SAMPLE_WEIGHT_CLIP:-20}"

# Curva solicitada por el tutor: por época/iteración de HistGradientBoosting.
# 0 en LOSS_CURVE_MAX_ROWS usa todo train/test; por defecto se muestrea para no duplicar horas de corrida.
LOSS_CURVE_ENABLED="${LOSS_CURVE_ENABLED:-1}"
LOSS_CURVE_MAX_ROWS="${LOSS_CURVE_MAX_ROWS:-50000}"
LOSS_CURVE_MIN_ROWS_PER_CLASS="${LOSS_CURVE_MIN_ROWS_PER_CLASS:-200}"
LOSS_CURVE_EVERY_N_ITER="${LOSS_CURVE_EVERY_N_ITER:-1}"

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
FI_MIN_ROWS_PER_CLASS="${FI_MIN_ROWS_PER_CLASS:-200}"
FI_PER_CLASS="${FI_PER_CLASS:-1}"
FI_PER_CLASS_N_REPEATS="${FI_PER_CLASS_N_REPEATS:-3}"
FI_PER_CLASS_N_JOBS="${FI_PER_CLASS_N_JOBS:-1}"
FI_BINARY_NORMAL_ATTACK="${FI_BINARY_NORMAL_ATTACK:-1}"
FI_BINARY_N_REPEATS="${FI_BINARY_N_REPEATS:-3}"
OPERATIONAL_MAX_ROWS="${OPERATIONAL_MAX_ROWS:-0}"  # 0 = medir todo el test
SAMPLE_N="${SAMPLE_N:-0}"
CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${PROJECT_DIR}"

mkdir -p "${RESULT_DIR}/multiclase/torpeda/logs" "${RESULT_DIR}/multiclase/harvard/logs" "${RESULT_DIR}/multiclase/egipcios/logs"
LOG_DIR="${RESULT_DIR}/multiclase/logs"
mkdir -p "${LOG_DIR}"
LOG_PREFIX="${LOG_DIR}/3datasetsMulticlase2_${RUN_DATASET}_${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_PREFIX}.out") 2> >(tee -a "${LOG_PREFIX}.err" >&2)

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] No existe TRAIN_PY=${TRAIN_PY}" >&2
  echo "Copiá 3datasetsMulticlase2.py a PROJECT_DIR o exportá TRAIN_PY=/ruta/al/3datasetsMulticlase2.py" >&2
  exit 2
fi
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[ERROR] No existe ${VENV_DIR}/bin/activate" >&2
  exit 2
fi
case "${RUN_DATASET}" in
  all|three|3datasets|both|torpeda|harvard|egipcios) ;;
  *) echo "[ERROR] RUN_DATASET debe ser all, three, 3datasets, both, torpeda, harvard o egipcios; recibido: ${RUN_DATASET}" >&2; exit 2 ;;
esac

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

should_run_dataset() {
  local ds="$1"
  case "${RUN_DATASET}" in
    all|three|3datasets) return 0 ;;
    both) [[ "${ds}" == "torpeda" || "${ds}" == "harvard" ]] ;;
    *) [[ "${RUN_DATASET}" == "${ds}" ]] ;;
  esac
}

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
    case "${ds}" in
      harvard) echo "[INFO] HARVARD_TABLE_COUNT=${count} en ${dir}" ;;
      egipcios) echo "[INFO] EGIPCIOS_TABLE_COUNT=${count} en ${dir}" ;;
      *) echo "[INFO] ${ds}_TABLE_COUNT=${count} en ${dir}" ;;
    esac
  fi
  if [[ "${count}" == "0" ]]; then
    echo "[ERROR] No encontré archivos de entrada para ${ds} en ${dir}" >&2
    exit 2
  fi
}

if should_run_dataset "torpeda"; then
  check_dir_for_dataset "torpeda" "${TORPEDA_RAW_DIR}"
fi
if should_run_dataset "harvard"; then
  check_dir_for_dataset "harvard" "${HARVARD_RAW_DIR}"
fi
if should_run_dataset "egipcios"; then
  check_dir_for_dataset "egipcios" "${EGIPCIOS_RAW_DIR}"
fi

cat <<EOF
================================================================
[START] $(date '+%F %T')
[INFO] Host=$(hostname) JOB=${SLURM_JOB_ID:-manual} PARTITION=${SLURM_JOB_PARTITION:-manual}
[INFO] RUN_DATASET=${RUN_DATASET}
[INFO] PROJECT_DIR=${PROJECT_DIR}
[INFO] TRAIN_PY=${TRAIN_PY}
[INFO] RESULT_DIR=${RESULT_DIR}
[INFO] TORPEDA_RAW_DIR=${TORPEDA_RAW_DIR}
[INFO] HARVARD_RAW_DIR=${HARVARD_RAW_DIR}
[INFO] EGIPCIOS_RAW_DIR=${EGIPCIOS_RAW_DIR}
[INFO] Modelo fijo: HistGradientBoostingClassifier
[INFO] Weight mode: vacío => TorpEda none, Harvard/Egipcios sqrt_balanced; SAMPLE_WEIGHT_MODE='${SAMPLE_WEIGHT_MODE}'
[INFO] Harvard label mode=${HARVARD_LABEL_MODE} (top10-request-groups => top14 + agrupación operativa en 10 clases finales)
[INFO] Egipcios CSV: sep=${EGIPCIOS_SEP}, request_col=${EGIPCIOS_REQUEST_COL}, label_col=${EGIPCIOS_LABEL_COL}, min_class_count=${EGIPCIOS_MIN_CLASS_COUNT}
[INFO] Features: 58 features validadas + 14 features manuales faltantes del paper (72 total), iguales para los 3 datasets
[INFO] FI global: repeats=${FI_N_REPEATS}, max_rows=${FI_MAX_ROWS}, scoring=${FI_SCORING}
[INFO] FI por clase: enabled=${FI_PER_CLASS}, repeats=${FI_PER_CLASS_N_REPEATS}, min_rows_per_class=${FI_MIN_ROWS_PER_CLASS}
[INFO] FI binaria NORMAL-vs-ataque: enabled=${FI_BINARY_NORMAL_ATTACK}, repeats=${FI_BINARY_N_REPEATS}
[INFO] Curva pérdida/error: enabled=${LOSS_CURVE_ENABLED}, max_rows=${LOSS_CURVE_MAX_ROWS}, min_rows_per_class=${LOSS_CURVE_MIN_ROWS_PER_CLASS}, every_n_iter=${LOSS_CURVE_EVERY_N_ITER}
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
required = ["numpy", "pandas", "sklearn", "joblib", "tqdm", "matplotlib"]
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
  --result-dir "${RESULT_DIR}"
  --torpeda-inputs "${TORPEDA_RAW_DIR}"
  --harvard-inputs "${HARVARD_RAW_DIR}"
  --egipcios-inputs "${EGIPCIOS_RAW_DIR}"
  --harvard-label-mode "${HARVARD_LABEL_MODE}"
  --harvard-multiclass-strategy "${HARVARD_MULTICLASS_STRATEGY}"
  --harvard-min-class-count "${HARVARD_MIN_CLASS_COUNT}"
  --torpeda-min-class-count "${TORPEDA_MIN_CLASS_COUNT}"
  --egipcios-min-class-count "${EGIPCIOS_MIN_CLASS_COUNT}"
  --sep "${HARVARD_SEP}"
  --method-col "${HARVARD_METHOD_COL}"
  --uri-col "${HARVARD_URI_COL}"
  --body-col "${HARVARD_BODY_COL}"
  --normal-col "${HARVARD_NORMAL_COL}"
  --egipcios-sep "${EGIPCIOS_SEP}"
  --egipcios-request-col "${EGIPCIOS_REQUEST_COL}"
  --egipcios-label-col "${EGIPCIOS_LABEL_COL}"
  --egipcios-normal-regex "${EGIPCIOS_NORMAL_REGEX}"
  --max-normal-rows "${MAX_NORMAL_ROWS}"
  --keep-labels-regex "${KEEP_LABELS_REGEX}"
  --drop-labels-regex "${DROP_LABELS_REGEX}"
  --test-size "${TEST_SIZE}"
  --seed "${SEED}"
  --histgb-preset "${HISTGB_PRESET}"
  --sample-weight-clip "${SAMPLE_WEIGHT_CLIP}"
  --loss-curve-max-rows "${LOSS_CURVE_MAX_ROWS}"
  --loss-curve-min-rows-per-class "${LOSS_CURVE_MIN_ROWS_PER_CLASS}"
  --loss-curve-every-n-iter "${LOSS_CURVE_EVERY_N_ITER}"
  --use-gpu "${USE_GPU}"
  --feature-workers "${FEATURE_WORKERS}"
  --feature-chunksize "${FEATURE_CHUNKSIZE}"
  --fi-n-repeats "${FI_N_REPEATS}"
  --fi-max-rows "${FI_MAX_ROWS}"
  --fi-scoring "${FI_SCORING}"
  --fi-n-jobs "${FI_N_JOBS}"
  --fi-min-rows-per-class "${FI_MIN_ROWS_PER_CLASS}"
  --fi-per-class-n-repeats "${FI_PER_CLASS_N_REPEATS}"
  --fi-per-class-n-jobs "${FI_PER_CLASS_N_JOBS}"
  --fi-binary-n-repeats "${FI_BINARY_N_REPEATS}"
  --operational-max-rows "${OPERATIONAL_MAX_ROWS}"
)

if [[ "${TORPEDA_ONLY_COMMON_LABELS}" == "1" ]]; then
  ARGS+=(--torpeda-only-common-labels)
else
  ARGS+=(--no-torpeda-only-common-labels)
fi
if [[ "${LOSS_CURVE_ENABLED}" == "1" ]]; then
  ARGS+=(--loss-curve)
else
  ARGS+=(--no-loss-curve)
fi
if [[ "${FI_PER_CLASS}" == "1" ]]; then
  ARGS+=(--fi-per-class)
else
  ARGS+=(--no-fi-per-class)
fi
if [[ "${FI_BINARY_NORMAL_ATTACK}" == "1" ]]; then
  ARGS+=(--fi-binary-normal-attack)
else
  ARGS+=(--no-fi-binary-normal-attack)
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
[OK] $(date '+%F %T') 3datasetsMulticlase2 multiclase finalizado.
[OUT_ROOT] ${RESULT_DIR}
[ARTIFACTS PRINCIPALES]
  comparison_csv: ${RESULT_DIR}/multiclase/model_comparison_multiclass_3datasetsMulticlase2.csv
  comparison_json: ${RESULT_DIR}/multiclase/model_comparison_multiclass_3datasetsMulticlase2.json
  manifest: ${RESULT_DIR}/manifest_3datasetsMulticlase2.json
  torpeda_metrics: ${RESULT_DIR}/multiclase/torpeda/metrics_multiclass_histgb_useful72_wamm_manual_request10.json
  harvard_metrics: ${RESULT_DIR}/multiclase/harvard/metrics_multiclass_histgb_useful72_wamm_manual_request10.json
  egipcios_metrics: ${RESULT_DIR}/multiclase/egipcios/metrics_multiclass_histgb_useful72_wamm_manual_request10.json
  torpeda_loss_curve_csv: ${RESULT_DIR}/multiclase/torpeda/loss_curve_histgb.csv
  torpeda_loss_curve_png: ${RESULT_DIR}/multiclase/torpeda/loss_curve_histgb.png
  harvard_loss_curve_csv: ${RESULT_DIR}/multiclase/harvard/loss_curve_histgb.csv
  harvard_loss_curve_png: ${RESULT_DIR}/multiclase/harvard/loss_curve_histgb.png
  egipcios_loss_curve_csv: ${RESULT_DIR}/multiclase/egipcios/loss_curve_histgb.csv
  egipcios_loss_curve_png: ${RESULT_DIR}/multiclase/egipcios/loss_curve_histgb.png
  torpeda_fi_per_class: ${RESULT_DIR}/multiclase/torpeda/feature_importance_per_class.csv
  harvard_fi_per_class: ${RESULT_DIR}/multiclase/harvard/feature_importance_per_class.csv
  egipcios_fi_per_class: ${RESULT_DIR}/multiclase/egipcios/feature_importance_per_class.csv
================================================================
EOF

find "${RESULT_DIR}" -maxdepth 3 -type f \( -name 'metrics_multiclass_histgb_useful72_wamm_manual_request10.json' -o -name 'metrics_flat.csv' -o -name 'feature_importance.csv' -o -name 'operational_metrics.json' -o -name 'model_comparison_multiclass_3datasetsMulticlase2.csv' -o -name 'feature_importance_per_class.csv' -o -name 'feature_importance_binary_normal_vs_attack.csv' -o -name 'selected_model_summary.txt' -o -name 'loss_curve_histgb.csv' -o -name 'loss_curve_by_epoch.csv' -o -name 'error_curve_mse_rmse_by_epoch.csv' -o -name 'loss_curve_histgb.png' -o -name 'loss_curve_log_loss.png' -o -name 'loss_curve_mse.png' -o -name 'loss_curve_rmse.png' -o -name 'prediction_error_curve_histgb.png' -o -name 'loss_curve_metadata.json' \) -print | sort
