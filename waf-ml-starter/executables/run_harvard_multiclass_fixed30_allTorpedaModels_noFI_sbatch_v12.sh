#!/usr/bin/env bash
#SBATCH --job-name=harvard_mc30_allmodels
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --chdir=/home_data/aroman/TFG/waf-ml-starter
#SBATCH --output=/home_data/aroman/TFG/waf-ml-starter/harvard_mc30_allmodels_%j.out
#SBATCH --error=/home_data/aroman/TFG/waf-ml-starter/harvard_mc30_allmodels_%j.err

set -Eeuo pipefail

on_error() {
  local code=$?
  echo ""
  echo "[ERROR] Job falló con exit_code=${code}"
  echo "[ERROR] Línea: ${BASH_LINENO[0]:-unknown}"
  echo "[ERROR] Comando: ${BASH_COMMAND}"
  echo "[ERROR] Revisá:"
  echo "  sacct -j ${SLURM_JOB_ID:-JOBID} --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Reason%80"
  echo "  cat ${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}/harvard_mc30_allmodels_${SLURM_JOB_ID:-JOBID}.err"
  exit "${code}"
}
trap on_error ERR

DATASET="harvard"
PROJECT_DIR="${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
RAW_DIR="${RAW_DIR:-${HARVARD_RAW_DIR:-${PROJECT_DIR}/data/raw/harvard}}"
TRAIN_PY="${TRAIN_PY:-${PROJECT_DIR}/train_waf_multiclass_fixed30_resultsOptimo_sbatch_v12_noFI_allTorpedaModelsHarvard.py}"
LABEL_PREFIX="${LABEL_PREFIX:-HARVARD}"
PROCESSED_OUT="${PROCESSED_OUT:-${PROJECT_DIR}/data/processed/harvard/harvard_features.parquet}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/resultsOptimo/harvard/multiclass}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"

# Harvard optimizado: no usa HARVARD-ANOMALOUS; agrupa CAPECs cercanos en clases explícitas.
HARVARD_LABEL_MODE="${HARVARD_LABEL_MODE:-optimized-family}"
HARVARD_MULTICLASS_STRATEGY="${HARVARD_MULTICLASS_STRATEGY:-severity}"
HARVARD_SEP="${HARVARD_SEP:-auto}"
HARVARD_METHOD_COL="${HARVARD_METHOD_COL:-request_http_method}"
HARVARD_URI_COL="${HARVARD_URI_COL:-request_http_request}"
HARVARD_BODY_COL="${HARVARD_BODY_COL:-request_body}"
HARVARD_NORMAL_COL="${HARVARD_NORMAL_COL:-000 - Normal}"
HARVARD_LABEL_COLS="${HARVARD_LABEL_COLS:-}"

# Competencia Harvard multiclase: probar el menú TorpEda también y elegir el mejor, sin feature importance.
PRESET="${PRESET:-strong}"                    # fast | strong | max
USE_GPU="${USE_GPU:-auto}"                    # auto | on | off
SEED="${SEED:-42}"
TEST_SIZE="${TEST_SIZE:-0.20}"
VAL_SIZE="${VAL_SIZE:-0.15}"
METRIC_SELECT="${METRIC_SELECT:-f1_macro}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
SAMPLE_N="${SAMPLE_N:-0}"
MIN_CLASS_COUNT="${MIN_CLASS_COUNT:-20}"      # elimina clases no estratificables, e.g. 1 muestra
MAX_NORMAL_ROWS="${MAX_NORMAL_ROWS:-0}"        # 0 = no downsample; mantener NORMAL completo
CANDIDATE_WEIGHT_MODES="${CANDIDATE_WEIGHT_MODES:-none,sqrt_balanced,balanced_clipped}"
SAMPLE_WEIGHT_CLIP="${SAMPLE_WEIGHT_CLIP:-20}"
NORMAL_THRESHOLD_GRID="${NORMAL_THRESHOLD_GRID:-0.02:0.98:0.01}"
ADD_TWO_STAGE="${ADD_TWO_STAGE:-1}"
TWO_STAGE_ONLY="${TWO_STAGE_ONLY:-0}"
TWO_STAGE_BINARY_WEIGHT_MODE="${TWO_STAGE_BINARY_WEIGHT_MODE:-sqrt_balanced}"
TWO_STAGE_ATTACK_WEIGHT_MODE="${TWO_STAGE_ATTACK_WEIGHT_MODE:-balanced_clipped}"
SKIP_LOGREG="${SKIP_LOGREG:-0}"               # TorpEda probó LogReg; por eso aquí NO se saltea por defecto
SKIP_HISTGB="${SKIP_HISTGB:-0}"               # TorpEda ganó con HistGradientBoosting; NO se saltea
SKIP_EXTRA_TREES="${SKIP_EXTRA_TREES:-0}"         # TorpEda también probó ExtraTrees
SKIP_RANDOM_FOREST="${SKIP_RANDOM_FOREST:-1}"
XGB_CPU_FALLBACK="${XGB_CPU_FALLBACK:-1}"        # TorpEda comparó XGB GPU/CPU; aquí también
FORCE_REQUIRE_XGBOOST="${FORCE_REQUIRE_XGBOOST:-1}"
CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-0}"        # 0 permite reusar candidate_scores.csv previo y añadir los modelos faltantes
CHECK_ONLY="${CHECK_ONLY:-0}"
CHECKPOINT_CANDIDATES="${CHECKPOINT_CANDIDATES:-1}"
RESUME_CANDIDATES="${RESUME_CANDIDATES:-1}"
REQUIRE_CANDIDATE_FAMILIES="${REQUIRE_CANDIDATE_FAMILIES:-xgboost_d3,xgboost_d5,xgboost_d7,two_stage_xgboost_d5,two_stage_xgboost_d7,hist_gradient_boosting,extra_trees,logistic_regression}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

FI_KIND="${FI_KIND:-none}"                     # sin feature importance en esta corrida
FI_N_REPEATS="${FI_N_REPEATS:-0}"
FI_MAX_ROWS="${FI_MAX_ROWS:-0}"                 # sin FI; 0 = no muestrear para FI
FI_SCORING="${FI_SCORING:-${METRIC_SELECT}}"
FI_N_JOBS="${FI_N_JOBS:-1}"

export REQUIRE_XGBOOST="${FORCE_REQUIRE_XGBOOST}"

cd "${PROJECT_DIR}"
if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] No existe TRAIN_PY=${TRAIN_PY}" >&2
  exit 2
fi
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[ERROR] No existe ${VENV_DIR}/bin/activate" >&2
  exit 2
fi
if [[ ! -d "${RAW_DIR}" && ! -f "${RAW_DIR}" ]]; then
  echo "[ERROR] No existe RAW_DIR=${RAW_DIR}" >&2
  exit 2
fi

if [[ "${CLEAN_OUTPUT_DIR}" == "1" && "${CHECK_ONLY}" != "1" ]]; then
  case "${OUT_DIR}" in
    "${PROJECT_DIR}/resultsOptimo/harvard/multiclass"|"${PROJECT_DIR}/resultsOptimo/harvard/multiclass/"*)
      echo "[CLEAN] Borrando resultados anteriores en ${OUT_DIR}"
      rm -rf "${OUT_DIR}"
      ;;
    *)
      echo "[ERROR] CLEAN_OUTPUT_DIR=1 pero OUT_DIR no es la carpeta Harvard esperada: ${OUT_DIR}" >&2
      exit 2
      ;;
  esac
fi

mkdir -p "${LOG_DIR}" "$(dirname "${PROCESSED_OUT}")"
LOG_PREFIX="${LOG_DIR}/harvard_multiclass_allmodels_noFI_${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_PREFIX}.out") 2> >(tee -a "${LOG_PREFIX}.err" >&2)

echo "================================================================"
echo "[START] $(date '+%F %T')"
echo "[INFO] Host=$(hostname) DATASET=${DATASET} JOB=${SLURM_JOB_ID:-manual} PARTITION=${SLURM_JOB_PARTITION:-manual}"
echo "[INFO] GPU Slurm: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] Recursos pedidos: partition=normal gres/gpu=1 cpus=${SLURM_CPUS_PER_TASK:-8} mem=32G time=16:00:00"
echo "[INFO] Modo: probar menú TorpEda completo en Harvard, sin feature importance"
echo "[INFO] Skips: logreg=${SKIP_LOGREG} histgb=${SKIP_HISTGB} extra_trees=${SKIP_EXTRA_TREES} random_forest=${SKIP_RANDOM_FOREST}"
echo "[INFO] Candidates: XGB GPU/CPU + two-stage + HistGB + ExtraTrees + LogReg; weights=${CANDIDATE_WEIGHT_MODES}; threshold_grid=${NORMAL_THRESHOLD_GRID}"
echo "[INFO] Resume=${RESUME_CANDIDATES} checkpoint=${CHECKPOINT_CANDIDATES} require_families=${REQUIRE_CANDIDATE_FAMILIES}"
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

echo "[INFO] VIRTUAL_ENV=${VIRTUAL_ENV}"
echo "[INFO] command -v python=$(command -v python || true)"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
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
has_xgb = importlib.util.find_spec("xgboost") is not None
print(f"[INFO] Dependency check: parquet={has_parquet} xgboost={has_xgb}", flush=True)
if has_xgb:
    import xgboost
    print(f"[INFO] xgboost version={xgboost.__version__}", flush=True)
if missing:
    raise SystemExit("[ERROR] Faltan dependencias base: " + ", ".join(missing))
if not has_parquet:
    raise SystemExit("[ERROR] Falta pyarrow o fastparquet para reemplazar parquet procesado.")
if os.environ.get("REQUIRE_XGBOOST", "0") == "1" and not has_xgb:
    raise SystemExit("[ERROR] REQUIRE_XGBOOST=1 pero xgboost no está instalado en este .venv.")
PYCODE

"${PYTHON_BIN}" -u -m py_compile "${TRAIN_PY}"

INPUT_COUNT="$(find "${RAW_DIR}" -type f \( -iname '*.csv' -o -iname '*.tsv' -o -iname '*.csv.gz' -o -iname '*.tsv.gz' \) | wc -l | tr -d ' ')"
echo "[INFO] HARVARD_FILE_COUNT=${INPUT_COUNT} en ${RAW_DIR}"
if [[ "${INPUT_COUNT}" == "0" ]]; then
  echo "[ERROR] No encontré CSV/TSV/CSV.GZ/TSV.GZ en ${RAW_DIR}" >&2
  exit 2
fi

if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "${SLURM_JOB_ID}" | egrep "JobState|Partition|ReqTRES|AllocTRES|TresPerNode|NodeList|NumCPUs|MinMemory" || true
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "[WARN] nvidia-smi no está disponible; el .py detecta GPU también por variables Slurm/CUDA."
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[OK] CHECK_ONLY=1: entorno/rutas/dependencias OK. No entreno."
  exit 0
fi

ARGS=(
  --dataset "${DATASET}"
  --project-dir "${PROJECT_DIR}"
  --inputs "${RAW_DIR}"
  --processed-out "${PROCESSED_OUT}"
  --output-dir "${OUT_DIR}"
  --label-prefix "${LABEL_PREFIX}"
  --harvard-label-mode "${HARVARD_LABEL_MODE}"
  --harvard-multiclass-strategy "${HARVARD_MULTICLASS_STRATEGY}"
  --sep "${HARVARD_SEP}"
  --method-col "${HARVARD_METHOD_COL}"
  --uri-col "${HARVARD_URI_COL}"
  --body-col "${HARVARD_BODY_COL}"
  --normal-col "${HARVARD_NORMAL_COL}"
  --preset "${PRESET}"
  --use-gpu "${USE_GPU}"
  --test-size "${TEST_SIZE}"
  --val-size "${VAL_SIZE}"
  --seed "${SEED}"
  --metric-select "${METRIC_SELECT}"
  --n-jobs "${N_JOBS}"
  --min-class-count "${MIN_CLASS_COUNT}"
  --max-normal-rows "${MAX_NORMAL_ROWS}"
  --candidate-weight-modes "${CANDIDATE_WEIGHT_MODES}"
  --sample-weight-clip "${SAMPLE_WEIGHT_CLIP}"
  --normal-threshold-grid "${NORMAL_THRESHOLD_GRID}"
  --two-stage-binary-weight-mode "${TWO_STAGE_BINARY_WEIGHT_MODE}"
  --two-stage-attack-weight-mode "${TWO_STAGE_ATTACK_WEIGHT_MODE}"
  --fi-kind "${FI_KIND}"
  --fi-n-repeats "${FI_N_REPEATS}"
  --fi-max-rows "${FI_MAX_ROWS}"
  --fi-scoring "${FI_SCORING}"
  --fi-n-jobs "${FI_N_JOBS}"
  --skip-feature-importance
  --require-candidate-families "${REQUIRE_CANDIDATE_FAMILIES}"
)

if [[ "${CLEAN_OUTPUT_DIR}" == "1" ]]; then
  ARGS+=(--clean-output-dir)
fi
if [[ "${CHECKPOINT_CANDIDATES}" == "1" ]]; then
  ARGS+=(--checkpoint-candidate-scores)
fi
if [[ "${RESUME_CANDIDATES}" == "1" ]]; then
  ARGS+=(--resume-candidate-scores)
fi
if [[ -n "${HARVARD_LABEL_COLS}" ]]; then
  ARGS+=(--label-cols "${HARVARD_LABEL_COLS}")
fi
if [[ "${SAMPLE_N}" != "0" ]]; then
  ARGS+=(--sample-n "${SAMPLE_N}")
fi
if [[ "${XGB_CPU_FALLBACK}" == "1" ]]; then
  ARGS+=(--xgb-cpu-fallback)
fi
if [[ "${ADD_TWO_STAGE}" == "1" ]]; then
  ARGS+=(--add-two-stage)
fi
if [[ "${TWO_STAGE_ONLY}" == "1" ]]; then
  ARGS+=(--two-stage-only)
fi
if [[ "${SKIP_LOGREG}" == "1" ]]; then
  ARGS+=(--skip-logreg)
fi
if [[ "${SKIP_HISTGB}" == "1" ]]; then
  ARGS+=(--skip-histgb)
fi
if [[ "${SKIP_EXTRA_TREES}" == "1" ]]; then
  ARGS+=(--skip-extra-trees)
fi
if [[ "${SKIP_RANDOM_FOREST}" == "1" ]]; then
  ARGS+=(--skip-random-forest)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=( ${EXTRA_ARGS} )
  ARGS+=("${EXTRA_ARRAY[@]}")
fi

echo "[RUN] ${PYTHON_BIN} -u ${TRAIN_PY} ${ARGS[*]}"
"${PYTHON_BIN}" -u "${TRAIN_PY}" "${ARGS[@]}"

SUMMARY_TXT="${OUT_DIR}/selected_model_summary.txt"
SUMMARY_JSON="${OUT_DIR}/selected_model_summary.json"
COMPARE_JSON="${PROJECT_DIR}/resultsOptimo/multiclass_fixed30_model_comparison.json"
COMPARE_CSV="${PROJECT_DIR}/resultsOptimo/multiclass_fixed30_model_comparison.csv"

echo ""
echo "================================================================"
echo "[OK] $(date '+%F %T') Harvard all-models noFI finalizado. Resultados en ${OUT_DIR}"
if [[ -f "${SUMMARY_TXT}" ]]; then
  echo "[BEST] Resumen compacto:"
  cat "${SUMMARY_TXT}"
fi
echo "[ARTIFACTS]"
echo "  metrics: ${OUT_DIR}/metrics_multiclass_fixed30.json"
echo "  summary_json: ${SUMMARY_JSON}"
echo "  candidate_scores_ranked: ${OUT_DIR}/candidate_scores_ranked.csv
  candidate_manifest: ${OUT_DIR}/candidate_manifest.csv"
echo "  model: ${OUT_DIR}/model_multiclass_fixed30.joblib"
echo "  comparison_json: ${COMPARE_JSON}"
echo "  comparison_csv: ${COMPARE_CSV}"
echo "================================================================"
