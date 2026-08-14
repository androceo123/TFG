#!/usr/bin/env bash
#SBATCH --job-name=ml_optimo_safe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home_data/aroman/TFG/waf-ml-starter
#SBATCH --output=/home_data/aroman/TFG/waf-ml-starter/resultsOptimo/slurm/DEFINITIVO_multilabel_harvard_multifamily_resume_%j.out
#SBATCH --error=/home_data/aroman/TFG/waf-ml-starter/resultsOptimo/slurm/DEFINITIVO_multilabel_harvard_multifamily_resume_%j.err

# Launcher DEFINITIVO multietiqueta Harvard/SR-BH multifamilia fixed30 resume-safe.
# Variante ARANDU-SAFE: no fija --partition ni --gres dentro del archivo.
# Guarda todos los resultados bajo /home_data/aroman/TFG/waf-ml-starter/resultsOptimo/.
# Nota Slurm: creá resultsOptimo/slurm antes de sbatch para que --output/--error puedan abrir los logs.
# Prueba varias familias One-vs-Rest, selecciona el mejor candidato en validación
# y reanuda tras timeout reutilizando processed + checkpoints por candidato.
#
# Submit seguro sin GPU fija:
#   sbatch run_DEFINITIVO_multilabel_harvard_multifamily_resume_resultsOptimo_arandu_safe_fixed30.sh
#
# Si tu cuenta requiere partición explícita:
#   sbatch -p normal run_DEFINITIVO_multilabel_harvard_multifamily_resume_resultsOptimo_arandu_safe_fixed30.sh
#
# Si sabés que la cola/GRES GPU disponible se llama gpu:1, pedila desde CLI:
#   sbatch -p normal --gres=gpu:1 run_DEFINITIVO_multilabel_harvard_multifamily_resume_resultsOptimo_arandu_safe_fixed30.sh
#
# Si querés probar recursos más altos, también hacelo desde CLI:
#   sbatch --cpus-per-task=12 --mem=64G --time=36:00:00 run_DEFINITIVO_multilabel_harvard_multifamily_resume_resultsOptimo_arandu_safe_fixed30.sh

set -Eeuo pipefail

on_error() {
  local code=$?
  echo ""
  echo "[ERROR] Job falló con exit_code=${code}"
  echo "[ERROR] Línea: ${BASH_LINENO[0]:-unknown}"
  echo "[ERROR] Comando: ${BASH_COMMAND}"
  echo "[ERROR] Revisá:"
  echo "  sacct -j ${SLURM_JOB_ID:-JOBID} --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Reason%80"
  local err_hint="${LOG_PREFIX:-${RESULTS_ROOT:-/home_data/aroman/TFG/waf-ml-starter/resultsOptimo}/multietiqueta/logs/definitivo_multilabel_harvard_multifamily_${SLURM_JOB_ID:-JOBID}}"
  echo "  cat ${err_hint}.err"
  exit "${code}"
}
trap on_error ERR

PROJECT_DIR="${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
TRAIN_PY="${TRAIN_PY:-${PROJECT_DIR}/train_waf_multilabel_DEFINITIVO_multifamily_resume_resultsOptimo_fixed30.py}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/resultsOptimo}"
DEFINITIVO_DIR="${RESULTS_ROOT}"
HARVARD_RAW_DIR="${HARVARD_RAW_DIR:-${PROJECT_DIR}/data/raw/harvard}"

# Harvard/SR-BH schema.
HARVARD_SEP="${HARVARD_SEP:-auto}"
HARVARD_METHOD_COL="${HARVARD_METHOD_COL:-request_http_method}"
HARVARD_URI_COL="${HARVARD_URI_COL:-request_http_request}"
HARVARD_BODY_COL="${HARVARD_BODY_COL:-request_body}"
HARVARD_NORMAL_COL="${HARVARD_NORMAL_COL:-000 - Normal}"
HARVARD_LABEL_COLS="${HARVARD_LABEL_COLS:-}"

# Multietiqueta: native conserva CAPEC Harvard; optimized-family agrupa como el multiclase.
LABEL_MODE="${LABEL_MODE:-native}"                    # native | optimized-family | family
MIN_POSITIVE_COUNT="${MIN_POSITIVE_COUNT:-20}"
MAX_NEGATIVE_POSITIVE_RATIO="${MAX_NEGATIVE_POSITIVE_RATIO:-0}"
MAX_NORMAL_ROWS="${MAX_NORMAL_ROWS:-0}"
KEEP_LABELS_REGEX="${KEEP_LABELS_REGEX:-}"
DROP_LABELS_REGEX="${DROP_LABELS_REGEX:-}"

# Búsqueda multifamilia.
# all/max incluye: XGBoost, LightGBM si está, CatBoost si está, HistGB, ExtraTrees,
# RandomForest, LogisticRegression, LinearSVC y SGD-logloss. LightGBM/CatBoost se omiten si no están instalados.
MODEL_BACKEND="${MODEL_BACKEND:-all}"                 # all | auto | compare | xgboost,histgb,extra_trees,...
REQUIRE_XGBOOST="${REQUIRE_XGBOOST:-1}"               # 1 = fallar si no está XGBoost; recomendado para usar GPU.
USE_GPU="${USE_GPU:-auto}"                            # auto | on | off
SEARCH_LEVEL="${SEARCH_LEVEL:-max}"                   # none | standard | max
CANDIDATE_WEIGHT_MODES="${CANDIDATE_WEIGHT_MODES:-none,sqrt,cuberoot,balanced}"
SELECTION_METRIC="${SELECTION_METRIC:-f1_macro}"
SCALE_POS_WEIGHT_CAP="${SCALE_POS_WEIGHT_CAP:-100}"
REFIT_FULL_AFTER_THRESHOLDS="${REFIT_FULL_AFTER_THRESHOLDS:-1}"

# Presets por familia. Para evitar un coste descomunal, bosques quedan en strong por defecto.
XGB_PRESET="${XGB_PRESET:-max}"                       # fast | strong | max
XGB_EARLY_STOPPING_ROUNDS="${XGB_EARLY_STOPPING_ROUNDS:-80}"
LGBM_PRESET="${LGBM_PRESET:-max}"
LGBM_EARLY_STOPPING_ROUNDS="${LGBM_EARLY_STOPPING_ROUNDS:-80}"
CATBOOST_PRESET="${CATBOOST_PRESET:-max}"
CATBOOST_EARLY_STOPPING_ROUNDS="${CATBOOST_EARLY_STOPPING_ROUNDS:-80}"
HISTGB_PRESET="${HISTGB_PRESET:-max}"
FOREST_PRESET="${FOREST_PRESET:-strong}"
LINEAR_PRESET="${LINEAR_PRESET:-max}"

# Split/umbrales.
TEST_SIZE="${TEST_SIZE:-0.20}"
VALID_SIZE="${VALID_SIZE:-0.20}"
SEED="${SEED:-42}"
THRESHOLD_OBJECTIVE="${THRESHOLD_OBJECTIVE:-f1}"
THRESHOLD_GRID_SIZE="${THRESHOLD_GRID_SIZE:-199}"
THRESHOLD_MIN="${THRESHOLD_MIN:-0.01}"
THRESHOLD_MAX="${THRESHOLD_MAX:-0.99}"
THRESHOLD_MIN_RECALL="${THRESHOLD_MIN_RECALL:-0.0}"
MIN_ATTACK_SCORE="${MIN_ATTACK_SCORE:-0.0}"

# Performance / feature importance / pruebas.
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
FEATURE_WORKERS="${FEATURE_WORKERS:-${N_JOBS}}"
FEATURE_CHUNKSIZE="${FEATURE_CHUNKSIZE:-256}"
FI_N_REPEATS="${FI_N_REPEATS:-3}"
FI_MAX_ROWS="${FI_MAX_ROWS:-10000}"
FI_SCORING="${FI_SCORING:-f1_macro}"
OPERATIONAL_MAX_ROWS="${OPERATIONAL_MAX_ROWS:-0}"
SAMPLE_N="${SAMPLE_N:-0}"
CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-0}"   # 0 por defecto para preservar checkpoints si Slurm corta por timeout.
RESUME="${RESUME:-1}"                     # 1 = reusa processed/checkpoints existentes; resume/checkpoint por defecto.
RESUME_SKIP_COMPLETE="${RESUME_SKIP_COMPLETE:-1}" # 1 = si ya terminó, no recomputa.
FORCE_REPROCESS="${FORCE_REPROCESS:-0}"   # 1 = ignora processed previo, pero mantiene checkpoints de candidatos.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFINITIVO_DIR}/multietiqueta/harvard/checkpoints}"
CHECKPOINT_COMPRESS="${CHECKPOINT_COMPRESS:-3}"
CHECK_ONLY="${CHECK_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${PROJECT_DIR}"
mkdir -p "${RESULTS_ROOT}" "${RESULTS_ROOT}/slurm"

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] No existe TRAIN_PY=${TRAIN_PY}" >&2
  echo "Copiá train_waf_multilabel_DEFINITIVO_multifamily_resume_resultsOptimo_fixed30.py a PROJECT_DIR o exportá TRAIN_PY=/ruta/al/script.py" >&2
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

mkdir -p "${DEFINITIVO_DIR}" "${DEFINITIVO_DIR}/multietiqueta/harvard/logs" "${DEFINITIVO_DIR}/multietiqueta/logs"
LOG_DIR="${DEFINITIVO_DIR}/multietiqueta/logs"
LOG_PREFIX="${LOG_DIR}/definitivo_multilabel_harvard_multifamily_${SLURM_JOB_ID:-manual}"
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
export REQUIRE_XGBOOST MODEL_BACKEND

cat <<EOF
================================================================
[START] $(date '+%F %T')
[INFO] Host=$(hostname) JOB=${SLURM_JOB_ID:-manual} PARTITION=${SLURM_JOB_PARTITION:-manual}
[INFO] PROJECT_DIR=${PROJECT_DIR}
[INFO] TRAIN_PY=${TRAIN_PY}
[INFO] RESULTS_ROOT=${RESULTS_ROOT}
[INFO] DEFINITIVO_DIR=${DEFINITIVO_DIR}
[INFO] HARVARD_RAW_DIR=${HARVARD_RAW_DIR}
[INFO] Task: multietiqueta Harvard/SR-BH multifamilia
[INFO] Features: fixed30 intra-request, mismas que multiclase
[INFO] Backend=${MODEL_BACKEND} SEARCH_LEVEL=${SEARCH_LEVEL} CANDIDATES=${CANDIDATE_WEIGHT_MODES} SELECTION_METRIC=${SELECTION_METRIC}
[INFO] Resume=${RESUME} ResumeSkipComplete=${RESUME_SKIP_COMPLETE} CLEAN_OUTPUT_DIR=${CLEAN_OUTPUT_DIR} CHECKPOINT_DIR=${CHECKPOINT_DIR}
[INFO] Presets: XGB=${XGB_PRESET} LGBM=${LGBM_PRESET} CATBOOST=${CATBOOST_PRESET} HISTGB=${HISTGB_PRESET} FOREST=${FOREST_PRESET} LINEAR=${LINEAR_PRESET}
[INFO] GPU request: no fijo en este .sh; si se pidió por CLI, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}
[INFO] CPU threads: N_JOBS=${N_JOBS} OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}
================================================================
EOF

"${PYTHON_BIN}" -u - <<PYCODE
import os, sys, importlib.util
print(f"[INFO] sys.executable={sys.executable}", flush=True)
print(f"[INFO] sys.prefix={sys.prefix}", flush=True)
expected = os.environ.get("VIRTUAL_ENV")
if expected and os.path.realpath(sys.prefix) != os.path.realpath(expected):
    raise SystemExit(f"[ERROR] Python no usa el .venv esperado: {sys.prefix} != {expected}")
required = ["numpy", "pandas", "sklearn", "joblib", "tqdm"]
missing = [p for p in required if importlib.util.find_spec(p) is None]
if missing:
    raise SystemExit("[ERROR] Faltan dependencias base: " + ", ".join(missing))
xgb = importlib.util.find_spec("xgboost") is not None
lgbm = importlib.util.find_spec("lightgbm") is not None
cat = importlib.util.find_spec("catboost") is not None
parquet = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
print(f"[INFO] Dependency check: xgboost={xgb} lightgbm={lgbm} catboost={cat} parquet={parquet}", flush=True)
require_xgb = os.environ.get("REQUIRE_XGBOOST", "1") == "1"
model_backend = os.environ.get("MODEL_BACKEND", "all").lower()
needs_xgb = model_backend in {"all", "auto", "compare", "full", "multifamily", "families"} or "xgb" in model_backend or "xgboost" in model_backend
if require_xgb and needs_xgb and not xgb:
    raise SystemExit("[ERROR] Falta xgboost en el .venv. Instalalo antes del sbatch o exportá REQUIRE_XGBOOST=0 para continuar sólo con familias CPU/optativas.")
if not lgbm:
    print("[INFO] LightGBM no está instalado; el script omitirá esa familia opcional.", flush=True)
if not cat:
    print("[INFO] CatBoost no está instalado; el script omitirá esa familia opcional.", flush=True)
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
  echo "[WARN] nvidia-smi no está disponible; si no hay GPU visible, los backends GPU usarán CPU/fallback."
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[OK] CHECK_ONLY=1: entorno/rutas/dependencias OK. No entreno."
  exit 0
fi

ARGS=(
  --project-dir "${PROJECT_DIR}"
  --definitivo-dir "${DEFINITIVO_DIR}"
  --harvard-inputs "${HARVARD_RAW_DIR}"
  --sep "${HARVARD_SEP}"
  --method-col "${HARVARD_METHOD_COL}"
  --uri-col "${HARVARD_URI_COL}"
  --body-col "${HARVARD_BODY_COL}"
  --normal-col "${HARVARD_NORMAL_COL}"
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
  --lgbm-preset "${LGBM_PRESET}"
  --lgbm-early-stopping-rounds "${LGBM_EARLY_STOPPING_ROUNDS}"
  --catboost-preset "${CATBOOST_PRESET}"
  --catboost-early-stopping-rounds "${CATBOOST_EARLY_STOPPING_ROUNDS}"
  --histgb-preset "${HISTGB_PRESET}"
  --forest-preset "${FOREST_PRESET}"
  --linear-preset "${LINEAR_PRESET}"
  --scale-pos-weight-cap "${SCALE_POS_WEIGHT_CAP}"
  --n-jobs "${N_JOBS}"
  --threshold-objective "${THRESHOLD_OBJECTIVE}"
  --threshold-grid-size "${THRESHOLD_GRID_SIZE}"
  --threshold-min "${THRESHOLD_MIN}"
  --threshold-max "${THRESHOLD_MAX}"
  --threshold-min-recall "${THRESHOLD_MIN_RECALL}"
  --min-attack-score "${MIN_ATTACK_SCORE}"
  --feature-workers "${FEATURE_WORKERS}"
  --feature-chunksize "${FEATURE_CHUNKSIZE}"
  --fi-n-repeats "${FI_N_REPEATS}"
  --fi-max-rows "${FI_MAX_ROWS}"
  --fi-scoring "${FI_SCORING}"
  --operational-max-rows "${OPERATIONAL_MAX_ROWS}"
  --checkpoint-compress "${CHECKPOINT_COMPRESS}"
)

if [[ -n "${HARVARD_LABEL_COLS}" ]]; then
  ARGS+=(--label-cols "${HARVARD_LABEL_COLS}")
fi
if [[ "${SAMPLE_N}" != "0" ]]; then
  ARGS+=(--sample-n "${SAMPLE_N}")
fi
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  ARGS+=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi
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
  if [[ "${RESUME}" == "1" ]]; then
    echo "[WARN] CLEAN_OUTPUT_DIR=1 con RESUME=1: el Python desactivará la limpieza para conservar checkpoints."
  else
    echo "[WARN] CLEAN_OUTPUT_DIR=1 y RESUME=0: se borrará el output y checkpoints previos."
  fi
  ARGS+=(--clean-output-dir)
fi
if [[ "${REFIT_FULL_AFTER_THRESHOLDS}" == "1" ]]; then
  ARGS+=(--refit-full-after-thresholds)
else
  ARGS+=(--no-refit-full-after-thresholds)
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
[OK] $(date '+%F %T') DEFINITIVO multietiqueta Harvard multifamilia finalizado.
[OUT_ROOT] ${DEFINITIVO_DIR}
[ARTIFACTS PRINCIPALES]
  metrics: ${DEFINITIVO_DIR}/multietiqueta/harvard/metrics_multilabel_multifamily_fixed30.json
  model: ${DEFINITIVO_DIR}/multietiqueta/harvard/model_multilabel_multifamily_fixed30.joblib
  candidates: ${DEFINITIVO_DIR}/multietiqueta/harvard/candidate_validation_summary.csv
  partial_candidates: ${DEFINITIVO_DIR}/multietiqueta/harvard/candidate_validation_summary.partial.csv
  checkpoints: ${CHECKPOINT_DIR}
  comparison_csv: ${DEFINITIVO_DIR}/multietiqueta/model_comparison_multilabel_definitivo.csv
  manifest: ${DEFINITIVO_DIR}/manifest_definitivo_multilabel.json
================================================================
EOF

find "${DEFINITIVO_DIR}/multietiqueta" -maxdepth 3 -type f \( \
  -name 'metrics_multilabel_multifamily_fixed30.json' -o \
  -name 'candidate_validation_summary.csv' -o \
  -name 'candidate_validation_summary.partial.csv' -o \
  -name 'metrics_flat.csv' -o \
  -name 'per_label_metrics.csv' -o \
  -name 'thresholds_validation.csv' -o \
  -name 'feature_importance.csv' -o \
  -name 'feature_importance_xgb_gain_aggregated.csv' -o \
  -name 'operational_metrics.json' -o \
  -name 'model_comparison_multilabel_definitivo.csv' -o \
  -name 'selected_model_summary.txt' \
\) -print | sort
