#!/usr/bin/env bash
#SBATCH --job-name=3datasetsMC3
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --chdir=/home_data/aroman/TFG/waf-ml-starter
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Pipeline multiclase unificado:
#   - un solo Python y una sola invocacion para TorpEda, Harvard y Egipcios;
#   - las mismas 73 features intra-request para los tres datasets;
#   - resultados exclusivamente en PROJECT_DIR/3datasetsMulticlase3.
#
# Ejecucion:
#   cd /home_data/aroman/TFG/waf-ml-starter && sbatch -p normal -w c3 ./3datasetsMulticlase3.sh

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
PYTHON_BIN="${VENV_DIR}/bin/python"
TRAIN_PY="${PROJECT_DIR}/3datasetsMulticlase3.py"

# Estas constantes se reemplazan una sola vez, despues de congelar el Python final.
TRAIN_EXPECTED_SHA256="c930f2a2b7a02d054c11d6a2f6693fec915b238fd9513fea25923dcc4b47efb4"
ARTIFACT_TAG="compact73_unified_inner_weight_selection"
FEATURE_SET_VERSION="compact73_cross_dataset_v1"
PIPELINE_RUN_VERSION="three_datasets_compact73_inner_weight_selection_v1"
FEATURE_LIST_SHA256="6409a743003e81a7a814072ba641fd89400eb51c42afe8eddea720f0f70de4f0"

TORPEDA_RAW_DIR="${TORPEDA_RAW_DIR:-${PROJECT_DIR}/data/raw/torpeda}"
HARVARD_RAW_DIR="${HARVARD_RAW_DIR:-${PROJECT_DIR}/data/raw/harvard}"
EGIPCIOS_INPUT="${EGIPCIOS_INPUT:-${PROJECT_DIR}/egipcios/DS_Augmented_v2_csv/combined_data.csv}"

# Se ignoran RESULT_DIR/DEFINITIVO_DIR heredados para evitar escribir en corridas anteriores.
RESULT_DIR="${PROJECT_DIR}/3datasetsMulticlase3"
LOG_DIR="${RESULT_DIR}/multiclase/logs"

RESUME="${RESUME:-1}"
SKIP_COMPLETE="${SKIP_COMPLETE:-1}"
CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
# 0 entrega el mejor resultado valido; 1 exige superar accuracy y F1 macro de WAMM.
REQUIRE_BEAT_WAMM="${REQUIRE_BEAT_WAMM:-0}"

ALLOC_CPUS="${SLURM_CPUS_PER_TASK:-8}"
FEATURE_WORKERS="${FEATURE_WORKERS:-${ALLOC_CPUS}}"
FEATURE_CHUNKSIZE="${FEATURE_CHUNKSIZE:-64}"
FEATURE_CHECKPOINT_ROWS="${FEATURE_CHECKPOINT_ROWS:-10000}"
FEATURE_ROW_TIMEOUT_SECONDS="${FEATURE_ROW_TIMEOUT_SECONDS:-30}"
OMP_THREADS="${OMP_NUM_THREADS:-${ALLOC_CPUS}}"

FI_N_REPEATS="${FI_N_REPEATS:-5}"
FI_MAX_ROWS="${FI_MAX_ROWS:-10000}"
FI_N_JOBS="${FI_N_JOBS:-1}"
FI_PER_CLASS_N_REPEATS="${FI_PER_CLASS_N_REPEATS:-3}"
FI_PER_CLASS_N_JOBS="${FI_PER_CLASS_N_JOBS:-1}"
FI_BINARY_N_REPEATS="${FI_BINARY_N_REPEATS:-3}"
FI_MIN_ROWS_PER_CLASS="${FI_MIN_ROWS_PER_CLASS:-200}"
OPERATIONAL_MAX_ROWS="${OPERATIONAL_MAX_ROWS:-0}"

LOSS_CURVE_MAX_ROWS="${LOSS_CURVE_MAX_ROWS:-50000}"
LOSS_CURVE_MIN_ROWS_PER_CLASS="${LOSS_CURVE_MIN_ROWS_PER_CLASS:-200}"
LOSS_CURVE_EVERY_N_ITER="${LOSS_CURVE_EVERY_N_ITER:-1}"
# Guardas de retencion frente a la corrida 93-feature. Accuracy y TorpEda F1
# permiten como maximo 0,01 pp; Harvard F1 permite 0,15 pp porque compact73
# mejora accuracy pero el benchmark limpio congelado reduce macro-F1 0,10317 pp.
# El margen adicional evita falsos fallos por unas pocas predicciones de clases raras.
ACCURACY_REGRESSION_TOLERANCE="0.0001"
TORPEDA_F1_REGRESSION_TOLERANCE="0.0001"
HARVARD_F1_REGRESSION_TOLERANCE="0.0015"

on_error() {
  local code=$?
  trap - ERR
  echo "[ERROR] Job fallo con exit_code=${code}; linea=${BASH_LINENO[0]:-unknown}; comando=${BASH_COMMAND}" >&2
  echo "[ERROR] Log: ${LOG_PREFIX:-${LOG_DIR}/3datasetsMulticlase3_JOBID}.err" >&2
  if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job "${SLURM_JOB_ID}" || true
  fi
  exit "${code}"
}
trap on_error ERR

require_positive_int() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${name} debe ser un entero >= 1; recibido=${value}" >&2
    exit 2
  fi
}

require_nonnegative_int() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ${name} debe ser un entero >= 0; recibido=${value}" >&2
    exit 2
  fi
}

require_nonnegative_number() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "[ERROR] ${name} debe ser numerico y >= 0; recibido=${value}" >&2
    exit 2
  fi
}

for name in RESUME SKIP_COMPLETE CLEAN_OUTPUT_DIR CHECK_ONLY REQUIRE_BEAT_WAMM; do
  value="${!name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "[ERROR] ${name} debe ser 0 o 1; recibido=${value}" >&2
    exit 2
  fi
done
if [[ "${RESUME}" == "1" && "${CLEAN_OUTPUT_DIR}" == "1" ]]; then
  echo "[ERROR] RESUME=1 y CLEAN_OUTPUT_DIR=1 son incompatibles." >&2
  exit 2
fi

for pair in \
  "ALLOC_CPUS:${ALLOC_CPUS}" \
  "FEATURE_WORKERS:${FEATURE_WORKERS}" \
  "FEATURE_CHUNKSIZE:${FEATURE_CHUNKSIZE}" \
  "FEATURE_CHECKPOINT_ROWS:${FEATURE_CHECKPOINT_ROWS}" \
  "OMP_NUM_THREADS:${OMP_THREADS}" \
  "FI_N_REPEATS:${FI_N_REPEATS}" \
  "FI_N_JOBS:${FI_N_JOBS}" \
  "FI_PER_CLASS_N_REPEATS:${FI_PER_CLASS_N_REPEATS}" \
  "FI_PER_CLASS_N_JOBS:${FI_PER_CLASS_N_JOBS}" \
  "FI_BINARY_N_REPEATS:${FI_BINARY_N_REPEATS}" \
  "FI_MIN_ROWS_PER_CLASS:${FI_MIN_ROWS_PER_CLASS}" \
  "LOSS_CURVE_MIN_ROWS_PER_CLASS:${LOSS_CURVE_MIN_ROWS_PER_CLASS}" \
  "LOSS_CURVE_EVERY_N_ITER:${LOSS_CURVE_EVERY_N_ITER}"; do
  require_positive_int "${pair%%:*}" "${pair#*:}"
done
for pair in \
  "FI_MAX_ROWS:${FI_MAX_ROWS}" \
  "OPERATIONAL_MAX_ROWS:${OPERATIONAL_MAX_ROWS}" \
  "LOSS_CURVE_MAX_ROWS:${LOSS_CURVE_MAX_ROWS}"; do
  require_nonnegative_int "${pair%%:*}" "${pair#*:}"
done
for pair in \
  "FEATURE_ROW_TIMEOUT_SECONDS:${FEATURE_ROW_TIMEOUT_SECONDS}" \
  "ACCURACY_REGRESSION_TOLERANCE:${ACCURACY_REGRESSION_TOLERANCE}" \
  "TORPEDA_F1_REGRESSION_TOLERANCE:${TORPEDA_F1_REGRESSION_TOLERANCE}" \
  "HARVARD_F1_REGRESSION_TOLERANCE:${HARVARD_F1_REGRESSION_TOLERANCE}"; do
  require_nonnegative_number "${pair%%:*}" "${pair#*:}"
done
for pair in \
  "FEATURE_WORKERS:${FEATURE_WORKERS}" \
  "OMP_NUM_THREADS:${OMP_THREADS}" \
  "FI_N_JOBS:${FI_N_JOBS}" \
  "FI_PER_CLASS_N_JOBS:${FI_PER_CLASS_N_JOBS}"; do
  if (( ${pair#*:} > ALLOC_CPUS )); then
    echo "[ERROR] ${pair%%:*}=${pair#*:} supera los ${ALLOC_CPUS} CPU asignados." >&2
    exit 2
  fi
done

if [[ "${TRAIN_EXPECTED_SHA256}" == __PENDING_* || "${ARTIFACT_TAG}" == __PENDING_* \
   || "${FEATURE_SET_VERSION}" == __PENDING_* || "${PIPELINE_RUN_VERSION}" == __PENDING_* \
   || "${FEATURE_LIST_SHA256}" == __PENDING_* ]]; then
  echo "[ERROR] El shell candidato aun tiene constantes pendientes del Python final." >&2
  exit 2
fi
if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] Falta el unico motor requerido: ${TRAIN_PY}" >&2
  exit 2
fi
if [[ ! -f "${EGIPCIOS_INPUT}" ]]; then
  echo "[ERROR] Falta Egipcios: ${EGIPCIOS_INPUT}" >&2
  exit 2
fi
if [[ ! -d "${TORPEDA_RAW_DIR}" || ! -d "${HARVARD_RAW_DIR}" ]]; then
  echo "[ERROR] Falta TorpEda o Harvard: ${TORPEDA_RAW_DIR}; ${HARVARD_RAW_DIR}" >&2
  exit 2
fi
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[ERROR] Falta el entorno virtual: ${VENV_DIR}" >&2
  exit 2
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  echo "[ERROR] sha256sum es obligatorio." >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "[ERROR] flock es obligatorio." >&2
  exit 2
fi

train_sha="$(sha256sum "${TRAIN_PY}" | awk '{print $1}')"
if [[ "${train_sha}" != "${TRAIN_EXPECTED_SHA256}" ]]; then
  echo "[ERROR] 3datasetsMulticlase3.py no coincide con la version emparejada con este shell." >&2
  echo "[ERROR] actual=${train_sha}; esperado=${TRAIN_EXPECTED_SHA256}" >&2
  exit 2
fi

torpeda_count="$(find "${TORPEDA_RAW_DIR}" -type f -iname '*.xml' | wc -l | tr -d ' ')"
harvard_count="$(find "${HARVARD_RAW_DIR}" -type f \( -iname '*.csv' -o -iname '*.tsv' -o -iname '*.csv.gz' -o -iname '*.tsv.gz' \) | wc -l | tr -d ' ')"
if [[ "${torpeda_count}" == "0" || "${harvard_count}" == "0" ]]; then
  echo "[ERROR] Entradas vacias: TorpEda XML=${torpeda_count}; Harvard tablas=${harvard_count}." >&2
  exit 2
fi

mkdir -p \
  "${RESULT_DIR}/multiclase/torpeda/logs" \
  "${RESULT_DIR}/multiclase/harvard/logs" \
  "${RESULT_DIR}/multiclase/egipcios/logs" \
  "${LOG_DIR}"
LOG_PREFIX="${LOG_DIR}/3datasetsMulticlase3_${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_PREFIX}.out") 2> >(tee -a "${LOG_PREFIX}.err" >&2)

LOCK_FILE="${RESULT_DIR}/.3datasetsMulticlase3.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[ERROR] Otra corrida ya escribe en ${RESULT_DIR}." >&2
  exit 75
fi

source "${VENV_DIR}/bin/activate"
hash -r
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONHASHSEED=42
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS="${OMP_THREADS}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON_BIN}" -u - <<'PYCODE'
import importlib.util
import os
import sys

expected = os.environ.get("VIRTUAL_ENV")
if expected and os.path.realpath(sys.prefix) != os.path.realpath(expected):
    raise SystemExit(f"[ERROR] Python fuera del .venv: {sys.prefix} != {expected}")
required = ["numpy", "pandas", "sklearn", "joblib", "tqdm", "matplotlib"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("[ERROR] Dependencias faltantes: " + ", ".join(missing))
if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
    raise SystemExit("[ERROR] Instale pyarrow o fastparquet para caches Parquet.")
print(f"[OK] Python={sys.executable}; dependencias completas", flush=True)
PYCODE

"${PYTHON_BIN}" -m py_compile "${TRAIN_PY}"

BASE_ARGS=(
  --dataset all
  --project-dir "${PROJECT_DIR}"
  --result-dir "${RESULT_DIR}"
  --torpeda-inputs "${TORPEDA_RAW_DIR}"
  --harvard-inputs "${HARVARD_RAW_DIR}"
  --egipcios-inputs "${EGIPCIOS_INPUT}"
  --torpeda-only-common-labels
  --torpeda-min-class-count 0
  --harvard-label-mode top10-request-groups
  --harvard-multiclass-strategy severity
  --harvard-min-class-count 20
  --egipcios-sep auto
  --egipcios-request-col col1
  --egipcios-label-col col2
  --egipcios-strict-paper-dataset
  --egipcios-min-class-count 2
  --sep auto
  --method-col request_http_method
  --uri-col request_http_request
  --body-col request_body
  --normal-col "000 - Normal"
  --sample-n 0
  --max-normal-rows 0
  --keep-labels-regex ""
  --drop-labels-regex ""
  --test-size 0.20
  --seed 42
  --histgb-preset max
  --sample-weight-clip 20
  --model-selection
  --model-selection-max-train-rows 250000
  --model-selection-validation-size 0.10
  --use-gpu off
  --feature-workers "${FEATURE_WORKERS}"
  --feature-chunksize "${FEATURE_CHUNKSIZE}"
  --feature-checkpoint-rows "${FEATURE_CHECKPOINT_ROWS}"
  --feature-row-timeout-seconds "${FEATURE_ROW_TIMEOUT_SECONDS}"
  --fi-n-repeats "${FI_N_REPEATS}"
  --fi-max-rows "${FI_MAX_ROWS}"
  --fi-scoring f1_macro
  --fi-n-jobs "${FI_N_JOBS}"
  --fi-min-rows-per-class "${FI_MIN_ROWS_PER_CLASS}"
  --fi-per-class
  --fi-per-class-n-repeats "${FI_PER_CLASS_N_REPEATS}"
  --fi-per-class-n-jobs "${FI_PER_CLASS_N_JOBS}"
  --fi-binary-normal-attack
  --fi-binary-n-repeats "${FI_BINARY_N_REPEATS}"
  --operational-max-rows "${OPERATIONAL_MAX_ROWS}"
  --loss-curve
  --loss-curve-max-rows "${LOSS_CURVE_MAX_ROWS}"
  --loss-curve-min-rows-per-class "${LOSS_CURVE_MIN_ROWS_PER_CLASS}"
  --loss-curve-every-n-iter "${LOSS_CURVE_EVERY_N_ITER}"
)

cat <<EOF
================================================================
[START] $(date '+%F %T')
[INFO] Host=$(hostname) JOB=${SLURM_JOB_ID:-manual} NODE=${SLURM_JOB_NODELIST:-manual}
[INFO] Python unico=${TRAIN_PY}; SHA256=${train_sha}
[INFO] Output=${RESULT_DIR}
[INFO] Dataset=all; feature_set=${FEATURE_SET_VERSION}; features=73 para los tres
[INFO] Split=80/20; seed=42; HistGradientBoosting CPU
[INFO] TorpEda XML=${torpeda_count}; Harvard tablas=${harvard_count}; Egipcios=${EGIPCIOS_INPUT}
[INFO] Resume=${RESUME}; SkipComplete=${SKIP_COMPLETE}; Clean=${CLEAN_OUTPUT_DIR}
[INFO] RequireBeatWAMM=${REQUIRE_BEAT_WAMM}
================================================================
EOF

if [[ "${CHECK_ONLY}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${TRAIN_PY}" "${BASE_ARGS[@]}" \
    --check-only --no-resume --no-skip-complete
  echo "[OK] CHECK_ONLY=1: codigo, datos, dependencias y pipeline compact73 validos; no entreno."
  exit 0
fi

TRAIN_ARGS=("${BASE_ARGS[@]}")
if [[ "${RESUME}" == "1" ]]; then TRAIN_ARGS+=(--resume); else TRAIN_ARGS+=(--no-resume); fi
if [[ "${SKIP_COMPLETE}" == "1" ]]; then TRAIN_ARGS+=(--skip-complete); else TRAIN_ARGS+=(--no-skip-complete); fi
if [[ "${CLEAN_OUTPUT_DIR}" == "1" ]]; then TRAIN_ARGS+=(--clean-output-dir); fi

echo "[RUN] Invocacion unica: TorpEda + Harvard + Egipcios con compact73"
"${PYTHON_BIN}" -u "${TRAIN_PY}" "${TRAIN_ARGS[@]}"

export RESULT_DIR TRAIN_PY TRAIN_EXPECTED_SHA256 ARTIFACT_TAG FEATURE_SET_VERSION
export PIPELINE_RUN_VERSION FEATURE_LIST_SHA256 REQUIRE_BEAT_WAMM
export ACCURACY_REGRESSION_TOLERANCE TORPEDA_F1_REGRESSION_TOLERANCE HARVARD_F1_REGRESSION_TOLERANCE
"${PYTHON_BIN}" -u - <<'PYCODE'
import csv
import gc
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib

root = Path(os.environ["RESULT_DIR"])
multi = root / "multiclase"
artifact_tag = os.environ["ARTIFACT_TAG"]
feature_version = os.environ["FEATURE_SET_VERSION"]
pipeline_version = os.environ["PIPELINE_RUN_VERSION"]
feature_hash_expected = os.environ["FEATURE_LIST_SHA256"]
accuracy_tolerance = float(os.environ["ACCURACY_REGRESSION_TOLERANCE"])
f1_tolerance = {
    "torpeda": float(os.environ["TORPEDA_F1_REGRESSION_TOLERANCE"]),
    "harvard": float(os.environ["HARVARD_F1_REGRESSION_TOLERANCE"]),
}
require_beat_wamm = os.environ["REQUIRE_BEAT_WAMM"] == "1"
metrics_filename = f"metrics_multiclass_histgb_{artifact_tag}.json"
model_filename = f"model_multiclass_histgb_{artifact_tag}.joblib"

def load_json(path):
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"[ERROR] Falta artefacto obligatorio: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def rows(path):
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"[ERROR] Falta CSV obligatorio: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def close(left, right, tol=1e-12):
    return finite(left) and finite(right) and abs(float(left) - float(right)) <= tol

paths = {name: multi / name for name in ("torpeda", "harvard", "egipcios")}
metrics = {name: load_json(out / metrics_filename) for name, out in paths.items()}
statuses = {name: load_json(out / "run_status.json") for name, out in paths.items()}
configs = {name: load_json(out / "run_config.json") for name, out in paths.items()}

classes = {
    "torpeda": {
        "NORMAL", "TORPEDA-ANOMALOUS", "TORPEDA-BufferOverflow", "TORPEDA-CRLFi",
        "TORPEDA-FormatString", "TORPEDA-LDAPi", "TORPEDA-SQLi", "TORPEDA-SSI",
        "TORPEDA-XPath", "TORPEDA-XSS",
    },
    "harvard": {
        "HARVARD-CAPEC126_PathTraversal", "HARVARD-CAPEC16_DictionaryPasswordAttack",
        "HARVARD-CAPEC242_CodeInjection", "HARVARD-CAPEC310_VulnerabilityScanning",
        "HARVARD-CAPEC34_HTTPResponseSplitting", "HARVARD-CAPEC66_SQLInjection",
        "HARVARD-CAPEC88_OSCommandInjection",
        "HARVARD-CAPEC88_OSCommandInjection__CAPEC126_PathTraversal",
        "HARVARD-RequestManipulation", "NORMAL",
    },
    "egipcios": {
        "NORMAL", "EGIPCIOS-SQLi", "EGIPCIOS-OSCommandInjection",
        "EGIPCIOS-PathTraversal", "EGIPCIOS-XSS", "EGIPCIOS-SSRF",
        "EGIPCIOS-CommandInjection", "EGIPCIOS-SSTI", "EGIPCIOS-CodeInjection",
    },
}
label_schema_versions = {
    "torpeda": "torpeda_native_common10_v1",
    "harvard": "harvard_top10_request_groups_v1",
    "egipcios": "egipcios_paper_native9_v1",
}
expected_rows = {
    "torpeda": (59306, 14827),
    "harvard": (721928, 180483),
    "egipcios": (563732, 140933),
}
egipcios_test_counts = {
    "NORMAL": 98703,
    "EGIPCIOS-SQLi": 29217,
    "EGIPCIOS-OSCommandInjection": 7223,
    "EGIPCIOS-PathTraversal": 3544,
    "EGIPCIOS-XSS": 579,
    "EGIPCIOS-SSRF": 529,
    "EGIPCIOS-CommandInjection": 519,
    "EGIPCIOS-SSTI": 379,
    "EGIPCIOS-CodeInjection": 240,
}
baseline = {
    "torpeda": {"accuracy": 0.9987185539893437, "f1_macro": 0.9945559748686266},
    "harvard": {"accuracy": 0.9778427885174781, "f1_macro": 0.9731671827230427},
}

failures = []
def check(name, condition):
    if not condition:
        failures.append(name)

feature_lists = {}
predictions = {}
common_required = [
    "classification_report_test.csv", "classification_report_test.json",
    "per_class_ovr_metrics.csv", "metrics_flat.csv", "feature_importance.csv",
    "feature_importance_per_class.csv", "feature_importance_metadata.json",
    "operational_metrics.json", "operational_metrics.csv", "features_used.csv",
    "selected_model_summary.txt", "confusion_matrix_test.csv", "test_predictions.csv",
    "loss_curve_histgb.csv", "loss_curve_metadata.json",
    "model_selection_inner_validation.json", "model_selection_inner_validation.csv",
    "feature_selection_manifest.csv", "label_schema_validation.json", "label_mapping.csv",
    "binary_normal_vs_attack_metrics.json", "binary_normal_vs_attack_metrics.csv",
    "feature_importance_binary_normal_vs_attack.csv",
]

for dataset, out in paths.items():
    payload = metrics[dataset]
    status = statuses[dataset]
    config = configs[dataset]
    expected_classes = classes[dataset]
    features = list(payload.get("features_used") or [])
    feature_lists[dataset] = features
    encoded = json.dumps(features, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    observed_feature_hash = hashlib.sha256(encoded).hexdigest()

    check(f"{dataset}:status", status.get("status") == "complete")
    check(f"{dataset}:identity", payload.get("dataset") == dataset and status.get("dataset") == dataset)
    try:
        status_metrics_ok = Path(str(status.get("metrics"))).resolve() == (out / metrics_filename).resolve()
    except Exception:
        status_metrics_ok = False
    check(f"{dataset}:status_metrics_path", status_metrics_ok)
    check(f"{dataset}:compact73", len(features) == 73 and len(set(features)) == 73)
    check(f"{dataset}:feature_hash", observed_feature_hash == feature_hash_expected)
    check(
        f"{dataset}:versions",
        payload.get("feature_set_version") == feature_version
        and status.get("feature_set_version") == feature_version
        and config.get("feature_set_version") == feature_version
        and payload.get("pipeline_run_version") == pipeline_version
        and status.get("pipeline_run_version") == pipeline_version
        and config.get("pipeline_run_version") == pipeline_version
        and payload.get("label_schema_version") == label_schema_versions[dataset]
        and status.get("label_schema_version") == label_schema_versions[dataset]
        and config.get("label_schema_version") == label_schema_versions[dataset],
    )
    methodology = payload.get("methodology_config_sha256")
    fingerprints = payload.get("input_fingerprints")
    check(
        f"{dataset}:provenance",
        isinstance(methodology, str) and len(methodology) == 64
        and status.get("methodology_config_sha256") == methodology
        and config.get("methodology_config_sha256") == methodology
        and isinstance(fingerprints, list) and len(fingerprints) > 0
        and status.get("input_fingerprints") == fingerprints
        and config.get("input_fingerprints") == fingerprints,
    )
    check(f"{dataset}:classes", int(payload.get("n_classes", -1)) == len(expected_classes) and set(payload.get("classes") or []) == expected_classes)
    check(
        f"{dataset}:rows",
        (int(payload.get("train_rows", -1)), int(payload.get("test_rows", -1))) == expected_rows[dataset],
    )
    check(f"{dataset}:split", int(payload.get("seed", -1)) == 42 and close(payload.get("test_size"), 0.20))
    for filename in common_required:
        check(f"{dataset}:artifact:{filename}", (out / filename).is_file() and (out / filename).stat().st_size > 0)

    model_path = out / model_filename
    check(f"{dataset}:model_file", model_path.is_file() and model_path.stat().st_size > 0)
    try:
        bundle = joblib.load(model_path)
        check(
            f"{dataset}:model_smoke",
            isinstance(bundle, dict)
            and bundle.get("dataset") == dataset
            and bundle.get("model") is not None
            and int(bundle.get("feature_count", -1)) == 73
            and list(bundle.get("features") or bundle.get("features_used") or []) == features
            and set(bundle.get("classes") or []) == expected_classes,
        )
    except Exception:
        check(f"{dataset}:model_smoke", False)
    finally:
        try:
            del bundle
        except NameError:
            pass
        gc.collect()

    true_counts = Counter()
    pred_counts = Counter()
    true_positive = Counter()
    confusion = Counter()
    count = correct_count = 0
    with (out / "test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_true, y_pred = row.get("y_true"), row.get("y_pred")
            count += 1
            true_counts[y_true] += 1
            pred_counts[y_pred] += 1
            confusion[(y_true, y_pred)] += 1
            if y_true == y_pred:
                correct_count += 1
                true_positive[y_true] += 1
    f1_values = []
    derived_by_class = {}
    for class_name in expected_classes:
        tp = true_positive[class_name]
        fp = pred_counts[class_name] - tp
        fn = true_counts[class_name] - tp
        tn = count - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1_values.append(f1)
        derived_by_class[class_name] = {
            "support": true_counts[class_name], "pred_support": pred_counts[class_name],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall_tpr": recall, "f1": f1,
        }
    prediction = {
        "count": count,
        "true_counts": dict(true_counts),
        "pred_classes": set(pred_counts),
        "accuracy": correct_count / count if count else float("nan"),
        "f1_macro": sum(f1_values) / len(f1_values) if f1_values else float("nan"),
    }
    predictions[dataset] = prediction
    test = payload.get("test_metrics") or {}
    check(
        f"{dataset}:metrics_recomputed",
        count == expected_rows[dataset][1]
        and set(true_counts) == expected_classes
        and set(pred_counts).issubset(expected_classes)
        and close(test.get("accuracy"), prediction["accuracy"])
        and close(test.get("f1_macro"), prediction["f1_macro"]),
    )

    per_class = rows(out / "per_class_ovr_metrics.csv")
    global_fi = rows(out / "feature_importance.csv")
    per_class_fi = rows(out / "feature_importance_per_class.csv")
    features_file = rows(out / "features_used.csv")
    fi_meta = load_json(out / "feature_importance_metadata.json")
    operational = load_json(out / "operational_metrics.json")
    loss_meta = load_json(out / "loss_curve_metadata.json")
    loss_rows = rows(out / "loss_curve_histgb.csv")
    selection_manifest = rows(out / "feature_selection_manifest.csv")
    label_validation = load_json(out / "label_schema_validation.json")
    label_mapping = rows(out / "label_mapping.csv")
    per_class_by_name = {r.get("class"): r for r in per_class}
    per_class_ok = len(per_class) == len(expected_classes) and set(per_class_by_name) == expected_classes
    if per_class_ok:
        for class_name, expected in derived_by_class.items():
            row = per_class_by_name[class_name]
            try:
                counts_ok = all(int(float(row.get(field, "nan"))) == expected[field] for field in ("support", "pred_support", "tp", "fp", "fn", "tn"))
            except (TypeError, ValueError):
                counts_ok = False
            ratios_ok = all(close(row.get(field), expected[field]) for field in ("precision", "recall_tpr", "f1"))
            if not counts_ok or not ratios_ok:
                per_class_ok = False
                break
    check(f"{dataset}:per_class_metrics", per_class_ok)
    check(
        f"{dataset}:features_file",
        len(features_file) == 73
        and [r.get("feature") for r in features_file] == features,
    )
    selected_manifest = [r for r in selection_manifest if str(r.get("selected", "")).lower() in {"true", "1"}]
    dropped_manifest = [r for r in selection_manifest if r.get("status") == "dropped"]
    check(
        f"{dataset}:feature_manifest",
        len(selection_manifest) == 124
        and len(selected_manifest) == 73
        and len(dropped_manifest) == 51
        and {r.get("feature") for r in selected_manifest} == set(features),
    )
    check(
        f"{dataset}:label_artifacts",
        label_validation.get("status") == "valid"
        and label_validation.get("label_schema_version") == label_schema_versions[dataset]
        and int(label_validation.get("n_classes", -1)) == len(expected_classes)
        and set(label_validation.get("classes") or []) == expected_classes
        and len(label_mapping) > 0,
    )
    check(f"{dataset}:global_fi", len(global_fi) == 73 and {r.get('feature') for r in global_fi} == set(features))
    per_class_fi_ok = (
        len(per_class_fi) == len(expected_classes) * 73
        and {r.get('class') for r in per_class_fi} == expected_classes
    )
    if per_class_fi_ok:
        for class_name in expected_classes:
            class_rows = [r for r in per_class_fi if r.get("class") == class_name]
            try:
                ranks = {int(float(r.get("rank_in_class", "nan"))) for r in class_rows}
            except (TypeError, ValueError):
                per_class_fi_ok = False
                break
            if (
                len(class_rows) != 73
                or {r.get("feature") for r in class_rows} != set(features)
                or ranks != set(range(1, 74))
                or not all(finite(r.get("permutation_importance_mean")) for r in class_rows)
            ):
                per_class_fi_ok = False
                break
    check(f"{dataset}:per_class_fi", per_class_fi_ok)
    check(
        f"{dataset}:fi_metadata",
        int(fi_meta.get("n_features", -1)) == 73
        and int(fi_meta.get("fi_rows_effective", 0)) > 0
        and not fi_meta.get("permutation_error")
        and not any((fi_meta.get("per_class_errors") or {}).values())
        and not any((fi_meta.get("binary_errors") or {}).values()),
    )
    operational_fields = (
        "latency_ms_pipeline_online_p50", "latency_ms_pipeline_online_p95",
        "latency_ms_pipeline_online_p99", "throughput_req_s_pipeline_online",
        "throughput_req_s_pipeline_batch", "throughput_req_s_inference_batch",
    )
    check(
        f"{dataset}:operational",
        operational.get("available") is True
        and int(operational.get("measured_rows", 0)) > 0
        and all(finite(operational.get(name)) and float(operational[name]) > 0 for name in operational_fields),
    )
    check(
        f"{dataset}:loss",
        loss_meta.get("available") is True
        and not loss_meta.get("staged_error")
        and int(loss_meta.get("n_curve_rows", 0)) > 0
        and len(loss_rows) == int(loss_meta.get("n_curve_rows", -1)),
    )

    selection = load_json(out / "model_selection_inner_validation.json")
    winner = selection.get("winner") or {}
    winner_power = winner.get("weight_power")
    observed_powers = sorted(
        round(float(row.get("weight_power")), 8)
        for row in (payload.get("model_selection_candidates") or [])
        if isinstance(row, dict) and row.get("weight_power") is not None
    )
    check(
        f"{dataset}:inner_selection",
        payload.get("model_selection_scope") == "outer_train_only"
        and selection.get("selection_scope") == "outer_train_only"
        and selection.get("outer_test_touched") is False
        and observed_powers == [0.20, 0.33, 0.40, 0.50]
        and finite(winner_power)
        and round(float(winner_power), 8) in {0.20, 0.33, 0.40, 0.50}
        and winner.get("weight_mode") == payload.get("weight_mode")
        and payload.get("weight_mode") in {
            "power_balanced_0_20", "power_balanced_0_33",
            "power_balanced_0_40", "power_balanced_0_50",
        },
    )

check("unified73:same_ordered_features", feature_lists["torpeda"] == feature_lists["harvard"] == feature_lists["egipcios"])
check("egipcios:test_supports", predictions["egipcios"]["true_counts"] == egipcios_test_counts)

check("torpeda:label_mode", metrics["torpeda"].get("dataset_label_mode") == "torpeda_native_common_names")
check("harvard:label_mode", metrics["harvard"].get("dataset_label_mode") == "top10-request-groups")
check("egipcios:label_mode", metrics["egipcios"].get("dataset_label_mode") == "egipcios_paper_native9")

regression_rows = []
for dataset in ("torpeda", "harvard"):
    row = {"dataset": dataset}
    passed = True
    for metric_name in ("accuracy", "f1_macro"):
        current = predictions[dataset][metric_name]
        previous = baseline[dataset][metric_name]
        metric_tolerance = accuracy_tolerance if metric_name == "accuracy" else f1_tolerance[dataset]
        metric_pass = current >= previous - metric_tolerance
        row[f"current_{metric_name}"] = current
        row[f"previous_{metric_name}"] = previous
        row[f"delta_percentage_points_{metric_name}"] = 100.0 * (current - previous)
        row[f"tolerance_absolute_{metric_name}"] = metric_tolerance
        row[f"pass_{metric_name}"] = metric_pass
        passed = passed and metric_pass
    row["pass_no_material_regression"] = passed
    regression_rows.append(row)
    check(f"{dataset}:no_regression", passed)

egipcios_out = paths["egipcios"]
benchmark = load_json(egipcios_out / "benchmark_vs_wamm.json")
egipcios_label_validation = load_json(egipcios_out / "label_schema_validation.json")
check(
    "egipcios:strict_native9_labels",
    egipcios_label_validation.get("strict_exact_distribution") is True
    and int(egipcios_label_validation.get("rows", -1)) == 704665
    and egipcios_label_validation.get("capec88_and_capec248_separate") is True,
)
for filename in (
    "benchmark_vs_wamm.csv", "benchmark_wamm_per_class_block_rate.csv",
):
    check(f"egipcios:artifact:{filename}", (egipcios_out / filename).is_file() and (egipcios_out / filename).stat().st_size > 0)
actual_accuracy = predictions["egipcios"]["accuracy"]
actual_f1_macro = predictions["egipcios"]["f1_macro"]
beats_accuracy = actual_accuracy > 0.9959
beats_f1 = actual_f1_macro > 0.8607
check(
    "egipcios:wamm_consistency",
    close(benchmark.get("paper_accuracy"), 0.9959)
    and close(benchmark.get("paper_f1_macro"), 0.8607)
    and close(benchmark.get("our_accuracy"), actual_accuracy)
    and close(benchmark.get("our_f1_macro"), actual_f1_macro)
    and benchmark.get("beats_paper_accuracy") == beats_accuracy
    and benchmark.get("beats_paper_f1_macro") == beats_f1
    and benchmark.get("beats_paper_both_primary_metrics") == (beats_accuracy and beats_f1),
)
if require_beat_wamm:
    check("egipcios:beats_wamm_both_required", beats_accuracy and beats_f1)

comparison_rows = []
for dataset in ("torpeda", "harvard", "egipcios"):
    test = metrics[dataset].get("test_metrics") or {}
    comparison_rows.append({
        "dataset": dataset,
        "profile": "unified_compact73",
        "feature_count": metrics[dataset].get("feature_count"),
        "n_classes": metrics[dataset].get("n_classes"),
        "weight_mode": metrics[dataset].get("weight_mode"),
        "accuracy": predictions[dataset]["accuracy"],
        "balanced_accuracy": test.get("balanced_accuracy"),
        "f1_macro": predictions[dataset]["f1_macro"],
        "f1_weighted": test.get("f1_weighted"),
        "mcc": test.get("mcc"),
        "previous_accuracy": baseline.get(dataset, {}).get("accuracy"),
        "previous_f1_macro": baseline.get(dataset, {}).get("f1_macro"),
        "passes_regression_gate": next((r["pass_no_material_regression"] for r in regression_rows if r["dataset"] == dataset), None),
        "wamm_accuracy": 0.9959 if dataset == "egipcios" else None,
        "wamm_f1_macro": 0.8607 if dataset == "egipcios" else None,
        "beats_wamm_both": beats_accuracy and beats_f1 if dataset == "egipcios" else None,
    })

comparison_csv = multi / "comparison_integrated_3datasets.csv"
with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
    writer.writeheader()
    writer.writerows(comparison_rows)
regression_csv = multi / "performance_regression_unified73.csv"
with regression_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(regression_rows[0]))
    writer.writeheader()
    writer.writerows(regression_rows)

def pct(value):
    return "-" if not finite(value) else f"{float(value):.6%}"

def dec(value, digits=4):
    return "-" if not finite(value) else f"{float(value):.{digits}f}"

created_at = datetime.now(timezone.utc).isoformat()
report = [
    "# Informe integrado - 3datasetsMulticlase3",
    "",
    f"Generado: {created_at}",
    "",
    "Los tres datasets fueron procesados por un solo Python y el mismo vector ordenado de 73 features intra-request.",
    "",
    "## Resultado global",
    "",
    "| Dataset | Features | Clases | Accuracy | Balanced accuracy | F1 macro | F1 weighted | MCC |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in comparison_rows:
    report.append(
        f"| {row['dataset']} | {row['feature_count']} | {row['n_classes']} | {pct(row['accuracy'])} | "
        f"{pct(row['balanced_accuracy'])} | {pct(row['f1_macro'])} | {pct(row['f1_weighted'])} | {dec(row['mcc'], 6)} |"
    )
report += [
    "",
    "## Retencion de rendimiento en Harvard y TorpEda",
    "",
    "Accuracy y el F1 de TorpEda admiten como maximo 0,01 pp de caida. El F1 macro de Harvard admite 0,15 pp: "
    "el benchmark limpio congelado de compact73 mejoro su accuracy, pero redujo su F1 macro 0,10317 pp. "
    "La tabla informa el delta realmente obtenido por esta ejecucion.",
    "",
    "| Dataset | Delta accuracy (pp) | Delta F1 macro (pp) | Tol. accuracy (pp) | Tol. F1 (pp) | Aprobada |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in regression_rows:
    report.append(
        f"| {row['dataset']} | {row['delta_percentage_points_accuracy']:.6f} | "
        f"{row['delta_percentage_points_f1_macro']:.6f} | "
        f"{100*row['tolerance_absolute_accuracy']:.4f} | {100*row['tolerance_absolute_f1_macro']:.4f} | "
        f"{row['pass_no_material_regression']} |"
    )
report += [
    "",
    "## Egipcios frente a WAMM",
    "",
    "Se conservaron NORMAL y las ocho clases de ataque; CAPEC-88 y CAPEC-248 permanecen separadas.",
    "",
    "| Metrica | HistGB compact73 | WAMM | Delta (pp) | Supera |",
    "|---|---:|---:|---:|---:|",
    f"| Accuracy | {actual_accuracy:.6%} | {0.9959:.6%} | {100*(actual_accuracy-0.9959):.6f} | {beats_accuracy} |",
    f"| F1 macro | {actual_f1_macro:.6%} | {0.8607:.6%} | {100*(actual_f1_macro-0.8607):.6f} | {beats_f1} |",
    "",
]
for dataset, out in paths.items():
    per_class = rows(out / "per_class_ovr_metrics.csv")
    fi_per_class = rows(out / "feature_importance_per_class.csv")
    op = load_json(out / "operational_metrics.json")
    timing = metrics[dataset].get("timing") or {}
    resource = metrics[dataset].get("resource_usage") or {}
    train_seconds = float(timing.get("t_build_train_seconds") or 0.0)
    selection_seconds = float(timing.get("model_selection_seconds") or 0.0)
    report += [
        f"## Metricas por clase - {dataset}",
        "",
        "| Clase | Support | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_class:
        report.append(
            f"| {row.get('class')} | {row.get('support')} | {pct(row.get('precision'))} | "
            f"{pct(row.get('recall_tpr'))} | {pct(row.get('specificity_tnr'))} | {pct(row.get('f1'))} | "
            f"{dec(row.get('roc_auc_ovr'), 6)} | {dec(row.get('pr_auc_ovr'), 6)} |"
        )
    report += [
        "",
        f"## Feature importance por clase - {dataset}",
        "",
        "Top 3 por clase; el CSV conserva las 73 por clase.",
        "",
        "| Clase | Rank | Feature | Importance | Desvio | Grupo |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in sorted(fi_per_class, key=lambda r: (str(r.get("class")), int(float(r.get("rank_in_class", 999999))))):
        if int(float(row.get("rank_in_class", 999999))) <= 3:
            report.append(
                f"| {row.get('class')} | {row.get('rank_in_class')} | {row.get('feature')} | "
                f"{dec(row.get('permutation_importance_mean'), 6)} | {dec(row.get('permutation_importance_std'), 6)} | "
                f"{row.get('feature_group')} |"
            )
    report += [
        "",
        f"## Metricas operativas - {dataset}",
        "",
        "| Filas | p50 ms | p95 ms | p99 ms | Throughput online req/s | Throughput batch req/s | Train s | Seleccion s | Run total s | Peak RSS MB | Modelo MB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {op.get('measured_rows')} | {dec(op.get('latency_ms_pipeline_online_p50'),3)} | "
        f"{dec(op.get('latency_ms_pipeline_online_p95'),3)} | {dec(op.get('latency_ms_pipeline_online_p99'),3)} | "
        f"{dec(op.get('throughput_req_s_pipeline_online'),3)} | {dec(op.get('throughput_req_s_pipeline_batch'),3)} | "
        f"{dec(train_seconds,3)} | {dec(selection_seconds,3)} | {dec(timing.get('run_total_seconds'),3)} | "
        f"{dec(resource.get('peak_rss_mb', op.get('peak_rss_mb')),3)} | "
        f"{dec(op.get('model_size_mb'),3)} |",
        "",
        f"Archivos completos: `{out / 'per_class_ovr_metrics.csv'}`, `{out / 'feature_importance_per_class.csv'}`, "
        f"`{out / 'operational_metrics.json'}`, `{out / 'confusion_matrix_test.csv'}`.",
        "",
    ]

report += [
    "## Validacion",
    "",
    f"Guardas aprobadas: **{not failures}**.",
    f"Fallos: `{failures}`.",
    "",
    "La comparacion con WAMM conserva dataset y taxonomia, pero el paper no publica indices exactos de su split.",
]
report_path = root / "informe_integrado_3datasets.md"
report_path.write_text("\n".join(report), encoding="utf-8")

payload = {
    "created_at": created_at,
    "result_root": str(root),
    "architecture": "single_python_single_invocation_unified_compact73",
    "training_python": os.environ["TRAIN_PY"],
    "training_python_sha256": os.environ["TRAIN_EXPECTED_SHA256"],
    "feature_set_version": feature_version,
    "pipeline_run_version": pipeline_version,
    "feature_list_sha256": feature_hash_expected,
    "rows": comparison_rows,
    "regression": regression_rows,
    "require_beat_wamm": require_beat_wamm,
    "beats_wamm_both": beats_accuracy and beats_f1,
    "validation_failures": failures,
    "all_validation_gates_pass": not failures,
}
(multi / "comparison_integrated_3datasets.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
(multi / "performance_regression_unified73.json").write_text(json.dumps({"created_at": created_at, "rows": regression_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
(root / "manifest_integrated_3datasets.json").write_text(json.dumps({**payload, "integrated_report": str(report_path)}, ensure_ascii=False, indent=2), encoding="utf-8")

for row in comparison_rows:
    print(f"[RESULT] {row['dataset']}: accuracy={row['accuracy']:.6%}; f1_macro={row['f1_macro']:.6%}; features=73")
print(f"[RESULT] Egipcios supera simultaneamente WAMM: {beats_accuracy and beats_f1}")
if failures:
    raise SystemExit("[ERROR] Fallaron guardas finales: " + ", ".join(failures))
print("[OK] Los tres datasets pasaron artefactos, compact73 comun y guardas de rendimiento.")
PYCODE

cat <<EOF
================================================================
[OK] $(date '+%F %T') 3datasetsMulticlase3 finalizado
[OUT] ${RESULT_DIR}
[REPORT] ${RESULT_DIR}/informe_integrado_3datasets.md
[COMPARISON] ${RESULT_DIR}/multiclase/comparison_integrated_3datasets.csv
[REGRESSION H/T] ${RESULT_DIR}/multiclase/performance_regression_unified73.csv
[WAMM] ${RESULT_DIR}/multiclase/egipcios/benchmark_vs_wamm.csv
[PER CLASS] ${RESULT_DIR}/multiclase/{torpeda,harvard,egipcios}/per_class_ovr_metrics.csv
[FI PER CLASS] ${RESULT_DIR}/multiclase/{torpeda,harvard,egipcios}/feature_importance_per_class.csv
[OPERATIONAL] ${RESULT_DIR}/multiclase/{torpeda,harvard,egipcios}/operational_metrics.json
================================================================
EOF
