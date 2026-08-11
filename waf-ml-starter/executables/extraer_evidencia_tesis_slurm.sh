#!/usr/bin/env bash
# Auditoria historica y actual del entorno Slurm de la tesis.
# No ejecuta modelos ni modifica el repositorio: solo consulta metadatos,
# inventaria c1/c2/c3 y copia artefactos textuales pequenos que ya existen.

set -uo pipefail

AUDIT_USER="${SLURM_AUDIT_USER:-aroman}"
START_DATE="${SLURM_AUDIT_START:-2026-01-01}"
END_DATE="${SLURM_AUDIT_END:-2027-01-01}"
PARTITION="${SLURM_AUDIT_PARTITION:-normal}"
PROJECT_ROOT="${SLURM_AUDIT_PROJECT_ROOT:-/home_data/aroman/TFG/waf-ml-starter}"
NODE_LIST="${SLURM_AUDIT_NODES:-c1,c2,c3}"
RUN_NODE_INVENTORY="${SLURM_AUDIT_NODE_INVENTORY:-1}"
RECOVER_BATCH_SCRIPTS="${SLURM_AUDIT_RECOVER_BATCH:-1}"
COPY_PROJECT_EVIDENCE="${SLURM_AUDIT_COPY_PROJECT_EVIDENCE:-1}"

JOB_IDS=(
  2312 2313 2314 2315 2399 2400 2401 2402 2403 2404 2405 2406 2407
  2408 2409 2410 2411 2412 2413 2414 2416 2418 2419 2420 2421 2422
  2426 2427 2428 2429 2430 2431 2432 2433 2434 2440 2448 2455 2457
  2458 2459 2460 2462 2464 2465 2466 2467 2468 2469 2470 2471 2472
  2473 2474 2475 2476 2477 2478 2488 2489 2490 2491 2496 2497 2498
  2499 2508 2515 2518 2528 2529 2530 2531 2532 2533 2534 2564 2565
  2569 2585 2586 2593 2594 2625 2657 2687 2766 2883 2927 2960 2962
  2963 2964 2965 2966 3009 3010 3011 3012 3013 3045 3046 3053 3065
  3066 3067 3068 3101 3102 3103 3104 3148 3149 3150 3151 3289 3296
  3305 3315 3331 3332 3357 3358 3359 3360 3361 3442 3443 3444 3445
  3446 3453 3454
)

timestamp() {
  date --iso-8601=seconds 2>/dev/null || date
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

section() {
  printf '\n[%s] %s\n' "$(timestamp)" "$*"
}

capture_command() {
  local output_file="$1"
  shift
  {
    printf 'capturado_en=%s\n' "$(timestamp)"
    printf 'comando='
    printf '%q ' "$@"
    printf '\n\n'
    "$@"
  } >"$output_file" 2>&1 || true
}

python_report() {
  local python_executable="$1"

  printf 'python_executable=%s\n' "$python_executable"
  "$python_executable" -VV 2>&1 || true
  "$python_executable" - <<'PY' 2>&1 || true
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

print("sys.executable=", sys.executable)
print("python=", sys.version.replace("\n", " "))
print("platform=", platform.platform())
print("machine=", platform.machine())
print("processor=", platform.processor())

packages = (
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib",
    "psutil", "pyarrow", "matplotlib", "seaborn", "torch", "tensorflow",
    "cupy", "cuml", "PyYAML", "tqdm", "fastapi", "uvicorn"
)
for package in packages:
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        pass
PY
  "$python_executable" -m pip --version 2>&1 || true
  "$python_executable" -m pip freeze 2>/dev/null | LC_ALL=C sort || true
}

collect_node_inventory() {
  local node
  node="$(hostname -s 2>/dev/null || hostname)"

  echo "tipo_evidencia=inventario_actual_del_nodo"
  echo "advertencia=no_prueba_por_si_solo_la_configuracion_historica"
  printf 'capturado_en=%s\n' "$(timestamp)"
  printf 'nodo=%s\n' "$node"
  printf 'directorio=%s\n' "$PWD"

  echo
  echo "IDENTIFICACION_Y_SO"
  hostname 2>/dev/null || true
  hostnamectl 2>/dev/null || true
  uname -a 2>/dev/null || true
  cat /etc/os-release 2>/dev/null || true

  echo
  echo "ASIGNACION_SLURM_DE_ESTE_INVENTARIO"
  env | LC_ALL=C sort | grep -E \
    '^(SLURM|CUDA|ROCR|OMP|MKL|OPENBLAS|NUMEXPR|PYTHONHASHSEED)' || true
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol show job -dd "$SLURM_JOB_ID" 2>/dev/null || true
  fi
  scontrol show node -dd "$node" 2>/dev/null || true

  echo
  echo "CPU_Y_AFINIDAD"
  lscpu 2>/dev/null || true
  printf 'nproc_total=%s\n' "$(nproc --all 2>/dev/null || echo N_D)"
  printf 'nproc_disponible=%s\n' "$(nproc 2>/dev/null || echo N_D)"
  grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status 2>/dev/null || true
  numactl --hardware 2>/dev/null || true

  echo
  echo "MEMORIA"
  free -h 2>/dev/null || true
  free -b 2>/dev/null || true
  grep -E 'MemTotal|SwapTotal|HugePages_Total' /proc/meminfo 2>/dev/null || true

  echo
  echo "GPU"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-no_definido}"
  printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS:-no_definido}"
  printf 'SLURM_GPUS_ON_NODE=%s\n' "${SLURM_GPUS_ON_NODE:-no_definido}"
  lspci 2>/dev/null | grep -Ei 'vga|3d|display|nvidia|amd' || true
  if command_exists nvidia-smi; then
    nvidia-smi -L 2>/dev/null || true
    nvidia-smi \
      --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total \
      --format=csv,noheader 2>/dev/null || true
    nvidia-smi 2>/dev/null || true
  else
    echo "nvidia-smi=no_disponible"
  fi
  nvcc --version 2>/dev/null || true

  echo
  echo "ALMACENAMIENTO_Y_LIMITES"
  df -hT "$PWD" 2>/dev/null || true
  lsblk -o NAME,TYPE,SIZE,ROTA,MODEL,MOUNTPOINTS 2>/dev/null || true
  ulimit -a 2>/dev/null || true

  echo
  echo "COMPILADORES_Y_MODULOS"
  module -t list 2>&1 || module list 2>&1 || true
  gcc --version 2>/dev/null | head -n 1 || true
  ldd --version 2>/dev/null | head -n 1 || true
  conda info 2>/dev/null || true

  echo
  echo "PYTHON_ACTUAL_DEL_SHELL"
  if command_exists python; then
    python_report "$(command -v python)"
  elif command_exists python3; then
    python_report "$(command -v python3)"
  else
    echo "python=no_disponible"
  fi

  echo
  echo "VENV_ACTUAL_DEL_PROYECTO"
  if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    python_report "$PROJECT_ROOT/.venv/bin/python"
  else
    echo "venv_proyecto=no_encontrado_en_$PROJECT_ROOT/.venv"
  fi

  echo
  echo "GIT_ACTUAL_DEL_PROYECTO"
  echo "advertencia=el_commit_actual_no_equivale_al_commit_historico_de_cada_job"
  if [ -d "$PROJECT_ROOT/.git" ]; then
    git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true
    git -C "$PROJECT_ROOT" branch --show-current 2>/dev/null || true
    git -C "$PROJECT_ROOT" status --short 2>/dev/null || true
  else
    echo "git=no_encontrado_en_$PROJECT_ROOT"
  fi
}

if [ "${1:-}" = "--node-only" ]; then
  collect_node_inventory
  exit 0
fi

if ! command_exists sacct; then
  echo "ERROR: sacct no esta disponible. Ejecuta este script en el cluster Arandu."
  exit 1
fi

STAMP="$(date +%Y%m%dT%H%M%S)"
OUT_DIR="${SLURM_AUDIT_OUTPUT:-${PWD}/evidencia_tesis_slurm_${STAMP}}"
SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")"

mkdir -p \
  "$OUT_DIR/sacct" \
  "$OUT_DIR/slurm_actual" \
  "$OUT_DIR/inventario_actual_nodos" \
  "$OUT_DIR/batch_scripts_recuperados" \
  "$OUT_DIR/proyecto_actual" \
  "$OUT_DIR/logs_preservados"

exec > >(tee -a "$OUT_DIR/ejecucion_auditoria.log") 2>&1

section "Inicio de la auditoria"
printf 'usuario=%s\n' "$AUDIT_USER"
printf 'periodo=%s a %s\n' "$START_DATE" "$END_DATE"
printf 'jobs_esperados=%s\n' "${#JOB_IDS[@]}"
printf 'nodos=%s\n' "$NODE_LIST"
printf 'proyecto=%s\n' "$PROJECT_ROOT"
printf 'salida=%s\n' "$OUT_DIR"

{
  echo "Este paquete separa evidencia historica de inventario actual."
  echo "Los archivos sacct_*.psv corresponden a registros historicos conservados por Slurm."
  echo "Los archivos inventario_actual_*.txt describen el nodo el dia de esta auditoria;"
  echo "no prueban por si solos que el nodo tuviera la misma configuracion en mayo-julio de 2026."
  echo "ReqTRES/AllocTRES prueban solicitud/asignacion, no uso efectivo del recurso."
  echo "Los jobs que nunca iniciaron deben documentarse como 'No aplica: no recibio nodo'."
} >"$OUT_DIR/LEEME.txt"

printf '%s\n' "${JOB_IDS[@]}" >"$OUT_DIR/job_ids_tesis.txt"
JOB_CSV="$(IFS=,; printf '%s' "${JOB_IDS[*]}")"

section "Version y campos admitidos por Slurm"
capture_command "$OUT_DIR/slurm_actual/versiones_comandos.txt" bash -c \
  'sacct --version; scontrol --version; sinfo --version; srun --version; sbatch --version'
sacct --helpformat >"$OUT_DIR/sacct/campos_sacct_disponibles.txt" 2>&1 || true

mapfile -t AVAILABLE_FIELDS < <(
  tr '[:space:]' '\n' <"$OUT_DIR/sacct/campos_sacct_disponibles.txt" |
    sed '/^$/d' | LC_ALL=C sort -u
)

field_available() {
  local candidate="$1"
  local available
  for available in "${AVAILABLE_FIELDS[@]}"; do
    if [ "$available" = "$candidate" ]; then
      return 0
    fi
  done
  return 1
}

select_fields() {
  local candidate
  local selected=()
  for candidate in "$@"; do
    if field_available "$candidate"; then
      selected+=("$candidate")
    else
      printf '%s\n' "$candidate" >>"$OUT_DIR/sacct/campos_solicitados_no_disponibles.txt"
    fi
  done
  local IFS=,
  printf '%s' "${selected[*]}"
}

: >"$OUT_DIR/sacct/campos_solicitados_no_disponibles.txt"

ALLOCATION_CANDIDATES=(
  JobIDRaw JobID JobName User UID Group Account Cluster Partition QOS State Reason
  ExitCode DerivedExitCode Submit Eligible Start End Elapsed ElapsedRaw Timelimit
  TimelimitRaw Suspended NodeList NNodes ReqNodes NTasks NCPUS ReqCPUS AllocCPUS
  ReqMem ReqTRES AllocTRES Constraints Exclusive OverSubscribe Priority Reservation
  ReqReservation ReqCPUFreq ReqCPUFreqMin ReqCPUFreqMax ReqCPUFreqGov ConsumedEnergy
  ConsumedEnergyRaw WorkDir StdIn StdOut StdErr SubmitLine Container Comment
  AdminComment SystemComment
)

STEP_CANDIDATES=(
  JobIDRaw JobID JobName State ExitCode Start End Elapsed ElapsedRaw NodeList NNodes
  NTasks NCPUS ReqCPUS AllocCPUS ReqMem ReqTRES AllocTRES TotalCPU UserCPU SystemCPU
  CPUTime CPUTimeRAW AveCPU MinCPU MinCPUNode MinCPUTask AveRSS MaxRSS MaxRSSNode
  MaxRSSTask AveVMSize MaxVMSize MaxVMSizeNode MaxVMSizeTask AvePages MaxPages
  AveDiskRead MaxDiskRead MaxDiskReadNode MaxDiskReadTask AveDiskWrite MaxDiskWrite
  MaxDiskWriteNode MaxDiskWriteTask ConsumedEnergy ConsumedEnergyRaw TRESUsageInAve
  TRESUsageInMax TRESUsageInMaxNode TRESUsageInMaxTask TRESUsageInTot TRESUsageOutAve
  TRESUsageOutMax TRESUsageOutMaxNode TRESUsageOutMaxTask TRESUsageOutTot
)

ALLOCATION_FIELDS="$(select_fields "${ALLOCATION_CANDIDATES[@]}")"
STEP_FIELDS="$(select_fields "${STEP_CANDIDATES[@]}")"

section "Extraccion historica de asignaciones"
sacct \
  -u "$AUDIT_USER" \
  -S "$START_DATE" \
  -E "$END_DATE" \
  -j "$JOB_CSV" \
  -D -X -P --units=K \
  --format="$ALLOCATION_FIELDS" \
  >"$OUT_DIR/sacct/sacct_jobs_asignacion.psv" 2>"$OUT_DIR/sacct/sacct_jobs_asignacion.err" \
  || true

section "Extraccion historica de jobs y pasos"
sacct \
  -u "$AUDIT_USER" \
  -S "$START_DATE" \
  -E "$END_DATE" \
  -j "$JOB_CSV" \
  -D -P --units=K \
  --format="$STEP_FIELDS" \
  >"$OUT_DIR/sacct/sacct_jobs_y_steps.psv" 2>"$OUT_DIR/sacct/sacct_jobs_y_steps.err" \
  || true

if field_available JobIDRaw && field_available JobName && \
   field_available WorkDir && field_available StdOut && field_available StdErr; then
  sacct \
    -u "$AUDIT_USER" \
    -S "$START_DATE" \
    -E "$END_DATE" \
    -j "$JOB_CSV" \
    -D -X -n -P \
    --format=JobIDRaw,JobName,WorkDir,StdOut,StdErr \
    >"$OUT_DIR/sacct/rutas_salida_por_job.psv" \
    2>"$OUT_DIR/sacct/rutas_salida_por_job.err" || true
fi

capture_command "$OUT_DIR/sacct/sacct_resumen_legible.txt" sacct \
  -u "$AUDIT_USER" -S "$START_DATE" -E "$END_DATE" -j "$JOB_CSV" -D -X \
  --format=JobID,JobName%50,Partition,State%25,Submit,Start,End,Elapsed,NodeList%30,ExitCode

section "Instantanea actual del controlador y de c1/c2/c3"
capture_command "$OUT_DIR/slurm_actual/sinfo_nodos.txt" sinfo -Nel
capture_command "$OUT_DIR/slurm_actual/sinfo_resumen.txt" sinfo -N -o '%N|%P|%c|%m|%G|%f|%t'
capture_command "$OUT_DIR/slurm_actual/scontrol_config.txt" scontrol show config
capture_command "$OUT_DIR/slurm_actual/scontrol_particiones.txt" scontrol show partitions -dd
capture_command "$OUT_DIR/slurm_actual/scontrol_jobs_retenidos.txt" scontrol show jobs -dd

IFS=',' read -r -a NODES <<<"$NODE_LIST"
for node in "${NODES[@]}"; do
  capture_command "$OUT_DIR/slurm_actual/scontrol_nodo_${node}.txt" \
    scontrol show node -dd "$node"
done

if [ "$RECOVER_BATCH_SCRIPTS" = "1" ]; then
  section "Intento de recuperar los batch scripts originales"
  printf 'JobID\tResultado\n' >"$OUT_DIR/batch_scripts_recuperados/resultado.tsv"
  for job_id in "${JOB_IDS[@]}"; do
    script_file="$OUT_DIR/batch_scripts_recuperados/job_${job_id}.sh"
    error_file="$OUT_DIR/batch_scripts_recuperados/job_${job_id}.err"
    if scontrol write batch_script "$job_id" "$script_file" 2>"$error_file"; then
      printf '%s\trecuperado\n' "$job_id" >>"$OUT_DIR/batch_scripts_recuperados/resultado.tsv"
    else
      printf '%s\tno_retenido_o_sin_permiso\n' "$job_id" >>"$OUT_DIR/batch_scripts_recuperados/resultado.tsv"
    fi
    sleep 0.10
  done
fi

if [ "$COPY_PROJECT_EVIDENCE" = "1" ] && [ -d "$PROJECT_ROOT" ]; then
  section "Inventario del repositorio y copia de evidencia textual preservada"
  {
    echo "tipo_evidencia=estado_actual_del_repositorio"
    echo "advertencia=no_prueba_el_commit_historico_de_cada_job"
    git -C "$PROJECT_ROOT" remote -v 2>/dev/null || true
    git -C "$PROJECT_ROOT" branch -a -vv 2>/dev/null || true
    git -C "$PROJECT_ROOT" status --short 2>/dev/null || true
    git -C "$PROJECT_ROOT" log --all --date=iso-strict \
      --pretty=format:'%H|%ad|%an|%D|%s' \
      --since='2026-05-01' --until='2026-08-01 23:59:59' 2>/dev/null || true
  } >"$OUT_DIR/proyecto_actual/git_actual_e_historial_candidato.txt"

  : >"$OUT_DIR/proyecto_actual/archivos_codigo_config_y_resultados.tsv"
  : >"$OUT_DIR/proyecto_actual/hashes_codigo_config_y_resultados.sha256"

  while IFS= read -r -d '' source_file; do
    relative_path="${source_file#"$PROJECT_ROOT"/}"
    size_bytes="$(stat -c '%s' "$source_file" 2>/dev/null || echo 0)"
    modified="$(stat -c '%y' "$source_file" 2>/dev/null || echo N_D)"
    printf '%s\t%s\t%s\n' "$relative_path" "$size_bytes" "$modified" \
      >>"$OUT_DIR/proyecto_actual/archivos_codigo_config_y_resultados.tsv"
    sha256sum "$source_file" \
      >>"$OUT_DIR/proyecto_actual/hashes_codigo_config_y_resultados.sha256" 2>/dev/null || true

    if [ "$size_bytes" -le 20971520 ]; then
      destination="$OUT_DIR/proyecto_actual/copias/$relative_path"
      mkdir -p "$(dirname "$destination")"
      cp -p -- "$source_file" "$destination" 2>/dev/null || true
    fi
  done < <(
    find "$PROJECT_ROOT" \
      \( -path '*/.git' -o -path '*/.venv' -o -path '*/venv' -o \
         -path '*/node_modules' -o -path '*/__pycache__' \) -prune -o \
      -type f \
      \( -name '*.sh' -o -name '*.py' -o -name 'requirements*.txt' -o \
         -name 'environment*.yml' -o -name 'environment*.yaml' -o \
         -name 'run_config*.json' -o -name '*metric*.json' -o \
         -name '*metric*.csv' -o -name '*params*.json' -o \
         -name '*params*.csv' -o -name '*config*.json' -o \
         -name '*config*.yml' -o -name '*config*.yaml' \) \
      -print0 2>/dev/null
  )

  while IFS= read -r -d '' log_file; do
    log_size="$(stat -c '%s' "$log_file" 2>/dev/null || echo 0)"
    if [ "$log_size" -le 104857600 ]; then
      relative_path="${log_file#"$PROJECT_ROOT"/}"
      destination="$OUT_DIR/logs_preservados/$relative_path"
      mkdir -p "$(dirname "$destination")"
      cp -p -- "$log_file" "$destination" 2>/dev/null || true
    fi
  done < <(
    find "$PROJECT_ROOT" \
      \( -path '*/.git' -o -path '*/.venv' -o -path '*/venv' -o \
         -path '*/node_modules' -o -path '*/__pycache__' \) -prune -o \
      -type f \
      \( -name 'slurm-*.out' -o -name 'slurm-*.err' -o -name '*.stderr' \) \
      -print0 2>/dev/null
  )

  if [ -s "$OUT_DIR/sacct/rutas_salida_por_job.psv" ]; then
    while IFS='|' read -r path_job_id path_job_name path_workdir path_stdout path_stderr; do
      for recorded_path in "$path_stdout" "$path_stderr"; do
        case "$recorded_path" in
          ''|'None'|'Unknown'|'N/A'|'null'|'(null)') continue ;;
        esac

        resolved_path="$recorded_path"
        resolved_path="${resolved_path//%j/$path_job_id}"
        resolved_path="${resolved_path//%J/$path_job_id}"
        resolved_path="${resolved_path//%A/$path_job_id}"
        resolved_path="${resolved_path//%x/$path_job_name}"
        resolved_path="${resolved_path//%u/$AUDIT_USER}"

        if [[ "$resolved_path" != /* ]]; then
          resolved_path="${path_workdir%/}/$resolved_path"
        fi

        if [ -f "$resolved_path" ]; then
          log_size="$(stat -c '%s' "$resolved_path" 2>/dev/null || echo 0)"
          if [ "$log_size" -le 104857600 ]; then
            destination="$OUT_DIR/logs_preservados/por_sacct/job_${path_job_id}/$(basename "$resolved_path")"
            mkdir -p "$(dirname "$destination")"
            cp -p -- "$resolved_path" "$destination" 2>/dev/null || true
          fi
        fi
      done
    done <"$OUT_DIR/sacct/rutas_salida_por_job.psv"
  fi
else
  printf 'Proyecto no encontrado o copia desactivada: %s\n' "$PROJECT_ROOT" \
    >"$OUT_DIR/proyecto_actual/NO_ENCONTRADO.txt"
fi

if [ "$RUN_NODE_INVENTORY" = "1" ] && command_exists srun; then
  section "Inventario actual de los nodos mediante trabajos Slurm breves"
  SRUN_WAIT=()
  if srun --help 2>&1 | grep -q -- '--immediate'; then
    SRUN_WAIT=(--immediate=60)
  fi

  for node in "${NODES[@]}"; do
    echo "Inventariando CPU/SO/RAM actuales de $node"
    srun "${SRUN_WAIT[@]}" \
      -p "$PARTITION" -w "$node" -N1 -n1 -c1 --mem=1G -t 00:05:00 \
      env SLURM_AUDIT_PROJECT_ROOT="$PROJECT_ROOT" \
      bash "$SCRIPT_PATH" --node-only \
      >"$OUT_DIR/inventario_actual_nodos/inventario_actual_${node}_cpu.txt" 2>&1 \
      || true

    node_description="$(scontrol show node "$node" -o 2>/dev/null || true)"
    if printf '%s\n' "$node_description" | grep -Eiq 'Gres=[^ ]*gpu'; then
      echo "Inventariando GPU actual asignada brevemente en $node"
      srun "${SRUN_WAIT[@]}" \
        -p "$PARTITION" -w "$node" -N1 -n1 -c1 --mem=1G -t 00:05:00 \
        --gres=gpu:1 \
        env SLURM_AUDIT_PROJECT_ROOT="$PROJECT_ROOT" \
        bash "$SCRIPT_PATH" --node-only \
        >"$OUT_DIR/inventario_actual_nodos/inventario_actual_${node}_gpu.txt" 2>&1 \
        || true
    fi
  done
fi

section "Manifest y paquete final"
find "$OUT_DIR" -type f ! -name 'manifest_sha256.txt' -print0 2>/dev/null |
  LC_ALL=C sort -z |
  xargs -0 -r sha256sum >"$OUT_DIR/manifest_sha256.txt" 2>/dev/null || true

ARCHIVE="${OUT_DIR}.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
sha256sum "$ARCHIVE" >"${ARCHIVE}.sha256"

printf '\nAuditoria terminada. Enviame estos dos archivos:\n'
printf '  %s\n' "$ARCHIVE"
printf '  %s.sha256\n' "$ARCHIVE"
