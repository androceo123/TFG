#!/usr/bin/env bash
# Submit helper para Arandu/SLURM.
# Ejecutar en el login node, NO con sbatch directo:
#   ./submit_10claseResults_best_node.sh
#
# Este wrapper mira los nodos de la partición, elige el que tenga más recursos libres
# para el job 10claseResults.sh, y recién después llama a sbatch --nodelist=<nodo>.

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home_data/aroman/TFG/waf-ml-starter}"
JOB_SCRIPT="${JOB_SCRIPT:-${PROJECT_DIR}/10claseResults.sh}"
PARTITION="${PARTITION:-normal}"

# Candidatos opcionales. Si está vacío, se descubren desde sinfo -p ${PARTITION}.
# Ejemplo: NODES_CANDIDATES="c1 c2 c3" ./submit_10claseResults_best_node.sh
NODES_CANDIDATES="${NODES_CANDIDATES:-}"

# Nodos a evitar. Ejemplo: EXCLUDE_NODES="c2 c3" ./submit_10claseResults_best_node.sh
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

# Si no se fijan, se leen desde las líneas #SBATCH del job script.
REQ_CPUS="${REQ_CPUS:-}"
REQ_MEM_MB="${REQ_MEM_MB:-}"

# Args extra para sbatch. Ejemplo: EXTRA_SBATCH_ARGS="--qos=debug" ./submit_10claseResults_best_node.sh
EXTRA_SBATCH_ARGS="${EXTRA_SBATCH_ARGS:-}"

# Si DRY_RUN=1, muestra qué haría pero no envía el job.
DRY_RUN="${DRY_RUN:-0}"

mem_to_mb() {
  local raw="${1:-}"
  raw="${raw//[[:space:]]/}"
  if [[ -z "${raw}" ]]; then
    echo "0"
    return 0
  fi
  # Soporta formatos típicos de SLURM: 64G, 64000M, 65536, 1T.
  awk -v s="${raw}" 'BEGIN {
    gsub(/B$/, "", s)
    unit = substr(s, length(s), 1)
    val = s
    if (unit ~ /[KkMmGgTt]/) val = substr(s, 1, length(s)-1)
    if (val == "") val = 0
    val = val + 0
    if (unit ~ /[Kk]/) printf "%d\n", int((val / 1024) + 0.999)
    else if (unit ~ /[Mm]/) printf "%d\n", int(val)
    else if (unit ~ /[Gg]/) printf "%d\n", int(val * 1024)
    else if (unit ~ /[Tt]/) printf "%d\n", int(val * 1024 * 1024)
    else printf "%d\n", int(val)
  }'
}

extract_sbatch_value() {
  local key="$1"
  local file="$2"
  awk -v key="${key}" '
    $1 == "#SBATCH" {
      for (i = 2; i <= NF; i++) {
        if ($i ~ "^" key "=") {
          sub("^" key "=", "", $i)
          print $i
          exit
        }
      }
    }
  ' "${file}"
}

contains_word() {
  local needle="$1"
  local haystack="$2"
  for x in ${haystack}; do
    [[ "${x}" == "${needle}" ]] && return 0
  done
  return 1
}

field_value() {
  local line="$1"
  local key="$2"
  sed -n "s/.*${key}=\([^ ]*\).*/\1/p" <<< "${line}" | head -n1
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "[ERROR] Este helper debe ejecutarse en el login node antes de crear el job, no dentro de un job SLURM." >&2
  echo "[ERROR] Usá: ./submit_10claseResults_best_node.sh" >&2
  exit 2
fi

for cmd in sinfo scontrol sbatch awk sed; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ERROR] No encuentro comando requerido: ${cmd}" >&2
    exit 2
  fi
done

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "[ERROR] No existe JOB_SCRIPT=${JOB_SCRIPT}" >&2
  echo "[ERROR] Copiá 10claseResults.sh a ${PROJECT_DIR} o exportá JOB_SCRIPT=/ruta/al/10claseResults.sh" >&2
  exit 2
fi

if [[ -z "${REQ_CPUS}" ]]; then
  REQ_CPUS="$(extract_sbatch_value "--cpus-per-task" "${JOB_SCRIPT}")"
  REQ_CPUS="${REQ_CPUS:-8}"
fi

if [[ -z "${REQ_MEM_MB}" ]]; then
  req_mem_raw="$(extract_sbatch_value "--mem" "${JOB_SCRIPT}")"
  req_mem_raw="${req_mem_raw:-64G}"
  REQ_MEM_MB="$(mem_to_mb "${req_mem_raw}")"
fi

if [[ -z "${NODES_CANDIDATES}" ]]; then
  # -N lista nodos individualmente en la mayoría de SLURM; show hostnames expande rangos si aparecen.
  NODES_CANDIDATES="$(sinfo -N -h -p "${PARTITION}" -o "%N" | xargs -r scontrol show hostnames | tr '\n' ' ')"
fi

if [[ -z "${NODES_CANDIDATES}" ]]; then
  echo "[ERROR] No pude descubrir nodos en partition=${PARTITION}" >&2
  exit 2
fi

printf '[INFO] Partition=%s\n' "${PARTITION}"
printf '[INFO] Job script=%s\n' "${JOB_SCRIPT}"
printf '[INFO] Requiere aprox: CPUs=%s MEM=%s MB\n' "${REQ_CPUS}" "${REQ_MEM_MB}"
printf '[INFO] Candidatos: %s\n' "${NODES_CANDIDATES}"
[[ -n "${EXCLUDE_NODES}" ]] && printf '[INFO] Excluidos: %s\n' "${EXCLUDE_NODES}"
echo
printf '%-8s %-14s %9s %9s %9s %9s %9s %10s %s\n' "NODE" "STATE" "CPUAlloc" "CPUTot" "IdleCPU" "AllocMem" "RealMem" "AvailMem" "Decision"

best_node=""
best_score="-999999999999999"

for node in ${NODES_CANDIDATES}; do
  if contains_word "${node}" "${EXCLUDE_NODES}"; then
    line="$(scontrol show node -o "${node}" 2>/dev/null || true)"
    state="$(field_value "${line}" "State")"
    printf '%-8s %-14s %9s %9s %9s %9s %9s %10s %s\n' "${node}" "${state:-?}" "-" "-" "-" "-" "-" "-" "excluido"
    continue
  fi

  line="$(scontrol show node -o "${node}" 2>/dev/null || true)"
  if [[ -z "${line}" ]]; then
    printf '%-8s %-14s %9s %9s %9s %9s %9s %10s %s\n' "${node}" "?" "-" "-" "-" "-" "-" "-" "sin_info"
    continue
  fi

  state="$(field_value "${line}" "State")"
  cpu_alloc="$(field_value "${line}" "CPUAlloc")"
  cpu_tot="$(field_value "${line}" "CPUTot")"
  cpu_load="$(field_value "${line}" "CPULoad")"
  real_mem="$(field_value "${line}" "RealMemory")"
  alloc_mem="$(field_value "${line}" "AllocMem")"

  cpu_alloc="${cpu_alloc:-0}"
  cpu_tot="${cpu_tot:-0}"
  cpu_load="${cpu_load:-9999}"
  real_mem="${real_mem:-0}"
  alloc_mem="${alloc_mem:-0}"

  idle_cpu=$(( cpu_tot - cpu_alloc ))
  avail_mem=$(( real_mem - alloc_mem ))
  decision="ok"

  if [[ "${state}" =~ DOWN|DRAIN|DRAINING|FAIL|MAINT|RESERVED|POWER|INVAL|UNKNOWN ]]; then
    decision="estado_no_apto"
  elif (( idle_cpu < REQ_CPUS )); then
    decision="pocas_cpu"
  elif (( avail_mem < REQ_MEM_MB )); then
    decision="poca_mem_slurm"
  fi

  printf '%-8s %-14s %9s %9s %9s %9s %9s %10s %s\n' \
    "${node}" "${state}" "${cpu_alloc}" "${cpu_tot}" "${idle_cpu}" "${alloc_mem}" "${real_mem}" "${avail_mem}" "${decision}"

  if [[ "${decision}" == "ok" ]]; then
    score="$(awk -v idle="${idle_cpu}" -v avail="${avail_mem}" -v load="${cpu_load}" 'BEGIN {
      if (load == "N/A" || load == "") load = 9999
      # Prioriza CPU libre, luego memoria asignable SLURM, y penaliza carga actual.
      printf "%.6f\n", (idle * 1000000000.0) + (avail * 1000.0) - (load * 100000.0)
    }')"
    if awk -v a="${score}" -v b="${best_score}" 'BEGIN { exit !(a > b) }'; then
      best_score="${score}"
      best_node="${node}"
    fi
  fi
done

echo
if [[ -z "${best_node}" ]]; then
  echo "[WARN] No encontré un nodo que cumpla CPUs=${REQ_CPUS} y MEM=${REQ_MEM_MB}MB."
  echo "[WARN] Envío sin --nodelist para que SLURM decida o lo deje en cola."
  # shellcheck disable=SC2206
  extra=( ${EXTRA_SBATCH_ARGS} )
  cmd=(sbatch -p "${PARTITION}" --export=ALL "${extra[@]}" "${JOB_SCRIPT}")
else
  echo "[OK] Nodo elegido: ${best_node}"
  # shellcheck disable=SC2206
  extra=( ${EXTRA_SBATCH_ARGS} )
  cmd=(sbatch -p "${PARTITION}" --export=ALL "${extra[@]}" --nodelist="${best_node}" "${JOB_SCRIPT}")
fi

printf '[RUN]'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[OK] DRY_RUN=1: no envío el job."
  exit 0
fi

"${cmd[@]}"
