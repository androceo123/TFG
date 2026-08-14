#!/usr/bin/env bash
#SBATCH --job-name=duplicadosHarvard
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-duplicadosHarvard-%j.out

set -Eeuo pipefail

# Uso normal desde la raíz de waf-ml-starter:
#   sbatch duplicadosHarvard.sh
#
# Usa automáticamente:
#   entrada: data/raw/harvard/data_capec_multilabel.csv
#   salidas:
#     duplicadosHarvard/harvard_duplicados.csv
#     duplicadosHarvard/harvard_duplicados.xlsx
#
# También se pueden sobrescribir las rutas:
#   sbatch duplicadosHarvard.sh \
#     data/raw/harvard/data_capec_multilabel.csv \
#     duplicadosHarvard/harvard_duplicados.csv
#
# Controles opcionales:
#   STRICT_OFFICIAL=1  exige hash, esquema y conteos de la copia oficial.
#   FORCE=0            impide reemplazar salidas existentes.
#                      Por defecto las salidas derivadas se reemplazan para
#                      permitir ejecutar simplemente: sbatch este_script.sh
#   PYTHON_BIN=/ruta/python3.11  selecciona el intérprete.

usage() {
  echo "Uso: $0 [data_capec_multilabel.csv[.gz]] [harvard_duplicados.csv]" >&2
}

if (( $# > 2 )); then
  usage
  exit 2
fi

# Slurm copia el .sh a /var/spool/slurmd/jobXXXXX antes de ejecutarlo.
# Por eso el .py se resuelve desde el directorio donde se lanzó sbatch y no
# desde BASH_SOURCE[0], que dentro del nodo apunta a esa copia temporal.
project_dir="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
cd -- "${project_dir}"
python_script="${project_dir}/duplicadosHarvard.py"
source_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${python_script}" && -f "${source_script_dir}/duplicadosHarvard.py" ]]; then
  # Conveniencia para una ejecución local con rutas explícitas.
  python_script="${source_script_dir}/duplicadosHarvard.py"
fi

input_path="${1:-data/raw/harvard/data_capec_multilabel.csv}"
output_path="${2:-duplicadosHarvard/harvard_duplicados.csv}"
excel_output_path="${output_path%.*}.xlsx"

if [[ ! -f "${python_script}" ]]; then
  echo "[ERROR] No existe el programa Python: ${python_script}" >&2
  echo "[ERROR] Coloque el .py y el .sh en la raíz de waf-ml-starter." >&2
  exit 2
fi

if [[ ! -f "${input_path}" ]]; then
  echo "[ERROR] No existe el archivo de entrada: ${input_path}" >&2
  echo "[ERROR] Ejecute este .sh desde la raíz de waf-ml-starter." >&2
  exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x ".venv/bin/python3.11" ]]; then
  python_bin=".venv/bin/python3.11"
elif [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  echo "[ERROR] No se encontró Python 3. Defina PYTHON_BIN=/ruta/al/python." >&2
  exit 2
fi

if [[ ! -x "${python_bin}" ]] && ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "[ERROR] El intérprete no es ejecutable: ${python_bin}" >&2
  exit 2
fi

mkdir -p -- "$(dirname -- "${output_path}")"
mkdir -p -- "$(dirname -- "${excel_output_path}")"

python_args=(
  "${python_script}"
  --input "${input_path}"
  --output "${output_path}"
  --excel-output "${excel_output_path}"
  --sep auto
)

if [[ "${STRICT_OFFICIAL:-0}" == "1" ]]; then
  python_args+=(--strict-official)
fi

if [[ "${FORCE:-1}" == "1" ]]; then
  python_args+=(--force)
fi

echo "[AUDITORÍA] Entrada: ${input_path}"
echo "[AUDITORÍA] CSV:     ${output_path}"
echo "[AUDITORÍA] Excel:   ${excel_output_path}"
echo "[AUDITORÍA] Python:  ${python_bin}"
echo "[AUDITORÍA] No se eliminarán ni consolidarán filas del archivo fuente."

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  srun --ntasks=1 "${python_bin}" "${python_args[@]}"
else
  "${python_bin}" "${python_args[@]}"
fi

echo "[OK] CSV disponible en:   ${output_path}"
echo "[OK] Excel disponible en: ${excel_output_path}"
echo "[INFO] Use el Excel para revisión manual; contiene la hoja Diccionario_auditoria."
