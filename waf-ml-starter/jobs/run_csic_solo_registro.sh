#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=ocsvm_solo_registro
#SBATCH --output=slurm-solo-registro-%j.out
#SBATCH --gres=gpu:1

# =============================================================================
# Experimento: OCSVM entrenado SOLO sobre peticiones de registro.jsp
#
# Motivacion:
#   Nico/Ralf (OCS-WAF 2017) entrenaban un modelo POR GRUPO (URL+metodo).
#   Los grupos c12, c13 y t01 corresponden exactamente a /registro.jsp.
#   Ellos lograron TPR=0.93, FPR=0.03, F1=0.95.
#
#   Al entrenar con el dataset completo (mezcla de todos los endpoints),
#   nuestro modelo tiene mucha mayor varianza en los datos y la frontera
#   de decision es menos precisa.
#
#   Este experimento replica parcialmente su enfoque:
#   - Filtra solo las filas de registro.jsp del parquet CSIC completo
#   - Entrena OCSVM con nu=0.05, gamma=0.1 (mejor calibracion encontrada)
#   - Compara con los resultados de Nico/Ralf para esos mismos grupos
#
# Si F1 se acerca a 0.95 => el problema era la heterogeneidad de datos
# Si F1 sigue bajo => el problema son los features (ver Paso B)
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p data/tmp
mkdir -p resultsOptimo/csic_solo_registro/nu05

# Verificar GPU
srun $PYTHON -c "import sys; print('Python:', sys.executable); import cuml.accel; cuml.accel.install(); print('GPU cuml.accel OK')"
nvidia-smi || true

# =============================================================================
# PASO 1: Filtrar parquet completo CSIC para quedarse SOLO con registro.jsp
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 1: Filtrar filas de registro.jsp"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import pandas as pd
from pathlib import Path

src = Path("data/processed/csic/csic_features.parquet")
out = Path("data/tmp/csic_solo_registro.parquet")

df = pd.read_parquet(src)
print(f"[FILTER] Dataset completo CSIC: {len(df)} filas")

# Filtrar filas cuya URI contiene "registro"
# La columna request_http_request contiene la URI raw (path + query)
mask = df["request_http_request"].str.contains("registro", case=False, na=False)
df_registro = df[mask].reset_index(drop=True)

print(f"[FILTER] Filas con 'registro' en URI: {len(df_registro)}")

if "label_binary" in df_registro.columns:
    dist = df_registro["label_binary"].value_counts().sort_index()
    print(f"[FILTER] Distribucion label_binary:")
    print(dist.to_string())

if len(df_registro) < 50:
    raise RuntimeError(f"Muy pocas filas de registro ({len(df_registro)}). Verificar columna URI.")

out.parent.mkdir(parents=True, exist_ok=True)
df_registro.to_parquet(out, index=False)
print(f"[FILTER] Guardado en: {out}")
PY

echo "[PASO 1 COMPLETO]"

# =============================================================================
# PASO 2: Entrenar OCSVM solo con datos de registro.jsp
#         nu=0.05, gamma=0.1 (mejor calibracion del experimento anterior)
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 2: Entrenar OCSVM nu=0.05, gamma=0.1 (solo registro.jsp)"
echo "======================================================"

PYTHONPATH=src srun $PYTHON src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data data/tmp/csic_solo_registro.parquet \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --final-train-on train_normals \
  --tune none \
  --nu 0.05 \
  --gamma 0.1 \
  --require-gpu \
  --out resultsOptimo/csic_solo_registro/nu05/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_solo_registro/nu05/metrics.json \
  --pred-out resultsOptimo/csic_solo_registro/nu05/pred.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_solo_registro/nu05/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 0 \
  --benchmark-out resultsOptimo/csic_solo_registro/nu05/benchmark.json

echo "[PASO 2 COMPLETO] Resultados en resultsOptimo/csic_solo_registro/nu05/"

# =============================================================================
# PASO 3: Comparacion con resultados anteriores y referencia Nico/Ralf
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 3: Comparacion de resultados"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

runs = [
    ("BASE completo nu=0.001 (con registro)",    "resultsOptimo/csic/oneclass/metrics.json"),
    ("BASE sin registro nu=0.001",               "resultsOptimo/csic_sin_registro/oneclass/metrics.json"),
    ("sin registro nu=0.05 (mejor calibrado)",   "resultsOptimo/csic_sin_registro/nicoRalf_nu05/metrics.json"),
    ("SOLO registro nu=0.05 (este experimento)", "resultsOptimo/csic_solo_registro/nu05/metrics.json"),
]

def get_m(path, key):
    p = Path(path)
    if not p.exists():
        return "N/A"
    with open(p) as f:
        d = json.load(f)
    # buscar en evaluation o directamente
    for sub in ("evaluation", "eval", "metrics", "test"):
        if sub in d and isinstance(d[sub], dict) and key in d[sub]:
            return f"{d[sub][key]:.4f}"
    v = d.get(key)
    if v is not None and not isinstance(v, dict):
        return f"{v:.4f}"
    return "N/A"

print(f"\n{'Experimento':<45} {'F1':>8} {'Recall':>8} {'Precision':>10} {'FPR':>8} {'BAcc':>8}")
print("-" * 92)
for name, path in runs:
    f1  = get_m(path, "f1")
    rec = get_m(path, "recall")
    pre = get_m(path, "precision")
    fpr = get_m(path, "fpr")
    ba  = get_m(path, "balanced_accuracy")
    print(f"{name:<45} {f1:>8} {rec:>8} {pre:>10} {fpr:>8} {ba:>8}")

print()
print("Referencia Nico/Ralf (OCS-WAF 2017) grupos c12/c13/t01:")
print("  F1=0.95   Recall(TPR)=0.93   FPR=0.03")
print()
print("INTERPRETACION:")
print("  Si F1 solo_registro >> F1 sin_registro  =>  el gap era por heterogeneidad de datos (arquitectura)")
print("  Si F1 solo_registro ~= F1 sin_registro  =>  el gap es por diferencia de features")
PY
