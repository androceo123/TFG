#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --job-name=csic_supervised_binary
#SBATCH --output=slurm-csic-supervised-binary-%j.out

# =============================================================================
# Experimento: Clasificador supervisado binario sobre CSIC 2010
#
# Motivacion:
#   El OCSVM global (one-class, application-independent) alcanza F1=0.787 con
#   57 features. Ninguna ingenieria de features adicional mejora este resultado.
#   El limite es el PARADIGMA, no los features:
#     - One-class: aprende solo de normales, no puede usar la senal de ataques
#     - Supervisado: aprende la frontera normal/ataque directamente
#
#   CSIC 2010 tiene etiquetas binarias (label_binary: 0=normal, 1=ataque).
#   Este experimento responde PI-1 del protocolo experimental:
#     "Que diferencia de desempeno se observa entre OCSVM calibrado y
#      modelos supervisados sobre los mismos datos HTTP?"
#
#   Usa exactamente los mismos 57 features y el mismo parquet que el
#   experimento OCSVM global (csic_sin_registro_v2.parquet).
#
# Sin GPU: LogReg es CPU, corre rapido (~2-5 min para 86.864 filas)
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p resultsOptimo/csic_sin_registro/supervised_binary

# Verificar que el parquet v2 existe
PYTHONPATH=src srun $PYTHON - <<'PY'
from pathlib import Path
import pandas as pd

p = Path("data/tmp/csic_sin_registro_v2.parquet")
if not p.exists():
    raise FileNotFoundError(
        f"No encontrado: {p}\n"
        "Ejecutar primero: sbatch jobs/run_csic_global_v2.sh"
    )
df = pd.read_parquet(p)
print(f"[CHECK] Parquet OK: {len(df)} filas, {len(df.columns)} columnas")
print(f"[CHECK] label_binary: {df['label_binary'].value_counts().sort_index().to_dict()}")
PY

echo ""
echo "======================================================"
echo " Entrenando LogReg binario (57 features, C=1.0)"
echo "======================================================"

PYTHONPATH=src srun $PYTHON src/waf_ml/scripts/train_supervised.py \
  --data data/tmp/csic_sin_registro_v2.parquet \
  --task multiclass \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --C 1.0 \
  --class-weight balanced \
  --max-iter 2000 \
  --tune none \
  --fi-kind coef \
  --fi-topk 25 \
  --out resultsOptimo/csic_sin_registro/supervised_binary/model.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/supervised_binary/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/supervised_binary/pred.csv \
  --benchmark-mode auto \
  --benchmark-max-rows 5000 \
  --benchmark-out resultsOptimo/csic_sin_registro/supervised_binary/benchmark.json

echo "[ENTRENAMIENTO COMPLETO]"

# =============================================================================
# Comparacion final: OCSVM vs supervisado, mismos datos y features
# =============================================================================
echo ""
echo "======================================================"
echo " Comparacion definitiva: paradigma one-class vs supervisado"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

def get_m(path, key):
    p = Path(path)
    if not p.exists():
        return "N/A"
    with open(p) as f:
        d = json.load(f)
    for sub in ("evaluation", "eval", "metrics", "test"):
        if sub in d and isinstance(d[sub], dict) and key in d[sub]:
            v = d[sub][key]
            if isinstance(v, (int, float)):
                return f"{v:.4f}"
            # para metricas por clase (dict), buscar clase "1" o "attack"
            if isinstance(v, dict):
                for cls in ("1", "attack", "anomaly"):
                    if cls in v:
                        return f"{v[cls]:.4f}"
    v = d.get(key)
    if v is not None and not isinstance(v, dict):
        return f"{float(v):.4f}"
    return "N/A"

runs = [
    ("OCSVM one-class  nu=0.001 auto-tuning (26 feat)", "resultsOptimo/csic_sin_registro/oneclass/metrics.json"),
    ("OCSVM one-class  nu=0.05, gamma=0.1  (26 feat)",  "resultsOptimo/csic_sin_registro/nicoRalf_nu05/metrics.json"),
    ("OCSVM one-class  nu=0.05, gamma=0.1  (57 feat)",  "resultsOptimo/csic_sin_registro/nu05_v2/metrics.json"),
    ("LogReg supervisado binario            (57 feat)",  "resultsOptimo/csic_sin_registro/supervised_binary/metrics.json"),
]

print(f"\n{'Experimento':<50} {'F1':>8} {'Recall':>8} {'Prec.':>8} {'FPR':>8} {'BAcc':>8}")
print("-" * 96)
for name, path in runs:
    f1  = get_m(path, "f1")
    rec = get_m(path, "recall")
    pre = get_m(path, "precision")
    fpr = get_m(path, "fpr")
    ba  = get_m(path, "balanced_accuracy")
    print(f"{name:<50} {f1:>8} {rec:>8} {pre:>8} {fpr:>8} {ba:>8}")

print()
print("Referencia Nico/Ralf (OCSVM per-group, app-dependent):")
print("  F1=0.95  Recall=0.93  FPR=0.03")
print()
print("INTERPRETACION:")
print("  Supervisado >> OCSVM  =>  el limite era el paradigma one-class, no los features")
print("  Supervisado ~= OCSVM  =>  los features son insuficientes para separar las clases")
PY
