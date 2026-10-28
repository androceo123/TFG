#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=ocsvm_registro_v2
#SBATCH --output=slurm-registro-v2-%j.out
#SBATCH --gres=gpu:1

# =============================================================================
# Experimento: OCSVM solo registro.jsp con 51 features (v2 char distribution)
#
# Motivacion:
#   Experimento anterior (nu05_entropy, 34 features): F1=0.457 -- igual que 26 feat.
#   El profesor confirmo: Nico/Ralf probaron "ambas formas" (por campo y por
#   toda la peticion). El libro (Tabla 2.3, pag. 17) detalla 10 features por
#   valor: 5 intervalos de distribucion + entropia + longitud + digitos + letras
#   + otros. La distribucion por intervalos captura la FORMA de la distribucion
#   (caida gradual normal vs caida abrupta en buffer overflow o SQLi).
#
#   Mejora implementada (application-independent):
#     - _char_dist_intervals(): 5 bins sobre frecuencias ordenadas descendente
#     - Aplicado a: query completo, body completo, param de mayor entropia (query),
#       param de mayor entropia (body)
#     - Estadisticas: mean + std de entropia entre params (no solo max)
#   Total: 34 -> 51 features, sin nombres de parametros (portabilidad total).
#
# Comparacion esperada:
#   Si F1 sube: los intervalos de distribucion explican el gap residual
#   Si F1 sigue igual: el gap es puramente arquitectural (per-group vs global)
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p data/tmp
mkdir -p resultsOptimo/csic_solo_registro/nu05_v2

# Verificar GPU
srun $PYTHON -c "import sys; print('Python:', sys.executable); import cuml.accel; cuml.accel.install(); print('GPU cuml.accel OK')"
nvidia-smi || true

# =============================================================================
# PASO 1: Re-extraer 51 features desde parquet existente (solo registro.jsp)
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 1: Re-extraer features v2 (51 features)"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
import pandas as pd
from pathlib import Path
from waf_ml.features.http_features import extract_http_features

def _to_bytes(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return b""
    s = str(v)
    if s in {"", "nan", "None"}:
        return b""
    return s.encode("utf-8", errors="ignore")

def _json_to_headers(s):
    if not s:
        return {}
    try:
        return json.loads(str(s))
    except Exception:
        return {}

src = Path("data/processed/csic/csic_features.parquet")
out = Path("data/tmp/csic_solo_registro_v2.parquet")

print(f"[STEP1] Leyendo parquet CSIC: {src}")
df = pd.read_parquet(src)
print(f"[STEP1] Total filas: {len(df)}")

mask = df["request_http_request"].str.contains("registro", case=False, na=False)
df_reg = df[mask].reset_index(drop=True)
print(f"[STEP1] Filas registro.jsp: {len(df_reg)}")

if "label_binary" in df_reg.columns:
    dist = df_reg["label_binary"].value_counts().sort_index()
    print(f"[STEP1] Distribucion label_binary:\n{dist.to_string()}")

print("[STEP1] Re-extrayendo 51 features...")
new_feats = df_reg.apply(
    lambda r: extract_http_features(
        method=str(r.get("request_http_method", "")),
        uri=str(r.get("request_http_request", "")),
        headers=_json_to_headers(r.get("request_headers_json", "")),
        body=_to_bytes(r.get("request_body", None)),
    ),
    axis=1,
)
feat_df = pd.DataFrame(list(new_feats))
print(f"[STEP1] Features extraidos: {len(feat_df.columns)}")

label_cols = [c for c in df_reg.columns if c.startswith("label_")]
meta_cols = [c for c in ["source_file", "dataset_name", "split", "label_type_raw"]
             if c in df_reg.columns and c not in label_cols]
keep_cols = label_cols + meta_cols

result = pd.concat([df_reg[keep_cols].reset_index(drop=True), feat_df], axis=1)
result.to_parquet(out, index=False)
print(f"[STEP1] Guardado: {out}  filas={len(result)}  columnas={len(result.columns)}")

# Estadisticas de los features nuevos (intervalos de distribucion)
new_cols = [c for c in feat_df.columns if "char_dist" in c or "mean_" in c or "std_" in c]
print(f"\n[STEP1] Estadisticas de los {len(new_cols)} features de distribucion (v2):")
print(result[new_cols].describe().round(4).to_string())
PY

echo "[PASO 1 COMPLETO]"

# =============================================================================
# PASO 2: OCSVM con 51 features, solo registro.jsp
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 2: OCSVM nu=0.05, gamma=0.1 con 51 features"
echo "======================================================"

PYTHONPATH=src srun $PYTHON src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data data/tmp/csic_solo_registro_v2.parquet \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --final-train-on train_normals \
  --tune none \
  --nu 0.05 \
  --gamma 0.1 \
  --require-gpu \
  --out resultsOptimo/csic_solo_registro/nu05_v2/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_solo_registro/nu05_v2/metrics.json \
  --pred-out resultsOptimo/csic_solo_registro/nu05_v2/pred.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_solo_registro/nu05_v2/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 0 \
  --benchmark-out resultsOptimo/csic_solo_registro/nu05_v2/benchmark.json

echo "[PASO 2 COMPLETO] Resultados en resultsOptimo/csic_solo_registro/nu05_v2/"

# =============================================================================
# PASO 3: Comparacion de los 4 escenarios de registro.jsp
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 3: Comparacion de escenarios registro.jsp"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

runs = [
    ("sin registro nu=0.05 (26 feat, MEJOR previo)",    "resultsOptimo/csic_sin_registro/nicoRalf_nu05/metrics.json"),
    ("SOLO registro nu=0.05 (26 feat)",                 "resultsOptimo/csic_solo_registro/nu05/metrics.json"),
    ("SOLO registro nu=0.05 (34 feat + entropia)",      "resultsOptimo/csic_solo_registro/nu05_entropy/metrics.json"),
    ("SOLO registro nu=0.05 (51 feat + char_dist v2)",  "resultsOptimo/csic_solo_registro/nu05_v2/metrics.json"),
]

def get_m(path, key):
    p = Path(path)
    if not p.exists():
        return "N/A"
    with open(p) as f:
        d = json.load(f)
    for sub in ("evaluation", "eval", "metrics", "test"):
        if sub in d and isinstance(d[sub], dict) and key in d[sub]:
            return f"{d[sub][key]:.4f}"
    v = d.get(key)
    if v is not None and not isinstance(v, dict):
        return f"{v:.4f}"
    return "N/A"

print(f"\n{'Experimento':<52} {'F1':>8} {'Recall':>8} {'Precision':>10} {'FPR':>8} {'BAcc':>8}")
print("-" * 100)
for name, path in runs:
    f1  = get_m(path, "f1")
    rec = get_m(path, "recall")
    pre = get_m(path, "precision")
    fpr = get_m(path, "fpr")
    ba  = get_m(path, "balanced_accuracy")
    print(f"{name:<52} {f1:>8} {rec:>8} {pre:>10} {fpr:>8} {ba:>8}")

print()
print("Referencia Nico/Ralf (OCS-WAF 2017) grupos c12/c13/t01:")
print("  F1=0.95   Recall(TPR)=0.93   FPR=0.03")
print()
print("INTERPRETACION:")
print("  51 feat >> 34 feat  =>  los intervalos de distribucion capturan el gap")
print("  51 feat ~= 34 feat  =>  el gap es puramente arquitectural (per-group vs global)")
PY

# =============================================================================
# PASO 4: Top features por SHAP en v2
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 4: Top 15 features SHAP (v2)"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import pandas as pd
from pathlib import Path

fi_path = Path("resultsOptimo/csic_solo_registro/nu05_v2/feature_importance_shap.csv")
if fi_path.exists():
    fi = pd.read_csv(fi_path)
    print("\nTop 15 features por importancia SHAP (nu05_v2, 51 features):")
    print(fi.head(15).to_string(index=False))
    # Filtrar solo los features nuevos de distribucion
    new_feats = fi[fi["feature"].str.contains("char_dist|mean_|std_body|std_param|max_body_param")]
    if not new_feats.empty:
        print(f"\nFeatures de distribucion v2 que aparecen en el ranking:")
        print(new_feats.to_string(index=False))
    else:
        print("\n[INFO] Ningun feature de distribucion v2 aparece en el top SHAP.")
else:
    print(f"[WARN] No encontrado: {fi_path}")
PY
