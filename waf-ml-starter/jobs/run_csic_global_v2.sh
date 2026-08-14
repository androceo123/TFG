#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --job-name=ocsvm_global_v2
#SBATCH --output=slurm-global-v2-%j.out
#SBATCH --gres=gpu:1

# =============================================================================
# Experimento: OCSVM global sin registro, 51 features (v2 char distribution)
#
# Motivacion:
#   Los experimentos sobre solo registro.jsp muestran F1=0.457 invariante con
#   26, 34 o 51 features. El gap vs Nico/Ralf es arquitectural para ese subconjunto.
#   Este experimento prueba si los 51 features mejoran el modelo GLOBAL completo
#   (86.864 filas sin registro.jsp), que previamente obtuvo F1=0.787 con 26 features.
#
#   Si los intervalos de distribucion de caracteres ayudan al modelo global,
#   veremos F1 > 0.787. Si el resultado es igual, la conclusion es que ninguna
#   ingenieria de features (application-independent) supera el limite del modelo
#   global one-class, y el gap restante vs Nico/Ralf es puramente arquitectural.
#
# Estrategia:
#   Re-extrae 51 features desde los campos raw del parquet existente
#   (request_http_method, request_http_request, request_body, request_headers_json).
#   No necesita los archivos .txt originales.
# =============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

PYTHON=".venv/bin/python3.11"

mkdir -p data/tmp
mkdir -p resultsOptimo/csic_sin_registro/nu05_v2

# Verificar GPU
srun $PYTHON -c "import sys; print('Python:', sys.executable); import cuml.accel; cuml.accel.install(); print('GPU cuml.accel OK')"
nvidia-smi || true

# =============================================================================
# PASO 1: Re-extraer 51 features del dataset global sin registro
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 1: Re-extraer 51 features (86.864 filas)"
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
out = Path("data/tmp/csic_sin_registro_v2.parquet")

print(f"[STEP1] Leyendo parquet CSIC completo: {src}")
df = pd.read_parquet(src)
print(f"[STEP1] Total filas: {len(df)}")

# Filtrar sin registro (igual que el experimento de recalibracion)
mask = df["request_http_request"].str.contains("registro", case=False, na=False)
df_fil = df[~mask].reset_index(drop=True)
print(f"[STEP1] Filas sin registro.jsp: {len(df_fil)}")

if "label_binary" in df_fil.columns:
    dist = df_fil["label_binary"].value_counts().sort_index()
    print(f"[STEP1] Distribucion label_binary:\n{dist.to_string()}")

print("[STEP1] Re-extrayendo 51 features (puede tardar varios minutos)...")
new_feats = df_fil.apply(
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

label_cols = [c for c in df_fil.columns if c.startswith("label_")]
meta_cols = [c for c in ["source_file", "dataset_name", "split", "label_type_raw"]
             if c in df_fil.columns and c not in label_cols]
keep_cols = label_cols + meta_cols

result = pd.concat([df_fil[keep_cols].reset_index(drop=True), feat_df], axis=1)
result.to_parquet(out, index=False)
print(f"[STEP1] Guardado: {out}  filas={len(result)}  columnas={len(result.columns)}")

# Estadisticas de los features de distribucion
dist_cols = [c for c in feat_df.columns if "char_dist" in c]
print(f"\n[STEP1] Estadisticas de los 20 intervalos de distribucion:")
print(result[dist_cols].describe().round(4).to_string())
PY

echo "[PASO 1 COMPLETO]"

# =============================================================================
# PASO 2: OCSVM global con 51 features, nu=0.05, gamma=0.1
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 2: OCSVM global nu=0.05, gamma=0.1, 51 features"
echo "======================================================"

PYTHONPATH=src srun $PYTHON src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data data/tmp/csic_sin_registro_v2.parquet \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --final-train-on train_normals \
  --tune none \
  --nu 0.05 \
  --gamma 0.1 \
  --require-gpu \
  --out resultsOptimo/csic_sin_registro/nu05_v2/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/nu05_v2/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/nu05_v2/pred.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_sin_registro/nu05_v2/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 0 \
  --benchmark-out resultsOptimo/csic_sin_registro/nu05_v2/benchmark.json

echo "[PASO 2 COMPLETO] Resultados en resultsOptimo/csic_sin_registro/nu05_v2/"

# =============================================================================
# PASO 3: Comparacion global 26 feat vs 51 feat
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 3: Comparacion modelo global (26 vs 51 features)"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import json
from pathlib import Path

runs = [
    ("BASE nu=0.001 auto-tuning (26 feat)",          "resultsOptimo/csic_sin_registro/oneclass/metrics.json"),
    ("GLOBAL nu=0.05, gamma=0.1 (26 feat) [MEJOR]",  "resultsOptimo/csic_sin_registro/nicoRalf_nu05/metrics.json"),
    ("GLOBAL nu=0.05, gamma=0.1 (51 feat v2)",        "resultsOptimo/csic_sin_registro/nu05_v2/metrics.json"),
    ("SOLO registro nu=0.05 (51 feat v2)",            "resultsOptimo/csic_solo_registro/nu05_v2/metrics.json"),
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

print(f"\n{'Experimento':<48} {'F1':>8} {'Recall':>8} {'Precision':>10} {'FPR':>8} {'BAcc':>8}")
print("-" * 96)
for name, path in runs:
    f1  = get_m(path, "f1")
    rec = get_m(path, "recall")
    pre = get_m(path, "precision")
    fpr = get_m(path, "fpr")
    ba  = get_m(path, "balanced_accuracy")
    print(f"{name:<48} {f1:>8} {rec:>8} {pre:>10} {fpr:>8} {ba:>8}")

print()
print("Referencia Nico/Ralf (OCS-WAF 2017, per-group):")
print("  F1=0.95   Recall(TPR)=0.93   FPR=0.03")
print()
print("INTERPRETACION:")
print("  51 feat >> 26 feat global  =>  los features v2 mejoran el modelo global")
print("  51 feat ~= 26 feat global  =>  limite del modelo one-class global alcanzado")
PY

# =============================================================================
# PASO 4: SHAP del modelo global v2 -- ver si char_dist aparece
# =============================================================================
echo ""
echo "======================================================"
echo " PASO 4: Top 20 features SHAP (modelo global v2)"
echo "======================================================"

PYTHONPATH=src srun $PYTHON - <<'PY'
import pandas as pd
from pathlib import Path

fi_path = Path("resultsOptimo/csic_sin_registro/nu05_v2/feature_importance_shap.csv")
if fi_path.exists():
    fi = pd.read_csv(fi_path)
    print("\nTop 20 features SHAP (global nu05_v2, 51 features):")
    print(fi.head(20).to_string(index=False))

    new_feats = fi[fi["feature"].str.contains("char_dist|mean_|std_param|max_body_param")]
    if not new_feats.empty:
        print(f"\nFeatures de distribucion v2 en el ranking:")
        print(new_feats.to_string(index=False))
    else:
        print("\n[INFO] Ningun feature de distribucion v2 aparece en el top SHAP.")
else:
    print(f"[WARN] No encontrado: {fi_path}")
PY
