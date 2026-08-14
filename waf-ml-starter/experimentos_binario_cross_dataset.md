# Experimentos Binarios Cross-Dataset

**Fecha:** 2026-06-08 | **Branch:** mejoras-profesor-ja

---

## Objetivo

Verificar si el mismo pipeline RF que obtuvo **F1=0.947 en CSIC 2010** mantiene
buenas métricas al aplicarse sin cambios sobre **TorpEda** y **Harvard SR-BH 2020**.

Misma lógica experimental que PI-1, ahora extendida para responder:
> *"¿Generaliza el pipeline application-independent entre datasets?"*

---

## Prerrequisitos

Antes de empezar, verificar que existen estos archivos:

```bash
# Desde waf-ml-starter/
ls data/processed/torpeda/torpeda_features.parquet   # debe existir
ls data/processed/harvard/harvard.parquet            # debe existir
ls data/tmp/csic_sin_registro_v2.parquet             # debe existir (CSIC baseline)
ls resultsOptimo/csic_sin_registro/rf_supervised/pred.csv  # resultados CSIC
```

Si `csic_sin_registro_v2.parquet` no existe:
```bash
sbatch jobs/run_csic_global_v2.sh   # reconstruye el parquet CSIC
```

---

## Archivos creados

| Archivo | Descripción |
|---------|-------------|
| `src/waf_ml/scripts/train_binary_rf.py` | Script Python reutilizable — RF binario agnóstico al dataset |
| `jobs/run_torpeda_binary_rf.sh` | Job SLURM para TorpEda (30 min, 16 GB) |
| `jobs/run_harvard_binary_rf.sh` | Job SLURM para Harvard (1.5h, 32 GB) |
| `jobs/run_compare_binary.sh` | Job SLURM tabla comparativa final (10 min) |

---

## Orden de ejecución

### Paso 1 — TorpEda (lanzar primero, es más rápido)

```bash
cd waf-ml-starter
sbatch jobs/run_torpeda_binary_rf.sh
```

**Qué hace:**
1. Verifica el parquet `data/processed/torpeda/torpeda_features.parquet`
2. Re-extrae las 57 features desde las columnas HTTP crudas (paralelo, 8 workers, ~2-3 min)
3. Entrena RF: 500 árboles, balanced, seed=42, split 80/20 (74K filas → rápido)
4. Threshold sweep 0.05-0.95, selecciona thr óptimo por max F1
5. Imprime comparativa CSIC vs TorpEda en el log

**Salidas:**
```
results/torpeda/binary/
    model.joblib
    pred.csv
    metrics.json                  ← thr=0.50
    confusion_matrix.csv
    classification_report.txt
    feature_importance_rf.csv
    features_used.csv
    model_config.json
results/torpeda/binary_threshold/
    metrics.json                  ← thr=OPTIMO (el importante)
    threshold_sweep.csv
    confusion_matrix.csv
    classification_report.txt
```

**Tiempo estimado:** ~10-15 min

---

### Paso 2 — Harvard/SR-BH (lanzar en paralelo o después)

```bash
sbatch jobs/run_harvard_binary_rf.sh
```

**Qué hace:**
1. Verifica `data/processed/harvard/harvard.parquet` (907K filas)
2. Re-extrae las 57 features en paralelo (8 workers, ~10-15 min)
3. Guarda parquet enriquecido en `data/tmp/harvard_binary_v2.parquet`
4. Entrena RF (misma config exacta que CSIC y TorpEda)
5. Threshold sweep + métricas completas

**Salidas:**
```
data/tmp/harvard_binary_v2.parquet     ← parquet con 57 features
results/harvard/binary/
    model.joblib
    pred.csv
    metrics.json
    confusion_matrix.csv
    classification_report.txt
    feature_importance_rf.csv
    features_used.csv
    model_config.json
results/harvard/binary_threshold/
    metrics.json                  ← thr=OPTIMO
    threshold_sweep.csv
    confusion_matrix.csv
    classification_report.txt
```

**Tiempo estimado:** ~40-60 min

---

### Paso 3 — Tabla comparativa final (después de Paso 1 y 2)

```bash
sbatch jobs/run_compare_binary.sh
```

**Qué hace:**
- Carga `pred.csv` de los 3 datasets (CSIC + TorpEda + Harvard)
- Recomputa **todas** las métricas desde cero para consistencia total
- Genera tabla con: accuracy, balanced\_acc, F1, recall, precision, ROC-AUC, PR-AUC, MCC, FPR, FNR
- Muestra matrices de confusión para cada dataset
- Guarda `results/comparison_binary_rf.csv`

**Tiempo estimado:** ~5 min

---

## Cómo monitorear los jobs

```bash
# Ver estado de todos tus jobs
squeue -u $USER

# Ver log en tiempo real (reemplazar JOBID)
tail -f slurm-torpeda-binary-rf-JOBID.out
tail -f slurm-harvard-binary-rf-JOBID.out

# Ver si terminaron correctamente (última línea del log)
tail -1 slurm-torpeda-binary-rf-JOBID.out
# debe mostrar: [EXPERIMENTO TORPEDA RF COMPLETO]
```

---

## Métricas que se generan (por dataset)

Cada `metrics.json` en `*_threshold/` contiene todos estos campos:

```json
{
  "dataset":          "torpeda",
  "model":            "RandomForest_57feat_balanced_threshold",
  "n_features":       57,
  "threshold":        0.XX,
  "oob_score":        0.XXXX,
  "train_time_s":     XX.X,
  "predict_time_s":   X.XXXX,
  "train_normal":     XXXX,
  "train_attack":     XXXXX,
  "test_normal":      XXXX,
  "test_attack":      XXXXX,
  "accuracy":         0.XXXX,
  "balanced_accuracy":0.XXXX,
  "precision":        0.XXXX,
  "recall":           0.XXXX,
  "f1":               0.XXXX,
  "roc_auc":          0.XXXX,
  "pr_auc":           0.XXXX,
  "mcc":              0.XXXX,
  "fpr":              0.XXXX,
  "fnr":              0.XXXX,
  "tn": XXXX, "fp": XXX, "fn": XXX, "tp": XXXXX
}
```

---

## Tabla comparativa esperada (referencia)

| Dataset | N feat | F1 (esperado) | FPR | Notas |
|---------|:------:|:-------------:|:---:|-------|
| CSIC 2010 | 57 | 0.947 | 0.007 | Baseline confirmado |
| TorpEda | 57 | ~0.97-0.99? | < 0.05? | 88.7% son ataques → RF debería tenerlo fácil |
| Harvard SR-BH | 57 | ? | ? | Dataset real de honeypot, más difícil |
| Nico/Ralf (ref.) | — | 0.950 | 0.030 | OCSVM per-group, app-dependent |

> **Hipótesis:** TorpEda debería obtener F1 muy alto (dataset dominado por ataques, fácil de separar).
> Harvard podría ser más difícil al ser tráfico real con más ruido.

---

## Ejecución local (sin SLURM, para pruebas rápidas)

Si quieres probar el script directamente sin enviar a HPC:

```bash
cd waf-ml-starter

# TorpEda (directo, sin srun)
PYTHONPATH=src .venv/bin/python3.11 \
    src/waf_ml/scripts/train_binary_rf.py \
    --parquet   data/processed/torpeda/torpeda_features.parquet \
    --dataset   torpeda \
    --out-dir   results/torpeda/binary \
    --feat-set  57 \
    --rebuild-features

# Harvard (puede tardar 15-30 min en local)
PYTHONPATH=src .venv/bin/python3.11 \
    src/waf_ml/scripts/train_binary_rf.py \
    --parquet   data/processed/harvard/harvard.parquet \
    --dataset   harvard \
    --out-dir   results/harvard/binary \
    --feat-set  57 \
    --rebuild-features

# Tabla comparativa
PYTHONPATH=src .venv/bin/python3.11 - < jobs/run_compare_binary.sh
# (O copiar el bloque PY del script y ejecutarlo directamente)
```

---

## Posibles problemas y soluciones

| Problema | Causa | Solución |
|----------|-------|----------|
| `FileNotFoundError: torpeda_features.parquet` | Parquet no existe | Verificar `data/processed/torpeda/` |
| `FileNotFoundError: harvard.parquet` | Parquet no existe | Verificar `data/processed/harvard/` |
| Job cancelado por tiempo (Harvard) | 907K filas tarda mucho | Aumentar `--time=0-02:00:00` en el .sh |
| `MemoryError` en Harvard | RAM insuficiente | Aumentar `--mem=48G` en el .sh |
| `KeyError: label_binary` | Parquet sin columna binaria | El parquet de Harvard ya la tiene; revisar el archivo |
| Feature importance vacío | pred.csv no generado | Verificar que el job de TorpEda/Harvard terminó antes de ejecutar compare |

---

## Próximos pasos (después de obtener resultados)

1. **Documentar resultados** en un nuevo `.md` tipo `resultados_cross_dataset.md`
2. **Responder la pregunta de investigación:**
   - ¿Generaliza el pipeline CSIC a otros datasets?
   - ¿Qué dataset es más difícil para el modelo?
3. **Ablación cross-dataset (opcional):** repetir con 25 y 34 features para ver si la entropía sigue siendo igual de importante en TorpEda/Harvard
4. **Redacción en el libro del TFG:** sección de experimentos comparativos

---

*Creado: 2026-06-08 | Branch: mejoras-profesor-ja*
