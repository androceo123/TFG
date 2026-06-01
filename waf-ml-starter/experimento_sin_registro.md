# Experimento: CSIC sin Aplicación "Registro"

## Motivación

El dataset CSIC 2010 contiene peticiones HTTP a una tienda online (`tienda1`), organizada en múltiples módulos:

| Módulo | Path |
|--------|------|
| Añadir al carrito | `/tienda1/publico/anadir.jsp` |
| Autenticar | `/tienda1/publico/autenticar.jsp` |
| **Registro de usuarios** | **`/tienda1/publico/registro.jsp`** |
| Editar miembro (modo registro) | `/tienda1/miembros/editar.jsp?modo=registro` |
| Pagar | `/tienda1/publico/pagar.jsp` |
| ... | ... |

El **Prof. Cristian Cappo** identificó que la aplicación de **registro** no era coherente con el resto del dataset, lo cual podría distorsionar los resultados del modelo. El objetivo de este experimento es verificar si eliminarla mejora la detección.

---

## Cambios realizados

### Dataset filtrado

Se eliminaron todas las filas cuya URI contiene `registro` (tanto en el path como en los parámetros de query):

| | Filas |
|---|---|
| Dataset original | 97.065 |
| Filas eliminadas (registro) | **10.201** |
| Dataset filtrado | **86.864** |

Distribución tras el filtrado:
- Normal: 66.000
- Anomalous: 20.864

El dataset filtrado se guarda en:
```
data/processed/csic_sin_registro/csic_features_sin_registro.parquet
```

### Archivos nuevos

```
jobs/
  run_csic_sin_registro.sh          # Job SLURM completo (dataset completo, GPU, 32 iters)
  run_csic_sin_registro_1000.sh     # Job SLURM de prueba (1000 filas, GPU, 8 iters)

src/waf_ml/optimo/
  compare_results.py                # Script de comparacion de metricas entre experimentos

data/processed/
  csic_sin_registro/
    csic_features_sin_registro.parquet   # Dataset CSIC sin aplicacion registro
```

---

## Cómo ejecutar

### En el clúster HPC (SLURM)

**Experimento de prueba (rapido, 1000 filas):**
```bash
cd /path/to/TFG-JA
sbatch waf-ml-starter/jobs/run_csic_sin_registro_1000.sh
```

**Experimento completo:**
```bash
sbatch waf-ml-starter/jobs/run_csic_sin_registro.sh
```

### En local (sin SLURM, sin GPU)

```bash
cd waf-ml-starter
source .venv/bin/activate
mkdir -p resultsOptimo/csic_sin_registro/oneclass

PYTHONPATH=src python src/waf_ml/optimo/ocsvmOptimo.py \
  --mode train \
  --backend sgd_ocsvm \
  --kernel-approx nystroem \
  --data data/processed/csic_sin_registro/csic_features_sin_registro.parquet \
  --label-col label_binary \
  --test-size 0.2 \
  --seed 42 \
  --final-train-on train_normals \
  --tune fast \
  --cv 3 \
  --tune-metric auto \
  --tune-n-iter 32 \
  --tune-n-jobs 4 \
  --out resultsOptimo/csic_sin_registro/oneclass/ocsvm.joblib \
  --metrics-out resultsOptimo/csic_sin_registro/oneclass/metrics.json \
  --pred-out resultsOptimo/csic_sin_registro/oneclass/pred.csv \
  --tune-results-out resultsOptimo/csic_sin_registro/oneclass/search.csv \
  --fi-kind shap \
  --fi-out resultsOptimo/csic_sin_registro/oneclass/feature_importance_shap.csv \
  --fi-max-rows 500 \
  --shap-background-rows 100 \
  --benchmark-mode auto \
  --benchmark-max-rows 500 \
  --benchmark-out resultsOptimo/csic_sin_registro/oneclass/benchmark.json
```

> **Nota**: sin `--require-gpu` el script corre en CPU automáticamente.

---

## Cómo comparar los resultados

Una vez ejecutados ambos experimentos, comparar con:

```bash
cd waf-ml-starter
PYTHONPATH=src python src/waf_ml/optimo/compare_results.py \
  --base    resultsOptimo/csic/oneclass/metrics.json \
  --exp     resultsOptimo/csic_sin_registro/oneclass/metrics.json \
  --out     resultsOptimo/comparacion_csic_vs_sin_registro.json \
  --out-csv resultsOptimo/comparacion_csic_vs_sin_registro.csv
```

El script genera:
- **Tabla en consola** con todas las métricas comparadas y flechas de mejora/empeora
- **JSON** con diferencias absolutas y porcentuales
- **CSV** exportable para incluir en la memoria del TFG

### Métricas clave a observar

| Métrica | ¿Qué indica? |
|---------|--------------|
| F1 Score | Balance precision/recall — la más importante para detección de ataques |
| Balanced Accuracy | Desempeño equilibrado entre clases (normal vs ataque) |
| MCC | Correlación de Matthews — robusto con clases desbalanceadas |
| ROC-AUC / PR-AUC | Capacidad discriminativa general del modelo |
| FPR | Tasa de falsas alarmas (tráfico normal bloqueado) |
| FNR | Tasa de ataques no detectados |

---

## Hipótesis del experimento

El Prof. Cappo sugiere que la aplicación de registro era **incoherente**: sus patrones de tráfico podían ser atípicos respecto al resto de la aplicación, haciendo que el modelo aprenda una frontera de decisión menos generalizable.

**Si la hipótesis es correcta**, se esperaría ver tras eliminar registro:
- ↑ F1, Balanced Accuracy, MCC
- ↓ FPR (menos falsas alarmas)
- Los resultados del CSIC se acercarían más a los de Torpeda y Harvard

**Si no mejora**, los features del modelo son lo suficientemente robustos como para no verse afectados por esa aplicación, lo cual también es un resultado válido y positivo para el TFG.

---

## Estructura de resultados

```
resultsOptimo/
  csic/
    oneclass/                        # Experimento BASE (dataset completo)
      metrics.json
      pred.csv
      search.csv
      feature_importance_shap.csv
      benchmark.json
  csic_sin_registro/
    oneclass/                        # Experimento SIN REGISTRO (dataset filtrado)
      metrics.json
      pred.csv
      search.csv
      feature_importance_shap.csv
      benchmark.json
    oneclass_test1000/               # Prueba rapida con 1000 filas
      metrics.json
      ...
  comparacion_csic_vs_sin_registro.json   # Comparacion automatica
  comparacion_csic_vs_sin_registro.csv
```
