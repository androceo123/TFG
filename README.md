# Guía de Comandos - Proyecto TFG WAF-ML

Documentación de comandos para entrenar y procesar modelos de machine learning en el proyecto de TFG.

---

## 1. Activación del Entorno Virtual

Activar el entorno virtual y actualizar dependencias:

```bash
source .venv/bin/activate
```

Actualizar pip:

```bash
python -m pip install --upgrade pip
```

Instalar dependencias del proyecto:

```bash
pip install -r requirements.txt
```

---

## 2. Conversión de Archivos

### Convertir todos los archivos .parquet a .csv

Convierte todos los archivos parquet del directorio `results/` a formato CSV:

```bash
find results -name "*.parquet" -print0 | while IFS= read -r -d '' f; do python3 -c "import pandas as pd; pd.read_parquet(r'''$f''').to_csv(r'''${f%.parquet}.csv''', index=False)"; done
```

---

## 3. Generación de Datasets Procesados

### CSIC Dataset

**Generar dataset CSIC procesado:**

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && python -m waf_ml.scripts.build_features_csic --inputs data/raw/csic --output data/processed/csic/csic_features.parquet --out-format parquet --use-headers
```

**Ver dataset CSIC completo (convertir a CSV):**

```bash
python -c "import pandas as pd; df=pd.read_parquet('data/processed/csic/csic_features.parquet'); df.to_csv('data/processed/csic/csic_features.csv', index=False); print('OK -> data/processed/csic/csic_features.csv', 'rows=', len(df), 'cols=', df.shape[1])"
```

---

### Torpeda Dataset

**Generar dataset Torpeda procesado:**

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && PYTHONPATH=src python src/waf_ml/scripts/build_features_torpeda.py --inputs data/raw/torpeda --output data/processed/torpeda/torpeda_features.parquet --out-format parquet --use-headers --label-prefix TORPEDA
```

**Ver dataset Torpeda completo (convertir a CSV):**

```bash
python -c "import pandas as pd; df=pd.read_parquet('data/processed/torpeda/torpeda_features.parquet'); df.to_csv('data/processed/torpeda/torpeda_features.csv', index=False); print('OK -> data/processed/torpeda/torpeda_features.csv', 'rows=', len(df), 'cols=', df.shape[1])"
```

---

### Harvard Dataset

**Generar dataset Harvard procesado:**

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && python -m waf_ml.scripts.build_features --inputs data/raw/harvard --output data/processed/harvard/harvard.parquet --sep auto
```

**Ver dataset Harvard completo (convertir a CSV):**

```bash
python -c "import pandas as pd; df=pd.read_parquet('data/processed/harvard/harvard.parquet'); df.to_csv('data/processed/harvard/harvard_features.csv', index=False); print('OK -> data/processed/harvard/harvard_features.csv', 'rows=', len(df), 'cols=', df.shape[1])"
```

---

## 4. Modelos OCSVM (One-Class SVM)

### Harvard - OCSVM

Entrenar y testear OCSVM con Harvard (base + sin tokens sospechosos):

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && mkdir -p models/harvard/ocsvm results/harvard/ocsvm && python src/waf_ml/scripts/train_ocsvm_updated.py --backend sgd_ocsvm --kernel-approx nystroem --mode train --data data/processed/harvard/harvard.parquet --label-col label_binary --test-size 0.2 --seed 42 --final-train-on train_normals --tune random --tune-budget 4 --tune-max-train 60000 --tune-max-eval 15000 --tune-metric auto --tune-nu-grid 0.01,0.05 --tune-gamma-grid 0.01,0.1 --tune-n-components-grid 128,256 --tune-max-iter-grid 1000,2000 --out models/harvard/ocsvm/ocsvm_harvard_tuned.joblib --metrics-out results/harvard/ocsvm/tuned.metrics.json --tune-results-out results/harvard/ocsvm/tuned.search.csv --benchmark-mode auto --benchmark-max-rows 400 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --benchmark-out results/harvard/ocsvm/tuned.benchmark.json && python src/waf_ml/scripts/train_ocsvm_updated.py --backend sgd_ocsvm --kernel-approx nystroem --mode train --data data/processed/harvard/harvard.parquet --label-col label_binary --test-size 0.2 --seed 42 --final-train-on train_normals --tune random --tune-budget 4 --tune-max-train 60000 --tune-max-eval 15000 --tune-metric auto --tune-nu-grid 0.01,0.05 --tune-gamma-grid 0.01,0.1 --tune-n-components-grid 128,256 --tune-max-iter-grid 1000,2000 --drop-features suspicious_tokens_count,has_suspicious_tokens,body_suspicious_tokens_count,body_has_suspicious_tokens --out models/harvard/ocsvm/ocsvm_harvard_tuned_no_suspicious.joblib --metrics-out results/harvard/ocsvm/tuned_no_suspicious.metrics.json --tune-results-out results/harvard/ocsvm/tuned_no_suspicious.search.csv --benchmark-mode auto --benchmark-max-rows 400 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --benchmark-out results/harvard/ocsvm/tuned_no_suspicious.benchmark.json
```

---

### CSIC - OCSVM

Entrenar y testear OCSVM con CSIC (base + sin tokens sospechosos):

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && mkdir -p models/csic/ocsvm results/csic/ocsvm && python src/waf_ml/scripts/train_ocsvm_updated.py --backend sgd_ocsvm --kernel-approx nystroem --mode train --data data/processed/csic/csic_features.parquet --label-col label_binary --test-size 0.2 --seed 42 --final-train-on train_normals --tune random --tune-budget 4 --tune-max-train 60000 --tune-max-eval 15000 --tune-metric auto --tune-nu-grid 0.01,0.05 --tune-gamma-grid 0.01,0.1 --tune-n-components-grid 128,256 --tune-max-iter-grid 1000,2000 --out models/csic/ocsvm/ocsvm_csic_tuned.joblib --metrics-out results/csic/ocsvm/tuned.metrics.json --tune-results-out results/csic/ocsvm/tuned.search.csv --benchmark-mode auto --benchmark-max-rows 400 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --benchmark-out results/csic/ocsvm/tuned.benchmark.json && python src/waf_ml/scripts/train_ocsvm_updated.py --backend sgd_ocsvm --kernel-approx nystroem --mode train --data data/processed/csic/csic_features.parquet --label-col label_binary --test-size 0.2 --seed 42 --final-train-on train_normals --tune random --tune-budget 4 --tune-max-train 60000 --tune-max-eval 15000 --tune-metric auto --tune-nu-grid 0.01,0.05 --tune-gamma-grid 0.01,0.1 --tune-n-components-grid 128,256 --tune-max-iter-grid 1000,2000 --drop-features suspicious_tokens_count,has_suspicious_tokens,body_suspicious_tokens_count,body_has_suspicious_tokens --out models/csic/ocsvm/ocsvm_csic_tuned_no_suspicious.joblib --metrics-out results/csic/ocsvm/tuned_no_suspicious.metrics.json --tune-results-out results/csic/ocsvm/tuned_no_suspicious.search.csv --benchmark-mode auto --benchmark-max-rows 400 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --benchmark-out results/csic/ocsvm/tuned_no_suspicious.benchmark.json
```

---

### Torpeda - OCSVM

Entrenar y testear OCSVM con Torpeda (base + sin tokens sospechosos):

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && mkdir -p models/torpeda/ocsvm results/torpeda/ocsvm && python src/waf_ml/scripts/train_ocsvm_updated.py --backend sgd_ocsvm --kernel-approx nystroem --mode train --data data/processed/torpeda/torpeda_features.parquet --label-col label_binary --test-size 0.2 --seed 42 --final-train-on train_normals --tune random --tune-budget 4 --tune-max-train 60000 --tune-max-eval 15000 --tune-metric auto --tune-nu-grid 0.01,0.05 --tune-gamma-grid 0.01,0.1 --tune-n-components-grid 128,256 --tune-max-iter-grid 1000,2000 --out models/torpeda/ocsvm/ocsvm_torpeda_tuned.joblib --metrics-out results/torpeda/ocsvm/tuned.metrics.json --tune-results-out results/torpeda/ocsvm/tuned.search.csv --benchmark-mode auto --benchmark-max-rows 400 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --benchmark-out results/torpeda/ocsvm/tuned.benchmark.json && python src/waf_ml/scripts/train_ocsvm_updated.py --backend sgd_ocsvm --kernel-approx nystroem --mode train --data data/processed/torpeda/torpeda_features.parquet --label-col label_binary --test-size 0.2 --seed 42 --final-train-on train_normals --tune random --tune-budget 4 --tune-max-train 60000 --tune-max-eval 15000 --tune-metric auto --tune-nu-grid 0.01,0.05 --tune-gamma-grid 0.01,0.1 --tune-n-components-grid 128,256 --tune-max-iter-grid 1000,2000 --drop-features suspicious_tokens_count,has_suspicious_tokens,body_suspicious_tokens_count,body_has_suspicious_tokens --out models/torpeda/ocsvm/ocsvm_torpeda_tuned_no_suspicious.joblib --metrics-out results/torpeda/ocsvm/tuned_no_suspicious.metrics.json --tune-results-out results/torpeda/ocsvm/tuned_no_suspicious.search.csv --benchmark-mode auto --benchmark-max-rows 400 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --benchmark-out results/torpeda/ocsvm/tuned_no_suspicious.benchmark.json
```

---

## 5. Modelos Multiclass

### Harvard - Multiclass

Entrenar y testear Multiclass con Harvard (base + sin tokens sospechosos):

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && mkdir -p models/harvard/multiclass results/harvard/multiclass && PYTHONPATH=src python src/waf_ml/scripts/train_supervised.py --data data/processed/harvard/harvard.parquet --task multiclass --label-col label_multiclass --test-size 0.2 --seed 42 --solver lbfgs --penalty l2 --C 1.0 --class-weight balanced --max-iter 500 --rare-class-policy keep --rare-class-min-count 2 --tune random --cv 3 --tune-budget 2 --tune-sample-n 10000 --tune-solvers lbfgs --tune-penalties l2 --tune-class-weights balanced,none --tune-c-grid 0.1,1 --fi-kind coef --pred-out results/harvard/multiclass/harvard_multiclass_lr_fast.pred.csv --metrics-out results/harvard/multiclass/harvard_multiclass_lr_fast.metrics.json --tune-results-out results/harvard/multiclass/harvard_multiclass_lr_fast.search.csv --benchmark-mode auto --benchmark-max-rows 600 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --benchmark-out results/harvard/multiclass/harvard_multiclass_lr_fast.bench.json --out models/harvard/multiclass/harvard_multiclass_lr_fast.joblib && PYTHONPATH=src python src/waf_ml/scripts/train_supervised.py --data data/processed/harvard/harvard.parquet --task multiclass --label-col label_multiclass --test-size 0.2 --seed 42 --solver lbfgs --penalty l2 --C 1.0 --class-weight balanced --max-iter 500 --rare-class-policy keep --rare-class-min-count 2 --tune random --cv 3 --tune-budget 2 --tune-sample-n 10000 --tune-solvers lbfgs --tune-penalties l2 --tune-class-weights balanced,none --tune-c-grid 0.1,1 --drop-features suspicious_tokens_count,has_suspicious_tokens,body_suspicious_tokens_count,body_has_suspicious_tokens --fi-kind coef --pred-out results/harvard/multiclass/harvard_multiclass_lr_no_suspicious_fast.pred.csv --metrics-out results/harvard/multiclass/harvard_multiclass_lr_no_suspicious_fast.metrics.json --tune-results-out results/harvard/multiclass/harvard_multiclass_lr_no_suspicious_fast.search.csv --benchmark-mode auto --benchmark-max-rows 600 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --benchmark-out results/harvard/multiclass/harvard_multiclass_lr_no_suspicious_fast.bench.json --out models/harvard/multiclass/harvard_multiclass_lr_no_suspicious_fast.joblib
```

---

### CSIC - Multiclass

Entrenar y testear Multiclass con CSIC (base + sin tokens sospechosos):

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && mkdir -p models/csic/multiclass results/csic/multiclass && PYTHONPATH=src python src/waf_ml/scripts/train_supervised.py --data data/processed/csic/csic_features.parquet --task multiclass --label-col label_multiclass --test-size 0.2 --seed 42 --solver lbfgs --penalty l2 --C 1.0 --class-weight balanced --max-iter 500 --rare-class-policy keep --rare-class-min-count 2 --tune random --cv 3 --tune-budget 2 --tune-sample-n 10000 --tune-solvers lbfgs --tune-penalties l2 --tune-class-weights balanced,none --tune-c-grid 0.1,1 --fi-kind coef --pred-out results/csic/multiclass/csic_multiclass_lr_fast.pred.csv --benchmark-mode auto --benchmark-max-rows 600 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --out models/csic/multiclass/csic_multiclass_lr_fast.joblib --metrics-out results/csic/multiclass/csic_multiclass_lr_fast.metrics.json --tune-results-out results/csic/multiclass/csic_multiclass_lr_fast.search.csv --benchmark-out results/csic/multiclass/csic_multiclass_lr_fast.bench.json && PYTHONPATH=src python src/waf_ml/scripts/train_supervised.py --data data/processed/csic/csic_features.parquet --task multiclass --label-col label_multiclass --test-size 0.2 --seed 42 --solver lbfgs --penalty l2 --C 1.0 --class-weight balanced --max-iter 500 --rare-class-policy keep --rare-class-min-count 2 --tune random --cv 3 --tune-budget 2 --tune-sample-n 10000 --tune-solvers lbfgs --tune-penalties l2 --tune-class-weights balanced,none --tune-c-grid 0.1,1 --drop-features suspicious_tokens_count,has_suspicious_tokens,body_suspicious_tokens_count,body_has_suspicious_tokens --fi-kind coef --pred-out results/csic/multiclass/csic_multiclass_lr_no_suspicious_fast.pred.csv --benchmark-mode auto --benchmark-max-rows 600 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --out models/csic/multiclass/csic_multiclass_lr_no_suspicious_fast.joblib --metrics-out results/csic/multiclass/csic_multiclass_lr_no_suspicious_fast.metrics.json --tune-results-out results/csic/multiclass/csic_multiclass_lr_no_suspicious_fast.search.csv --benchmark-out results/csic/multiclass/csic_multiclass_lr_no_suspicious_fast.bench.json
```

---

### Torpeda - Multiclass

Entrenar y testear Multiclass con Torpeda (base + sin tokens sospechosos):

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && mkdir -p models/torpeda/multiclass results/torpeda/multiclass && PYTHONPATH=src python src/waf_ml/scripts/train_supervised.py --data data/processed/torpeda/torpeda_features.parquet --task multiclass --label-col label_multiclass --test-size 0.2 --seed 42 --solver lbfgs --penalty l2 --C 1.0 --class-weight balanced --max-iter 500 --rare-class-policy keep --rare-class-min-count 2 --tune random --cv 3 --tune-budget 2 --tune-sample-n 10000 --tune-solvers lbfgs --tune-penalties l2 --tune-class-weights balanced,none --tune-c-grid 0.1,1 --fi-kind coef --pred-out results/torpeda/multiclass/torpeda_multiclass_lr_fast.pred.csv --benchmark-mode auto --benchmark-max-rows 600 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --out models/torpeda/multiclass/torpeda_multiclass_lr_fast.joblib --metrics-out results/torpeda/multiclass/torpeda_multiclass_lr_fast.metrics.json --tune-results-out results/torpeda/multiclass/torpeda_multiclass_lr_fast.search.csv --benchmark-out results/torpeda/multiclass/torpeda_multiclass_lr_fast.bench.json && PYTHONPATH=src python src/waf_ml/scripts/train_supervised.py --data data/processed/torpeda/torpeda_features.parquet --task multiclass --label-col label_multiclass --test-size 0.2 --seed 42 --solver lbfgs --penalty l2 --C 1.0 --class-weight balanced --max-iter 500 --rare-class-policy keep --rare-class-min-count 2 --tune random --cv 3 --tune-budget 2 --tune-sample-n 10000 --tune-solvers lbfgs --tune-penalties l2 --tune-class-weights balanced,none --tune-c-grid 0.1,1 --drop-features suspicious_tokens_count,has_suspicious_tokens,body_suspicious_tokens_count,body_has_suspicious_tokens --fi-kind coef --pred-out results/torpeda/multiclass/torpeda_multiclass_lr_no_suspicious_fast.pred.csv --benchmark-mode auto --benchmark-max-rows 600 --benchmark-warmup-rows 50 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics --out models/torpeda/multiclass/torpeda_multiclass_lr_no_suspicious_fast.joblib --metrics-out results/torpeda/multiclass/torpeda_multiclass_lr_no_suspicious_fast.metrics.json --tune-results-out results/torpeda/multiclass/torpeda_multiclass_lr_no_suspicious_fast.search.csv --benchmark-out results/torpeda/multiclass/torpeda_multiclass_lr_no_suspicious_fast.bench.json
```

---

## 6. Modelos Multilabel

### Harvard - Multilabel

Entrenar y testear Multilabel con Harvard:

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && mkdir -p models/harvard/multilabel results/harvard/multilabel && python src/waf_ml/scripts/train_multilabel_lr_harvard_thesis.py --data data/processed/harvard/harvard.parquet --label-col label_multilabel --out models/harvard/multilabel/ovr_logreg_harvard_thesis.joblib --metrics-out results/harvard/multilabel/ovr_logreg_harvard_thesis.metrics.json --benchmark-out results/harvard/multilabel/ovr_logreg_harvard_thesis.benchmark.json --tune random --cv 3 --tune-budget 2 --tune-sample-n 10000 --tune-results-out results/harvard/multilabel/ovr_logreg_harvard_thesis.search.csv --solver lbfgs --penalty l2 --class-weight balanced --tune-solvers lbfgs --tune-penalties l2 --tune-class-weights balanced,none --tune-c-grid 0.1,1 --max-iter 300 --tol 1e-3 --n-jobs -1 --tune-n-jobs 1 --fi-kind coef --run-suspicious-ablation --ablation-comparison-out results/harvard/multilabel/ovr_logreg_harvard_thesis.ablation_comparison.json --benchmark-mode auto --benchmark-max-rows 600 --benchmark-warmup-rows 80 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics
```

---

### CSIC - Multilabel

Entrenar y testear Multilabel con CSIC:

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && mkdir -p models/csic/multilabel results/csic/multilabel && python src/waf_ml/scripts/train_multilabel_lr_harvard_thesis.py --data data/processed/csic/csic_features.parquet --label-col label_multilabel --reduced-binary-mode force --out models/csic/multilabel/ovr_logreg_csic_thesis.joblib --metrics-out results/csic/multilabel/ovr_logreg_csic_thesis.metrics.json --benchmark-out results/csic/multilabel/ovr_logreg_csic_thesis.benchmark.json --tune random --cv 3 --tune-budget 2 --tune-sample-n 10000 --tune-results-out results/csic/multilabel/ovr_logreg_csic_thesis.search.csv --solver lbfgs --penalty l2 --class-weight balanced --tune-solvers lbfgs --tune-penalties l2 --tune-class-weights balanced,none --tune-c-grid 0.1,1 --max-iter 300 --tol 1e-3 --n-jobs -1 --tune-n-jobs 1 --fi-kind coef --run-suspicious-ablation --ablation-comparison-out results/csic/multilabel/ovr_logreg_csic_thesis.ablation_comparison.json --benchmark-mode auto --benchmark-max-rows 600 --benchmark-warmup-rows 80 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics
```

---

### Torpeda - Multilabel

Entrenar y testear Multilabel con Torpeda:

```bash
cd /home/andres/tesis/TFG/waf-ml-starter && mkdir -p models/torpeda/multilabel results/torpeda/multilabel && python src/waf_ml/scripts/train_multilabel_lr_harvard_thesis.py --data data/processed/torpeda/torpeda_features.parquet --label-col label_multilabel --reduced-binary-mode auto --out models/torpeda/multilabel/ovr_logreg_torpeda_thesis.joblib --metrics-out results/torpeda/multilabel/ovr_logreg_torpeda_thesis.metrics.json --benchmark-out results/torpeda/multilabel/ovr_logreg_torpeda_thesis.benchmark.json --tune random --cv 3 --tune-budget 2 --tune-sample-n 10000 --tune-results-out results/torpeda/multilabel/ovr_logreg_torpeda_thesis.search.csv --solver lbfgs --penalty l2 --class-weight balanced --tune-solvers lbfgs --tune-penalties l2 --tune-class-weights balanced,none --tune-c-grid 0.1,1 --max-iter 300 --tol 1e-3 --n-jobs -1 --tune-n-jobs 1 --fi-kind coef --run-suspicious-ablation --ablation-comparison-out results/torpeda/multilabel/ovr_logreg_torpeda_thesis.ablation_comparison.json --benchmark-mode auto --benchmark-max-rows 600 --benchmark-warmup-rows 80 --benchmark-repeats 1 --benchmark-load-levels 1,2 --benchmark-load-rows 100 --require-resource-metrics
```

---

## 📋 Notas Importantes

- Todos los comandos están organizados en bloques copiables para facilitar su uso
- Asegúrate de estar en el directorio correcto antes de ejecutar los comandos
- Los modelos y resultados se guardan en los directorios `models/` y `results/` respectivamente
- Para más información sobre los parámetros, consulta la documentación del proyecto

---

**Última actualización:** Abril 2026
