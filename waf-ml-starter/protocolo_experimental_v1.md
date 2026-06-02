# Protocolo Experimental v1 -- TFG WAF-ML

**Version:** 1.1
**Fecha:** 2026-06-02
**Estado:** Aprobado
**Autores:** Andres Roman, Juan Gonzalez
**Tutor:** Dr. Cristian Cappo

> Este documento define el alcance cerrado de la fase experimental. Una vez aprobado, no se agregan nuevos experimentos sin modificar explicitamente este protocolo.

---

## 1. Objetivo operativo de la fase experimental

Evaluar y comparar, sobre tres datasets publicos de trafico HTTP, la capacidad de deteccion de anomalias de un enfoque **one-class no supervisado** (OCSVM) frente a enfoques **supervisados** (clasificacion multiclase y multietiqueta), midiendo tanto metricas predictivas como metricas operativas de viabilidad tipo WAF, con resultados reproducibles y documentados que permitan responder las preguntas de investigacion de la tesis.

**Condicion de cierre:** El objetivo esta cumplido cuando todos los experimentos marcados como `OBLIGATORIO` en la Seccion 4 tienen sus archivos de resultado (`metrics.json`, `benchmark.json` donde aplica) commitados o preservados con trazabilidad completa, y se ha completado la tabla maestra de comparacion (Seccion 5).

---

## 2. Preguntas de investigacion

**PI-1 -- Brecha entre enfoques:**
?Que diferencia de desempeno en deteccion (F1, Recall, FPR) se observa entre un modelo OCSVM calibrado y modelos supervisados (multiclase y multietiqueta) cuando se entrena sobre los mismos datos HTTP?

**PI-2 -- Generalizacion entre datasets:**
?Los patrones de desempeno (que enfoque funciona mejor y por que margen) son consistentes entre datasets con caracteristicas distintas: sintetico/controlado (CSIC), mixto (TorpEda) y real/heterogeneo (SR-BH)?

**PI-3 -- Viabilidad operativa:**
?Alguno de los enfoques evaluados alcanza, en el entorno experimental, una latencia p99 <= 10ms por solicitud y throughput >= 100 req/s, umbrales de referencia internos que orientan la discusion sobre viabilidad en un contexto tipo WAF?

**PI-4 -- Valor del enfoque multietiqueta:**
?Agrega valor clasificatorio real el enfoque multietiqueta (clasificar tipo de ataque segun CAPEC) frente al binario, considerando las limitaciones de ruido de etiquetas documentadas en SR-BH 2020?

**PI-5 -- Impacto de features de seguridad:**
?Que features son los mas determinantes para la deteccion en cada enfoque, segun el analisis de importancia (SHAP / coeficientes), y son consistentes entre datasets?

---

## 3. Hipotesis de trabajo

**HP -- Principal:**
Se espera que los modelos supervisados (multiclase y multietiqueta) obtengan mejor desempeno predictivo --en terminos de F1, Recall y balance entre falsos positivos y falsos negativos-- que el OCSVM calibrado, cuando se dispone de etiquetas de ataque en el entrenamiento. El OCSVM aporta valor diferencial en escenarios donde solo se cuenta con trafico normal para entrenar, pagando un costo en recall.

> *Nota metodologica:* Esta hipotesis es de direccion esperada, no una hipotesis nula para prueba estadistica formal. La evaluacion es comparativa y descriptiva, basada en diferencias observadas en las metricas definidas.

**HS1 -- Viabilidad operativa:**
Al menos uno de los tres enfoques alcanzara, en el entorno experimental, latencia p99 <= 10ms y throughput >= 100 req/s en inferencia individual, lo que orientaria teoricamente su integracion en un componente de inspeccion de trafico HTTP en linea. Estos umbrales son criterios internos de referencia para el analisis experimental, no estandares universales de produccion.

**HS2 -- Clases CAPEC dificiles:**
El clasificador multietiqueta sobre SR-BH 2020 producira F1 por debajo de 0.70 en al menos algunas clases CAPEC minoritarias (especificamente las identificadas en la literatura como problematicas: CAPEC-33, CAPEC-16, CAPEC-274), lo que confirma las limitaciones de observabilidad desde una request aislada y acota el alcance de las conclusiones sobre el enfoque multietiqueta.

**HS3 -- Consistencia de features entre datasets:**
Se espera que los features de codificacion (`encoded`, `body_encoded`) y los de longitud anomala (`uri_len`, `max_param_value_len`) aparezcan entre los mas relevantes en todos los datasets, dado que son senales comunes de payloads de ataque HTTP independientemente del entorno de captura.

---

## 4. Matriz minima de experimentos

> **Regla de esta matriz:** un experimento es una combinacion unica de dataset + tarea + modelo. Se define un unico modelo por tarea. El objetivo es cubrir las preguntas de investigacion con el minimo de experimentos defendibles, no maximizar la cantidad de resultados.

| ID | Dataset | Tarea | Modelo / Script | Metricas predictivas | Metricas operativas | Prioridad | Justificacion |
|----|---------|-------|-----------------|----------------------|---------------------|-----------|---------------|
| **E01** | CSIC 2010 sin registro (86.864 filas) | Binaria one-class | `ocsvmOptimo.py` nu=0.05, gamma=0.1 | F1, Recall, Precision, FPR, BAcc, MCC, ROC-AUC | Throughput, latencia p50/p99, CPU%, RSS, model size | **OBLIGATORIO** | Experimento base. Recalibracion ya ejecutada en HPC. Falta versionar metrics.json. |
| **E02** | TorpEda 2012 | Binaria one-class | `ocsvmOptimo.py` nu=0.05, gamma=0.1 | Idem E01 | Idem E01 | **OBLIGATORIO** | Segundo dataset. Evalua generalizacion del OCSVM a un dataset distinto del mismo grupo CSIC. |
| **E03** | SR-BH 2020 | Binaria one-class | `ocsvmOptimo.py` nu=0.05, gamma=0.1 | Idem E01 | Idem E01 | **OBLIGATORIO** | Necesario para comparar OCSVM vs supervisado sobre el mismo corpus SR-BH (responde PI-1). |
| **E04** | TorpEda 2012 | Multiclase supervisada | `train_supervised.py` LogReg, C=1.0, balanced | F1 macro/weighted, Accuracy, BAcc, MCC, per-class F1 | Throughput, latencia p50/p99, CPU%, RSS, model size | **OBLIGATORIO** | TorpEda tiene etiquetas de tipo de ataque nativas. Permite contrastar OCSVM (E02) vs supervisado en el mismo dataset. Cubre PI-1 y PI-2. |
| **E05** | SR-BH 2020 | Multiclase supervisada | `train_supervised.py` LogReg, C=1.0, balanced | Idem E04 | Idem E04 | **OBLIGATORIO** | Enfoque supervisado sobre el dataset central. Par obligatorio de E03 para PI-1 y PI-2. |
| **E06** | SR-BH 2020 | Multietiqueta | `train_multilabel_lr_harvard_thesis.py` OvR+LogReg | F1 micro, F1 macro, Hamming Loss, Jaccard, Exact Match, per-label F1 | Throughput, latencia p50/p99, CPU%, RSS, model size | **OBLIGATORIO** | Contribucion central de la tesis. Unico dataset publico multilabel con CAPEC. Responde PI-4 y HS2. |
| **E07** | CSIC 2010 sin registro (ablacion) | Binaria one-class | `ocsvmOptimo.py` + `--drop-features suspicious_*` | F1, Recall, FPR | No requeridas | **TRABAJO FUTURO** | Ablacion para PI-5. Recortable: el analisis de features se cubre con SHAP e interpretacion. Solo si sobra tiempo. |
| **E08** | SR-BH 2020 (ablacion) | Binaria one-class | `ocsvmOptimo.py` + `--drop-features suspicious_*` | F1, Recall, FPR | No requeridas | **TRABAJO FUTURO** | Contraparte de E07 sobre dataset real para HS3. Solo si sobra tiempo. |

**Experimentos OBLIGATORIOS:** E01, E02, E03, E04, E05, E06 (6 en total).
**Trabajo futuro:** E07, E08 -- se mencionan en el libro como extension posible, no como resultados pendientes.

**Nota sobre CSIC multiclase/multilabel:** CSIC 2010 es un dataset binario nativo. No se ejecutan experimentos multiclase ni multilabel sobre CSIC. Hacerlo implicaria construir etiquetas artificiales que no aportan valor cientifico a la comparacion.

---

## 5. Criterios de cierre

Un experimento se considera **terminado** cuando cumple TODOS los siguientes criterios:

### C1 -- Artefactos de resultado preservados

- [ ] `metrics.json` existe y esta **versionado en el repositorio** o preservado con checksum documentado
- [ ] `benchmark.json` existe y esta preservado (excepto E07/E08 donde no se requiere)
- [ ] `pred.csv` se conserva como artefacto experimental cuando el tamano lo permite (< 50 MB); si es mayor, se guarda comprimido o como release artifact y se documenta la ruta y el checksum SHA-256

### C2 -- Trazabilidad completa

- [ ] El comando exacto ejecutado esta documentado (en el MD del experimento o en el script SLURM correspondiente)
- [ ] La seed usada esta registrada (`seed=42` en todos los experimentos de esta tesis)
- [ ] El script exacto esta identificado: nombre del archivo y el commit del repositorio al momento de la ejecucion
- [ ] El dataset exacto esta identificado: nombre del `.parquet`, numero de filas, columna de etiqueta usada

### C3 -- Metricas verificables y completas

- [ ] `metrics.json` contiene como minimo: F1, Recall, Precision, FPR, BAcc, MCC
- [ ] Para multilabel (E06): F1 micro, F1 macro, Hamming Loss, metricas por etiqueta CAPEC
- [ ] Para benchmark: throughput (req/s), latencia mean/p50/p95/p99, CPU%, RSS, model size en bytes

### C4 -- Fila en la tabla maestra

- [ ] El experimento tiene una fila completa en `resultados_consolidados.md`
- [ ] La fila incluye: ID, dataset, modelo, parametros clave, metricas principales, latencia p99
- [ ] Si hubo warnings o incidencias durante la ejecucion, se registran en la columna de observaciones

### C5 -- Sin incidencias metodologicas abiertas

- [ ] Si hubo warnings relevantes (ej. `y_pred contains classes not in y_true`), estan explicados y clasificados como esperados o como problemas reales
- [ ] Si el optimizador no convergio (ej. `max_iter` alcanzado en LogReg), se registra y se evalua si afecta las metricas

---

## 6. Experimentos fuera de alcance

Los siguientes experimentos **no se realizaran en esta tesis**. Si el tutor los solicita, se discutira explicitamente modificando este protocolo.

| Fuera de alcance | Razon |
|------------------|-------|
| **Implementar un WAF proxy funcional** | La tesis evalua viabilidad teorica basada en metricas operativas medidas experimentalmente. Un WAF productivo requiere integracion de red, gestion de sesiones, politicas de bloqueo y hardening que exceden el alcance academico. El esqueleto FastAPI existe como prueba de concepto, no como entregable. |
| **Modelos de deep learning** (CNN, LSTM, Transformer) | Aumentaria el costo computacional, el tiempo de analisis e interpretacion, y el alcance de la comparacion. Los resultados del estado del arte con deep learning (E-WebGuard, WAMM) sirven como referencia bibliografica, no como linea base a superar experimentalmente. |
| **Tuning exhaustivo de hiperparametros** | El tuning de nu/gamma para OCSVM esta justificado, documentado y cerrado. Para los experimentos pendientes se usan parametros calibrados (nu=0.05, gamma=0.1) o valores estandar (C=1.0, balanced). No se realizan busquedas adicionales. |
| **Agregar nuevos datasets** | CSIC, TorpEda y SR-BH cubren el espacio experimental necesario (sintetico, mixto, real). La justificacion metodologica de SR-BH descarta explicitamente las alternativas disponibles. Agregar datasets reinicia el ciclo experimental sin aportar conclusiones nuevas a las preguntas de investigacion definidas. |
| **Ablaciones con/sin suspicious_tokens** (E07, E08) | El analisis de importancia de features se cubre con SHAP (ya ejecutado para CSIC) y con los coeficientes del modelo supervisado. Las ablaciones son trabajo futuro, no necesarias para cerrar la tesis. |
| **Medir robustez adversarial** | Requiere generacion de payloads adversariales, tecnicas de evasion y evaluacion especializada. Se menciona como limitacion explicita y como direccion de trabajo futuro en el libro. |
| **Comparacion experimental con ModSecurity/CRS** | No se tiene acceso controlado a ModSecurity en el mismo entorno de evaluacion. La comparacion se hace bibliograficamente con los valores reportados por Gniewkowski et al. (2024). |
| **Replicar arquitectura Nico/Ralf** (18 modelos per-grupo) | El gap de rendimiento respecto a Nico/Ralf (F1=0.95) se explica y documenta como diferencia arquitectural (global vs per-grupo). La replicacion exacta no responde las preguntas de investigacion de esta tesis. |

---

## 7. Riesgos metodologicos y mitigacion

| Riesgo | Severidad | Mitigacion concreta |
|--------|-----------|---------------------|
| **Resultados del HPC no versionados** | ALTA | Paso 1 del plan de ejecucion: descargar y commitear metricas del cluster antes de cualquier otro paso. Sin esto, los resultados de recalibracion CSIC son trazablemente incompletos. |
| **Warning `y_pred contains classes not in y_true`** | MEDIA | Durante el tuning one-class el score se basa en `normal_acceptance` sobre normales; el warning emerge de candidatos que predicen clases ausentes en el subset de CV. Documentar en el MD de E01 como advertencia esperada e inofensiva para el resultado final. Verificar que no aparezca en la evaluacion final sobre el holdout. |
| **Dos versiones de scripts coexistentes** | MEDIA | Regla fija: OCSVM -> `ocsvmOptimo.py` (salida en `resultsOptimo/`). Supervisados -> `train_supervised.py` y `train_multilabel_lr_harvard_thesis.py` (salida en `results/`). `train_ocsvm_updated.py` no se usa en ningun experimento nuevo. |
| **SR-BH con ~48K muestras benignas posiblemente mal etiquetadas** | MEDIA | Amenaza a la validez interna documentada y asumida. Los resultados sobre SR-BH se presentan con ese asterisco explicito en el libro. No se realiza re-etiquetado (fuera de alcance). Se priorizan metricas macro y per-label para detectar efectos por clase. |
| **CSIC sin etiquetas multiclase nativas** | BAJA | Multiclase sobre CSIC queda explicitamente fuera de alcance (Seccion 6). No se ejecuta. |
| **Datos raw no en el repositorio** | BAJA | Se documenta en el libro que los datasets se obtienen de fuentes publicas con DOI. Los `.parquet` procesados se incluyen como release artifact si el tamano lo permite; en caso contrario se documenta el script de reproduccion paso a paso. |
| **Alcance demasiado grande** | ALTA | Esta matriz de 6 experimentos obligatorios es el techo. El tutor debe aprobar este protocolo antes de ejecutar los pasos 2-7. Si algun obligatorio no puede completarse antes de la fecha limite, se convierte en trabajo futuro con justificacion explicita en el libro. |

---

## 8. Plan de ejecucion inmediato

### Paso 1 -- Recuperar y versionar resultados existentes *(bloqueante)*
**Objetivo:** Cerrar E01 completamente segun criterios C1-C5.
**Accion:** Descargar del cluster los `metrics.json`, `benchmark.json` y `pred.csv` (si tamano lo permite) de los 5 experimentos de recalibracion CSIC sin registro (`oneclass`, `nicoRalf_nu01`, `nicoRalf_nu05`, `nicoRalf_nu10`, `nicoRalf_tune_meanscore`). Commitear en `resultsOptimo/csic_sin_registro/*/`.
**Criterio de finalizacion:** `git log` muestra commit con estos archivos. E01 marcado como CERRADO en este protocolo.

### Paso 2 -- Preparar datos de TorpEda y SR-BH en el cluster
**Objetivo:** Verificar que los `.parquet` procesados existen en el cluster antes de enviar jobs.
**Accion:** Confirmar existencia de `data/processed/torpeda/torpeda_features.parquet` y `data/processed/harvard/harvard.parquet` en el cluster. Si no existen, ejecutar `build_features_torpeda.py` y `build_features.py` antes de continuar.
**Criterio de finalizacion:** Ambos archivos `.parquet` existen y su numero de filas es consistente con la documentacion (SR-BH: 907.814 filas segun justificacion del dataset).

### Paso 3 -- Crear y ejecutar jobs OCSVM para TorpEda y SR-BH (E02, E03)
**Objetivo:** Primeros resultados OCSVM en los datasets pendientes.
**Accion:** Crear `jobs/run_torpeda_ocsvm.sh` y `jobs/run_srbh_ocsvm.sh` basados en `run_csic_sin_registro.sh`, cambiando paths de dataset y output. Usar nu=0.05, gamma=0.1. Enviar con `sbatch`. Descargar y commitear resultados.
**Archivos nuevos:** `jobs/run_torpeda_ocsvm.sh`, `jobs/run_srbh_ocsvm.sh`, `resultsOptimo/torpeda/ocsvm_nu05/`, `resultsOptimo/srbh/ocsvm_nu05/`
**Criterio de finalizacion:** `metrics.json` y `benchmark.json` commitados para E02 y E03.

### Paso 4 -- Crear y ejecutar jobs supervisados para TorpEda y SR-BH (E04, E05, E06)
**Objetivo:** Los tres experimentos supervisados centrales de la tesis.
**Accion:** Crear `jobs/run_torpeda_supervised.sh` (E04), `jobs/run_srbh_supervised.sh` (E05 y E06 en secuencia). Los scripts supervisados corren en CPU; ajustar tiempo SLURM. Output a `results/torpeda/multiclass/` y `results/harvard/multiclass/` y `results/harvard/multilabel/`.
**Archivos nuevos:** 2 jobs SLURM, resultados en `results/torpeda/` y `results/harvard/`
**Criterio de finalizacion:** `metrics.json` y `benchmark.json` commitados para E04, E05 y E06.

### Paso 5 -- Tabla maestra de resultados
**Objetivo:** Un documento unico con todos los resultados para el libro.
**Accion:** Crear `waf-ml-starter/resultados_consolidados.md` con la tabla maestra: datasets x modelos x metricas predictivas x metricas operativas. Rellenar con todos los experimentos cerrados. Columna de observaciones para incidencias por experimento.
**Archivos nuevos:** `resultados_consolidados.md`
**Criterio de finalizacion:** Tabla completa para los 6 obligatorios. Revisada por ambos autores y tutor.

### Paso 6 -- Incorporar resultados al libro LaTeX
**Objetivo:** Capitulos de resultados y discusion completos en el LaTeX principal.
**Accion:** Incorporar tabla maestra, analisis SHAP, discusion del gap vs Nico/Ralf, analisis de clases CAPEC dificiles (HS2), viabilidad operativa (HS1), comparacion bibliografica con literatura (E-WebGuard, Gniewkowski et al.). Identificar el archivo LaTeX activo antes de editar.
**Criterio de finalizacion:** Capitulos de resultados y discusion completos, revisados por tutor.

### Paso 7 -- Limpieza final del repositorio
**Objetivo:** Repositorio reproducible y listo para entrega.
**Accion:** (a) Corregir rutas hardcodeadas del `README.md`. (b) Mergear `mejoras-profesor-ja` -> `master`. (c) Tagear version de entrega: `git tag v1.0-entrega`.
**Criterio de finalizacion:** `master` contiene todos los resultados, protocolo y el README es reproducible con rutas relativas correctas.

---

*Este protocolo fue aprobado por los autores el 2026-06-02. Cualquier modificacion al alcance debe registrarse en este documento con fecha y justificacion.*
