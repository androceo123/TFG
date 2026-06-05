# Resultados PI-1 — Dataset CSIC 2010

**Pregunta de Investigación 1 (PI-1):**
> *"¿Qué diferencia de desempeño se observa entre OCSVM calibrado y modelos supervisados sobre los mismos datos HTTP?"*

---

## Resumen ejecutivo

| Modelo | F1 | Recall | Precisión | FPR | BAcc |
|--------|-----|--------|-----------|-----|------|
| OCSVM one-class (26 feat, nu=0.001 auto) | 0.519 | 0.350 | 0.999 | 0.0002 | 0.675 |
| OCSVM one-class (26 feat, nu=0.05, γ=0.1) | 0.787 | 0.689 | 0.916 | 0.099 | 0.795 |
| OCSVM one-class (57 feat, nu=0.05, γ=0.1) | 0.787 | 0.689 | 0.916 | 0.099 | 0.795 |
| LogReg supervisado (57 feat, thr=óptimo) | 0.665 | 0.750 | 0.597 | 0.160 | 0.795 |
| **RandomForest supervisado (57 feat, thr=óptimo)** | **0.947** | **0.919** | **0.975** | **0.007** | **0.956** |
| Nico/Ralf (referencia, OCSVM per-group) | 0.95 | 0.93 | — | 0.03 | — |

**Conclusión: el RF application-independent con 57 features prácticamente iguala a Nico/Ralf (per-group, app-dependent).**

---

## Descripción del dataset

- **Dataset:** CSIC 2010 HTTP Dataset
- **Subconjunto:** Global sin registro.jsp (86.864 filas)
- **Split:** 80% entrenamiento / 20% test, semilla=42, estratificado
- **Distribución label_binary:** `{0: 66.000 (normal), 1: 20.864 (ataque)}`
- **Ratio normal/ataque:** 3.16:1

---

## Experimento E01 — OCSVM línea base

**Job:** `run_csic_oneclass_global.sh` (experimento original)

- **Configuración:** OCSVM global, 26 features, nu=0.001 (auto-tuning por `normal_acceptance`)
- **Resultado:** F1=0.519, Recall=0.350, FPR=0.0002
- **Diagnóstico:** El auto-tuner maximiza aceptación de normales, no detecta ataques. Resultado ultra-conservador: preciso pero con recall muy bajo.

---

## Experimento E02 — OCSVM recalibrado (parámetros Nico/Ralf)

**Job:** `run_csic_nicoRalf_nu05.sh`

- **Configuración:** OCSVM global, 26 features, nu=0.05, gamma=0.1 (rango del libro, pág. 58)
- **Resultado:** F1=0.787, Recall=0.689, Precision=0.916, FPR=0.099, BAcc=0.795
- **Conclusión:** La recalibración mejora F1 en +0.27 puntos. Este es el techo del OCSVM con 26 features application-independent.

---

## Experimento E03 — Ingeniería de features (26 → 34 → 57)

### E03a — Entropía y ratios (26 → 34 features)

**Job:** `run_csic_solo_registro_entropy.sh`

Nuevos features añadidos (8):
- `uri_entropy`, `query_entropy`, `max_param_value_entropy`
- `query_pct_digit`, `query_pct_alpha`, `body_entropy`, `body_pct_digit`, `body_pct_alpha`

**Resultado (solo registro.jsp):** F1=0.457 — idéntico a 26 features
**Diagnóstico SHAP:** Importancia ≈ 0 para todos los nuevos features. El 75% de las peticiones GET no tienen query params ni body, haciendo que la distribución de features nuevos sea indistinguible entre normales y ataques.

### E03b — Distribución de caracteres por intervalos (34 → 57 features)

**Job:** `run_csic_solo_registro_v2.sh` / `run_csic_global_v2.sh`

Inspirado en Tabla 2.3 del libro Nico/Ralf: 5 bins sobre frecuencias de caracteres ordenadas descendentemente `[(0,1),(1,3),(3,6),(6,10),(10+)]`.

Nuevos features añadidos (23):
- `query_char_dist_i0..i4`: distribución del query string completo
- `max_param_char_dist_i0..i4`: parámetro query de mayor entropía
- `body_char_dist_i0..i4`: distribución del body completo
- `max_body_param_char_dist_i0..i4`: parámetro body de mayor entropía
- `mean_param_value_entropy`, `std_param_value_entropy`, `max_body_param_value_entropy`, `mean_body_param_value_entropy`

**Resultado (global, 57 feat):** F1=0.787 — idéntico a 26 features
**Resultado (solo registro.jsp, 57 feat):** F1=0.457 — idéntico
**Diagnóstico SHAP:** Ningún feature v2 aparece en el top de importancia SHAP. El problema es estructural, no de features.

### Conclusión de E03

> **El techo del OCSVM application-independent es F1=0.787, independientemente del número y sofisticación de los features (26, 34 ó 57). La ingeniería de features está agotada para este paradigma.**

---

## Experimento E04 — LogReg supervisado binario

**Job:** `run_csic_supervised_binary.sh` / `run_csic_logreg_threshold.sh` / `run_csic_logreg_nobalanced.sh`

**Configuración:**
- Modelo: LogisticRegression (lineal)
- Features: 57 (mismos que E03b)
- Datos: `csic_sin_registro_v2.parquet` (mismos que OCSVM)
- C=1.0, max_iter=2000, solver por defecto

**Variantes probadas:**

| Variante | thr | F1 | Recall | Precisión | FPR | ROC-AUC |
|----------|-----|-----|--------|-----------|-----|---------|
| class_weight=balanced, thr=0.50 | 0.50 | 0.639* | 0.705 | 0.585 | 0.158 | 0.861 |
| class_weight=balanced, thr=óptimo | 0.48 | 0.665 | 0.750 | 0.597 | 0.160 | 0.861 |
| class_weight=None, thr=óptimo | 0.47 | 0.665 | 0.750 | 0.597 | 0.160 | 0.861 |

*Valor calculado manualmente del classification report; el metrics.json guarda F1 por clase como dict.

**Observaciones:**
- `balanced` y `no-balanced` producen resultados idénticos (ratio 3.16:1 no es suficientemente extremo)
- ROC-AUC LogReg (0.861) > OCSVM (~0.840), pero en el punto de operación F1 el OCSVM gana
- **Causa:** LogReg busca frontera lineal en R^57. El espacio de features no es linealmente separable
- **LogReg F1=0.665 < OCSVM F1=0.787** — el modelo supervisado lineal NO supera al one-class con kernel

**Conclusión parcial:** La limitación no es el paradigma one-class, sino la **linealidad del modelo supervisado**. El kernel RBF del OCSVM captura no-linealidades que LogReg no puede.

---

## Experimento E05 — RandomForest supervisado (no lineal)

**Job:** `run_csic_random_forest.sh`

**Configuración:**
- Modelo: RandomForestClassifier
- n_estimators=500, max_depth=None, min_samples_leaf=1
- class_weight=balanced, n_jobs=8, random_state=42
- Features: 57 (mismos que todos los experimentos anteriores)
- OOB score: disponible en métricas

**Resultados:**

| Umbral | F1 | Recall | Precisión | FPR | BAcc |
|--------|-----|--------|-----------|-----|------|
| thr=0.50 (default) | ~0.93 | — | — | — | — |
| **thr=óptimo** | **0.947** | **0.919** | **0.975** | **0.007** | **0.956** |

**Top features por importancia Gini (RF):**
```
uri_len:              7.659
query_len:            4.074
n_query_params:       2.135
body_encoded:         1.458
path_depth:           1.361
encoded:              0.863
uri_pct_non_alnum:    0.777
has_suspicious_tokens: 0.460
suspicious_tokens_count: 0.365
max_param_value_len:  0.329
```
Los mismos features estructurales dominan — los nuevos features v2 siguen siendo secundarios, pero el RF los aprovecha mejor que el OCSVM.

**Comparación definitiva:**

| Modelo | F1 | Δ vs OCSVM | FPR |
|--------|-----|-----------|-----|
| OCSVM nu=0.05 (57 feat) | 0.787 | baseline | 0.099 |
| LogReg thr=óptimo (57 feat) | 0.665 | **−0.122** | 0.160 |
| **RF thr=óptimo (57 feat)** | **0.947** | **+0.160** | **0.007** |
| Nico/Ralf referencia | 0.950 | — | 0.030 |

---

## Conclusiones

### PI-1: ¿Supera el supervisado al OCSVM calibrado?

**Sí, pero solo con modelo no lineal.**

- **LogReg (lineal):** F1=0.665 < OCSVM F1=0.787. El modelo supervisado lineal es PEOR que el one-class.
- **RandomForest (no lineal):** F1=0.947 >> OCSVM F1=0.787. El modelo supervisado no lineal supera claramente.

### Hallazgo 1 — El límite era el paradigma, no los features

Los 57 features application-independent son **suficientes** para separar normal/ataque cuando el modelo es no lineal. El OCSVM no podía usarlos al máximo porque:
1. Entrena solo con normales (no ve la señal de ataques)
2. Su frontera de decisión (kernel RBF) es una hipersuperficie en R^57 que no puede adaptarse a las regiones de ataque sin ejemplos negativos

### Hallazgo 2 — RandomForest iguala a Nico/Ralf

Con los **mismos features globales** (application-independent, sin nombres de parámetros, sin modelos por URL) el RF alcanza F1=0.947, prácticamente idéntico a F1=0.950 de Nico/Ralf, que usaba:
- 18 modelos separados por URL+método
- Features por campo individual (application-dependent)
- OCSVM per-group con parámetros afinados manualmente

**El RF application-independent resuelve el gap arquitectural sin necesitar segmentación por aplicación.**

### Hallazgo 3 — El FPR mejora incluso sobre la referencia

- OCSVM global: FPR=0.099 (9.9% falsos positivos)
- Nico/Ralf: FPR=0.030
- **RF application-independent: FPR=0.007** (0.7% falsos positivos)

El RF no solo iguala el recall de la referencia sino que tiene **4× menos falsos positivos**.

### Hallazgo 4 — Linealidad, no paradigma

La secuencia LogReg → RF muestra que el paradigma supervisado requiere un modelo con capacidad no lineal para aprovechar las etiquetas de ataque. Con LogReg, el kernel RBF del OCSVM es más eficiente que la frontera lineal aunque no vea ataques.

---

## Retroalimentación del profesor (2026-06-02)

Tras ver los resultados del RF (F1=0.947), el profesor señaló:

1. **Entropía es importante** — `uri_entropy` es el feature #1 en importancia Gini (0.141), confirmando que añadirla fue correcto.
2. **Char distribution puede no usarse** — los features `*_char_dist_i*` (Nico/Ralf Tabla 2.3) tienen baja importancia individual. Validar si son necesarios.
3. **Multiclase con RF** — RandomForest es el más utilizado para clasificación multiclase. Avanzar con TorpEda.
4. **Balanceo de clases** — importante para multiclase, usar `class_weight='balanced'`.
5. **Feature selection** — demasiados features = más lento y difícil de explicar. Quedarse con los más importantes.

---

## Experimento E06 — Ablación de features (CSIC, RF)

**Objetivo:** Aislar la contribución de cada grupo de features en el RF.

**Jobs:**
- `run_csic_rf_25feat.sh` → RF con 25 features estructurales originales
- `run_csic_rf_34feat.sh` → RF con 34 features (25 + 8 entropía v1)
- `run_csic_rf_top10.sh` → RF con top-10 del ranking Gini (feature selection)

**Estructura de features:**

| Grupo | Features | Descripción |
|-------|----------|-------------|
| Estructurales (25) | uri_len, path_depth, query_len, ... + method one-hot | Features básicos del proyecto |
| Entropía v1 (8) | uri_entropy, query_entropy, body_entropy, ... | Añadidos por nosotros |
| Char dist v2 (23) | query_char_dist_i*, body_char_dist_i*, ... | Inspirados en Nico/Ralf Tabla 2.3 |

**Tabla de ablación (pendiente tras ejecutar jobs):**

| Experimento | Features | F1 | Recall | Precisión | FPR | BAcc |
|-------------|----------|-----|--------|-----------|-----|------|
| RF thr=OPTIMO (baseline) | 57 | 0.947 | 0.919 | 0.975 | 0.007 | 0.956 |
| RF thr=OPTIMO | 34 (25+entropía) | — | — | — | — | — |
| RF thr=OPTIMO | 25 (solo estructurales) | — | — | — | — | — |
| RF thr=OPTIMO | top-10 (feature selection) | — | — | — | — | — |

**Hipótesis:**
- Si RF-34 ≈ RF-57: los 23 char_dist de Nico/Ralf son prescindibles
- Si RF-25 ≈ RF-34: la entropía tampoco aporta mucho al RF (aunque es top feature Gini)
- Si RF-top10 ≈ RF-57: podemos explicar la detección con solo 10 features

---

## Siguiente paso — E07: TorpEda multiclase

**Job:** `run_torpeda_rf_multiclass.sh`

**Dataset TorpEda:**
- 74.133 muestras
- Clases: NORMAL (8.363), SQLi (43k), XSS (4.8k), SSI, BufferOverflow, CRLFi, XPath, LDAPi, FormatString, ANOMALOUS

**Objetivo:** Ver si RF distingue entre tipos de ataque (no solo normal/ataque).
El job reconstruye el parquet con 57 features y entrena RF con 25, 34 y 57 feat.

---

*Actualizado: 2026-06-05 | Branch: mejoras-profesor-ja*
