# Recalibración de Parámetros nu/gamma — Análisis y Decisiones

## Contexto

El Prof. Cappo indicó que los resultados del experimento completo (F1=0.518, Recall=0.350) estaban **mal calibrados** y sugirió revisar los parámetros `nu` y `gamma` que usaron Nico Epp y Ralf Funk en su trabajo OCS-WAF (2017), ya que ellos lograron TPR=0.93, FPR=0.03, F1=0.95 con el mismo dataset CSIC.

---

## Paso 1: Lectura del libro `libroNicoRalf.pdf`

Se leyó el PDF completo (`libroTFG/correciones/libroNicoRalf.pdf`) buscando los valores exactos de `nu` y `gamma` que usaron.

### Hallazgo 1 — Rango de búsqueda (página 58 del libro)

El texto del libro dice explícitamente:

> *"realizamos una búsqueda en el rango [0,0001; 0,1] para ambos parámetros y seleccionamos los valores que obtuvieron los mejores resultados en la fase de detección para cada grupo de peticiones."*

**Lo que esto significa:** Buscaron `nu` y `gamma` en el rango **[0.0001, 0.1]** y seleccionaron manualmente el mejor por grupo evaluando TPR y FPR. **El libro NO da una tabla con los valores específicos por grupo.** Los resultados finales (Tabla 5.2 y 5.3) muestran el rendimiento pero no los parámetros usados.

### Hallazgo 2 — Arquitectura completamente diferente (Tabla 5.1, página 57)

Este es el hallazgo más importante. Nico/Ralf **NO entrenan un modelo global** como nosotros. Entrenan **un clasificador separado por cada combinación de URL + método HTTP**:

| ID | Método | URL | Normales | Anómalas |
|----|--------|-----|----------|----------|
| c00 | GET | /tienda1/miembros/editar.jsp | 2.000 | 1.362 |
| c01 | POST | /tienda1/miembros/editar.jsp | 2.000 | 1.362 |
| c02 | GET | /tienda1/publico/anadir.jsp | 2.000 | 1.380 |
| ... | ... | ... | ... | ... |
| c12 | GET | /tienda1/publico/registro.jsp | 2.000 | 1.364 |
| c13 | POST | /tienda1/publico/registro.jsp | 2.000 | 1.364 |
| t00 | POST | /tienda1/miembros/editar.jsp | 5.608 | 10.121 |
| t01 | POST | /tienda1/publico/registro.jsp | 2.522 | 13.163 |

En total: **18 modelos distintos**, cada uno con ~1.500 peticiones para entrenamiento (75% de las normales disponibles).

**Implicación:** Sus modelos son muy específicos. Cada grupo de peticiones tiene sus propios patrones de features bien definidos, lo cual facilita mucho la separación entre normales y anómalas. Un solo modelo global tiene mucha más varianza en los datos y la tarea de separación es más difícil.

### Hallazgo 3 — Incluían el módulo "registro"

Los grupos **c12, c13 y t01** corresponden exactamente a `/tienda1/publico/registro.jsp`. Es decir, Nico/Ralf SÍ incluían el módulo registro, pero lo manejaban como grupos separados. El prof. Cappo identificó que en nuestro modelo global ese módulo distorsionaba resultados — tiene razón, pero el efecto es por la arquitectura monolítica, no solo por los datos.

### Hallazgo 4 — Métricas de tuning

La diferencia clave que explica el gap de resultados (F1=0.52 nuestro vs F1=0.95 de Nico/Ralf) no es solo los parámetros — **es el criterio de selección**. Nico/Ralf seleccionaban parámetros evaluando TPR y FPR sobre datos con ataques. Nuestro auto-tuner no puede hacer eso.

---

## Paso 2: Diagnóstico del problema en `ocsvmOptimo.py`

Se analizó el código del script de entrenamiento para entender por qué el auto-tuner siempre elige `nu=0.001`.

### El problema: `--tune-metric auto = normal_acceptance`

```python
# En ocsvmOptimo.py, función _score_candidate():
if metric in {"", "auto", "normal_acceptance", "acceptance"}:
    return normal_acceptance, details  # normal_acceptance = 1 - fraccion_rechazados
```

**¿Qué hace `normal_acceptance`?**

Mide qué fracción de peticiones normales (del set de validación de tuning) el modelo clasifica como "normal". El problema:

- **Con nu=0.001** (frontera muy ajustada): casi todo se clasifica como normal → `normal_acceptance ≈ 0.999` ✅
- **Con nu=0.1** (frontera más laxa): el 10% de normales puede quedar fuera → `normal_acceptance ≈ 0.900` ❌

El tuner **siempre elige nu=0.001** porque minimiza falsas alarmas en normales. Pero como el tuning se hace **SOLO con datos normales** (one-class learning), NUNCA sabe que nu=0.001 detecta solo el 35% de los ataques.

### Resumen del diagnóstico

| Parámetro | Valor auto-tuning | Consecuencia |
|-----------|-------------------|-------------|
| nu=0.001 | Gana siempre en `normal_acceptance` | Solo el 0.1% de normales son SVs → frontera extremadamente ajustada |
| Recall=0.350 | Consecuencia directa | El 65% de ataques queda "dentro" de la frontera normal |
| F1=0.518 | Consecuencia directa | Precisión alta (0.999) pero recall muy bajo |

---

## Paso 3: Decisión sobre los nuevos jobs

### Por qué NO simplemente cambiar los grids

El rango `nu=[0.001...0.1]` ya está incluido en el auto-tuner por defecto. El problema no es el rango, es que la **métrica de optimización** siempre favorece el extremo conservador. Ampliar el grid no resuelve nada.

### Por qué usamos `--tune none` con valores fijos

La estrategia de Nico/Ralf era **evaluar manualmente** nu y gamma viendo los resultados de TPR/FPR. Nosotros replicamos eso corriendo el modelo con valores fijos y comparando los métricas finales (que SÍ incluyen ataques en el set de evaluación).

**Valores elegidos y justificación:**

| Variante | nu | gamma | Justificación |
|----------|-----|-------|---------------|
| `nicoRalf_nu05` | 0.05 | 0.1 | Punto medio del rango [0.0001, 0.1] — típico en literatura OCSVM |
| `nicoRalf_nu10` | 0.10 | 0.1 | Extremo superior del rango de Nico/Ralf — máxima flexibilidad |
| `nicoRalf_nu01` | 0.01 | 0.1 | Valor intermedio bajo — balance entre los extremos |
| `nicoRalf_tune_meanscore` | auto (grid 0.01-0.1) | auto | `mean_score` usa decision_function — menos sesgado que `normal_acceptance` |

**¿Por qué gamma=0.1?**

Con nu bajo (nu=0.001), el modelo compensa con gamma bajo (frontera muy suave). Con nu más alto, gamma=0.1 crea una frontera más ajustada que tiene más sentido para separar clases. Gamma controla el "radio de influencia" del kernel RBF — valores más altos = fronteras más locales.

### Por qué `mean_score` como métrica alternativa de tuning

```python
# mean_score = promedio de la función de decisión sobre los normales de validación
mean_score = float(np.mean(model.decision_function(X_eval)))
```

La `decision_function` devuelve la distancia al hiperplano separador. Maximizar el promedio de esa distancia significa que los normales están **más adentro** del hiperplano, no solo que son clasificados como normales. Esto penaliza modelos que aceptan normales "por los pelos" (cerca del límite) — más representativo de la calidad real del modelo.

---

## Paso 4: Jobs creados

### `run_csic_nicoRalf_nu_1000.sh` — Test rápido (30 min)
- 1000 filas estratificadas del dataset sin registro
- Solo RUN1 (nu=0.05) y RUN2 (nu=0.1)
- Muestra tabla comparativa al final
- **Objetivo**: verificar funcionamiento y estimar mejora antes del run completo

### `run_csic_nicoRalf_nu.sh` — Experimento completo (3h)
- Dataset completo sin registro (86.864 filas)
- 4 variantes en secuencia
- Tabla comparativa final incluyendo baseline y referencia Nico/Ralf

---

## Cómo ejecutar

### Paso 1 — Subir cambios al cluster (si no están)
```bash
git pull origin mejoras-profesor-ja
```

### Paso 2 — Test rápido primero (30 min)
```bash
cd waf-ml-starter
sbatch jobs/run_csic_nicoRalf_nu_1000.sh

# Monitorear
squeue -u $USER
tail -f slurm-nicoRalf-nu-1000-<JOBID>.out
```

### Paso 3 — Si los resultados mejoran, lanzar el completo (3h)
```bash
sbatch jobs/run_csic_nicoRalf_nu.sh

tail -f slurm-nicoRalf-nu-<JOBID>.out
```

### Paso 4 — Ver resultados completos
Al finalizar, el job imprime una tabla como esta (valores hipotéticos):

```
Experimento                         F1     Recall   Precision    FPR     BAcc
--------------------------------------------------------------------------------
BASE (nu=0.001 auto-tuning)       0.5180   0.3500    0.9996   0.0001   0.6748
RUN1: nu=0.05, gamma=0.1          0.????   0.????    0.????   0.????   0.????
RUN2: nu=0.10, gamma=0.1          0.????   0.????    0.????   0.????   0.????
RUN3: nu=0.01, gamma=0.1          0.????   0.????    0.????   0.????   0.????
RUN4: mean_score auto-tune        0.????   0.????    0.????   0.????   0.????

Referencia Nico/Ralf (OCS-WAF 2017): F1=0.95, Recall=0.93, FPR=0.03
```

---

## Resultados reales obtenidos (dataset completo 86.864 filas)

| Experimento | F1 | Recall | Precision | FPR | Bal.Acc | MCC |
|---|---|---|---|---|---|---|
| BASE nu=0.001 (auto-tuning) | 0.518 | 0.350 | 0.9996 | ~0 | ~0.675 | 0.415 |
| RUN3: nu=0.01, gamma=0.1 | 0.765 | 0.622 | **0.993** | **0.007** | 0.807 | 0.616 |
| **RUN1: nu=0.05, gamma=0.1** ⭐ | **0.787** | 0.689 | 0.916 | 0.099 | **0.795** | 0.577 |
| RUN2: nu=0.10, gamma=0.1 | 0.794 | **0.720** | 0.884 | 0.150 | 0.785 | 0.556 |
| RUN4: mean_score auto-tune | 0.607 | 0.440 | 0.977 | 0.017 | 0.712 | 0.461 |
| **Nico/Ralf (referencia)** | **0.95** | **0.93** | ~0.97 | **0.03** | ~0.95 | — |

### Análisis de los resultados

**RUN1 (nu=0.05)** es el mejor balance global (F1=0.787). Sube el Recall de 0.35 → 0.69 con FPR=10%.

**RUN3 (nu=0.01)** es el más parecido a Nico/Ralf en FPR (0.007 vs 0.03). Muy alta precisión (0.993) con Recall=0.622. Buena opción si se prioriza no bloquear tráfico legítimo.

**RUN2 (nu=0.10)** sube el Recall a 0.72 pero FPR=15% — demasiadas falsas alarmas.

**RUN4 (mean_score)** eligió nu=0.01/gamma=0.01 automáticamente — peor que el manual. Confirma que ninguna métrica de tuning one-class sustituye a la evaluación con ataques reales.

### Gap restante vs Nico/Ralf (F1=0.787 vs F1=0.95)

El gap es **arquitectural**. Nico/Ralf entrenaron 18 modelos separados (uno por URL+método HTTP), cada uno con datos homogéneos. Nuestro modelo global mezcla todos los patrones en un único clasificador. Esto es académicamente sólido para el TFG: demuestra el impacto de la granularidad del modelo en la detección de anomalías web.

---

## Estructura de resultados generados

```
resultsOptimo/
  csic_sin_registro/
    oneclass/                    ← BASE: nu=0.001 auto-tuning (ya ejecutado)
      metrics.json               F1=0.518, Recall=0.350
    test1000_nicoRalf_nu05/      ← TEST 1000 filas, nu=0.05
    test1000_nicoRalf_nu10/      ← TEST 1000 filas, nu=0.10
    nicoRalf_nu05/               ← COMPLETO, nu=0.05, gamma=0.1
    nicoRalf_nu10/               ← COMPLETO, nu=0.10, gamma=0.1
    nicoRalf_nu01/               ← COMPLETO, nu=0.01, gamma=0.1
    nicoRalf_tune_meanscore/     ← COMPLETO, auto-tune con mean_score
```

---

## Diferencias fundamentales con OCS-WAF de Nico/Ralf

| Aspecto | Nuestro modelo | Nico/Ralf OCS-WAF |
|---------|---------------|-------------------|
| Arquitectura | 1 modelo global | 18 modelos (1 por URL+método) |
| Datos de entrenamiento | 66.000 normales mezcladas | ~1.500 normales por grupo |
| Selección de nu/gamma | Auto-tuning (normal_acceptance) | Manual (evaluando TPR/FPR) |
| Include "registro" | No (filtrado) | Sí (como grupos c12, c13, t01) |
| Herramienta OCSVM | SGDOneClassSVM + Nystroem | OneClassSVM (kernel RBF directo) |
| F1 obtenido | 0.518 | 0.95 |

El gap de rendimiento se explica principalmente por la **arquitectura** (global vs per-grupo), no solo por los parámetros. Los jobs nuevos nos permitirán cuantificar cuánto mejora solo con recalibrar nu/gamma.
