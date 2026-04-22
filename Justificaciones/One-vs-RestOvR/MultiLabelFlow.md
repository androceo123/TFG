## 4. Flujo multilabel usado en este proyecto

En este TFG, la formulación **multietiqueta** se aplica principalmente sobre el dataset **SR-BH / Harvard**, porque es el que realmente contiene **múltiples etiquetas CAPEC por request**.

Esto significa que una misma petición puede pertenecer simultáneamente a varias categorías de ataque, por ejemplo:

- SQL Injection
- Path Traversal
- HTTP Response Splitting
- técnicas de evasión u otras CAPEC

Por eso, SR-BH / Harvard es el caso más representativo para evaluar una solución multilabel real.

---

## 5. Diferencia con OCSVM

Es importante no confundir las tareas:

### Multilabel
- Se resuelve con **One-vs-Rest + Logistic Regression**
- Cada etiqueta se modela como un clasificador binario
- El resultado final es un vector de etiquetas activas

### One-class / anomalías
- Se resuelve con **OCSVM / SGDOneClassSVM**
- El modelo aprende solo con tráfico normal
- Luego detecta desviaciones o anomalías
- No predice múltiples etiquetas CAPEC

Por tanto:

- **multilabel = OvR**
- **anomalías = OCSVM**

---

## 6. Flujo completo de multilabel con SR-BH / Harvard

El proceso que seguimos es el siguiente:

### Paso 1: Datos crudos
Se parte del dataset SR-BH / Harvard en formato CSV o TSV.

### Paso 2: Carga del archivo
El loader de SR-BH:
- detecta automáticamente el separador si hace falta;
- normaliza columnas;
- identifica las columnas de CAPEC;
- verifica que existan las columnas principales de request HTTP:
  - método
  - URI
  - body

### Paso 3: Construcción de etiquetas
A partir de las columnas CAPEC del dataset:
- se crea `label_multilabel` como una lista de etiquetas por fila;
- se crea `label_binary` para el caso normal/anómalo;
- se crea `label_multiclass` cuando se necesita una sola etiqueta primaria;
- se conserva `label_multiclass_first` como referencia de auditoría.

### Paso 4: Extracción de features HTTP
Cada request se transforma en un vector numérico con variables como:
- longitud de la URI;
- profundidad del path;
- número de parámetros;
- percent-encoding;
- presencia de tokens sospechosos;
- método HTTP en one-hot;
- longitud del body;
- content-length.

Estas features son las que alimentan al clasificador.

### Paso 5: Entrenamiento
Se usa:

- `StandardScaler`
- `OneVsRestClassifier`
- `LogisticRegression`

Es decir, un clasificador binario por etiqueta.

### Paso 6: Evaluación
Se calculan métricas multilabel como:
- `F1 micro`
- `F1 macro`
- `Hamming Loss`
- `Jaccard`
- `Exact Match Ratio`

Y además, cuando hay probabilidades o scores:
- `PR-AUC`
- `ROC-AUC`

### Paso 7: Resultados y benchmark
El script guarda:
- el modelo entrenado (`.joblib`)
- métricas (`.json`)
- predicciones (`.csv`)
- benchmark operativo (`.json`)
- importancia de features

---

## 7. Script principal que se usa para multilabel

El script más importante para este flujo es:

- `waf_ml/scripts/train_multilabel_lr_harvard_thesis.py`

Ese script:
- toma el dataset procesado;
- separa train/test;
- entrena OvR;
- evalúa métricas;
- hace tuning interno;
- permite ablación de tokens sospechosos;
- guarda los resultados.

---

## 8. Cómo se prepara el dataset SR-BH / Harvard

Antes de entrenar, el dataset debe pasar por una preparación que deja una tabla homogénea.

### Entrada
El archivo original contiene:
- requests HTTP;
- columnas de etiquetas CAPEC;
- campos de método, URI, body y headers.

### Salida procesada
El dataset final debe tener:
- `request_http_method`
- `request_http_request`
- `request_body`
- `request_headers_json`
- `label_multilabel`
- `label_binary`
- `label_multiclass`
- features numéricas HTTP

Ese archivo procesado es el que se usa para entrenamiento.

---

## 9. Resumen operativo del flujo
text SR-BH / Harvard raw → loader SRBH → normalización de columnas → generación de label_multilabel → extracción de features HTTP → matriz numérica → One-vs-Rest + Logistic Regression → métricas multilabel → modelo y resultados


---

## 10. Cómo ejecutarlo

La ejecución depende de que primero tengas el dataset procesado.

### 10.1 Generar features / dataset procesado
Primero se debe construir el dataset procesado desde SR-BH / Harvard.

### 10.2 Entrenar multilabel
Luego se ejecuta el script multilabel principal, apuntando al archivo procesado.

Ejemplo general:

```
    bash python src/waf_ml/scripts/train_multilabel_lr_harvard_thesis.py --data <archivo_procesado> --label-col label_multilabel --out <modelo.joblib>
```

### 10.3 Ver resultados
Después de correrlo, se obtienen:
- el modelo entrenado;
- métricas de evaluación;
- predicciones;
- benchmark de rendimiento;
- análisis por etiqueta.

---

## 11. Idea clave para la tesis

La razón por la que usamos **SR-BH / Harvard** en multilabel es que es el dataset más adecuado para representar el caso real de **una request con múltiples patrones CAPEC simultáneos**.  
Eso justifica metodológicamente el uso de **OvR**, porque el problema no es elegir una sola clase, sino decidir la presencia o ausencia de cada etiqueta.

---

## 12. Conclusión

En este proyecto:

- **SR-BH / Harvard** se usa para la parte **multilabel**;
- **OvR + Logistic Regression** es el modelo principal para esa tarea;
- **OCSVM** queda reservado para la tarea **one-class/anomalías**.

Esto permite mantener una comparación clara entre:
- detección de anomalías;
- clasificación multiclase;
- clasificación multietiqueta.

Y, sobre todo, permite trabajar con un flujo reproducible desde datos crudos hasta resultados finales.

