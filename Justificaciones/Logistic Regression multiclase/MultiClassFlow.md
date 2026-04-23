## 4. Flujo multiclase usado en este proyecto

En este TFG, la formulación **multiclase** se aplica sobre los datasets que pueden representarse con **una sola clase final por request**.

Eso ocurre principalmente en dos casos:

- **TorpEda**, donde la clase multiclase se obtiene de forma bastante natural a partir de la categoría nativa del tráfico: normal, anómalo o tipo de ataque;
- **SR-BH / Harvard**, donde la clase multiclase se deriva a partir del problema multilabel, escogiendo una sola etiqueta primaria por request.

En **CSIC 2010** también existe `label_multiclass`, pero allí el resultado queda reducido en la práctica a dos clases (`NORMAL` y la clase anómala), por lo que sirve más como compatibilidad del pipeline que como caso multiclase rico.

---

## 5. Diferencia con multilabel y con OCSVM

Es importante no confundir las tareas:

### Multiclase
- Se resuelve con **StandardScaler + LogisticRegression**
- Cada request termina en **una sola clase**
- El resultado final es una única etiqueta predicha

### Multilabel
- Se resuelve con **One-vs-Rest + Logistic Regression**
- Una request puede activar varias etiquetas
- El resultado final es un vector de etiquetas

### One-class / anomalías
- Se resuelve con **OCSVM / SGDOneClassSVM**
- El modelo aprende solo con tráfico normal
- Luego detecta desviaciones o anomalías
- No asigna una clase concreta de ataque

Por tanto:

- **multiclase = una clase final**
- **multilabel = varias etiquetas posibles**
- **anomalías = normal vs desvío**

---

## 6. Flujo completo de multiclase en el proyecto

El proceso que seguimos es el siguiente:

### Paso 1: Datos crudos
Se parte del dataset crudo correspondiente:
- **SR-BH / Harvard** en CSV o TSV
- **TorpEda** en XML
- **CSIC** en TXT

### Paso 2: Carga del archivo
Cada dataset pasa por su loader específico:

- el loader de **SR-BH** detecta separador, normaliza columnas y encuentra las columnas CAPEC;
- el loader de **TorpEda** interpreta el XML, extrae request y labels nativos;
- el loader de **CSIC** reconstruye requests HTTP desde archivos de texto.

### Paso 3: Construcción de etiquetas
A partir de cada loader se crean:
- `label_binary`
- `label_multiclass`
- `label_multilabel`

Pero para esta tarea interesa sobre todo `label_multiclass`.

Según el dataset:

#### SR-BH / Harvard
- se crea `label_multilabel` con todas las CAPEC activas;
- luego se deriva `label_multiclass`;
- además puede guardarse `label_multiclass_first` como referencia de auditoría;
- cuando se usa la estrategia por severidad, también puede quedar `label_primary_severity`.

#### TorpEda
- `label_multiclass` representa:
  - `NORMAL`
  - `{PREFIX}-ANOMALOUS`
  - `{PREFIX}-{ATTACK_NAME}`

#### CSIC
- `label_multiclass` queda básicamente como:
  - `NORMAL`
  - `{PREFIX}` para tráfico anómalo

### Paso 4: Extracción de features HTTP
Cada request se transforma en un vector numérico con variables como:
- longitud de la URI;
- profundidad del path;
- número de parámetros;
- ratio de caracteres no alfanuméricos;
- percent-encoding;
- presencia de tokens sospechosos;
- método HTTP en one-hot;
- longitud del body;
- content-length.

Estas features son las que alimentan al clasificador.

### Paso 5: Definición del problema supervisado
Se toma:
- `X =` features HTTP
- `y = label_multiclass`

Cada fila tiene una sola clase.

### Paso 6: Manejo de clases raras
Antes del split, el script puede:
- conservar clases raras;
- fusionarlas en `OTHER_RARE`;
- eliminarlas.

Eso sirve para evitar problemas cuando hay clases con muy pocas muestras.

### Paso 7: Entrenamiento
Se usa:

- `StandardScaler`
- `LogisticRegression`

Es decir, una pipeline lineal supervisada para clasificación multiclase.

### Paso 8: Tuning
Si se activa tuning, el script:
- trabaja solo dentro del bloque de entrenamiento;
- usa grid, random o halving;
- explora combinaciones válidas de solver, penalty, `C`, `class_weight` y `l1_ratio`;
- usa como métrica principal de selección **`f1_weighted`**.

### Paso 9: Evaluación
Se calculan métricas multiclase como:
- `accuracy`
- `balanced_accuracy`
- `F1 micro`
- `F1 macro`
- `F1 weighted`
- `precision`
- `recall`
- `MCC`
- matriz de confusión

Y además, cuando hay probabilidades:
- `ROC-AUC`
- `PR-AUC`

### Paso 10: Resultados y benchmark
El script guarda:
- el modelo entrenado (`.joblib`)
- métricas (`.json`)
- predicciones (`.csv`)
- benchmark operativo (`.json`)
- importancia de features por coeficientes
- importancia por permutación si se activa

---

## 7. Script principal que se usa para multiclase

El script más importante para este flujo es:

- `waf_ml/scripts/train_supervised.py`

Para la tarea multiclase se ejecuta con:

- `--task multiclass`
- `--label-col label_multiclass`

Ese script:
- toma el dataset procesado;
- separa train/test;
- maneja clases raras;
- entrena la regresión logística multiclase;
- evalúa métricas;
- hace tuning interno;
- permite ablación vía `--drop-features`;
- guarda los resultados.

---

## 8. Cómo se prepara el dataset para multiclase

Antes de entrenar, el dataset debe pasar por una preparación que deja una tabla homogénea.

### Entrada
El archivo original contiene:
- requests HTTP;
- labels o columnas fuente para construir labels;
- campos de método, URI, body y headers.

### Salida procesada
El dataset final debe tener:
- `request_http_method`
- `request_http_request`
- `request_body`
- `request_headers_json`
- `label_multiclass`
- `label_binary`
- `label_multilabel`
- features numéricas HTTP

Ese archivo procesado es el que se usa para entrenamiento.

---

## 9. Resumen operativo del flujo

**SR-BH raw / TorpEda raw / CSIC raw → loader específico → normalización de columnas → generación de `label_multiclass` → extracción de features HTTP → matriz numérica → StandardScaler + LogisticRegression → métricas multiclase → modelo y resultados**

---

## 10. Cómo ejecutarlo

La ejecución depende de que primero tengas el dataset procesado.

### 10.1 Generar features / dataset procesado
Primero se debe construir el dataset procesado desde el dataset crudo correspondiente.

### 10.2 Entrenar multiclase
Luego se ejecuta el script principal, apuntando al archivo procesado.

Ejemplo general:

```bash
python src/waf_ml/scripts/train_supervised.py   --data <archivo_procesado>   --task multiclass   --label-col label_multiclass   --out <modelo.joblib>
```

### 10.3 Ver resultados
Después de correrlo, se obtienen:
- el modelo entrenado;
- métricas de evaluación;
- predicciones;
- benchmark de rendimiento;
- análisis de importancia de variables.

---

## 11. Idea clave para la tesis

La razón por la que usamos **multiclase** es que en esta formulación cada request debe terminar con **una sola clase final**, lo que permite comparar directamente ataques o categorías mutuamente excluyentes dentro del problema supervisado.

En **SR-BH / Harvard**, esto requiere derivar una clase primaria desde el esquema multilabel.
En **TorpEda**, la formulación es mucho más natural porque el propio dataset ya distingue categorías de tráfico y tipos de ataque.
En **CSIC**, la estructura es compatible, aunque el resultado práctico siga siendo binario.

---

## 12. Conclusión

En este proyecto:

- **TorpEda** y **SR-BH / Harvard derivado** son los casos más relevantes para la parte **multiclase**;
- **Logistic Regression + StandardScaler** es el modelo principal para esta tarea;
- **OvR** queda reservado como idea central del caso multilabel;
- **OCSVM** queda reservado para la tarea **one-class/anomalías**.

Esto permite mantener una comparación clara entre:
- detección de anomalías;
- clasificación multiclase;
- clasificación multietiqueta.

Y, sobre todo, permite trabajar con un flujo reproducible desde datos crudos hasta resultados finales.
