# Justificación del uso de Regresión Logística para clasificación multiclase

## 1. Contexto del problema

En este Trabajo Final de Grado se aborda la detección y clasificación de ataques en tráfico HTTP.

Dentro de este contexto, la tarea de **clasificación multiclase** aparece cuando una misma solicitud debe asignarse a **una sola clase final**.

Eso puede representar, por ejemplo:

- tráfico normal;
- una categoría concreta de ataque;
- una clase primaria derivada desde varias etiquetas posibles.

A diferencia de multilabel, acá no se busca decidir varias etiquetas activas al mismo tiempo, sino resolver una sola decisión final por request.

---

## 2. ¿Qué es la Regresión Logística multiclase?

La Regresión Logística multiclase es una extensión de la clasificación logística supervisada al caso en que existen más de dos clases posibles.

La idea general es:

- tomar un vector de features de entrada;
- calcular un puntaje para cada clase;
- transformar esos puntajes en valores comparables;
- elegir la clase con mayor valor final.

En términos prácticos:

- cada clase tiene pesos asociados a las variables;
- el modelo aprende esos pesos a partir de los datos;
- luego usa esos pesos para decidir cuál clase es la más probable para una request dada.

En el script del proyecto, este bloque se implementa con:

- `StandardScaler`
- `LogisticRegression`

Y se ejecuta dentro de `train_supervised.py` cuando se usa `--task multiclass`.

Matiz importante:
el script construye `LogisticRegression` con `multi_class="auto"`.
Eso significa que, según el solver elegido, scikit-learn puede:

- resolver el problema multiclase de forma nativa;
- o recurrir internamente a un esquema One-vs-Rest como compatibilidad.

Pero metodológicamente, en la tesis se trata como un baseline de **Regresión Logística multiclase**.

---

## 3. ¿Por qué se eligió este enfoque en el TFG?

La elección de Regresión Logística multiclase responde a criterios metodológicos, prácticos y operativos.

### 3.1. Simplicidad metodológica

Es un enfoque claro y relativamente fácil de explicar.

Eso es importante en una tesis de grado porque permite:

- justificar el procedimiento;
- mantener trazabilidad entre features, entrenamiento y salida;
- evitar complejidad innecesaria;
- concentrar la discusión en la comparación de formulaciones, no en la opacidad del modelo.

En investigación aplicada, esta simplicidad es una ventaja.

---

### 3.2. Coherencia con un baseline lineal

En este TFG interesa comparar enfoques de manera controlada.

La Regresión Logística ofrece un baseline lineal sólido porque:

- es ampliamente conocido;
- tiene comportamiento estable;
- permite interpretar coeficientes;
- es una referencia razonable frente a métodos más complejos.

Eso hace que sus resultados sean defendibles metodológicamente.

---

### 3.3. Interpretabilidad

Una de las ventajas más relevantes del enfoque es que permite relacionar cada clase con pesos y coeficientes.

Eso facilita responder preguntas como:

- ¿qué variables empujan la decisión hacia una clase concreta?
- ¿qué señales son más relevantes para determinada CAPEC o tipo de ataque?
- ¿qué efecto tienen los tokens sospechosos?
- ¿qué cambia cuando se eliminan ciertas variables?

En un trabajo académico, esa interpretabilidad tiene mucho valor.

---

### 3.4. Escalabilidad y costo computacional

En problemas de seguridad web es frecuente trabajar con:

- datasets grandes;
- clases desbalanceadas;
- varias categorías de ataque;
- restricciones de tiempo y recursos.

La Regresión Logística lineal es conveniente porque:

- suele entrenar más rápido que modelos más complejos;
- permite tuning acotado;
- mantiene inferencia relativamente barata;
- es compatible con una evaluación de viabilidad operativa tipo WAF.

Esto está alineado con tu tesis, donde también importan:
- latencia;
- throughput;
- tiempo de entrenamiento;
- tamaño del modelo.

---

### 3.5. Salida probabilística útil

Otra ventaja importante es que el modelo puede producir probabilidades por clase.

Eso permite:

- elegir la clase más probable;
- calcular ROC-AUC y PR-AUC cuando corresponde;
- estudiar confianza de predicción;
- analizar errores con más detalle que usando solo la clase dura.

Para una tesis comparativa, esto aporta más información que un clasificador que solo entrega una etiqueta final sin scores interpretables.

---

### 3.6. Compatibilidad con el pipeline del proyecto

La Regresión Logística encaja muy bien con el pipeline que ya construiste porque:

- trabaja directamente con las features HTTP numéricas;
- se beneficia del `StandardScaler`;
- permite ablaciones con `--drop-features`;
- genera artefactos claros (`joblib`, `metrics.json`, `pred.csv`, `benchmark.json`);
- soporta tuning acotado dentro del bloque de entrenamiento.

Eso reduce fricción metodológica y técnica.

---

### 3.7. Manejo razonable del desbalance y de clases raras

El script contempla:
- `class_weight`
- búsqueda de hiperparámetros
- manejo de clases raras con políticas `keep`, `merge` o `drop`

Eso es especialmente útil en multiclase, donde algunas clases pueden tener muy pocas muestras y complicar tanto el split como la validación.

No resuelve mágicamente el desbalance, pero sí da herramientas prácticas para tratarlo de forma explícita.

---

## 4. Alternativas existentes a la Regresión Logística multiclase

La Regresión Logística no es la única opción posible.
Existen varios modelos que también podrían haberse usado.

### 4.1. Random Forest

#### Idea
Construir múltiples árboles y combinar sus decisiones.

#### Ventajas
- puede capturar no linealidades;
- maneja interacciones complejas;
- suele rendir bien sin mucho preprocesamiento.

#### Desventajas
- menor interpretabilidad global;
- mayor costo de inferencia que un modelo lineal;
- importancia de variables menos directa que en coeficientes lineales;
- puede complicar la discusión metodológica si se busca un baseline simple.

#### En este TFG
Es una comparación útil, pero no necesariamente el mejor baseline principal si se prioriza interpretabilidad y costo operativo.

---

### 4.2. SVM multiclase

#### Idea
Separar clases maximizando márgenes entre ellas.

#### Ventajas
- buen rendimiento en algunos problemas de clasificación;
- formulación matemática fuerte.

#### Desventajas
- tuning más delicado;
- menor interpretabilidad;
- probabilidades menos naturales;
- peor escalabilidad práctica en conjuntos grandes, sobre todo con kernels.

#### En este TFG
Puede ser interesante como referencia, pero no es tan cómodo como baseline general con foco en viabilidad operativa.

---

### 4.3. k-NN

#### Idea
Clasificar según vecinos cercanos en el espacio de features.

#### Ventajas
- simple de entender;
- sin entrenamiento complejo.

#### Desventajas
- costo de inferencia alto;
- sensible a escala y ruido;
- menos adecuado cuando se piensa en escenarios tipo tiempo real;
- interpretabilidad limitada a nivel global.

#### En este TFG
No parece la opción más defendible si se quiere discutir despliegue y eficiencia.

---

### 4.4. Naive Bayes

#### Idea
Modelar probabilidades bajo supuestos de independencia condicional.

#### Ventajas
- muy rápido;
- fácil de implementar;
- buen baseline muy simple.

#### Desventajas
- supuestos demasiado fuertes;
- puede quedar corto frente a relaciones más complejas entre variables HTTP;
- interpretabilidad estadística sí, pero menor flexibilidad que LR en este contexto.

#### En este TFG
Puede servir como baseline mínimo, pero la Regresión Logística suele ser una referencia más robusta para datos tabulares con variables diseñadas manualmente.

---

### 4.5. Modelos más complejos

También existen enfoques como:

- gradient boosting;
- redes neuronales;
- transformers o modelos profundos específicos para tráfico HTTP;
- ensembles más sofisticados.

#### Ventajas
- pueden capturar relaciones más complejas;
- pueden superar a un modelo lineal en algunos escenarios.

#### Desventajas
- mayor costo computacional;
- menor interpretabilidad;
- tuning más costoso;
- más difícil aislar si la mejora viene del enfoque o del aumento de complejidad.

#### En este TFG
Son opciones válidas para trabajo futuro o comparación adicional, pero no necesariamente como baseline principal.

---

## 5. Comparación conceptual de alternativas

| Enfoque | Escalable | Interpretable | Probabilidades útiles | Adecuado como baseline |
|---|---:|---:|---:|---:|
| Regresión Logística multiclase | Alta | Alta | Alta | Sí |
| Random Forest | Media | Media | Media | Sí, como comparación |
| SVM multiclase | Media/Baja | Media/Baja | Media | Sí, pero más costoso |
| k-NN | Baja en inferencia | Baja | Baja | No ideal |
| Modelos profundos | Media/Baja | Baja | Variable | Más complejos para este TFG |

---

## 6. Justificación específica para este proyecto

En este TFG interesa comparar enfoques de detección y clasificación de ataques web considerando:

- efectividad predictiva;
- costo computacional;
- interpretabilidad;
- viabilidad operativa en un entorno tipo WAF.

Bajo esos criterios, la Regresión Logística multiclase es una elección muy razonable porque:

1. ofrece un baseline lineal claro;
2. funciona bien con features HTTP manuales;
3. permite interpretación por coeficientes;
4. produce probabilidades por clase;
5. mantiene entrenamiento e inferencia contenidos;
6. encaja con el pipeline reproducible del proyecto;
7. permite manejar de forma explícita clases raras y desbalance.

En otras palabras, no se elige porque sea el modelo más sofisticado, sino porque es uno de los **más defendibles metodológicamente** para este contexto.

---

## 7. Relación con la tesis

Desde el punto de vista de la tesis, este enfoque permite sostener una línea experimental coherente con los objetivos del trabajo:

- comparar formulaciones one-class, multiclase y multietiqueta;
- mantener trazabilidad en la evaluación;
- priorizar soluciones compatibles con tiempo real;
- conservar interpretabilidad en la discusión de resultados;
- evitar que la complejidad del modelo desplace la pregunta de investigación central.

Esto es especialmente importante porque el objetivo del proyecto no es solo obtener buenos números, sino justificar de forma académica **qué enfoque conviene y por qué**.

---

## 8. Conclusión

Se utiliza Regresión Logística para la clasificación multiclase porque ofrece un equilibrio adecuado entre:

- simplicidad;
- escalabilidad;
- interpretabilidad;
- compatibilidad con features tabulares;
- salida probabilística;
- viabilidad en un contexto tipo WAF.

Aunque existen alternativas como Random Forest, SVM o modelos más complejos, la Regresión Logística es una elección sólida como baseline principal para esta tarea.
Su uso está alineado con el carácter investigativo del proyecto, con las restricciones computacionales del entorno y con la necesidad de justificar cada decisión metodológica de forma clara y reproducible.

---

## 9. Referencias teóricas sugeridas

Para respaldar esta justificación en la tesis, conviene citar trabajos y documentación técnica sobre:

- regresión logística;
- clasificación multiclase;
- evaluación multiclase;
- interpretabilidad en modelos lineales;
- preprocesamiento con escalado.

Referencias útiles:

- documentación oficial de `scikit-learn` sobre `LogisticRegression`
- documentación oficial de `scikit-learn` sobre `StandardScaler`
- documentación oficial de `scikit-learn` sobre métricas de clasificación
- Hastie, Tibshirani y Friedman, *The Elements of Statistical Learning*
- Bishop, *Pattern Recognition and Machine Learning*
