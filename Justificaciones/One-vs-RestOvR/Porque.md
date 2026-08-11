# Justificación del uso de One-vs-Rest (OvR) para clasificación multietiqueta

## 1. Contexto del problema

En este Trabajo Final de Grado se aborda la detección y clasificación de ataques en tráfico HTTP.
Dentro de este contexto, la tarea de **clasificación multietiqueta** aparece cuando una misma solicitud puede estar asociada a **más de una etiqueta al mismo tiempo**.

Esto ocurre, por ejemplo, cuando una request presenta simultáneamente señales compatibles con distintos patrones de ataque, como:

- SQL Injection
- Cross-Site Scripting (XSS)
- Path Traversal
- técnicas de evasión u ofuscación

En este tipo de escenario, la formulación multiclase tradicional no es suficiente, ya que en multiclase cada instancia solo puede pertenecer a **una única clase**.
Por ello se necesita una estrategia específicamente diseñada para **salidas múltiples y no mutuamente excluyentes**.

---

## 2. ¿Qué es One-vs-Rest?

One-vs-Rest (OvR), también llamado en la literatura **One-vs-All**, es una estrategia de descomposición del problema multietiqueta en varios problemas binarios independientes.

La idea es simple:

- para cada etiqueta se entrena un clasificador binario independiente;
- ese clasificador responde si la etiqueta está presente o no en la instancia.

Si existen `L` etiquetas posibles, se entrena un conjunto de `L` clasificadores binarios.

En términos prácticos:

- un modelo aprende a decidir si una request pertenece a una etiqueta concreta;
- otro modelo hace lo mismo con la siguiente etiqueta;
- y así sucesivamente hasta cubrir todas las etiquetas.

---

## 3. ¿Por qué se eligió OvR en este TFG?

La elección de OvR responde a criterios metodológicos, prácticos y de eficiencia, especialmente relevantes para un trabajo de investigación aplicado a seguridad web.

### 3.1. Simplicidad metodológica

OvR es una estrategia conceptual y matemáticamente simple.
Esto resulta adecuado para una tesis de grado porque permite:

- explicar claramente el procedimiento;
- justificar cada decisión experimental;
- mantener trazabilidad en la construcción del modelo;
- evitar complejidad innecesaria en una etapa donde lo importante es comparar enfoques de forma rigurosa.

En un trabajo de investigación, esta simplicidad no es una debilidad: es una ventaja, porque facilita la interpretación de resultados.

---

### 3.2. Escalabilidad

En problemas de seguridad web es común trabajar con:

- datasets grandes;
- clases desbalanceadas;
- muchas etiquetas raras o poco frecuentes;
- restricciones de tiempo y recursos.

OvR escala bien porque:

- descompone el problema en múltiples clasificadores binarios;
- permite entrenamiento relativamente eficiente;
- es compatible con modelos lineales de bajo costo computacional;
- puede adaptarse mejor a escenarios donde la inferencia debe ser rápida.

Esto es particularmente importante en una tesis orientada a un contexto tipo **WAF**, donde la latencia y el costo operativo importan tanto como la precisión.

---

### 3.3. Interpretabilidad

Una de las ventajas más relevantes de OvR en este proyecto es que permite asociar cada etiqueta con un clasificador individual.
Eso facilita responder preguntas como:

- ¿qué variables empujan la predicción de una etiqueta concreta?
- ¿qué señales son más relevantes para SQLi?
- ¿qué features influyen más en XSS?
- ¿qué grupos de variables aportan valor real?

En un trabajo de investigación, esta trazabilidad es fundamental.
No solo interesa predecir bien, sino también poder explicar **por qué** el modelo toma ciertas decisiones.

---

### 3.4. Compatibilidad con modelos lineales

En este TFG se priorizan modelos lineales por su equilibrio entre:

- rendimiento;
- interpretabilidad;
- costo computacional;
- facilidad de despliegue.

OvR se integra naturalmente con modelos como la regresión logística, lo que permite:

- entrenar un modelo por etiqueta;
- analizar coeficientes;
- comparar importancia relativa de features;
- mantener una línea base robusta y comprensible.

---

### 3.5. Adecuación al dominio de seguridad

En clasificación multietiqueta de ataques web, las etiquetas no siempre son mutuamente excluyentes.
Un mismo request puede estar relacionado con más de un patrón de ataque o con combinaciones de técnicas.

OvR es útil porque:

- no fuerza a elegir una única clase;
- preserva la posibilidad de múltiples salidas activas;
- encaja con la estructura real del problema;
- evita simplificaciones excesivas del fenómeno.

---

## 4. Alternativas existentes a OvR

OvR no es la única estrategia posible para multietiqueta.
Existen otras formulaciones que pueden compararse desde un punto de vista metodológico.

### 4.1. Binary Relevance

Binary Relevance es probablemente la alternativa más cercana a OvR.
En la práctica, muchas veces se usan como enfoques equivalentes o muy similares.

#### Idea
Entrenar un clasificador binario por etiqueta.

#### Ventajas
- simple;
- escalable;
- fácil de implementar;
- buena línea base.

#### Desventajas
- no modela dependencias entre etiquetas;
- trata cada salida de forma independiente.

#### Relación con OvR
En multietiqueta, Binary Relevance y OvR suelen considerarse muy próximos.
Ambos adoptan la idea de descomponer el problema en múltiples decisiones binarias.

---

### 4.2. Classifier Chains

Classifier Chains es una alternativa más sofisticada.

#### Idea
Cada clasificador predice una etiqueta y además utiliza las predicciones de etiquetas anteriores como entrada.

#### Ventajas
- modela dependencias entre etiquetas;
- puede capturar correlaciones útiles entre ataques;
- en algunos casos mejora sobre OvR.

#### Desventajas
- más sensible al orden de las etiquetas;
- los errores pueden propagarse a lo largo de la cadena;
- es menos estable;
- su interpretación es más compleja.

#### En este TFG
Classifier Chains puede ser una buena comparación experimental, pero como baseline principal es menos conveniente que OvR porque añade complejidad y reduce trazabilidad.

---

### 4.3. Label Powerset

Label Powerset transforma cada combinación de etiquetas en una sola clase.

#### Idea
Si una instancia tiene la combinación `{XSS, SQLi}`, esa combinación se trata como una clase distinta.

#### Ventajas
- captura correlaciones entre etiquetas;
- modela combinaciones específicas.

#### Desventajas
- puede generar una explosión de clases;
- es poco escalable;
- muchas combinaciones pueden ser raras o incluso únicas;
- el problema crece rápidamente si hay muchas etiquetas.

#### En este TFG
Label Powerset no suele ser la mejor opción si el número de combinaciones es alto o si el dataset es disperso.
Puede ser útil como referencia, pero no como elección principal.

---

### 4.4. Métodos específicos de multietiqueta más complejos

También existen enfoques como:

- ML-kNN;
- redes neuronales multietiqueta;
- modelos profundos con atención;
- arquitecturas jerárquicas;
- ensembles especializados.

#### Ventajas
- pueden capturar relaciones más complejas;
- pueden rendir mejor en escenarios específicos.

#### Desventajas
- mayor costo computacional;
- menor interpretabilidad;
- mayor complejidad de implementación;
- más difícil justificar como baseline en una tesis de grado;
- requieren más ajuste de hiperparámetros.

#### En este TFG
Estos métodos pueden mencionarse como alternativas teóricas o trabajo futuro, pero no parecen la mejor base experimental principal dadas las prioridades del proyecto.

---

## 5. Comparación conceptual de alternativas

| Enfoque | Modela dependencias entre etiquetas | Escalable | Interpretable | Adecuado como baseline |
|---|---:|---:|---:|---:|
| OvR / Binary Relevance | Baja | Alta | Alta | Sí |
| Classifier Chains | Media/Alta | Media | Media | Sí, como comparación |
| Label Powerset | Alta | Baja | Media | Solo en escenarios pequeños |
| Modelos profundos multilabel | Alta | Media/Baja | Baja | Más complejos para este TFG |

---

## 6. Justificación específica para este proyecto

En este TFG interesa comparar enfoques de detección y clasificación de ataques web considerando:

- efectividad predictiva;
- costo computacional;
- interpretabilidad;
- viabilidad operativa en un entorno tipo WAF.

Bajo esos criterios, OvR es una elección muy razonable porque:

1. permite una formulación natural de la multietiqueta;
2. mantiene un costo de entrenamiento e inferencia contenido;
3. facilita la explicación por etiqueta;
4. se adapta bien a modelos lineales;
5. ofrece una base sólida para comparación con otras estrategias.

En otras palabras, OvR no se elige porque sea el modelo más sofisticado, sino porque es uno de los **más defendibles metodológicamente** para este contexto.

---

## 7. Relación con la tesis

Desde el punto de vista de la tesis, OvR permite sostener una línea experimental coherente con los objetivos del trabajo:

- comparar formulaciones one-class, multiclase y multietiqueta;
- mantener trazabilidad en la evaluación;
- priorizar soluciones compatibles con tiempo real;
- conservar interpretabilidad en la discusión de resultados;
- evitar que la complejidad del modelo desplace la pregunta de investigación central.

Esto es especialmente importante porque el objetivo del proyecto no es solo obtener buenos números, sino justificar de forma académica **qué enfoque conviene y por qué**.

---

## 8. Conclusión

Se utiliza One-vs-Rest (OvR) en la clasificación multietiqueta porque ofrece un equilibrio adecuado entre:

- simplicidad;
- escalabilidad;
- interpretabilidad;
- compatibilidad con modelos lineales;
- viabilidad en un contexto tipo WAF.

Aunque existen alternativas como Classifier Chains y Label Powerset, OvR es una elección sólida como baseline principal para este TFG.
Su uso está alineado con el carácter investigativo del proyecto, con las restricciones computacionales del entorno y con la necesidad de justificar cada decisión metodológica de forma clara y reproducible.

---

## 9. Referencias teóricas sugeridas

Para respaldar esta justificación en la tesis, conviene citar trabajos clásicos y documentación técnica sobre:

- clasificación multietiqueta;
- One-vs-Rest / One-vs-All;
- regresión logística;
- evaluación de multietiqueta;
- interpretabilidad en modelos lineales.

Referencias útiles:

- Rifkin & Klautau, *In Defense of One-Vs-All Classification*: https://www.jmlr.org/papers/volume5/rifkin04a/rifkin04a.pdf
- Tsoumakas & Katakis, *Multi-Label Classification: An Overview*: https://link.springer.com/chapter/10.1007/978-0-387-30424-3_12
- documentación oficial de `scikit-learn` sobre `OneVsRestClassifier`: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
- documentación de `scikit-learn` sobre métricas multilabel: https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics