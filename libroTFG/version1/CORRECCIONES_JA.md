# Correcciones aplicadas — `libroVersion2CorrecionJA.tex`

**Autor de las correcciones:** Juan González  
**Archivo base:** `libroVersion1CorrecionAndres.tex`  
**Archivo resultante:** `libroVersion2CorrecionJA.tex`  
**Revisiones aplicadas:** 1 a 26 (del tutor Dr. Cristian Cappo)

> Las correcciones del compañero Andrés Román (revisiones 27+) fueron conservadas intactas.

---

## Estructura de capítulos resultante

| # | Capítulo |
|---|----------|
| 1 | Introducción |
| 2 | Marco teórico |
| 3 | Estado del arte y trabajos relacionados |
| 4 | Metodología propuesta |
| 5 | Diseño experimental |
| 6 | Implementación (pipeline) |
| 7 | Resultados y discusión |
| 8 | Protocolo de evaluación de viabilidad en tiempo real en un entorno tipo WAF |
| 9 | Conclusiones |
| 10 | Trabajos futuros |

---

## Detalle de correcciones por revisión

### Rev. 1 — Resumen: problema antes que solución
- Se agregó un párrafo introductorio al inicio del Resumen que explica el problema (ataques web, variabilidad del tráfico, variantes novedosas, limitaciones de WAF basados en reglas) **antes** de presentar el pipeline y la solución propuesta.

### Rev. 2 — Ampliar Introducción / Contexto y motivación
- La sección se expandió de 3 párrafos breves a 5 párrafos completos.
- Se explica el problema de seguridad en aplicaciones web, la variabilidad del tráfico HTTP, el desbalance entre tráfico normal y malicioso, y por qué el aprendizaje automático es una alternativa válida de análisis.

### Rev. 3 — Definiciones informales para lectores no expertos
- Se creó la sección **"Conceptos fundamentales"** en la Introducción con las siguientes definiciones:
  - Petición HTTP (*request*)
  - WAF (*Web Application Firewall*)
  - Carga útil (*payload*)
  - Firma o regla
  - Anomalía
  - Clasificación binaria
  - Clasificación multiclase
  - Clasificación multietiqueta
  - Modelo de Machine Learning

### Rev. 4 — Explicar las tres formulaciones del problema
- Se creó la sección **"Formulaciones del problema"** en la Introducción explicando por qué son relevantes:
  - Detección binaria / *one-class*: útil sin catálogo completo de ataques.
  - Clasificación multiclase: tipifica el ataque con una única etiqueta.
  - Clasificación multietiqueta: permite múltiples etiquetas simultáneas, conserva riqueza semántica.

### Rev. 5 — Reordenar frases que presentan la solución antes del problema
- En el Resumen, la frase del pipeline fue reubicada después del párrafo que presenta las limitaciones de los WAF tradicionales.

### Rev. 6 — Mejorar redacción y coherencia en la zona inicial
- La sección de Contexto y motivación fue reescrita con mejor cohesión y transiciones entre párrafos.

### Rev. 7 — Definir *payload* como carga útil
- En la sección de Conceptos fundamentales, *payload* se define como **carga útil**: la parte de la petición donde pueden aparecer datos enviados por el cliente, parámetros, cuerpo o cadenas utilizadas por un atacante.
- Primera mención normalizada como `carga útil (payload)`.

### Rev. 8 — Explicar "ataques nuevos" como variantes sin cobertura
- En el Planteamiento del problema, "ataques emergentes" se aclaró como: *variantes o patrones de ataque para los cuales todavía no existen reglas, firmas o ejemplos suficientes en el sistema*.

### Rev. 9 — Justificar la mención de ModSecurity
- ModSecurity ya no aparece como ejemplo aislado. Se explica que es un **ejemplo representativo de WAF basado en reglas/firmas**, útil para contrastar el enfoque tradicional con enfoques basados en anomalías y aprendizaje automático.

### Rev. 10 — Mantener "Marco teórico" como capítulo
- El capítulo se conservó con ese nombre y se amplió para funcionar como fundamentación conceptual real.

### Rev. 11 — Describir HTTP de manera más formal
- La sección "El protocolo HTTP y la superficie de ataque" describe ahora el protocolo alineado al estándar: método, URI, versión, cabeceras y cuerpo opcional.
- Se aclara que en este trabajo se usan principalmente método, URI, cuerpo y, cuando está disponible, información derivada de cabeceras.

### Rev. 12 — Ampliar la comparación WAF firmas vs. anomalías
- La sección se organizó en dos subsecciones formales con sus respectivos párrafos de **ventajas y limitaciones**:
  - Enfoque basado en firmas o reglas
  - Enfoque basado en anomalías

### Rev. 13 — Integrar "Base matemática" dentro del Marco teórico
- El antiguo `\chapter{Base matemática de los modelos}` fue convertido a `\section{Fundamentos matemáticos de los modelos}` dentro del Marco teórico.
- Todas las secciones descendieron un nivel jerárquico (section → subsection, subsection → subsubsection).

### Rev. 14 — Marco teórico solo conceptual, sin implementación
- Se eliminaron de la parte teórica las referencias a scripts, nombres de archivos y decisiones experimentales concretas. Todo lo conceptual quedó en el Marco teórico.

### Rev. 15 — Datasets y splits movidos a Diseño experimental
- Las descripciones específicas de disponibilidad de etiquetas por dataset permanecen en el capítulo de Diseño experimental, no en la sección matemática.

### Rev. 16 — Normalizar "variables objetivo"
- Donde se usaba "objetivos" para referirse a `label_binary`, `label_multiclass` o `label_multilabel`, se reemplazó por **"variables objetivo"**, **"variables de salida"** o **"etiquetas"** para evitar confusión con los objetivos de la tesis.

### Rev. 17 — Normalizar idioma (primera mención)
- `petición HTTP (request)` → primera mención normalizada, luego se prefiere "petición HTTP".
- `características (features)` → primera mención normalizada.
- `carga útil (payload)` → primera mención normalizada.

### Rev. 18 — Aclarar qué significa "request" en el alcance del trabajo
- Se agregó un párrafo explícito en la sección de HTTP indicando que no se procesa necesariamente el mensaje HTTP completo, sino los campos disponibles y normalizados en los datasets: método, URI, cuerpo y campos derivados de cabeceras cuando existen.

### Rev. 19 — Definir CAPEC antes de usarlo
- Se creó la sección **"CAPEC como taxonomía de patrones de ataque"** en el Marco teórico, antes de cualquier uso matemático o metodológico del término.
- Explica que CAPEC es una taxonomía de patrones de ataque mantenida por MITRE.
- Se eliminó la sección CAPEC duplicada que existía en el capítulo de Datasets.

### Rev. 20 — Definir tarea de clasificación en Machine Learning
- Se creó la subsección **"Tarea de clasificación en aprendizaje automático"** en los fundamentos matemáticos, explicando qué aprende un modelo a partir de ejemplos etiquetados y cómo generaliza a nuevos ejemplos.

### Rev. 21 — Mover nota de disponibilidad de etiquetas
- La nota "no todos los datasets soportan todas las tareas" fue eliminada de la sección matemática y su contenido ya se trata en el capítulo de Diseño experimental.

### Rev. 22 — Justificar la función indicadora I[·]
- Se agregó una explicación de por qué se introduce: se usará para expresar predicciones binarias, coincidencias exactas y métricas de evaluación como exactitud y *Exact Match Ratio*.

### Rev. 23 — Justificar el detector one-class en Metodología
- En el capítulo de Diseño experimental, la sección "Modelos seleccionados" se amplió con:
  - **Justificación del enfoque one-class**: por qué es útil entrenar solo con tráfico normal frente a ataques sin cobertura.
  - **Justificación de la variante escalable**: por qué el One-Class SVM clásico es impracticable para grandes volúmenes y cómo la variante SGD resuelve ese problema.

### Rev. 24 — Tabla de símbolos al inicio de la sección matemática
- La tabla de símbolos se movió al inicio de la subsección de notación, antes de usar los símbolos en ecuaciones.

### Rev. 25 — Mover "correspondencia con la implementación" a Metodología
- El contenido sobre las 25 características y las variables objetivo (`label_binary`, `label_multiclass`, `label_multilabel`) fue movido al capítulo de Diseño experimental, dentro de la nueva subsección **"Correspondencia entre representación y variables objetivo"**.
- Se detalla la lógica de derivación de `label_multiclass` desde la anotación CAPEC multietiqueta de SR-BH.

### Rev. 26 — Pipeline concreto One-Class SVM en Metodología, no en Marco teórico
- La implementación concreta `StandardScaler → Nystroem → SGDOneClassSVM` fue movida al capítulo de Diseño experimental, dentro de la sección "Modelos seleccionados".
- En el Marco teórico quedó únicamente la formulación **conceptual** del detector one-class (intuición, formulación matemática general, justificación del enfoque).

---

## Correcciones adicionales de revisión final

| Problema | Corrección aplicada |
|----------|---------------------|
| Referencia hardcodeada `Sección~3.5` (capítulo eliminado) | Reemplazada por referencia descriptiva al Marco teórico |
| Título de subsección `comparativa` en minúscula | Corregido a `Comparativa` |
| Typo `extit{tuning}` | Corregido a `\textit{tuning}` |
| Sección CAPEC duplicada en capítulo de Datasets | Eliminada (ya definida en Marco teórico, Rev. 19) |
| Nombres de capítulos no alineados a la estructura obligatoria | Renombrados según la estructura requerida |
| Capítulo único "Conclusiones y trabajo futuro" | Separado en dos capítulos: **Conclusiones** y **Trabajos futuros** |

---

agregue definicion de capec y quite la parte que hablaba de implementacion en la parte teorica

## Qué NO se modificó

- Capítulos de Resultados y discusión, Implementación (pipeline) y Protocolo de evaluación WAF: conservados intactos (correcciones de Andrés).
- Valores numéricos, tablas de resultados, métricas y conclusiones experimentales.
- Bibliografía existente: no se agregaron ni eliminaron claves bibliográficas.
- Figuras, tablas, comandos LaTeX y paquetes del preámbulo.
