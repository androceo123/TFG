# Justificación Metodológica del Dataset SR-BH 2020
**DOI: [10.7910/DVN/OGOIXX](https://doi.org/10.7910/DVN/OGOIXX) — Harvard Dataverse**

---

## 1. Criterio General de Selección

En este trabajo se seleccionó el **SR-BH 2020 multi-label dataset**, publicado en Harvard Dataverse bajo el DOI `10.7910/DVN/OGOIXX`, como fuente principal para los experimentos de clasificación multietiqueta y como fuente complementaria para los experimentos binarios y multiclase. La elección se fundamenta en su alineación con el objetivo central del trabajo: evaluar enfoques de detección de anomalías y clasificación de ataques en tráfico HTTP a nivel de solicitud, comparando formulaciones one-class, multiclase y multietiqueta.

La decisión **no** se apoya en asumir que SR-BH 2020 sea un dataset libre de errores. Por el contrario, su uso se justifica por una combinación de ventajas metodológicas y limitaciones explícitamente declaradas. El dataset aporta una estructura que los conjuntos clásicos no ofrecen de forma suficiente: etiquetas asociadas a patrones CAPEC, granularidad a nivel de request HTTP, volumen considerable, disponibilidad pública irrestricta y la posibilidad de formular el problema como binario, multiclase o multietiqueta sobre un mismo corpus. Al mismo tiempo, la literatura reciente reporta ruido de etiquetado, redundancia estructural y categorías CAPEC discutibles. Por ello, se lo utiliza en este trabajo como **benchmark académico público y reproducible**, no como representación definitiva del tráfico productivo de una organización.

Esta posición permite sostener una justificación equilibrada y metodológicamente honesta: SR-BH 2020 es pertinente porque cubre dimensiones del problema que CSIC 2010, ECML/PKDD 2007 y otros datasets clásicos simplemente no pueden cubrir; pero sus resultados deben interpretarse dentro de las amenazas a la validez propias del dataset, que quedan explícitamente incorporadas al análisis.

---

## 2. Descripción Técnica del Dataset

SR-BH 2020 fue construido por investigadores de la Universidad de Alcalá y la Universidad Internacional de La Rioja (UNIR): Tomás Sureda Riera, Juan-Ramón Bermejo Higuera, Javier Bermejo Higuera, José-Javier Martínez Herráiz y Juan-Antonio Sicilia Montalvo. Fue publicado en Harvard Dataverse en 2022 bajo licencia **CC0 1.0** (dominio público).

| Atributo | Valor |
|---|---|
| **Período de captura** | 12 días de julio de 2020 |
| **Entorno de captura** | Honeypot real: WordPress + Apache expuesto a Internet |
| **Herramienta de captura** | ModSecurity v2.9.2 con OWASP CRS v3.3.0 en modo "Detection only" |
| **Total de solicitudes** | 907.814 |
| **Solicitudes normales** | 525.195 (57,8%) |
| **Solicitudes anómalas** | 382.619 (42,2%) |
| **Features por registro** | 24 |
| **Etiquetas (labels)** | 13 (1 normal + 12 categorías CAPEC) |
| **Tipo de clasificación** | Multi-label (una petición puede tener múltiples etiquetas simultáneas) |
| **Versión actual** | 1.2 |
| **Licencia** | CC0 1.0 |

Los logs generados diariamente por ModSecurity fueron revisados de forma **manual y semi-automática** para corregir la asignación normal/ataque y asegurar la clasificación CAPEC correspondiente. El resultado final es un dataset en formato CSV con 24 features y 13 etiquetas por registro, apuntando especialmente al entrenamiento de modelos de detección de ataques web.

Las 12 categorías de ataque corresponden a la taxonomía **CAPEC** (*Common Attack Pattern Enumeration and Classification*) de MITRE, incluyendo SQLi, XSS, Command Injection, Path Traversal, HTTP Request Smuggling, entre otras, alineadas con el **OWASP Top 10**.

---

## 3. Pertinencia Respecto al Objetivo del Trabajo

La pertinencia del SR-BH 2020 se explica por la naturaleza comparativa del trabajo. La tesis no se limita a responder si una solicitud es normal o maliciosa; también analiza la capacidad de distintos modelos para clasificar el **tipo** de ataque. En ese sentido, SR-BH 2020 permite construir tres vistas experimentales sobre un mismo corpus:

1. **Vista binaria:** normal frente a ataque/anomalía.
2. **Vista multiclase:** una clase principal por solicitud, derivada a partir de las etiquetas disponibles.
3. **Vista multietiqueta:** una o más etiquetas CAPEC por solicitud.

La tercera vista es especialmente relevante porque una misma solicitud HTTP puede contener indicios de más de una categoría de ataque simultáneamente. Por ejemplo, una request puede combinar codificación de caracteres, manipulación de parámetros, secuencias de path traversal y patrones compatibles con inyección de comandos. En un dataset estrictamente binario, esa complejidad queda colapsada en una única decisión normal/ataque. En SR-BH 2020, en cambio, se puede evaluar si el modelo identifica el conjunto completo de etiquetas asociadas a la solicitud.

Esta característica es coherente con un escenario tipo WAF, donde no solo interesa detectar una solicitud maliciosa, sino también producir información útil sobre la **naturaleza** del ataque. Aunque el trabajo no implementa un WAF productivo, la granularidad multietiqueta permite evaluar una dimensión más informativa que la simple clasificación binaria.

La implementación transforma SR-BH 2020 en un esquema común con tres columnas principales de etiqueta:
- `label_binary`: normal frente a ataque/anomalía.
- `label_multiclass`: clase principal por solicitud.
- `label_multilabel`: conjunto de etiquetas CAPEC asociadas a la solicitud.

Esta transformación permite ejecutar sobre el mismo corpus los tres enfoques comparados, con métricas diferenciadas para cada formulación.

---

## 4. Relación con CAPEC y su Valor Metodológico

SR-BH 2020 utiliza etiquetas basadas en **CAPEC** (*Common Attack Pattern Enumeration and Classification*), una taxonomía mantenida por MITRE para organizar patrones de ataque conocidos. El uso de CAPEC aporta trazabilidad conceptual porque las clases no dependen únicamente de nombres internos del dataset, sino de una clasificación externa ampliamente reconocida en la industria de la seguridad del software.

En el contexto de este trabajo, CAPEC permite relacionar los resultados del modelo con categorías interpretables de ataque, facilita el análisis por tipo de amenaza, el reporte de métricas por etiqueta y la comparación con trabajos que también utilizan taxonomías de ataques web.

No obstante, el uso de CAPEC también introduce una limitación importante: no todas las categorías CAPEC son igualmente observables a partir de una única solicitud HTTP aislada. Algunas categorías pueden depender de contexto de sesión, respuesta del servidor o comportamiento acumulado. Por esa razón, las etiquetas CAPEC de SR-BH 2020 se utilizan como referencia experimental, pero no se interpretan como una verdad operacional perfecta.

---

## 5. Referencias Favorables al Dataset

### 5.1 Artículo original: el dataset como contribución al estado del arte

**Sureda Riera, T. et al. (2022).** *A new multi-label dataset for Web attacks CAPEC classification using machine learning techniques.* Computers & Security, 120, 102788. https://doi.org/10.1016/j.cose.2022.102788

El artículo original presenta SR-BH 2020 como el primer dataset derivado de **tráfico HTTP real** diseñado específicamente para entrenamiento de modelos de clasificación multietiqueta con taxonomía CAPEC. Los autores destacan que los datasets existentes etiquetan las solicitudes únicamente como normales o ataques, sin indicar qué tipo de ataque está presente. SR-BH 2020 fue propuesto precisamente para cubrir esa brecha. Esta referencia favorece directamente la elección porque el objetivo del trabajo incluye comparar un enfoque multietiqueta: sin SR-BH 2020, ese experimento simplemente no puede realizarse con datos públicos.

El trabajo demuestra que usando encodings basados en valores ASCII y modelos de clasificación multi-label (LightGBM, CatBoost), se logran resultados prometedores. La combinación del modelo de dos fases con `MultiOutputClassifier` y CatBoost muestra la mayor superioridad entre escenarios de criticidad variados.

### 5.2 E-WebGuard: Mejoras con arquitecturas de deep learning

**Zhou, L. et al. (2024/2025).** *E-WebGuard: Enhanced neural architectures for precision web attack detection.* Computers & Security. https://doi.org/10.1016/j.cose.2024.104322

Este estudio emplea SR-BH 2020 junto con CSIC-2010 para evaluar cuatro arquitecturas (Char-SVM, Char-LSTM, CNN-SVM, CNN-Bi-LSTM). La elección del dataset se justifica por ser de los pocos conjuntos públicos con clasificación multi-clase de ataques web. El CNN-Bi-LSTM alcanza un **99,63% de accuracy** en clasificación multi-clase sobre SR-BH 2020, superando los resultados originales e indicando que el dataset tiene complejidad suficiente para entrenar modelos de alto rendimiento.

El mismo paper posiciona SR-BH 2020 como la única alternativa moderna a datasets obsoletos para el problema específico de detección de ataques web por tipo, y documenta que el accuracy promedio reportado en el trabajo original fue solo del 88,4%, lo cual subraya la necesidad de explorar el potencial de los enfoques de deep learning.

### 5.3 Feature selection basada en conocimiento experto en seguridad

**Gniewkowski, M. et al. (2024).** *Capturing the security expert knowledge in feature selection for web application attack detection.* Proceedings of LADC 2024, ACM. https://doi.org/10.1145/3697090.3699797

Este trabajo utiliza SR-BH 2020 para entrenar modelos One-Class SVM de detección de ataques web, reportando que con 100 features seleccionadas se logra un TPR de 78,87% con un FPR de 5,18% y un AUC de 0,84, superando las baselines de ModSecurity configurado con OWASP CRS estándar. Este resultado muestra que el dataset puede ser empleado directamente para el desarrollo y evaluación de sistemas WAF basados en aprendizaje automático.

### 5.4 Detección no supervisada con Gaussian Mixture Model

**Tran-Thi, M.H. et al. (2024).** *An Effective Unsupervised Cyber Attack Detection on Web Applications Using Gaussian Mixture Model.* CITA 2024, LNNS 882, Springer. https://doi.org/10.1007/978-3-031-74127-2_39

Un GMM entrenado exclusivamente sobre el tráfico normal de SR-BH 2020 logró detectar ataques no vistos con un **accuracy y F1-score del 91%**. Este experimento demuestra que el dataset es válido incluso para paradigmas sin etiquetas (anomaly detection), lo que amplía su utilidad más allá de la clasificación supervisada y refuerza la pertinencia de la elección para trabajos que comparan múltiples paradigmas de aprendizaje.

### 5.5 Reconocimiento en surveys sistemáticos de datasets NIDS

**Goldschmidt, O. et al. (2025).** *Network Intrusion Datasets: A Survey, Limitations, and Recommendations.* arXiv:2502.06688.

Este survey sistemático de 89 datasets públicos para NIDS reconoce SR-BH 2020 como una de las pocas fuentes para la evaluación de modelos de detección de intrusiones HTTP(S) multi-label, destacando su singularidad en el espacio de datasets de ataques web.

### 5.6 Uso posterior en múltiples trabajos de aprendizaje profundo

El paper WAMM (Osama et al., 2025/2026) elige SR-BH 2020 como base para su pipeline de mejora porque era el único dataset web con tráfico real y clasificación multi-label disponible públicamente. Tras las fases de limpieza, deduplicación, reetiquetado y augmentación, XGBoost sobre el dataset refinado alcanzó un **99,59% de accuracy** con latencia de inferencia a nivel de microsegundos, validando que SR-BH 2020 —una vez refinado— puede ser la base para sistemas WAF de producción.

---

## 6. Críticas y Limitaciones Documentadas en la Literatura

La elección de SR-BH 2020 requiere declarar explícitamente sus limitaciones. La crítica más sistemática y técnicamente detallada proviene del trabajo WAMM, que aplica un proceso de depuración, relabeling asistido por LLM y augmentación de datos sobre SR-BH 2020. Ese estudio no se limita a proponer un modelo alternativo; también reporta problemas concretos en la calidad del dataset original que deben incorporarse como amenazas a la validez.

### 6.1 Solicitudes maliciosas etiquetadas como benignas (WAMM, arXiv:2512.23610)

**Osama, H. et al. (2025/2026).** *Enhanced Web Payload Classification Using WAMM.* arXiv:2512.23610v2.

WAMM reporta que un script de detección basado en 127 expresiones regulares, aplicado sobre la clase benigna de SR-BH 2020, identificó posibles muestras mal etiquetadas. Según la Tabla I de ese trabajo, el conteo por tipo de ataque fue el siguiente:

| Tipo de ataque identificado en benignos | Filas reportadas como mal etiquetadas |
|---|---:|
| CMDi | 16.740 (3,348%) |
| Server-Side Template Injection (SSTI) | 2.500 (0,500%) |
| SQLi | 20.800 (4,160%) |
| XSS | 2.335 (0,467%) |
| Path Traversal | 8.000 (1,600%) |

Adicionalmente, WAMM reporta que la clase benigna completa fue procesada con el modelo Qwen3-8B y que la evaluación asistida por LLM identificó **48.522 solicitudes maliciosas etiquetadas incorrectamente como benignas**. Esta cifra es especialmente crítica para modelos one-class, cuyo entrenamiento se basa precisamente en la distribución de la clase normal: si esa distribución contiene payloads de ataque, el modelo aprende como normales patrones que en realidad corresponden a amenazas reales.

En este trabajo, esta limitación se incorpora como amenaza a la validez interna. Los resultados sobre SR-BH no se presentan como rendimiento esperado en producción, sino como evidencia experimental bajo las etiquetas disponibles.

### 6.2 Categorías CAPEC problemáticas o difíciles de observar

WAMM también reporta inconsistencias en categorías CAPEC específicas:

- **CAPEC-33 / HTTP Request Smuggling:** algunas muestras carecen de los headers requeridos (`Content-Length`, `Transfer-Encoding`), lo que dificultaría sostener la presencia del ataque desde la request aislada.
- **CAPEC-16 / Dictionary-based Password Attack:** ciertas solicitudes son difíciles de distinguir de intentos normales de login sin información de frecuencia, sesión o comportamiento acumulado.
- **CAPEC-274 / HTTP Verb Tampering:** la confirmación del ataque puede depender de la respuesta de la aplicación, no solo del método HTTP de la request.
- **CAPEC-194 / Fake the Source of Data:** WAMM indica que esta clase contendría ataques SSRF, lo que sugiere una discrepancia entre el nombre de la categoría y el patrón real representado.
- **CAPEC-248 / Command Injection:** se reporta una presencia extremadamente reducida o problemática en el análisis.

Estas observaciones son metodológicamente importantes porque muestran que no todas las etiquetas poseen el mismo grado de confiabilidad ni la misma observabilidad desde los campos disponibles. Por ello, el análisis de resultados no descansa únicamente en métricas agregadas; se priorizan métricas macro y métricas por clase/etiqueta.

### 6.3 Redundancia de muestras benignas

WAMM reporta un proceso de deduplicación basado en MinHash y Locality-Sensitive Hashing (LSH) para reducir redundancia estructural en la clase benigna. La existencia de redundancia puede inflar el rendimiento experimental si instancias muy similares aparecen en entrenamiento y evaluación, aun cuando no exista fuga directa de filas idénticas. Esta amenaza se declara como parte de la validez interna y se mitiga parcialmente mediante una separación entrenamiento/prueba fija y explícita.

### 6.4 Desbalance y clases de baja frecuencia

SR-BH 2020 presenta clases con soportes muy desiguales. El desbalance afecta especialmente a la interpretación de métricas agregadas: un modelo puede obtener resultados altos en micro-F1 o accuracy y, al mismo tiempo, tener bajo desempeño en clases minoritarias. En clasificación multietiqueta, cada etiqueta tiene su propio soporte y su propia dificultad. Por esta razón, el trabajo prioriza métricas complementarias: F1 micro, F1 macro, Hamming Loss, Jaccard, Exact Match Ratio y métricas por etiqueta. La inclusión de F1 macro permite detectar si el rendimiento se concentra en clases frecuentes.

### 6.5 Especificidad tecnológica y período de captura restringido

SR-BH 2020 fue recolectado sobre un servidor WordPress durante **12 días** de un único mes (julio de 2020). Esta configuración aporta realismo respecto a tráfico capturado en un servicio expuesto, pero limita la generalización en dos dimensiones: (a) los patrones observados pueden estar condicionados por el CMS, la configuración del servidor y las reglas activas, y (b) la brevedad del período puede sobrerepresentar campañas de escaneo masivo que ocurrieron durante esa semana específica, sin capturar variaciones estacionales de larga duración.

Por ese motivo, los resultados obtenidos sobre SR-BH no se extrapolan directamente a cualquier aplicación web, framework o infraestructura. El dataset se usa para comparar modelos bajo condiciones reproducibles, no para afirmar rendimiento universal en entornos productivos heterogéneos.

### 6.6 Cobertura limitada de ataques evasivos y ofuscados

Los ataques capturados en el honeypot corresponden mayoritariamente a payloads no ofuscados detectables por ModSecurity. WAMM añade una fase de augmentación de datos de ataque realistas para cubrir variantes ofuscadas y polimórficas que no están representadas en el dataset original, lo que limita la robustez de los modelos entrenados sobre él frente a técnicas de evasión modernas.

---

## 7. Panorama de Datasets HTTP Públicos: ¿Existen Alternativas Multi-label?

La siguiente sección investiga en profundidad si existen datasets públicos que puedan competir con SR-BH 2020 en el dominio específico de la detección multi-label de ataques web a nivel de solicitud HTTP. La conclusión anticipada, respaldada por la revisión de literatura, es que **no existe ningún dataset público que combine las cinco propiedades que SR-BH 2020 ofrece simultáneamente**: (1) tráfico HTTP real, (2) granularidad a nivel de request, (3) clasificación multi-label, (4) taxonomía CAPEC, (5) licencia abierta y volumen suficiente.

### 7.1 CSIC-2010 (HTTP Dataset CSIC 2010)

**Origen:** Consejo Superior de Investigaciones Científicas (CSIC), España, 2010.  
**Descripción:** Generado automáticamente mediante solicitudes simuladas a una aplicación de e-commerce. Contiene ~36.000 solicitudes normales y ~25.000 anómalas.  
**Clasificación:** **Binaria** (normal / anómalo). No especifica el tipo de ataque.  
**Limitaciones críticas:** Completamente sintético —el campo HOST es siempre el mismo y no incluye información temporal—, lo que impide analizar correlaciones entre intentos de ataque. La distribución de ataques no refleja el tráfico de Internet real. Aunque sigue siendo el dataset más citado para detección de anomalías web, el propio ecosistema de investigación reconoce que representa un entorno muy controlado y no actualizado.  
**Rol frente a SR-BH 2020:** CSIC-2010 es útil como referencia binaria clásica y para comparación con literatura histórica, pero es incapaz de satisfacer el objetivo de clasificación multietiqueta por tipo de ataque. Su uso en este trabajo es complementario, no sustitutivo.

### 7.2 ECML/PKDD 2007

**Origen:** ECML/PKDD 2007 Discovery Challenge, tráfico real del servidor web de la Universidad de Montpellier.  
**Descripción:** ~50.000 registros con tráfico web real, pero **anonimizado** mediante la sustitución de nombres y valores de parámetros por valores aleatorios.  
**Clasificación:** **Binaria** (normal / ataque). Sin especificación de tipo.  
**Limitaciones críticas:** La anonimización destruye la semántica de los payloads de ataque, eliminando la información que los modelos de análisis de contenido necesitan para distinguir tipos de ataque. El dataset tiene más de 17 años y no cubre los vectores de ataque modernos. Varios estudios lo describen como obsoleto para los estándares actuales.  
**Rol frente a SR-BH 2020:** Referencia histórica. No viable para multietiqueta.

### 7.3 TorpEda (ITEFI-CSIC, 2024)

**Origen:** Instituto de Técnicas y Tecnologías Físicas y de la Información (ITEFI), CSIC, España, 2024.  
**Descripción:** Dataset más reciente del grupo CSIC, mencionado en múltiples trabajos de 2024 como referencia alternativa a CSIC-2010. Disponible en https://www.tic.itefi.csic.es/torpeda/datasets.html  
**Clasificación:** Multi-clase (incluye tipos de ataque nativos como SQLi, XSS, etc.), con mayor granularidad que CSIC-2010.  
**Limitaciones críticas:** Aunque representa un avance claro frente a CSIC-2010, TorpEda **no es multi-label**: cada registro tiene una sola etiqueta de clase. Tampoco utiliza la taxonomía CAPEC, lo que reduce su comparabilidad con estándares industriales. Su adopción en la comunidad científica es aún limitada —aparece citado pero no como base experimental dominante en ningún paper de referencia— y la documentación pública sobre sus características técnicas exactas (volumen, features, distribución de clases) es considerablemente menos detallada que la de SR-BH 2020.  
**Rol frente a SR-BH 2020:** Complemento posible para validación multi-clase externa. No reemplaza la capacidad multi-label de SR-BH 2020.

### 7.4 KDD Cup 99 / NSL-KDD / DARPA 1999

**Descripción:** Los datasets más utilizados históricamente en IDS general. Basados en tráfico simulado de 1999, con distribuciones de clases ampliamente criticadas por contener artefactos artificiales.  
**Clasificación:** Multi-clase (pero no HTTP-específica ni CAPEC-alineada).  
**Limitaciones críticas:** El KDD Archive desaconseja activamente el uso de KDD Cup 99. No son específicos de tráfico web HTTP a nivel de request. La representación de ataques web modernos (XSS, SQLi, OWASP Top 10) es mínima o inexistente.  
**Rol frente a SR-BH 2020:** Inaplicables para el problema de detección de ataques web a nivel de aplicación. Se incluyen en esta comparación únicamente para contextualizar la evolución del campo.

### 7.5 CIC-IDS 2017 / CSE-CIC-IDS 2018

**Origen:** Canadian Institute for Cybersecurity.  
**Descripción:** Datasets más modernos con tráfico mixto (red y aplicación). CIC-IDS 2017 contiene 15 tipos de ataques; CSE-CIC-IDS 2018 amplía la cobertura.  
**Clasificación:** Multi-clase, pero no multi-label.  
**Limitaciones críticas:** No son específicos de tráfico HTTP a nivel de request. La fracción de ataques web dentro de estos datasets es un subconjunto pequeño del total. No ofrecen clasificación CAPEC. Su granularidad de features está orientada a flujos de red (NetFlow/CICFlowMeter), no a campos de la solicitud HTTP.  
**Rol frente a SR-BH 2020:** Útiles para IDS general de red, pero no alineados con el objetivo de detección de ataques HTTP a nivel de aplicación.

### 7.6 AIoT-Sol Dataset (2022)

**Origen:** Investigadores de seguridad IoT, 2022.  
**Descripción:** Contiene 17 tipos de ataque de categorías variadas: ataques de red, ataques web y ataques sobre protocolos IoT (MQTT, etc.).  
**Clasificación:** Multi-clase, con cobertura de ataques web como XSS y SQLi.  
**Limitaciones críticas:** No es específico de solicitudes HTTP a nivel de request (incluye tráfico de red e IoT en general). No es multi-label. No utiliza taxonomía CAPEC. Fue diseñado para el dominio IoT, por lo que la distribución de ataques web es marginal frente al tráfico total.  
**Rol frente a SR-BH 2020:** Comparable como referencia de detección web en contextos IoT, no como reemplazo para clasificación HTTP multi-label en WAFs.

### 7.7 "Thirty-Day Dataset of Malicious HTTP Requests" (Lucz, 2025)

**Origen:** Budapest University of Technology and Economics, publicado en Zenodo en 2025 (DOI: 10.5281/zenodo.17178461). Artículo en Data, MDPI 2025.  
**Descripción:** 30 días de solicitudes HTTP maliciosas bloqueadas por OWASP ModSecurity sobre un servidor de producción real. Incluye SQLi, XSS, LFI, scanner probes y otras formas de input malformado.  
**Clasificación:** **Solo malicious** (no incluye tráfico benigno). Etiquetado por regla CRS disparada, no por tipo de ataque en taxonomía estándar.  
**Limitaciones críticas:** La ausencia de tráfico benigno lo hace inutilizable para entrenamiento de modelos supervisados que distinguen normal de ataque. No es multi-label en el sentido CAPEC. No existe vista de clasificación binaria ni multiclase comparable a SR-BH 2020. Es un recurso valioso para análisis de payloads maliciosos y tuning de reglas WAF, pero no para los objetivos de clasificación multi-label de este trabajo.  
**Rol frente a SR-BH 2020:** Complementario para análisis de payloads, no sustitutivo.

### 7.8 WEB-IDS23

**Origen:** Technical Report (arXiv:2502.03909), laboratorio sintético con OWASP Juice Shop como objetivo.  
**Descripción:** Dataset generado en entorno de laboratorio controlado, con ataques HTTP sobre una aplicación deliberadamente vulnerable.  
**Clasificación:** Multi-clase (SQLi, XSS, DoS, Brute Force, SSRF, SSTI).  
**Limitaciones críticas:** Entorno completamente sintético. No multi-label. No usa taxonomía CAPEC. Disponibilidad pública y adopción por parte de la comunidad aún muy limitadas al momento de este trabajo.  
**Rol frente a SR-BH 2020:** Referencia emergente. Interesante para trabajos futuros, pero sin la madurez ni el volumen de SR-BH 2020.

---

## 8. Tabla Comparativa Exhaustiva

| Dataset | Año | Tráfico | Clasificación | Multi-label | CAPEC | Tráfico benigno | Volumen | Acceso libre |
|---|---|---|---|---|---|---|---|---|
| **SR-BH 2020** | 2020 (pub. 2022) | **Real** (honeypot) | **13 clases** | **✅** | **✅** | ✅ | 907.814 req. | **✅ CC0** |
| CSIC-2010 | 2010 | Sintético | Binaria | ❌ | ❌ | ✅ | ~61.000 req. | ✅ |
| ECML/PKDD 2007 | 2007 | Real (anonimizado) | Binaria | ❌ | ❌ | ✅ | ~50.000 req. | ✅ |
| TorpEda | 2024 | Mixto | Multi-clase | ❌ | ❌ | ✅ | No documentado públicamente | ✅ |
| KDD Cup 99 | 1999 | Simulado | Multi-clase (red) | ❌ | ❌ | ✅ | 4,9M registros | ✅ |
| NSL-KDD | 2009 | Simulado | Multi-clase (red) | ❌ | ❌ | ✅ | ~125.000 reg. | ✅ |
| CIC-IDS 2017 | 2017 | Mixto | Multi-clase (red) | ❌ | ❌ | ✅ | ~2,8M registros | ✅ |
| AIoT-Sol | 2022 | Mixto (IoT) | Multi-clase | ❌ | ❌ | ✅ | No publicado | Limitado |
| Thirty-Day HTTP | 2025 | **Real** | Por regla CRS | ❌ | ❌ | **❌** | No especificado | ✅ Zenodo |
| WEB-IDS23 | 2023 | Sintético | Multi-clase | ❌ | ❌ | ✅ | No publicado | Limitado |

**Conclusión de la tabla:** SR-BH 2020 es el **único dataset público que combina las cinco propiedades críticas** para el objetivo de este trabajo: tráfico real, granularidad HTTP a nivel de request, clasificación multi-label, taxonomía CAPEC y licencia abierta con volumen suficiente. Ningún dataset par cubre de manera equivalente la necesidad de evaluar clasificación multietiqueta de ataques web.

---

## 9. Síntesis de la Justificación

La elección de SR-BH 2020 se sustenta en tres pilares complementarios:

**Pilar 1 — Singularidad en el espacio de datasets públicos.** Es el único dataset disponible públicamente con tráfico HTTP real, clasificación multi-label y taxonomía CAPEC, en volumen suficiente para entrenar modelos de machine learning y deep learning. Sus alternativas más cercanas (CSIC-2010, TorpEda) son binarias o multi-clase pero no multi-label, sintéticas o de cobertura tecnológica diferente.

**Pilar 2 — Pertinencia directa con el objetivo del trabajo.** SR-BH 2020 permite formular el mismo problema en tres variantes experimentales (binaria, multi-clase, multi-label) sobre un corpus idéntico, facilitando la comparación controlada entre paradigmas de clasificación. Esta versatilidad es imposible de replicar con ningún otro dataset público en el dominio HTTP.

**Pilar 3 — Validación comunitaria y línea experimental activa.** Ha sido adoptado en múltiples estudios publicados en revistas de primer nivel (Computers & Security, LNNS, ACM) y reconocido en surveys sistemáticos. Incluso los trabajos que critican sus limitaciones (WAMM) lo eligen como punto de partida, confirmando que no existe alternativa superior en el espacio público.

**Limitaciones asumidas explícitamente.** SR-BH 2020 no se adopta por ser un ground truth perfecto. Presenta ruido en las etiquetas, duplicados potenciales, especificidad tecnológica (WordPress/ModSecurity) y categorías CAPEC de observabilidad variable. Estas limitaciones condicionan el alcance de las conclusiones y están explícitamente incorporadas al análisis metodológico.

---

## 10. Integración con la Metodología y Mitigación de Amenazas

Para mitigar las amenazas a la validez identificadas, la metodología de este trabajo incorpora las siguientes decisiones:

- **Separación fija entrenamiento/prueba** para evitar fugas de datos entre instancias redundantes.
- **Ajuste de transformaciones únicamente dentro del bloque de entrenamiento**, sin filtración de información del conjunto de evaluación.
- **Métricas multietiqueta y por clase** (F1 micro, F1 macro, Hamming Loss, Jaccard, Exact Match Ratio), que permiten detectar si el rendimiento se concentra en clases frecuentes o si el modelo falla sistemáticamente en etiquetas minoritarias.
- **Análisis de clases raras**, identificando explícitamente las categorías CAPEC con bajo soporte y las que presentan problemas de observabilidad.
- **Comparación con otros datasets** (CSIC-2010, TorpEda) para evaluar si los patrones de rendimiento se mantienen en conjuntos con características diferentes.
- **Evaluación con y sin features de tokens sospechosos**, para observar hasta qué punto el desempeño depende de patrones léxicos explícitos que podrían estar afectados por ruido de etiquetado o sesgos de captura.

Estas decisiones no eliminan el ruido del dataset original, pero hacen explícito el alcance de la evaluación y reducen el riesgo de interpretar una métrica agregada como evidencia suficiente de generalización robusta.

---

## 11. Amenazas a la Validez Asociadas al Uso de SR-BH 2020

Las principales amenazas a la validez vinculadas al uso de SR-BH 2020 son las siguientes:

1. **Ruido de etiquetas:** la presencia de solicitudes maliciosas dentro de la clase benigna (~48.522 según WAMM) puede afectar el entrenamiento y la evaluación, especialmente en modelos one-class.
2. **Observabilidad incompleta de algunas clases CAPEC:** ciertas categorías pueden requerir contexto adicional, respuesta del servidor o comportamiento temporal no disponible en una request aislada.
3. **Redundancia estructural:** instancias muy similares pueden facilitar resultados altos sin representar necesariamente generalización robusta.
4. **Desbalance de clases:** las clases frecuentes pueden dominar las métricas agregadas, ocultando bajo rendimiento en etiquetas minoritarias.
5. **Contexto tecnológico específico:** el dataset refleja tráfico capturado en un entorno WordPress con ModSecurity/CRS, lo que limita la generalización a otras tecnologías web.
6. **Período de captura restringido:** 12 días de julio de 2020 pueden no ser representativos de variaciones estacionales o campañas de ataque de largo plazo.
7. **Cobertura limitada de ataques evasivos:** los payloads no ofuscados dominan el dataset; variantes polimórficas y de evasión están subrepresentadas.

Estas amenazas no invalidan la elección del dataset, pero delimitan el alcance de las conclusiones. Los resultados obtenidos sobre SR-BH deben entenderse como evidencia comparativa dentro de un benchmark reproducible, no como garantía directa de rendimiento en despliegues productivos reales.

---

## 12. Conclusión Metodológica

SR-BH 2020 fue elegido porque permite evaluar una dimensión central del trabajo que no está adecuadamente cubierta por ningún otro dataset público disponible: la clasificación multietiqueta de ataques web a nivel de solicitud HTTP con taxonomía CAPEC. Su estructura permite comparar formulaciones binarias, multiclase y multietiqueta sobre el mismo corpus, y su publicación abierta bajo CC0 garantiza reproducibilidad total.

La elección es metodológicamente **defendible precisamente cuando se presenta de forma crítica**. SR-BH 2020 ofrece ventajas claras frente a los datasets binarios clásicos (CSIC-2010, ECML/PKDD 2007) y frente a los alternativos más recientes (TorpEda, WEB-IDS23, AIoT-Sol), que no ofrecen multi-label con CAPEC sobre tráfico real. Pero también presenta ruido de etiquetas cuantificable, redundancia potencial y especificidad de stack. Estas limitaciones están explícitamente incorporadas al análisis metodológico, con métricas que permiten observar el desempeño por clase y mitigaciones de diseño que reducen el impacto de las amenazas identificadas.

En síntesis, SR-BH 2020 no se adopta por ser un ground truth perfecto, sino porque **ofrece la base pública más amplia, más reciente y más granular disponible para estudiar el problema de clasificación multi-label de ataques HTTP**, con limitaciones documentadas que pueden gestionarse metodológicamente.

---

## 13. Referencias

**[1]** Sureda Riera, T.; Bermejo Higuera, J. R.; Bermejo Higuera, J.; Sicilia Montalvo, J. A.; Martínez Herráiz, J. J. **SR-BH 2020 multi-label dataset**. Harvard Dataverse, 2022. DOI: `10.7910/DVN/OGOIXX`.  
Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OGOIXX  
Ficha institucional UNIR: https://investigacion.unir.net/documentos/668fc479b9e7c03b01bde810

**[2]** Riera, T. S.; Bermejo Higuera, J. R.; Bermejo Higuera, J.; Martínez Herráiz, J. J.; Sicilia Montalvo, J. A. **A new multi-label dataset for Web attacks CAPEC classification using machine learning techniques**. *Computers & Security*, 120, 102788, 2022. DOI: `10.1016/j.cose.2022.102788`.  
https://www.sciencedirect.com/science/article/pii/S0167404822001833

**[3]** Osama, H.; Elebiary, O.; Qassim, Y.; Maghawry, M. A.; Saafan, A.; Ghalwash, H. **Enhanced Web Payload Classification Using WAMM: An AI-Based Framework for Dataset Refinement and Model Evaluation**. arXiv:2512.23610v2, 2025/2026.  
https://arxiv.org/abs/2512.23610

**[4]** Zhou, L.; Yau, W.-C.; Gan, Y. S.; Liong, S.-T. **E-WebGuard: Enhanced neural architectures for precision web attack detection**. *Computers & Security*, 2024. DOI: `10.1016/j.cose.2024.104322`.  
https://www.sciencedirect.com/science/article/abs/pii/S0167404824004322

**[5]** Gniewkowski, M. et al. **Capturing the security expert knowledge in feature selection for web application attack detection**. Proceedings of the 13th LADC 2024, ACM. DOI: `10.1145/3697090.3699797`.  
https://dl.acm.org/doi/10.1145/3697090.3699797

**[6]** Tran-Thi, M.H.; Ngo, T.K.; Le, X.H.; Nguyen, D.T.; Nguyen, X.H.; Le, K.H. **An Effective Unsupervised Cyber Attack Detection on Web Applications Using Gaussian Mixture Model**. In: CITA 2024. LNNS 882, Springer, 2024. DOI: `10.1007/978-3-031-74127-2_39`.

**[7]** Goldschmidt, O.; Chudá, D. **Network Intrusion Datasets: A Survey, Limitations, and Recommendations**. arXiv:2502.06688, 2025.  
https://arxiv.org/html/2502.06688v1

**[8]** Torrano-Gimenez, C.; Perez-Villegas, A.; Alvarez, G. **HTTP data set CSIC 2010**. Information Security Institute of CSIC, 2010.  
https://www.impactcybertrust.org/dataset_view?idDataset=940

**[9]** Raïssi, C. et al. **Web Analyzing Traffic: ECML/PKDD 2007 Discovery Challenge**. Warsaw, 2007.

**[10]** Tavallaee, M.; Bagheri, E.; Lu, W.; Ghorbani, A.A. **A detailed analysis of the KDD CUP 99 data set**. Proceedings of the Second IEEE Symposium on CISDA, 2009.

**[11]** ITEFI-CSIC. **TorpEda: dataset for testing and research on prevention, detection and analysis**. CSIC, 2024.  
https://www.tic.itefi.csic.es/torpeda/datasets.html

**[12]** Lucz, G. **A Thirty-Day Dataset of Malicious HTTP Requests Blocked by OWASP ModSecurity on a Production Web Server**. *Data*, 10(11), 186, 2025. DOI: `10.3390/data10110186`.  
https://zenodo.org/records/17178461

**[13]** MITRE. **Common Attack Pattern Enumeration and Classification (CAPEC)**.  
https://capec.mitre.org/

**[14]** Sharafaldin, I.; Habibi Lashkari, A.; Ghorbani, A.A. **Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization**. Proceedings of ICISSP 2018.

---

*Documento elaborado mediante revisión bibliográfica sistemática de fuentes académicas peer-reviewed, arXiv, y registros de repositorios de datos científicos. Fecha de consulta: abril de 2026.*
