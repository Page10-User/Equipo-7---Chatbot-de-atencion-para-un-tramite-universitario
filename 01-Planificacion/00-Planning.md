# PLANNING — Agente conversacional para la tramitación del Título Intermedio · FACENA UNNE

> Documento de planificación interna del equipo.
> **Nota:** Este documento refleja el estado final del proyecto.

---

## Índice

1. [Descripción del proyecto](#1-descripción-del-proyecto)
2. [Alcance](#2-alcance)
3. [Alineación con el TP Integrador](#3-alineación-con-el-tp-integrador)
4. [Intenciones identificadas](#4-intenciones-identificadas)
5. [Qué vamos a hacer](#5-qué-vamos-a-hacer)
6. [Datos](#6-datos)
7. [Herramientas y tecnologías](#7-herramientas-y-tecnologías)
8. [Experimentos planificados](#8-experimentos-planificados)
9. [Entregables requeridos](#9-entregables-requeridos)
10. [División de tareas (final)](#10-división-de-tareas-final)
11. [Decisiones y cambios durante el desarrollo](#11-decisiones-y-cambios-durante-el-desarrollo)

---

## 1. Descripción del proyecto

**Título:** Agente conversacional para la tramitación del Título Intermedio (Pregrado) — FACENA UNNE

**Descripción:**
Un chatbot basado en procesamiento de lenguaje natural (PLN) que guía al estudiante a través del proceso de tramitación del Título Intermedio (Pregrado) de la Facultad de Ciencias Exactas y Naturales y Agrimensura (FACENA) de la UNNE. El sistema recibe una consulta del usuario en lenguaje natural, clasifica su intención y devuelve la respuesta correspondiente extraída de la documentación oficial del trámite.

**Problema que resuelve:**
El proceso de tramitación del título intermedio involucra múltiples pasos, sistemas (SIU-Guaraní), instituciones (biblioteca de FACENA, biblioteca central de UNNE) y documentación específica. Si bien está documentado públicamente, la información es difícil de encontrar y de seguir en orden, lo que genera confusión, demoras y consultas reiteradas al personal administrativo. Un agente conversacional puede guiar al estudiante paso a paso de forma inmediata, consistente y disponible en cualquier momento.

**Fuente del dominio:**
Documentación oficial pública de FACENA-UNNE: sección *"Tramitación de Diploma de Pregrado y Grado"*.

**Ámbito de aplicación / validación:**
Facultad de Ciencias Exactas y Naturales y Agrimensura — UNNE.
Trámite: Título Intermedio (Pregrado). Proceso completo: Paso 1 y Paso 2.

---

## 2. Alcance

### V1.0 — Esta entrega cubre

- Proceso completo del trámite de **Título Intermedio (Pregrado)**: Paso 1 y Paso 2.
- Documentación requerida (DNI, fotos, libre deudas).
- Gestión dentro del SIU-Guaraní en el marco del trámite.
- Casos especiales documentados: estudiantes extranjeros y del MERCOSUR.
- Información general sobre qué es el título intermedio y para qué sirve.
- Intención de fallback: respuesta ante preguntas no reconocidas.

### Fuera de esta entrega

| No incluido | Motivo |
|---|---|
| Trámite de Título de Grado | Proceso similar pero no validado empíricamente por el equipo; se deja como evolución |
| Diploma decorativo | Documentado en la misma fuente; queda para una versión futura |
| Otros trámites administrativos | Fuera del alcance temporal del TP |
| Soporte técnico del SIU-Guaraní | No es un trámite de la facultad sino del sistema |
| Consultas académicas (correlativas, horarios, etc.) | Dominio diferente |

### Evolución natural (trabajos futuros)

La arquitectura del agente se diseña desde el inicio para ser **extensible por trámite**. Agregar el Título de Grado o el Diploma Decorativo en una versión futura implica únicamente incorporar nuevas intenciones al dataset y reentrenar el modelo, sin modificar la estructura del sistema. Esta es la línea de trabajo futuro que se menciona en el informe, incluyendo la expansión a otros trámites institucionales de la UNNE.

---

## 3. Alineación con el TP Integrador

| Requisito del TP | Cómo lo cubrimos |
|---|---|
| Modelo de IA para resolver un problema real | Clasificador de intenciones con PLN aplicado al trámite de título intermedio de FACENA-UNNE |
| Problema abstraído de la realidad y delimitado en un contexto | Trámite de Título Intermedio (Pregrado), contexto: FACENA - UNNE |
| Al menos tres experimentos | NB+TF-IDF, SVM+TF-IDF, SVM+Embeddings (ganador: 81.25%) |
| Tabla y métricas de resultados | Accuracy, Precision, Recall, F1-score por experimento, CV F1 (K=5) |
| Informe formato IEEE (máx. 4 páginas) | ✅ Completo — 06-Informe/informe_ieee.pdf (2 páginas) |
| Infografía | ✅ Completa — 06-Informe/Infografía/ |
| Código | Repositorio Git (este repo) — pipeline, modelos, interfaz Streamlit |
| Conclusiones / trabajos futuros | Extensión a Título de Grado, Diploma Decorativo y otros trámites institucionales |

---

## 4. Intenciones identificadas

Derivadas directamente del proceso documentado (Paso 1 y Paso 2). Cada intención se corresponde con un tipo de pregunta que un estudiante haría naturalmente.

| Intención | Pregunta representativa | Paso del trámite |
|---|---|---|
| `que_es_titulo_intermedio` | ¿Qué es el título intermedio? ¿Para qué sirve? | General |
| `inicio_tramite` | ¿Por dónde empiezo el trámite? ¿Cuál es el primer paso? | Paso 1 |
| `actualizar_siu` | ¿Cómo actualizo mis datos en el SIU-Guaraní? | Paso 1 |
| `documentacion_requerida` | ¿Qué documentos tengo que presentar? | Paso 1 |
| `formato_dni` | ¿Cómo tiene que ser la copia del DNI? ¿En qué formato? | Paso 1 |
| `libre_deuda_facena` | ¿Cómo saco el libre deuda de la biblioteca de FACENA? | Paso 1 |
| `libre_deuda_central` | ¿Cómo saco el libre deuda de la biblioteca central de la UNNE? | Paso 1 |
| `foto_carnet` | ¿Cómo tiene que ser la foto del carnet? | Paso 1 |
| `requisitos_extranjeros` | ¿Qué pasa si soy extranjero o del MERCOSUR? | Paso 1 |
| `continuidad_paso2` | ¿Cuándo paso al Paso 2? ¿Cómo sé si fue aceptado el Paso 1? | Paso 2 |
| `generar_solicitud_siu` | ¿Cómo genero la solicitud en el SIU? | Paso 2 |
| `estado_solicitud` | ¿Cómo sé si mi solicitud está pendiente o aceptada? | Paso 2 |
| `pago_arancel` | ¿Cómo pago el arancel? ¿Dónde? | Paso 2 |
| `descarga_constancia` | ¿Cómo descargo la constancia del título en trámite? | Paso 2 |
| `tiempo_espera` | ¿Cuánto tarda en estar disponible la constancia? | Paso 2 |
| `fallback` | Respuesta para preguntas no reconocidas | — |

**Total: 15 intenciones + fallback**, distribuidas entre los dos pasos del proceso.

---

## 5. Qué vamos a hacer

### Paso 1 — Construcción del dataset
Crear manualmente un conjunto de pares **intención → pregunta(s) → respuesta** basados en la documentación oficial del trámite.
Cada intención tendrá entre 5 y 8 variaciones de pregunta para que el modelo pueda generalizar.

Ejemplo de estructura:

```
Intención: libre_deuda_facena
Preguntas:
  - "¿Cómo saco el libre deuda de la biblioteca de FACENA?"
  - "¿Dónde pido el certificado de libre deuda de la facultad?"
  - "¿Qué tengo que hacer para el libre deuda de FACENA?"
  - "¿El libre deuda de la biblioteca lo pido en persona?"
  - "¿Me piden el libre deuda de FACENA para el título?"
Respuesta: "Para obtener el certificado de libre deuda de la Biblioteca de FACENA, debés dirigirte a ..."
```

### Paso 2 — Preprocesamiento del texto
- Limpieza: minúsculas, eliminación de signos de puntuación, stopwords en español.
- Vectorización: TF-IDF o embeddings (a definir en experimentos).

### Paso 3 — Entrenamiento del clasificador
Entrenar un modelo que, dada una pregunta nueva, prediga la intención correcta.
Modelos a comparar: Naive Bayes, SVM, y opcionalmente un modelo con embeddings preentrenados.

### Paso 4 — Interfaz del chatbot
Implementar un bucle simple de entrada/salida en consola (o una interfaz web mínima) que:
1. Recibe la pregunta del usuario.
2. Clasifica la intención.
3. Devuelve la respuesta asociada.
4. Indica cuando no entiende la pregunta (umbral de confianza bajo).

### Paso 5 — Evaluación
Separar el dataset en entrenamiento (80%) y prueba (20%).
Medir accuracy, precisión, recall y F1 por cada variante del experimento.

---

## 6. Datos

| Elemento | Detalle |
|---|---|
| Origen | Creados por el equipo basados en documentación oficial pública de FACENA-UNNE |
| Formato | JSON con columnas: `intencion`, `paso`, `pregunta`, `respuesta` |
| Cantidad final | **240 preguntas** distribuidas en 15 intenciones (16 variaciones por intención) |
| Versionado | v1.0 → v1.1 → v1.2 → v1.3 → **v1.4** (respuestas expandidas y actualizadas) |
| Privacidad | Sin datos sensibles; todo el contenido es información pública de la facultad |

### Historial de versiones del dataset

| Versión | Cambios |
|---------|---------|
| v1.0 | Estructura inicial, 15 intents + fallback, 90 preguntas |
| v1.1 | Expansión a 240 preguntas (16 por intent) |
| v1.2 | Corrección de respuestas |
| v1.3 | Refinamiento general |
| v1.4 | Respuestas expandidas: `inicio_tramite` (Paso 2 completo), `libre_deuda_central` (formulario web + 7 campos), `libre_deuda_facena` (datos requeridos), `generar_solicitud_siu` (docs adicionales) |

> **Importante:** Las respuestas se actualizan sin retreinar el modelo — el clasificador solo predice intenciones, las respuestas se buscan del JSON.

---

## 7. Herramientas y tecnologías

| Componente | Opción utilizada | Alternativa |
|---|---|---|
| Lenguaje | Python 3.12 | — |
| Preprocesamiento NLP | NLTK + cleaner propio | spaCy evaluado |
| Vectorización | TF-IDF (scikit-learn) + SentenceTransformer (multilingual) | — |
| Clasificador | MultinomialNB, SVC (scikit-learn) | Regresión logística (no usada) |
| Optimización | GridSearchCV + 5-Fold Cross Validation | — |
| Métricas | classification_report, f1_score, accuracy_score | — |
| Interfaz | **Streamlit 1.57** (final) | Consola Python (descartada) |
| Informe | **LaTeX + IEEEtran** (final) | Word (descartado) |
| Infografía | HTML/CSS → PDF | Canva (alternativa) |
| Gestión del proyecto | Git + GitHub | — |

---

## 8. Experimentos planificados

El TP exige **al menos tres experimentos**. Los variamos sobre el mismo dataset para poder comparar resultados de forma coherente.

| # | Variable modificada | Descripción |
|---|---|---|
| Exp. 1 | Algoritmo de clasificación | Naive Bayes multinomial con TF-IDF |
| Exp. 2 | Algoritmo de clasificación | SVM lineal con TF-IDF |
| Exp. 3 | Representación del texto | SVM con embeddings (SentenceTransformer en español) |

### Fases experimentales adicionales realizadas

| Fase | Descripción | Herramienta |
|---|---|---|
| Fase 3 — Preprocesamiento | Stemming, ngram_range, max_features | Scripts en `pruebas/` |
| Fase 4 — Optimización | GridSearchCV + K-Fold CV sobre hiperparámetros | GridSearchCV (scikit-learn) |

### Tabla de resultados finales

| Experimento | Vectorización | Modelo | Hiperparámetros | Accuracy | F1 (macro) | CV F1 (K=5) |
|---|---|---|---|---|---|---|
| Exp. 1 | TF-IDF (unigrams, 103 terms) | Naive Bayes | α=0.5 | 79.17% | 0.77 | 0.71 |
| Exp. 2 | TF-IDF (unigrams, 103 terms) | SVM lineal | C=1.0 | 75.00% | 0.73 | 0.69 |
| Exp. 3 | Embeddings (SentenceTransformer 512d) | **SVM RBF** | **C=10.0, γ=scale** | **81.25%** | **0.81** | **0.76** |

---

## 9. Entregables requeridos

Según el TP Integrador, los productos a entregar son:

- [x] **Código** — en este repositorio Git, organizado y comentado
- [x] **Informe** — máximo 4 páginas, formato IEEE (2 páginas final)
- [x] **Infografía** — resumen visual del proyecto

---

## 10. División de tareas (final)

| Tarea | Responsable | Estado |
|---|---|---|
| Construcción del dataset | Acosta Lopez Gonzalo | ✅ Completado (v1.4, 240 preguntas) |
| Preprocesamiento y pipeline | Acosta Lopez Gonzalo | ✅ Completado |
| Experimento 1 (Naive Bayes) | Acosta Lopez Gonzalo | ✅ Completado (79.17%) |
| Experimento 2 (SVM + TF-IDF) | Acosta Lopez Gonzalo | ✅ Completado (75.00%) |
| Experimento 3 (Embeddings + SVM RBF) | Acosta Lopez Gonzalo | ✅ Completado (**81.25%** — ganador) |
| Preprocesamiento avanzado (stemming, ngrams) | Acosta Lopez Gonzalo | ✅ Completado (documentado) |
| GridSearchCV + K-Fold Cross Validation | Acosta Lopez Gonzalo | ✅ Completado (α=0.5, C=10.0) |
| Modelo final + script de inferencia | Acosta Lopez Gonzalo | ✅ Completado (`clasificar.py`) |
| Interfaz web (Streamlit) | Acosta Lopez Gonzalo | ✅ Completada (tema oscuro, 3 modelos) |
| Dataset v1.4 — respuestas expandidas | Acosta Lopez Gonzalo | ✅ Completado |
| Redacción del informe IEEE | Acosta Lopez Gonzalo, Barrios Calathaki | ✅ Completo (LaTeX + PDF) |
| Infografía | Acosta Lopez Gonzalo | ✅ Completa (HTML + PDF) |


---

## 11. Decisiones y cambios durante el desarrollo

### Decisiones técnicas

| Decisión | Opción elegida | Alternativa descartada | Motivo |
|---|---|---|---|
| Interfaz | Streamlit (web) | Consola Python | Mejor experiencia de usuario, requerimiento del TP |
| Formato informe | LaTeX (IEEEtran) | Word | Formato IEEE profesional, control fino |
| Infografía | HTML/CSS → PDF | Canva | Auto-contenido, sin dependencias externas |
| ngram_range | (1, 1) — unigrams | (1, 2) — bigramas | Menor dispersión con 240 muestras |
| Stemming | No usar | Aplicar stemming | Destruye embeddings (−12.5%) |
| Optimización | GridSearchCV + 5-Fold CV | Validación simple | Mejor estimación del rendimiento real |

### Cambios durante el desarrollo

- **Dataset expandido:** de 90 a 240 preguntas (16 por intención) para mejor generalización
- **v1.4:** respuestas expandidas sin retreinar el modelo
- **Embeddings:** se confirmó como la mejor representación, superando a TF-IDF por ~6 puntos
- **Equipo:** se redujo a 2 integrantes activos (un miembro no pudo continuar, informado a la cátedra)
- **Notebook sobrescrito en 3 ocasiones** — se resolvió usando File → Checkpoint manualmente

### Próximos pasos

- Posible v2.0 del dataset (50 preguntas por intención, ~750 total) — a cargo del usuario

---

*Última actualización: Junio 2026*
