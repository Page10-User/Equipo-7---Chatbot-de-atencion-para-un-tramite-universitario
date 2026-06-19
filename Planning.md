# PLANNING — Agente conversacional para la tramitación del Título Intermedio · FACENA UNNE

> Documento de planificación interna del equipo.
> **No es el README final** — ese se redactará al cierre del proyecto.
> Actualizar este archivo a medida que avance el desarrollo.

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
10. [División de tareas (provisional)](#10-división-de-tareas-provisional)
11. [Pendientes y decisiones abiertas](#11-pendientes-y-decisiones-abiertas)

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

La arquitectura del agente se diseña desde el inicio para ser **extensible por trámite**. Agregar el Título de Grado o el Diploma Decorativo en una versión futura implica únicamente incorporar nuevas intenciones al dataset y reentrenar el modelo, sin modificar la estructura del sistema. Esta es la línea de trabajo futuro que se mencionará en el informe.

---

## 3. Alineación con el TP Integrador

| Requisito del TP | Cómo lo cubrimos |
|---|---|
| Modelo de IA para resolver un problema real | Clasificador de intenciones con PLN aplicado al trámite de título intermedio de FACENA-UNNE |
| Problema abstraído de la realidad y delimitado en un contexto | Trámite de Título Intermedio (Pregrado), contexto: FACENA - UNNE |
| Al menos tres experimentos | Ver sección 8 |
| Tabla y métricas de resultados | Accuracy, Precision, Recall, F1-score por experimento |
| Informe formato IEEE (máx. 4 páginas) | Secciones: Introducción, Método, Herramientas, Resultados, Conclusiones, Referencias |
| Infografía | Resumen visual del proyecto con las mismas secciones del informe |
| Código | Repositorio Git (este repo) |
| Exposición en clase | Demostración en vivo del chatbot + presentación de resultados |
| Conclusiones / trabajos futuros | Extensión a Título de Grado, Diploma Decorativo y otros trámites de la misma fuente |

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
| Formato | JSON o CSV con columnas: `intencion`, `paso`, `pregunta`, `respuesta` |
| Cantidad estimada | 90 a 120 preguntas distribuidas en 15 intenciones (5-8 variaciones por intención) |
| Herramienta de carga | Script Python propio o Google Sheets exportado a CSV |
| Privacidad | Sin datos sensibles; todo el contenido es información pública de la facultad |

---

## 7. Herramientas y tecnologías

> A confirmar según avance el desarrollo. Se irá completando.

| Componente | Opción principal | Alternativa |
|---|---|---|
| Lenguaje | Python 3.x | — |
| Preprocesamiento NLP | NLTK / spaCy (español) | — |
| Vectorización | TF-IDF (scikit-learn) | Sentence-Transformers |
| Clasificador | SVM / Naive Bayes (scikit-learn) | Regresión logística |
| Métricas | classification_report (scikit-learn) | — |
| Interfaz | A definir | Consola Python / Streamlit |
| Gestión del proyecto | Git + este repositorio | — |
| Informe | A definir | LaTeX IEEE template / Word |

---

## 8. Experimentos planificados

El TP exige **al menos tres experimentos**. Los variamos sobre el mismo dataset para poder comparar resultados de forma coherente.

| # | Variable modificada | Descripción |
|---|---|---|
| Exp. 1 | Algoritmo de clasificación | Naive Bayes multinomial con TF-IDF |
| Exp. 2 | Algoritmo de clasificación | SVM lineal con TF-IDF |
| Exp. 3 | Representación del texto | SVM con embeddings (sentence-transformers en español) |

**Métricas a reportar por experimento:**
- Accuracy global
- Precision, Recall y F1-score por intención
- Matriz de confusión

**Tabla de síntesis de resultados** (a completar durante el desarrollo):

| Experimento | Vectorización | Modelo | Accuracy | F1 (macro) |
|---|---|---|---|---|
| Exp. 1 | TF-IDF | Naive Bayes | — | — |
| Exp. 2 | TF-IDF | SVM lineal | — | — |
| Exp. 3 | Embeddings | SVM lineal | — | — |

---

## 9. Entregables requeridos

Según el TP Integrador, los productos a entregar son:

- [ ] **Código** — en este repositorio Git, organizado y comentado
- [ ] **Informe** — máximo 4 páginas, formato IEEE, con las secciones indicadas en el TP
- [ ] **Infografía** — resumen visual del proyecto (Introducción, Método, Herramientas, Resultados, Conclusiones)
- [ ] **Exposición en clase** — demostración del sistema funcionando + defensa de resultados

---

## 10. División de tareas (provisional)

> Completar con los nombres reales del equipo.

| Tarea | Responsable | Estado |
|---|---|---|
| Construcción del dataset | Acosta Lopez Gonzalo | En proceso |
| Preprocesamiento y pipeline | Acosta Lopez Gonzalo | Pendiente |
| Experimento 1 (Naive Bayes) | — | Pendiente |
| Experimento 2 (SVM + TF-IDF) | — | Pendiente |
| Experimento 3 (Embeddings) | — | Pendiente |
| Interfaz del chatbot | — | Pendiente |
| Redacción del informe | — | Pendiente |
| Infografía | — | Pendiente |
| Preparación de la exposición | — | Pendiente |

---

## 11. Pendientes y decisiones abiertas

- [ ] Confirmar URL oficial de la página "Tramitación de Diploma de Pregrado y Grado" para referenciar en el informe
- [ ] Completar las respuestas del dataset con información oficial verificada
- [ ] Decidir si la interfaz será consola o Streamlit
- [ ] Decidir si usar LaTeX o Word para el informe
- [ ] Completar nombres del equipo en la sección 10
- [ ] Revisar si el Experimento 3 está al alcance del equipo (requiere descargar un modelo de embeddings en español)
