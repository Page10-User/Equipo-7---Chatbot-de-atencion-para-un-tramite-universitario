# PLANNING — Agente conversacional para consultas de trámites universitarios

> Documento de planificación interna del equipo.
> **No es el README final** — ese se redactará al cierre del proyecto.
> Actualizar este archivo a medida que avance el desarrollo.

---

## Índice

1. [Descripción del proyecto](#1-descripción-del-proyecto)
2. [Alineación con el TP Integrador](#2-alineación-con-el-tp-integrador)
3. [Qué vamos a hacer](#3-qué-vamos-a-hacer)
4. [Datos](#4-datos)
5. [Herramientas y tecnologías](#5-herramientas-y-tecnologías)
6. [Experimentos planificados](#6-experimentos-planificados)
7. [Entregables requeridos](#7-entregables-requeridos)
8. [División de tareas (provisional)](#8-división-de-tareas-provisional)
9. [Pendientes y decisiones abiertas](#9-pendientes-y-decisiones-abiertas)

---

## 1. Descripción del proyecto

**Título:** Agente conversacional para la atención de consultas sobre trámites universitarios

**Descripción:**
Un chatbot basado en procesamiento de lenguaje natural (PLN) que responde preguntas frecuentes sobre un trámite específico de la facultad (por ejemplo: inscripción a materias, solicitud de certificados, o consulta de correlativas). El sistema recibe una consulta del usuario en lenguaje natural, la clasifica según su intención y devuelve la respuesta más adecuada de una base de conocimiento construida por el equipo.

**Problema que resuelve:**
Los estudiantes universitarios suelen tener dudas recurrentes sobre trámites y procedimientos que requieren consultar a personal administrativo o buscar información dispersa. Un agente automatizado puede responder estas consultas de forma inmediata y consistente.

**Ámbito de aplicación / validación:**
Facultad de Ciencias Exactas y Naturales y Agrimensura — UNNE.
Trámite específico a definir por el equipo (ver sección 9).

---

## 2. Alineación con el TP Integrador

| Requisito del TP | Cómo lo cubrimos |
|---|---|
| Modelo de IA para resolver un problema real | Clasificador de intenciones con PLN aplicado a consultas reales de la facultad |
| Problema abstraído de la realidad y delimitado en un contexto | Trámites universitarios, contexto: FACENA - UNNE |
| Al menos tres experimentos | Ver sección 6 |
| Tabla y métricas de resultados | Accuracy, Precision, Recall, F1-score por experimento |
| Informe formato IEEE (máx. 4 páginas) | Secciones: Introducción, Método, Herramientas, Resultados, Conclusiones, Referencias |
| Infografía | Resumen visual del proyecto con las mismas secciones del informe |
| Código | Repositorio Git (este repo) |
| Exposición en clase | Demostración en vivo del chatbot + presentación de resultados |

---

## 3. Qué vamos a hacer

### Paso 1 — Construcción del dataset
Crear manualmente un conjunto de pares **intención → pregunta(s) → respuesta** sobre el trámite elegido.
Cada intención tendrá al menos 5 variaciones de pregunta para que el modelo pueda generalizar.

Ejemplo de estructura:

```
Intención: consulta_correlativas
Preguntas:
  - "¿Qué materias necesito para anotarme en Algoritmos?"
  - "¿Cuáles son las correlativas de Análisis?"
  - "¿Tengo que aprobar algo antes de cursar Programación 2?"
Respuesta: "Para inscribirse a [materia], es necesario tener aprobadas las siguientes materias: ..."
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

## 4. Datos

| Elemento | Detalle |
|---|---|
| Origen | Creados por el equipo (no depende de fuentes externas) |
| Formato | JSON o CSV con columnas: `intencion`, `pregunta`, `respuesta` |
| Cantidad estimada | 80 a 120 pares pregunta-respuesta, distribuidos en 8 a 12 intenciones |
| Herramienta de carga | Script Python propio o Google Sheets exportado a CSV |
| Privacidad | Sin datos sensibles; todo el contenido es información pública de la facultad |

---

## 5. Herramientas y tecnologías

| Componente | Opción principal | Alternativa |
|---|---|---|
| Lenguaje | Python 3.x | — |
| Preprocesamiento NLP | NLTK / spaCy (español) | — |
| Vectorización | TF-IDF (scikit-learn) | Sentence-Transformers |
| Clasificador | SVM / Naive Bayes (scikit-learn) | Regresión logística |
| Métricas | classification_report (scikit-learn) | — |
| Interfaz | Consola Python | Streamlit (web simple) |
| Gestión del proyecto | Git + este repositorio | — |
| Informe | LaTeX (IEEE template) o Word | Google Docs |

---

## 6. Experimentos planificados

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

## 7. Entregables requeridos

Según el TP Integrador, los productos a entregar son:

- [ ] **Código** — en este repositorio Git, organizado y comentado
- [ ] **Informe** — máximo 4 páginas, formato IEEE, con las secciones indicadas en el TP
- [ ] **Infografía** — resumen visual del proyecto (Introducción, Método, Herramientas, Resultados, Conclusiones)
- [ ] **Exposición en clase** — demostración del sistema funcionando + defensa de resultados

---

## 8. División de tareas (provisional)

> Completar con los nombres reales del equipo.

| Tarea | Responsable | Estado |
|---|---|---|
| Construcción del dataset | — | Pendiente |
| Preprocesamiento y pipeline | — | Pendiente |
| Experimento 1 (Naive Bayes) | — | Pendiente |
| Experimento 2 (SVM + TF-IDF) | — | Pendiente |
| Experimento 3 (Embeddings) | — | Pendiente |
| Interfaz del chatbot | — | Pendiente |
| Redacción del informe | — | Pendiente |
| Infografía | — | Pendiente |
| Preparación de la exposición | — | Pendiente |

---

## 9. Pendientes y decisiones abiertas

- [ ] Confirmar el trámite universitario específico a abordar (inscripción a materias, certificados, correlativas, etc.)
- [ ] Decidir si la interfaz será consola o Streamlit
- [ ] Decidir si usar LaTeX o Word para el informe
- [ ] Completar nombres del equipo en la sección 8
- [ ] Revisar si el Experimento 3 está al alcance del equipo (requiere descargar un modelo de embeddings en español)
