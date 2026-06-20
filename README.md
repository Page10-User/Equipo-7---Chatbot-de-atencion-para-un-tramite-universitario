# 🎓 Agente Conversacional — Tramitación del Título Intermedio (Pregrado)

**Facultad de Ciencias Exactas y Naturales y Agrimensura (FACENA) — UNNE**  
**Asignatura:** Inteligencia Artificial  
**Año:** 2026

---

## 📋 Descripción

Chatbot basado en procesamiento de lenguaje natural (PLN) que guía al estudiante a través del proceso de tramitación del **Título Intermedio (Pregrado)** de FACENA-UNNE. El sistema recibe una consulta del usuario en lenguaje natural, clasifica su intención y devuelve la respuesta correspondiente extraída de la documentación oficial del trámite.

## 👥 Integrantes del equipo

| Nombre                         | Rol                              |
| ------------------------------ | -------------------------------- |
| Acosta Lopez Gonzalo Nahuel    | Dataset, pipeline y experimentos |
| Cesar Pietro Barrios Calathaki | —                                |
| Gastón Osvaldo Rodríguez       | —                                |

## 📂 Estructura del proyecto

| Carpeta                                                | Contenido                                                        |
| ------------------------------------------------------ | ---------------------------------------------------------------- |
| [`01-Planificacion/`](01-Planificacion/00-Planning.md) | Planning, alcance, intenciones, experimentos, división de tareas |
| [`02-Dataset/`](02-Dataset/)                           | Dataset en JSON, esqueleto, estructura propuesta                 |
| [`03-Referencias/`](03-Referencias/)                   | Fuente oficial de FACENA-UNNE, transcripción, repositorio GitHub |
| [`04-Consignas/`](04-Consignas/)                       | TP Integrador oficial (enunciado)                                |
| [`05-Desarrollo/`](05-Desarrollo/)                     | Pipeline de preprocesamiento, arquitectura y experimentos         |

## 🚦 Estado del proyecto

| Etapa                                | Estado                                                                                                         |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| **Dataset**                          | ✅ **v1.2 — Completado y verificado** (16 intenciones, 112 preguntas)                                           |
| **Arquitectura de preprocesamiento** | ✅ **Documentada** — [`00-Arquitectura-preprocesamiento.md`](05-Desarrollo/00-Arquitectura-preprocesamiento.md) |
| **Preprocesamiento (código)**        | ✅ **Implementado** — [`dataset.py`](05-Desarrollo/dataset.py) + [`cleaner.py`](05-Desarrollo/cleaner.py)      |
| **Vectorización**                    | ✅ **Implementada** — [`vectorizer.py`](05-Desarrollo/vectorizer.py) — TF-IDF (98 términos) + SentenceTransformer (512 dims) |
| **Experimento 1 — Naive Bayes + TF-IDF** | ✅ **Completado** — Accuracy: 73.91% — F1 macro: 0.58                                           |
| **Experimento 2 — SVM + TF-IDF**         | ✅ **Completado** — Accuracy: 82.61% — F1 macro: 0.67                                           |
| **Experimento 3 — SVM + Embeddings**     | ✅ **Implementado** — Pendiente de ejecución (requiere ~1 GB RAM)                               |
| Interfaz del chatbot                 | 🔲 Pendiente                                                                                                   |
| Informe IEEE                         | 🔲 Pendiente                                                                                                   |
| Infografía                           | 🔲 Pendiente                                                                                                   |
| Exposición                           | 🔲 Pendiente                                                                                                   |

## 📈 Resultados preliminares

| Experimento | Modelo | Vectorización | Accuracy | F1 (macro) |
|-------------|--------|---------------|----------|------------|
| Exp 1 | Naive Bayes | TF-IDF (500 terms) | 73.91% | 0.58 |
| Exp 2 | SVM (RBF) | TF-IDF (500 terms) | **82.61%** | **0.67** |
| Exp 3 | SVM (RBF) | SentenceTransformer (512d) | ⏳ Pendiente | ⏳ Pendiente |

> SVM con TF-IDF duplica el F1-score del Naive Bayes, lo que indica que las fronteras de decisión lineales con márgenes anchos capturan mejor la separabilidad de las intenciones.

## 🔗 Enlaces útiles

- [Repositorio GitHub](https://github.com/Page10-User/Equipo-7---Chatbot-de-atencion-para-un-tramite-universitario)
- [Notebook de experimentos](05-Desarrollo/experimentos.ipynb)
- [Pipeline completo](05-Desarrollo/pipeline.py)
- [Fuente oficial — FACENA UNNE](https://exa.unne.edu.ar/r/?page_id=9008)
- [Video explicativo del trámite](https://www.youtube.com/watch?v=KDsmSxQoqJg&feature=youtu.be)

---

*Este documento se actualiza a medida que avanza el proyecto.*
