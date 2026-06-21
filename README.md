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
| **Dataset**                          | ✅ **v1.3 — Completado y verificado** (15 intenciones + fallback, 240 preguntas)                                |
| **Arquitectura de preprocesamiento** | ✅ **Documentada** — [`00-Arquitectura-preprocesamiento.md`](05-Desarrollo/00-Arquitectura-preprocesamiento.md) |
| **Preprocesamiento (código)**        | ✅ **Implementado** — [`dataset.py`](05-Desarrollo/dataset.py) + [`cleaner.py`](05-Desarrollo/cleaner.py)      |
| **Vectorización**                    | ✅ **Implementada** — [`vectorizer.py`](05-Desarrollo/vectorizer.py) — TF-IDF (103 términos, unigrams) + SentenceTransformer (512 dims) |
| **Experimento 1 — Naive Bayes + TF-IDF** | ✅ **Completado** — Accuracy: **79.17%** — F1 macro: **0.77** (alpha=0.5, unigrams)               |
| **Experimento 2 — SVM + TF-IDF**         | ✅ **Completado** — Accuracy: **75.00%** — F1 macro: **0.73** (C=1.0, unigrams)                    |
| **Experimento 3 — SVM + Embeddings (RBF)** | ✅ **Completado** — Accuracy: **81.25%** — F1 macro: **0.81** (C=10.0, gamma='scale')           |
| Preprocesamiento avanzado (stemming, ngrams, max_features) | ✅ **Completado** — Resultados documentados en documento de pruebas          |
| Ajuste de hiperparámetros (GridSearchCV + K-Fold) | ✅ **Completado** — alpha=0.5, C=10.0, gamma='scale', CV F1: 0.76 (embeddings)    |
| **Modelo final entrenado y guardado** | ✅ **Listo** — [`modelo_final.py`](05-Desarrollo/modelo_final.py) + modelos en [`modelos/`](05-Desarrollo/modelos/) |
| **Script de inferencia**             | ✅ **Listo** — [`clasificar.py`](05-Desarrollo/clasificar.py) — `python clasificar.py "tu pregunta"` |
| Interfaz del chatbot                 | 🔲 Pendiente                                                                                                   |
| Informe IEEE                         | 🔲 Pendiente                                                                                                   |
| Infografía                           | 🔲 Pendiente                                                                                                   |
| Exposición                           | 🔲 Pendiente                                                                                                   |

## 📈 Resultados finales

| Experimento | Modelo | Vectorización | Hiperparámetros | Accuracy | F1 (macro) | CV F1 (K=5) |
|-------------|--------|---------------|-----------------|----------|------------|-------------|
| Exp 1 | Naive Bayes | TF-IDF (103 terms, unigrams) | alpha=0.5 | **79.17%** | 0.77 | 0.71 |
| Exp 2 | SVM (linear) | TF-IDF (103 terms, unigrams) | C=1.0 | 75.00% | 0.73 | 0.69 |
| Exp 3 | **SVM (RBF)** | **SentenceTransformer (512d)** | **C=10.0, gamma='scale'** | **81.25%** | **0.81** | **0.76** |

> **Ganador: SVM RBF + Embeddings (C=10.0).** Los embeddings capturan similitud semántica que TF-IDF no puede ver, y el kernel RBF + C elevado aprovecha la estructura no lineal del espacio de embeddings de alta calidad.

### Pruebas de preprocesamiento (Fase 3)

| Prueba | Mejor configuración | Impacto |
|--------|-------------------|---------|
| Stemming | **No usar** | Beneficia NB (+2%) pero destruye embeddings (−12.5%) |
| ngram_range | **(1, 1)** — unigrams | Supera a (1,2) en NB (+4.17%) y SVM (+2.08%) |
| max_features | **500** (min_df=2 limita a 103-163 términos) | Sin cambio real, el límite lo pone min_df |

> Documentación completa: [`Resultados-pruebas-preprocesamiento.md`](05-Desarrollo/pruebas/Resultados-pruebas-preprocesamiento.md) (9 secciones con teoría y resultados).

## 🔗 Enlaces útiles

- [Repositorio GitHub](https://github.com/Page10-User/Equipo-7---Chatbot-de-atencion-para-un-tramite-universitario)
- [Notebook de experimentos](05-Desarrollo/experimentos.ipynb)
- [Pipeline completo](05-Desarrollo/pipeline.py)
- [Fuente oficial — FACENA UNNE](https://exa.unne.edu.ar/r/?page_id=9008)
- [Video explicativo del trámite](https://www.youtube.com/watch?v=KDsmSxQoqJg&feature=youtu.be)

---

*Este documento se actualiza a medida que avanza el proyecto.*
