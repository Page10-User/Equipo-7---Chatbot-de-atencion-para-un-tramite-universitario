# 🎓 Agente Conversacional — Tramitación del Título Intermedio (Pregrado)

**Facultad de Ciencias Exactas y Naturales y Agrimensura (FACENA) — UNNE**  
**Asignatura:** Inteligencia Artificial  
**Equipo 7 — Año 2026**

---

## 📋 Descripción

Sistema de clasificación de intenciones en lenguaje natural que orienta a estudiantes de FaCENA-UNNE en el trámite del **Título Intermedio de Pregrado**. El usuario formula una consulta en lenguaje natural, el sistema clasifica su intención (15 intenciones + fallback) y devuelve la respuesta oficial correspondiente.

---

## 👥 Integrantes

| Nombre                         | Rol                                              |
| ------------------------------ | ------------------------------------------------ |
| Acosta Lopez Gonzalo Nahuel    | Dataset, pipeline, experimentos, interfaz, informe |
| Cesar Pietro Barrios Calathaki | Dataset, informe                                 |

---

## 📂 Estructura del proyecto

| Carpeta                                                          | Contenido                                                              |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [`01-Planificacion/`](01-Planificacion/00-Planning.md)           | Planning, alcance, intenciones, experimentos, división de tareas       |
| [`02-Dataset/`](02-Dataset/)                                     | Dataset en JSON (v1.4, 240 preguntas, 15 intents + fallback)           |
| [`03-Referencias/`](03-Referencias/)                             | Fuente oficial de FACENA-UNNE, transcripción, repositorio GitHub       |
| [`04-Consignas/`](04-Consignas/)                                 | TP Integrador oficial (enunciado)                                      |
| [`05-Desarrollo/`](05-Desarrollo/)                               | Pipeline, modelos, experimentos, script de inferencia                  |
| [`06-Informe/`](06-Informe/)                                     | Informe IEEE (LaTeX + PDF + Markdown) e infografía                     |
| [`interfaz/`](interfaz/)                                         | Aplicación web Streamlit                                                |

---

## 🚦 Estado del proyecto

| Etapa                                                    | Estado                                                                                                 |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Dataset**                                              | ✅ **v1.4 — Completo** (15 intenciones + fallback, 240 preguntas, respuestas actualizadas)              |
| **Arquitectura de preprocesamiento**                     | ✅ **Documentada** — [`00-Arquitectura-preprocesamiento.md`](05-Desarrollo/00-Arquitectura-preprocesamiento.md) |
| **Preprocesamiento (código)**                            | ✅ **Implementado** — [`dataset.py`](05-Desarrollo/dataset.py) + [`cleaner.py`](05-Desarrollo/cleaner.py) |
| **Vectorización**                                        | ✅ **Implementada** — TF-IDF (103 términos, unigrams) + SentenceTransformer (512 dims)                 |
| **Exp 1 — Naive Bayes + TF-IDF**                         | ✅ **Completado** — Accuracy: **79.17%** — F1 macro: **0.77** (α=0.5, unigrams)                       |
| **Exp 2 — SVM Lineal + TF-IDF**                          | ✅ **Completado** — Accuracy: **75.00%** — F1 macro: **0.73** (C=1.0, unigrams)                        |
| **Exp 3 — SVM RBF + Embeddings**                         | ✅ **Completado** — Accuracy: **81.25%** — F1 macro: **0.81** (C=10.0, γ=scale) [GANADOR]             |
| Preprocesamiento avanzado (stemming, ngrams, max_features) | ✅ **Completado** — Resultados documentados en documento de pruebas                                   |
| GridSearchCV + K-Fold Cross Validation                   | ✅ **Completado** — α=0.5, C=10.0, γ=scale, CV F1: 0.76 (embeddings)                                  |
| **Modelos finales entrenados y guardados**               | ✅ **Listos** — NB, SVM-L, SVM-Emb en [`modelos/`](05-Desarrollo/modelos/)                              |
| **Script de inferencia**                                 | ✅ **Listo** — [`clasificar.py`](05-Desarrollo/clasificar.py) — `python clasificar.py "tu pregunta"`    |
| **Interfaz Streamlit**                                   | ✅ **Completa** — [`interfaz/app.py`](interfaz/app.py) — 3 modelos seleccionables, historial chat       |
| **Informe IEEE**                                         | ✅ **Completo** — [`06-Informe/informe_ieee.pdf`](06-Informe/informe_ieee.pdf) (2 págs, LaTeX)          |
| **Infografía**                                           | ✅ **Completa** — [`06-Informe/Infografía/`](06-Informe/Infografía/) (HTML + PDF)                      |

---

## 📈 Resultados finales

| Experimento | Modelo              | Vectorización                | Hiperparámetros              | Accuracy   | F1 (macro) | CV F1 (K=5) |
|-------------|---------------------|------------------------------|------------------------------|------------|------------|-------------|
| Exp 1       | Naive Bayes         | TF-IDF (103 terms, unigrams) | α = 0.5                      | **79.17%** | 0.77       | 0.71        |
| Exp 2       | SVM Lineal          | TF-IDF (103 terms, unigrams) | C = 1.0                      | 75.00%     | 0.73       | 0.69        |
| Exp 3       | **SVM RBF**         | **SentenceTransformer (512d)**| **C = 10.0, γ = scale**      | **81.25%** | **0.81**   | **0.76**    |

> **🏆 Ganador: SVM RBF + Embeddings (C=10.0, γ=scale).** Los embeddings semánticos capturan similitud que TF-IDF no puede ver, y el kernel RBF con C elevado aprovecha la estructura no lineal del espacio de embeddings.

### Pruebas cualitativas

- **7 preguntas novedosas** (no incluidas en el dataset) → **6/7 correctas (85.71%)**
- El modelo generaliza bien a consultas reales de estudiantes

### Pruebas de preprocesamiento (Fase 3)

| Prueba       | Mejor configuración | Impacto                                                      |
|--------------|---------------------|--------------------------------------------------------------|
| Stemming     | **No usar**         | Beneficia NB (+2%) pero destruye embeddings (−12.5%)         |
| ngram_range  | **(1, 1)** — unigrams | Supera a (1,2) en NB (+4.17%) y SVM (+2.08%)              |
| max_features | **500**             | Sin efecto real; min_df=2 limita vocabulario a ~163 términos |

> Documentación completa: [`Resultados-pruebas-preprocesamiento.md`](05-Desarrollo/pruebas/Resultados-pruebas-preprocesamiento.md)

---

## 🖥️ Interfaz web (Streamlit)

Una aplicación web interactiva para probar el clasificador en tiempo real:

- **3 modelos seleccionables** desde la barra lateral (SVM+Embeddings como predeterminado)
- **Sugerencias de preguntas** (4 tarjetas en grilla 2×2)
- **Historial de conversación** con mensajes de usuario y asistente
- **Tema oscuro** (#0B0F19) consistente con la identidad visual del proyecto
- Botón para limpiar el chat

```bash
streamlit run interfaz/app.py
```

> Primera carga ~5 segundos (cacheo de SentenceTransformer). Posteriormente es instantáneo.

---

## 📄 Informe IEEE

Disponible en `06-Informe/` en tres formatos:

| Formato | Archivo |
|---------|---------|
| LaTeX   | [`informe_ieee.tex`](06-Informe/informe_ieee.tex) |
| PDF     | [`informe_ieee.pdf`](06-Informe/informe_ieee.pdf) |
| Markdown | [`Informe_IEEE.md`](06-Informe/Informe_IEEE.md) |

Secciones: Introducción (Problema, Alcance, Tema Disciplinar, Evolución), Método (Metodología CRISP-DM, Modelos, Experimentación), Herramientas, Resultados (Preprocesamiento, Optimización, Finales) y Conclusiones.

---

## 🖼️ Infografía

Disponible en [`06-Informe/Infografía/`](06-Informe/Infografía/):

- Versión HTML/CSS interactiva → [`infografia.html`](06-Informe/infografia.html)
- Versión PDF exportada → [`Infografía - Equipo 7 _ Chatbot Trámite UNNE.pdf`](06-Informe/Infografía/)

Diseño oscuro profesional con: pipeline visual, cards de modelos con resultados, tabla de métricas, tecnologías y equipo.

---

## 🔗 Enlaces útiles

- [Repositorio GitHub](https://github.com/Page10-User/Equipo-7---Chatbot-de-atencion-para-un-tramite-universitario)
- [Notebook de experimentos](05-Desarrollo/experimentos.ipynb)
- [Pipeline completo](05-Desarrollo/pipeline.py)
- [Fuente oficial — FACENA UNNE](https://exa.unne.edu.ar/r/?page_id=9008)

---

*Última actualización: Junio 2026*
