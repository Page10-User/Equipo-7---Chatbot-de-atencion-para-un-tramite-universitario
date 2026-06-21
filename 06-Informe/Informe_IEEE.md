---
title: "Sistema de Clasificacion de Intenciones para un Asistente Virtual de Tramites Universitarios"
author:
  - "Equipo 7"
  - "Gonzalo Nahuel Acosta Lopez <gonzalineritolol@gmail.com>"
  - "Cesar Pietro Barrios Calathaki <pietrocalathaki@gmail.com>"
date: "2026"
abstract: |
  Este trabajo presenta el desarrollo de un asistente virtual conversacional
  para orientar a estudiantes de la Facultad de Ciencias Exactas y Naturales
  y Agrimensura (FaCENA-UNNE) en el tramite del Titulo Intermedio de
  Pregrado. Se implementaron y compararon tres modelos de clasificacion de
  intenciones: Naive Bayes Multinomial con TF-IDF, SVM Lineal con TF-IDF
  y SVM con kernel RBF utilizando embeddings semanticos (SentenceTransformer).
  El modelo ganador alcanzo un 81.25% de exactitud y 0.81 de F1-macro,
  validado con 5-fold cross-validation. Se desarrollo una interfaz web
  funcional con Streamlit para su uso por parte de la comunidad estudiantil.
---

# Sistema de Clasificacion de Intenciones para un Asistente Virtual de Tramites Universitarios

## I. Introduccion

### _A. Problema y Contexto_

Los estudiantes de la Universidad Nacional del Nordeste (UNNE) que cursan
carreras en la Facultad de Ciencias Exactas y Naturales y Agrimensura
(FaCENA) enfrentan dificultades para comprender el proceso de obtencion
del Titulo Intermedio de Pregrado. El tramite involucra dos etapas
principales, cada una con requisitos, plazos y procedimientos especificos
que generan consultas recurrentes a la Direccion de Gestion de Estudios.
Actualmente no existe un canal automatizado que resuelva estas consultas
de forma inmediata, lo que deriva en sobrecarga administrativa y demoras
en la atencion.

### _B. Alcance_

El sistema desarrollado es un asistente virtual capaz de interpretar
preguntas en lenguaje natural formuladas por los estudiantes y clasificarlas
en 16 categorias (15 intenciones especificas del tramite + 1 fallback).
El asistente responde con informacion oficial basada en la resolucion
RES-2025-7936-REC-UNNE y fuentes institucionales de FaCENA. Capaz de respoder _en este caso_ consultas sobre el trámite de título intermedio.

### _C. Tema Disciplinar y Ambito de Aplicacion_

Este trabajo se enmarca en el area de Procesamiento de Lenguaje Natural
(NLP), especificamente en clasificacion de textos cortos. El ambito de
aplicacion es el dominio administrativo-universitario, con potencial
de transferencia a otros tramites institucionales.

### _D. Evolución_
Este trabajo fue hecho en base a la prueba de un modelo que no probamos durante la asignatura, el modelo SVM, además de aplicar herramientas nuevas relacionadas con el Procesamiento de lenguaje natural PLN como TF IDF (modelo 1 y 2) y Embeddings (modelo 3).
---

## II. Metodo

### _A. Metodologia_

Se adopto un proceso iterativo basado en CRISP-DM (Cross-Industry Standard
Process for Data Mining), adaptado para aprendizaje automatico supervisado:

1. **Comprension del negocio**: Analisis del tramite y relevamiento de
   consultas frecuentes.
2. **Preparacion de datos**: Curado y expansion del dataset, limpieza
   de texto (normalizacion, eliminacion de stopwords, stemming opcional).
3. **Modelado**: Entrenamiento de tres arquitecturas de clasificacion.
4. **Evaluacion**: Validacion con metricas de exactitud, F1-macro y
   matriz de confusion, complementadas con 5-fold cross-validation y
   GridSearchCV para optimizacion de hiperparametros.

### _B. Modelos de IA_

Se implementaron tres modelos supervisados de clasificacion multiclase:

| Modelo | Vectorizacion | Hiperparametros |
|--------|---------------|-----------------|
| **M1** - MultinomialNB | TF-IDF (unigramas, 103 terminos) | alpha = 0.5 |
| **M2** - SVM Lineal | TF-IDF (unigramas, 103 terminos) | C = 1.0, kernel = linear |
| **M3** - SVM RBF | SentenceTransformer (512 dim.) | C = 10.0, gamma = scale |

**M1** utiliza Naive Bayes Multinomial con suavizado de Laplace (alpha=0.5),
que modela la probabilidad condicional de cada termino dada una clase.
**M2** emplea Support Vector Machine con kernel lineal, que encuentra el
hiperplano de separacion optimo maximizando el margen entre clases.
**M3** utiliza SVM con kernel RBF sobre embeddings semanticos densos
generados por el modelo multilingue distiluse-base-multilingual-cased-v2,
capturando relaciones semanticas entre las consultas.

### _C. Experimentacion_

El dataset original de 240 preguntas (16 por cada una de las 15 intenciones)
se dividio en 80% entrenamiento y 20% prueba (192/48), estratificado para
preservar la proporcion de clases. Se realizaron tres fases experimentales:

- **Fase 1 — Preprocesamiento**: Evaluacion de stemming, rango de n-gramas
  y maximo de caracteristicas (max_features).
- **Fase 2 — Optimizacion**: GridSearchCV con 5-fold cross-validation
  sobre los hiperparametros de cada modelo.
- **Fase 3 — Validacion final**: Entrenamiento con dataset completo y
  evaluacion cualitativa con 7 preguntas novedosas.

---

## III. Herramientas

El sistema fue desarrollado integramente en **Python 3.12**. Las
principales librerias utilizadas fueron:

| Libreria | Version | Proposito |
|----------|---------|-----------|
| scikit-learn | 1.6 | Modelos (MultinomialNB, SVC), vectorizacion (TfidfVectorizer), validacion (GridSearchCV, cross_val_score) |
| sentence-transformers | 3.4 | Embeddings semanticos (distiluse-base-multilingual-cased-v2) |
| pandas / numpy | 2.2 / 2.0 | Manipulacion de datos y calculos numericos |
| joblib | 1.4 | Persistencia de modelos entrenados |
| streamlit | 1.57 | Interfaz web de usuario |

Para el preprocesamiento de texto se implementaron funciones de
normalizacion (minusculas, eliminacion de tildes, puntuacion y
stopwords en espanol) utilizando expresiones regulares y NLTK.

---

## IV. Resultados

### _A. Experimentos de Preprocesamiento_

| Experimento | Configuracion | NB+TF-IDF | SVM+TF-IDF | SVM+Emb. |
|-------------|--------------|-----------|------------|----------|
| Baseline | unigramas, sin stemming | 77.08% | 72.92% | 81.25% |
| Con stemming | unigramas + stemming | 79.17% | 70.83% | 68.75% |
| Bigramas | (1,2) + sin stemming | 75.00% | 72.92% | — |
| Sin acotar | max_features=500 | 77.08% | 72.92% | — |

El stemming beneficio levemente a NB (+2.09%) pero degrada
significativamente a los modelos basados en embeddings (-12.50%),
por lo que se descarto para el pipeline final. Los unigramas
superaron a los bigramas al generar representaciones menos dispersas
con 240 muestras.

### _B. Optimizacion de Hiperparametros (GridSearchCV + 5-Fold CV)_

| Modelo | Mejores params | CV F1-macro |
|--------|---------------|-------------|
| MultinomialNB | alpha = 0.5 | 0.7063 |
| SVM Lineal | C = 1.0 | 0.6872 |
| SVM RBF | C = 10.0, gamma = scale | **0.7600** |

### _C. Resultados Finales_

| Modelo | Exactitud | F1-macro | F1-ponderado |
|--------|-----------|----------|-------------|
| NB + TF-IDF (alpha=0.5) | 79.17% | 0.77 | 0.79 |
| SVM Lineal + TF-IDF (C=1.0) | 75.00% | 0.73 | 0.74 |
| SVM RBF + Embeddings (C=10.0) | **81.25%** | **0.81** | **0.81** |

El modelo SVM con embeddings semanticos (M3) obtuvo el mejor desempeno
general. En una prueba cualitativa con 7 preguntas novedosas no incluidas
en el dataset, clasifico correctamente 6 de 7 (85.71%), demostrando
capacidad de generalizacion.

---

## V. Conclusiones

### _A. Hallazgos Principales_

1. Los embeddings semanticos (SentenceTransformer) superan a TF-IDF
   cuando se combinan con un clasificador no lineal (SVM RBF),
   alcanzando 81.25% de exactitud y 0.81 de F1-macro.
2. El preprocesamiento agresivo (stemming) beneficia modelos
   probabilisticos como Naive Bayes pero perjudica gravemente a los
   modelos basados en embeddings (-12.5 puntos porcentuales).
3. Con conjuntos de datos pequenos (240 muestras, 15 clases), los
   unigramas son preferibles a bigramas, que generan demasiada
   dispersion.
4. La validacion con 5-fold CV y GridSearchCV permitio identificar
   configuraciones optimas (NB alpha=0.5, SVM RBF C=10.0) que
   mejoraron hasta 4 puntos respecto a los valores por defecto.

### _B. Trabajos Futuros_

- Expansion del dataset a 50+ preguntas por intencion (750+ total)
  para mejorar la robustez del clasificador.
- Incorporacion de tecnicas de data augmentation y balanceo de clases.
- Integracion con sistema de ticketing institucional para derivacion
  de consultas no resueltas.
- Evaluacion de modelos de lenguaje pre-entrenados (BERT, GPT) para
  clasificacion few-shot.
- Expansión a otros trámites institucionales.

---

## Referencias

[1] S. Raschka y V. Mirjalili, _Python Machine Learning_, 3ra ed.
    Birmingham: Packt Publishing, 2019.

[2] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python,"
    _Journal of Machine Learning Research_, vol. 12, pp. 2825-2830, 2011.

[3] N. Reimers e I. Gurevych, "Sentence-BERT: Sentence Embeddings using
    Siamese BERT-Networks," en _Proc. EMNLP-IJCNLP_, 2019, pp. 3982-3992.

[4] C. Cortes y V. Vapnik, "Support-vector networks," _Machine Learning_,
    vol. 20, no. 3, pp. 273-297, 1995.

[5] T. Mikolov et al., "Efficient Estimation of Word Representations in
    Vector Space," en _Proc. ICLR_, 2013.

[6] Resolucion RES-2025-7936-REC-UNNE. Universidad Nacional del Nordeste.
    Disponible: https://exa.unne.edu.ar/r/?page_id=9008

[7] S. Bird, E. Klein y E. Loper, _Natural Language Processing with
    Python_. O'Reilly Media, 2009.
