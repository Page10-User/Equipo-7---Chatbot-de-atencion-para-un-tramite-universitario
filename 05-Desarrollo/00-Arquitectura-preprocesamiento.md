# 🏗️ Arquitectura del Pipeline de Preprocesamiento

> Documento de diseño y fundamentos conceptuales.
> **Propósito:** Que el equipo entienda QUÉ hace cada pieza del pipeline y POR QUÉ existe, antes de escribir código.

---

## 📋 Índice

1. [El problema de fondo](#1-el-problema-de-fondo)
2. [Visión general del pipeline](#2-visión-general-del-pipeline)
3. [Etapa 1 — Carga del dataset](#3-etapa-1--carga-del-dataset)
4. [Etapa 2 — Limpieza y normalización](#4-etapa-2--limpieza-y-normalización)
5. [Etapa 3 — Tokenización y stopwords](#5-etapa-3--tokenización-y-stopwords)
6. [Etapa 4 — Vectorización](#6-etapa-4--vectorización)
7. [Conexión con los experimentos](#7-conexión-con-los-experimentos)
8. [Herramientas y justificación](#8-herramientas-y-justificación)
9. [Estructura del código](#9-estructura-del-código)
10. [Glosario rápido](#10-glosario-rápido)

---

## 1. El problema de fondo

Tenemos un dataset con **114 preguntas en español** distribuidas en **16 intenciones** (15 reales + fallback). Cada pregunta es texto escrito por humanos, con toda la variabilidad que eso implica.

### 🔬 El desafío

Un clasificador (Naive Bayes, SVM) **no entiende texto**. No sabe qué significa "título", "arancel" o "libre deuda". Solo entiende **números**.

El pipeline de preprocesamiento es el puente entre:
```
🌍 Lenguaje humano          🤖 Lenguaje de máquina
"¿Cómo saco el libre        →    [0.42, 0.0, 0.0, 0.87, ...]
 deuda de FACENA?"
```

Si este puente está mal construido, todos los experimentos que vienen después —accuracy, F1-score, matrices de confusión— **no valen nada**. Garbage in, garbage out.

### 🎯 Objetivos del preprocesamiento

| Objetivo                    | Descripción                                                                |
| --------------------------- | -------------------------------------------------------------------------- |
| **Reducir ruido**           | Sacar elementos que no aportan información (signos, caracteres especiales) |
| **Unificar vocabulario**    | Que "Título", "título" y "TITULO" sean la misma palabra                    |
| **Reducir dimensionalidad** | Menos palabras distintas = menos columnas = modelo más eficiente           |
| **Preservar significado**   | No sacar TANTO que perdamos la capacidad de distinguir intenciones         |

---

## 2. Visión general del pipeline

```
📁 02-Dataset/02-Dataset.json
            │
            ▼
┌─────────────────────────────────────────────┐
│  🔧 CARGA (dataset.py)                      │
│  JSON → pandas DataFrame (pregunta, intent) │
└─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────┐
│  🧹 LIMPIEZA Y NORMALIZACIÓN (cleaner.py)    │
│  • Minúsculas    • Sin tildes               │
│  • Sin puntuación • Sin URLs/emails          │
│  • Sin espacios múltiples                    │
└─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────┐
│  ✂️ TOKENIZACIÓN Y STOPWORDS (cleaner.py)    │
│  "como saco libre deuda"                     │
│       → ["saco", "libre", "deuda"]          │
└─────────────────────────────────────────────┘
            │
            ▼
        ╔════╤════╗
        ║  TF-IDF  ║     ──► Exp. 1 (Naive Bayes)
        ║          ║     ──► Exp. 2 (SVM lineal)
        ╚════╤════╝
            │
            ▼
    ┌───────────────┐
    │  Matriz TF-IDF │  (muestra: 114 filas × ~200 columnas)
    │  [0.0, 0.42,   │
    │   0.0, 0.0,    │
    │   0.87, ...]   │
    └───────────────┘

────────── O ──────────

        ╔═══════════╗
        ║ Embeddings ║     ──► Exp. 3 (SVM + embeddings)
        ╚════╤══════╝
            │
            ▼
    ┌───────────────────┐
    │  Matriz densa       │  (114 filas × 384 columnas)
    │  [0.23, -0.15,     │
    │   0.67, 0.02, ...] │
    └───────────────────┘
```

---

## 3. Etapa 1 — Carga del dataset

### 📥 Origen

El dataset está en `02-Dataset/02-Dataset.json`. La estructura es:

```json
{
  "intents": [
    {
      "intent": "libre_deuda_facena",
      "paso": "1",
      "preguntas": ["¿Cómo saco...?", "¿Dónde pido...?", ...],
      "respuesta": "Para obtener..."
    }
  ]
}
```

### 🔄 Transformación necesaria

Para entrenar un clasificador necesitamos un formato de dos columnas:

| texto | intent |
|-------|--------|
| "¿Cómo saco el libre deuda de FACENA?" | libre_deuda_facena |
| "¿Dónde pido el certificado de libre deuda?" | libre_deuda_facena |
| "¿Qué es el título intermedio?" | que_es_titulo_intermedio |

Cada pregunta individual se convierte en una fila. Esto se llama **"aplanar"** el JSON.

### 📊 Resultado esperado

- **114 filas** (una por pregunta)
- **2 columnas**: `texto` (la pregunta cruda) e `intent` (la intención)
- Listo para pasar al limpiador

### 💻 Código asociado

`dataset.py` — una función que:
1. Lee el JSON
2. Lo aplana a DataFrame
3. Devuelve `(X, y)` donde X son los textos e y son las etiquetas

---

## 4. Etapa 2 — Limpieza y normalización

### 🧹 ¿Por qué limpiar?

Mirá estas 3 preguntas reales de nuestro dataset:

```
"¿Cómo saco el libre deuda de la biblioteca de FACENA?"
"¿Dónde pido el certificado de libre deuda de la facultad?"
"¿Cómo obtengo el libre deuda de la biblioteca de FACENA?"
```

Para un ser humano son claramente la misma intención. Para una máquina, si no limpiamos:
- `"Cómo"` y `"cómo"` son tokens diferentes por la mayúscula
- `"FACENA?"` y `"FACENA"` son diferentes por el signo

Multiplicá eso por 114 preguntas y 300 palabras únicas y tenés un desastre.

### 🔧 Operaciones de limpieza

| Operación | Antes | Después | ¿Por qué? |
|-----------|-------|---------|-----------|
| **Minúsculas** | `"¿Cómo Saco?"` | `"cómo saco"` | Unifica vocabulario |
| **Sin puntuación** | `"¿Cómo saco?"` | `"Cómo saco"` | Los signos no aportan significado discriminativo |
| **Sin tildes** | `"cómo"` | `"como"` | Reduce variantes ortográficas |
| **Sin URLs** | `"...ver en https://..."` | `"...ver en "` | URLs no distinguen intenciones |
| **Sin espacios extra** | `"como   saco"` | `"como saco"` | Evita tokens vacíos |
| **Sin números** | `"Paso 1"` | `"Paso "` | Los números no aportan (depende del dominio) |

### ⚠️ Decisión de diseño: ¿tildes sí o no?

Para nuestro dominio (trámites universitarios), sacar tildes es beneficioso porque:
- `"cómo"` y `"como"` son la misma palabra para el trámite
- Reduce el vocabulario en ~5-10%
- No perdemos significado semántico relevante

### 💻 Código asociado

`cleaner.py` — función `limpiar(texto)` que aplica todas las operaciones en cadena.

---

## 5. Etapa 3 — Tokenización y stopwords

### ✂️ Tokenización

Es romper el texto en piezas individuales llamadas **tokens** (generalmente palabras).
	
```
"como saco el libre deuda"
    → ["como", "saco", "el", "libre", "deuda"]
```

Cada token será una **dimensión** en el espacio vectorial.

### 🛑 Stopwords (palabras vacías)

Ciertas palabras aparecen en TODAS las preguntas sin aportar información distintiva:

**Artículos:** el, la, los, las, un, una  
**Preposiciones:** de, en, para, con, por, sin  
**Conjunciones:** y, o, pero, que, como  
**Verbos comunes:** es, está, tiene, hacer, ser

Si dejamos estas palabras, el modelo les asigna importancia que no merecen.

```
🔴 SIN stopwords:    ["como", "saco", "el", "libre", "deuda", "la", "biblioteca", "de", "facena"]
🟢 CON stopwords:     ["saco", "libre", "deuda", "biblioteca", "facena"]
```

**Redujimos de 9 a 5 tokens**, y los 5 que quedan son los que realmente importan.

### 🤔 Pero ojo...

Hay casos donde "no" es una stopword pero sacarla es peligroso:
- `"¿Es obligatorio?"` vs `"¿No es obligatorio?"`
- `"¿Tiene costo?"` vs `"¿No tiene costo?"`

Para nuestro dataset, revisaremos si necesitamos conservar la negación. En los experimentos 1 y 2 con TF-IDF, el contexto suele compensar porque cada pregunta tiene suficientes palabras distintivas.

### 💻 Código asociado

`cleaner.py` — función `tokenizar(texto)` que tokeniza usando NLTK y filtra stopwords en español.

---

## 6. Etapa 4 — Vectorización

### 📐 ¿Qué es vectorizar?

Es convertir texto en números. La máquina necesita vectores de longitud fija para poder hacer matemática.

### 📊 Opción A: TF-IDF

TF-IDF = **Term Frequency × Inverse Document Frequency**

**TF (Frecuencia del Término):**
Cuántas veces aparece una palabra en una pregunta específica.

- Si `"deuda"` aparece 2 veces en la misma pregunta → TF alto.
- La mayoría de palabras aparecen 1 vez en preguntas cortas → TF = 1.

**IDF (Frecuencia Inversa de Documento):**
Qué tan rara es una palabra en todo el dataset.

```
IDF(t) = log(Total de preguntas / Preguntas que contienen t)
```

- `"título"` aparece en muchas preguntas → IDF bajo (~0.3) → **poco distintiva**
- `"arancel"` aparece en solo 8 preguntas → IDF alto (~2.5) → **muy distintiva**
- `"el"` aparece en TODAS las preguntas → IDF ~0 → **inservible** (por eso la sacamos como stopword)

**TF-IDF = TF × IDF**

Una palabra tiene alto TF-IDF si:
1. Aparece muchas veces en esta pregunta (TF alto), Y
2. Aparece en pocas preguntas del dataset (IDF alto)

**Ejemplo visual:**

| Palabra         | TF  | IDF  | TF-IDF  |
| --------------- | --- | ---- | ------- |
| "deuda"         | 1   | 2.3  | **2.3** |
| "título"        | 1   | 0.5  | **0.5** |
| "el" (stopword) | 1   | 0.01 | **0**   |

**Output:** Matriz dispersa de 114 filas × 250 columnas (una por palabra única). La mayoría de las celdas son 0 (palabras que no aparecen en esa pregunta).

### 🧠 Opción B: Embeddings (Sentence-Transformers)

Los embeddings resuelven una limitación importante de TF-IDF:

**TF-IDF:** No entiende sinónimos. `"¿Cómo saco el libre deuda?"` y `"¿Dónde pido el certificado?"` comparten **0 palabras** → TF-IDF las ve como completamente diferentes.

**Embeddings:** Cada palabra se convierte en un vector de 384 números que representan su **significado**. Palabras con significado similar tienen vectores cercanos.

```
"deuda" → [0.23, -0.15, 0.67, 0.02, ...]
"certificado" → [0.20, -0.10, 0.71, 0.05, ...]  ← parecido!
```

**Aún mejor:** Sentence-Transformers no vectoriza palabras sueltas, sino **oraciones completas**. La pregunta entera se convierte en un único vector de 384 dimensiones que captura el significado global.

**Output:** Matriz densa de 114 filas × 384 columnas. Sin ceros.

### 📊 Comparativa visual

```
TF-IDF (disperso):                    Embeddings (denso):
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ "deuda"  "pago"  "título"                        │       │ dim_0  dim_1  dim_2  dim_3                  │
│   2.3     0.0     0.0                                    │       │ 0.23  -0.15  0.67   0.02                            │
│   0.0     1.8     0.0                                     │       │ 0.20  -0.10  0.71   0.05                            │
│   0.0     0.0     0.5                                    │       │ 0.18  -0.12  0.65   0.01                             │
│   0.0     0.0     0.0                                    │       │ -0.05  0.30  0.12   0.45                           │
└─────────────────────────────┘       └─────────────────────────────┘
Cada celda es una palabra                                Cada celda es una dimensión
Muchos ceros                                                    Sin ceros
No entiende sinónimos                                     Captura significado semántico
```

### 💻 Código asociado

`vectorizer.py`:
- `vectorizar_tfidf(textos)` → devuelve matriz TF-IDF + el vectorizador
- `vectorizar_embeddings(textos)` → devuelve matriz densa de embeddings

---

## 7. Conexión con los experimentos

```
                        ┌──────────────────┐
                        │   DATASET (114)   │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Train/Test Split │
                        │   (80% / 20%)     │
                        └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
            ┌───────▼──────┐  ┌─▼──────────┐ │
            │   TF-IDF     │  │  TF-IDF    │ │
            └───────┬──────┘  └──────┬──────┘ │
                    │               │        │
            ┌───────▼──────┐  ┌─────▼──────┐  │
            │ Naive Bayes  │  │    SVM     │  │
            │ (Multinomial)│  │  (Lineal)  │  │
            └───────┬──────┘  └──────┬──────┘  │
                    │               │        │
            ┌───────▼──────┐  ┌─────▼──────┐  │
            │ Accuracy     │  │ Accuracy   │  │
            │ Precision    │  │ Precision  │  │
            │ Recall, F1   │  │ Recall, F1 │  │
            └──────────────┘  └────────────┘  │
                                       │
                                       │
                              ┌────────▼────────┐
                              │   Embeddings    │
                              │ (Sentence-      │
                              │  Transformer)   │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │      SVM        │
                              │   (Lineal)      │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │   Accuracy      │
                              │   Precision     │
                              │   Recall, F1    │
                              └─────────────────┘
```

**Variable controlada por experimento:**

| Experimento | Vectorización | Clasificador | Variable que cambia |
|------------|--------------|--------------|-------------------|
| **Exp. 1** | TF-IDF | Naive Bayes | — (baseline) |
| **Exp. 2** | TF-IDF | SVM lineal | **Clasificador** |
| **Exp. 3** | Embeddings | SVM lineal | **Vectorización** |

Así podemos aislar qué mejora (o no) el rendimiento.

---

## 8. Herramientas y justificación

| Herramienta | Versión | Función | ¿Por qué esta y no otra? |
|-------------|---------|---------|--------------------------|
| **Python** | 3.x | Lenguaje | Estándar en NLP/ML. Bibliotecas maduras. |
| **pandas** | 2.x | DataFrame | Aplana el JSON a tabla. `pd.json_normalize()` |
| **NLTK** | 3.x | Tokenización, stopwords | Tiene stopwords en español incluidas. Liviana. |
| **scikit-learn** | 1.x | TF-IDF, modelos, métricas | API unificada. `TfidfVectorizer`, `MultinomialNB`, `SVC`, `classification_report`. |
| **sentence-transformers** | 3.x | Embeddings | Modelos multilingüe. `distiluse-base-multilingual-cased` (384 dims). |
| **matplotlib / seaborn** | — | Gráficos | Matriz de confusión, barras comparativas. |
| **Jupyter** | — | Notebook | Código + resultados + gráficos + explicaciones en un solo archivo. |

---

## 9. Estructura del código

```
05-Desarrollo/
│
├── 00-Arquitectura-preprocesamiento.md   ← Este documento
│
├── dataset.py          ← Carga y aplana el JSON
├── cleaner.py          ← Limpieza, tokenización, stopwords
├── vectorizer.py       ← TF-IDF + sentence-transformers
│
├── pipeline.py         ← Orquesta todo: carga → limpia → vectoriza → entrena
│
└── experimentos.ipynb  ← Notebook con los 3 experimentos + métricas + gráficos
```

**Dependencias entre módulos:**
```
dataset.py  ──►  cleaner.py  ──►  vectorizer.py  ──►  pipeline.py
     │               │                │
     └───────────────┴────────────────┴──────────►  experimentos.ipynb
```

---

## 10. Glosario rápido

| Término | Significado |
|---------|-------------|
| **Token** | Unidad mínima de texto (generalmente una palabra) |
| **Stopword** | Palabra vacía sin valor discriminativo (el, la, de) |
| **TF-IDF** | Ponderación que mide importancia de una palabra en un documento dentro de un conjunto |
| **Vector disperso** | Vector con muchos ceros (típico de TF-IDF) |
| **Vector denso** | Vector con pocos o ningún cero (típico de embeddings) |
| **Dimensionalidad** | Cantidad de columnas del vector (features) |
| **Embedding** | Representación numérica densa que captura significado semántico |
| **Train/Test split** | División del dataset en entrenamiento (80%) y prueba (20%) |
| **Accuracy** | Porcentaje de aciertos sobre el total |
| **Precision** | De las que clasifiqué como X, ¿cuántas eran realmente X? |
| **Recall** | De las que eran realmente X, ¿cuántas clasifiqué bien? |
| **F1-score** | Media armónica entre precision y recall |

---

> **Próximo paso:** Implementar `dataset.py` — cargar el JSON y aplanarlo a un DataFrame de pandas con columnas `texto` e `intent`.
>
> *Este documento vive en el vault de Obsidian como referencia del equipo. Cuando el pipeline esté completo y verificado, se sube al repositorio de GitHub.*
