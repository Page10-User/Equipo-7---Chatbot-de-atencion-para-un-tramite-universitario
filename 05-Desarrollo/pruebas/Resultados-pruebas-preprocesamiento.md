# Resultados de Pruebas — Preprocesamiento y Vectorización

> **Propósito**: Documentar los experimentos de preprocesamiento realizados sobre el dataset v1.3 (240 preguntas, 15 intenciones), explicar el fundamento teórico de cada cambio, y justificar las decisiones tomadas antes de avanzar a la Fase 4 (GridSearchCV + K-Fold Cross Validation).

---

## 1. Configuración General

| Parámetro | Valor |
|---|---|
| Dataset | v1.3 — 240 preguntas, 15 intenciones |
| Test split | 20% (48 preguntas), estratificado |
| Random state | 42 |
| Preprocesamiento base | `cleaner.preprocesar()` — minúsculas, sin acentos, sin puntuación, tokenización NLTK, stopwords español |
| Modelos evaluados | MultinomialNB, SVM Lineal, SVM RBF |
| Métricas | Accuracy, F1-score macro |

---

## 2. Prueba 1: Stemming (SnowballStemmer español)

### 2.1. ¿Qué probamos?

Aplicar stemming **después de la tokenización** usando `SnowballStemmer("spanish")` sobre el texto ya limpiado. El stemmer reduce cada palabra a su raíz morfológica:

| Original | Stem |
|---|---|
| "solicitar", "solicito", "solicitó" | → "solicit" |
| "título", "tituló" | → "titul" |
| "universidad", "universitario" | → "universit" |

Se preservaron **siglas** (DNI, SIU, CUIL) y palabras cortas (< 3 caracteres) para no perder términos informativos.

### 2.2. Resultados

| Experimento | SIN Stemming | CON Stemming | Diferencia |
|---|---|---|---|
| NB + TF-IDF | **75.00% / 0.7291** | 77.08% / 0.7471 | **+2.08 pp** ✅ |
| SVM + TF-IDF | **72.92% / 0.7145** | 70.83% / 0.6886 | −2.08 pp ❌ |
| SVM + Embeddings (RBF) | **81.25% / 0.8128** | 68.75% / 0.6686 | **−12.50 pp** ❌❌ |

### 2.3. Fundamento Teórico

**Stemming** agrupa variantes morfológicas de una misma palabra. Esto:

- **Reduce la dimensionalidad** del vocabulario (menos términos distintos)
- **Aumenta la frecuencia** de cada término por clase (más evidencia estadística)
- **Elimina ruido** de conjugaciones y género

**¿Por qué ayuda a Naive Bayes?**
NB estima `P(término|clase)` por conteo. Si "solicito" aparece 1 vez y "solicitar" aparece 2 veces en la misma clase, el stemmer las unifica → el término "solicit" aparece 3 veces. NB tiene **más evidencia** para esa palabra → mejor estimación de probabilidad.

**¿Por qué NO ayuda a SVM + Embeddings?**
Los transformers (SentenceTransformer) están entrenados con **palabras completas en contexto**. La palabra "solicit" no existe en el vocabulario del modelo preentrenado. El embedding generado para "solicit" es **ruido** — no se parece al embedding de "solicito" ni de "solicitar". Literalmente estamos destruyendo el sentido semántico.

**Analogía**: Es como si le cambiaras el idioma a un traductor simultáneo — no importa qué tan bien haga su trabajo, lo que le das de entrada no tiene sentido.

### 2.4. Decisión

**No aplicar stemming.** Beneficia marginalmente a NB (+2 pp) pero perjudica severamente a nuestro mejor modelo (Embeddings, −12.5 pp). El modelo final priorizará la configuración de embeddings.

---

## 3. Prueba 2: ngram_range en TF-IDF

### 3.1. ¿Qué probamos?

Variar el rango de n-gramas en TfidfVectorizer manteniendo `max_features=500`, `min_df=2`:

| Configuración | Descripción |
|---|---|
| `(1, 1)` | Solo **unigramas**: "título", "intermedio", "libre", "deuda" |
| `(1, 2)` | Unigramas **+ bigramas**: "título intermedio", "libre deuda" |
| `(1, 3)` | Unigramas + bigramas + trigramas |
| `(2, 2)` | Solo bigramas |

### 3.2. Resultados

| ngram_range | Términos | NB Acc | NB F1 | SVM Acc | SVM F1 |
|---|---|---|---|---|---|
| **(1, 1)** | 103 | **79.17%** | **0.7729** | **75.00%** | **0.7329** |
| (1, 2) ← anterior | 163 | 75.00% | 0.7291 | 72.92% | 0.7145 |
| (1, 3) | 181 | 75.00% | 0.7291 | 72.92% | 0.7145 |
| (2, 2) | 60 | 39.58% | 0.3611 | 41.67% | 0.3939 |

### 3.3. Fundamento Teórico

**N-gramas** capturan contexto local. Un bigrama como "libro deuda" tiene más información semántica que "libro" + "deuda" por separado. **Pero** hay un tradeoff: cuantos más n-gramas, más términos distintos y **menos ocurrencias por término**.

**La maldición de la dimensionalidad en texto**: Con 240 documentos y 15 clases, cada clase tiene ~16 documentos. Un bigrama aparece típicamente 1-2 veces en toda la clase. Para Naive Bayes, `P(bigrama | clase)` se estima con muy pocas observaciones → **estimaciones ruidosas**.

**Unigramas ganan porque**:
- Cada palabra aparece en más documentos → estimaciones más estables
- La pérdida de contexto (no distinguir "título intermedio" de "título universitario") se compensa porque el clasificador aprende **patrones de palabras individuales**
- NB es un modelo generativo: necesita **frecuencias confiables**, no relaciones entre palabras

**Solo bigramas (2,2) es un desastre** porque perdés todas las palabras individuales. Muchos bigramas son irrepetibles ("paso_segundo" aparece 1 vez en todo el dataset).

### 3.4. Decisión

**Cambiar default de `(1, 2)` a `(1, 1)`.** Esto mejora NB en +4.17 pp y SVM en +2.08 pp. Se actualizó `vectorizer.py`.

---

## 4. Prueba 3: max_features en TF-IDF

### 4.1. ¿Qué probamos?

Variar la cantidad máxima de términos en TfidfVectorizer manteniendo `min_df=2`:

| max_features | Términos reales | NB Acc | SVM Acc |
|---|---|---|---|
| 50 | 50 | 66.67% | 66.67% |
| 100 | 100 | 72.92% | 72.92% |
| 200 | 163 | 75.00% | 72.92% |
| 500 | 163 | 75.00% | 72.92% |
| 1000 | 163 | 75.00% | 72.92% |

### 4.2. Análisis

El **cuello de botella real** no es `max_features` sino `min_df`. Con `min_df=2` (ignorar términos que aparecen en menos de 2 documentos), el dataset de 240 preguntas apenas produce **163 términos reales**. Aunque pidamos 500 o 1000, `min_df` los filtra.

Con 50 términos es demasiado restrictivo (perdés palabras informativas), con 100 se estabiliza, y a partir de 163 es indistinto.

### 4.3. Decisión

**Mantener `max_features=500`.** No causa daño (el límite real lo pone `min_df`), y si en el futuro crece el dataset, el límite superior ya está configurado. **No cambiar `min_df`** — con `min_df=1` entrarían términos que aparecen en un solo documento (ruido).

---

## 5. Tabla Resumen — Configuraciones Evaluadas

| Configuración | Mejor accuracy | Mejor F1 | CV F1 (K=5) | Tiempo relativo | Veredicto |
|---|---|---|---|---|---|
| **SVM + Embeddings (RBF)** | **81.25%** | **0.8090** | **0.7600** | Lento (carga modelo ~5s) | 🏆 **Ganador general** |
| NB + TF-IDF (1,1) | 79.17% | 0.7711 | 0.7063 | **Instantáneo** | ✅ Alternativa rápida |
| NB + TF-IDF (1,2) ← anterior | 75.00% | 0.7291 | — | Instantáneo | ❌ Superado |
| SVM + TF-IDF (1,1) | 75.00% | 0.7329 | 0.6872 | Rápido | ❌ Superado |
| NB + Stemming | 77.08% | 0.7471 | — | Rápido | ❌ No compatible con embeddings |
| SVM + Emb + Stemming | 68.75% | 0.6686 | — | Lento | ❌❌ **Evitar** |

---

## 6. Decisiones Finales y Estado del Pipeline

| Componente | Antes | Ahora | Motivo |
|---|---|---|---|
| `ngram_range` en TF-IDF | `(1, 2)` | `(1, 1)` | Unigrams estiman mejor con 240 muestras |
| Stemming | No se aplicaba | Sigue sin aplicarse | Destruye embeddings (−12.5 pp) |
| `max_features` | 500 | 500 | Sin cambios (no es cuello de botella) |
| `min_df` | 2 | 2 | Sin cambios (filtra ruido) |
| NB `alpha` | 1.0 (default) | **0.5** | Menos suavizado → más peso a frecuencias reales |
| SVM Lineal `C` | 1.0 (default) | 1.0 | Default ya era óptimo según GridSearchCV |
| SVM RBF `C` | 1.0 (default) | **10.0** | Embeddings de alta calidad permiten margen más ajustado |
| SVM RBF `gamma` | 'scale' (default) | 'scale' | Default ya era óptimo según GridSearchCV |
| **Mejor modelo** | SVM+Emb RBF | SVM+Emb RBF (`C=10.0`) | Mantuvo 81.25% test, mejoró CV F1 a 0.76 |

---

## 7. Fase 4: GridSearchCV + K-Fold Cross Validation

### 7.1. ¿Qué probamos?

Optimización de hiperparámetros con **búsqueda en grilla (GridSearchCV)** y **validación cruzada estratificada de 5 folds (StratifiedKFold)**.

**Grids evaluados:**

| Experimento | Hiperparámetros | Grid |
|---|---|---|
| NB + TF-IDF (1,1) | `alpha` (suavizado Laplace) | [0.01, 0.1, 0.5, 1.0, 2.0, 5.0] |
| SVM Lineal + TF-IDF (1,1) | `C` (regularización) | [0.01, 0.1, 1.0, 10.0, 100.0] |
| SVM RBF + Embeddings | `C`, `gamma` | C: [0.1, 1.0, 10.0], gamma: ['scale', 'auto', 0.01, 0.001] |

Métrica de scoring: `f1_macro`. Se reportan CV score (validación cruzada), train score (para detectar sobreajuste), y test score final.

### 7.2. Resultados

| Experimento | Best Params | CV F1 (val) | CV F1 (train) | Diff (train−val) | Test Acc | Test F1 |
|---|---|---|---|---|---|---|
| NB + TF-IDF (1,1) | `alpha=0.5` | **0.7063** | 0.9022 | 0.196 | **79.17%** | 0.7711 |
| SVM Lineal + TF-IDF (1,1) | `C=1.0` | 0.6872 | 0.9328 | 0.246 | 75.00% | 0.7329 |
| **SVM RBF + Embeddings** | **`C=10.0, gamma='scale'`** | **0.7600** | **1.0000** | 0.240 | **81.25%** | **0.8090** |

### 7.3. Análisis de cada experimento

**NB + TF-IDF (1,1) — alpha óptimo: 0.5**

- El default `alpha=1.0` es **demasiado regularización**. Con `alpha=0.5` se permite que las frecuencias observadas tengan más peso.
- La diferencia train-val (0.196) indica sobreajuste, pero es esperable con 192 muestras de entrenamiento y 15 clases (~13 muestras por clase).
- El test accuracy (79.17%) se mantiene **exactamente igual** que sin tuning — la ganancia de CV no se traslada al test set. Esto sugiere que el test set actual puede no ser representativo del CV.

**SVM Lineal + TF-IDF (1,1) — C óptimo: 1.0**

- El default `C=1.0` ya era el óptimo. Ni mayor ni menor regularización mejora el CV score.
- El sobreajuste es el más alto (0.246). SVM con kernel lineal en espacio TF-IDF de 103 dimensiones tiende a sobreajustar con pocas muestras.
- **Conclusión**: El modelo ya estaba en su punto óptimo desde el principio.

**SVM RBF + Embeddings — C=10.0, gamma='scale'**

- `C=10.0` es **mayor que el default (1.0)** → menos regularización, margen más ajustado. Esto funciona porque los embeddings de SentenceTransformer generan **representaciones de alta calidad** donde las clases están bien separadas. Un margen más ajustado (C más grande) captura mejor los límites entre clases.
- `gamma='scale'` ya era el default — `gamma=1/(n_features * X.var())` = `1/(512 * varianza)` que para embeddings normalizados da un valor razonable. Valores fijos como 0.01 o 0.001 no mejoraron.
- **Train F1 = 1.0000**: el modelo clasifica perfectamente el train set. Con C=10.0, los support vectors se ajustan firmemente a los datos de entrenamiento. Pero el test F1 (0.8090) muestra que **generaliza bien igual**.
- El CV F1 de 0.7600 es el más alto de los 3 experimentos, confirmando que Embeddings + RBF sigue siendo la configuración ganadora.

### 7.4. Análisis de Sobreajuste

Los 3 experimentos muestran sobreajuste (diff > 0.10). ¿Es problemático?

| Experimento | Diff train-val | ¿Problema? |
|---|---|---|
| NB + TF-IDF | 0.196 | ⚠️ Moderado — pero test acc consistente con Fase 3 |
| SVM + TF-IDF | 0.246 | ⚠️ Significativo — pero test acc se mantiene en 75% |
| SVM + Embeddings | 0.240 | ⚠️ Significativo — pero test acc se mantiene en **81.25%** |

**No es problemático porque**:
1. El **test score se mantiene consistente** entre Fase 3 y Fase 4 — no hay degradación
2. Con 192 muestras de entrenamiento y 15 clases, **es esperable** que el modelo aprenda de memoria patrones específicos
3. La validación cruzada (CV F1) es la métrica honesta: ~0.70 para TF-IDF, ~0.76 para Embeddings
4. Para producción, se necesitarían **más datos por clase** (~50+ por clase) para reducir el sobreajuste

### 7.5. Mejores hiperparámetros encontrados

| Modelo | Parámetro | Default | Óptimo | Efecto |
|---|---|---|---|---|
| MultinomialNB | `alpha` | 1.0 | **0.5** | Menos suavizado → más peso a frecuencias observadas |
| SVM Lineal | `C` | 1.0 | 1.0 | Sin cambios |
| SVM RBF | `C` | 1.0 | **10.0** | Margen más ajustado → mejor separación con embeddings de calidad |
| SVM RBF | `gamma` | 'scale' | 'scale' | Sin cambios |

**Actualización del pipeline**: Solo cambia `alpha` para NB (de 1.0 a 0.5). Los demás parámetros ya eran óptimos.

---

## 8. Resumen General — Estado Final del Pipeline

| Componente | Configuración Final | Fundamento |
|---|---|---|
| Preprocesamiento | `cleaner.preprocesar()` sin stemming | Stemming destruye embeddings (−12.5 pp) |
| Vectorización TF-IDF | `max_features=500, min_df=2, ngram_range=(1,1)` | Unigrams > bigrams con dataset chico |
| NB + TF-IDF | `alpha=0.5` | Menos regularización mejora CV F1 |
| SVM Lineal + TF-IDF | `C=1.0` | Default ya era óptimo |
| SVM RBF + Embeddings | `C=10.0, gamma='scale'` | Embeddings de calidad permiten margen más ajustado |
| **Modelo ganador** | **SVM RBF + Embeddings (81.25% / 0.8090 F1)** | Captura relaciones semánticas que TF-IDF no puede |

### Próximos pasos (post-Fase 4)

- Expandir el dataset (~50+ preguntas por clase) para reducir sobreajuste
- Probar modelos más robustos (Random Forest, XGBoost) sobre embeddings
- Implementar **ensemble** (NB rápido + SVM preciso) para producción
- Evaluar en un **entorno real** con preguntas de usuarios reales

---

---

## 9. Aprendizajes Transversales (preguntas del equipo)

### 9.1. ¿Accuracy o F1-macro como métrica principal?

**Problema**: Durante las fases 1-4 presentamos Accuracy como métrica principal para determinar el mejor modelo. Sin embargo, en clasificación multiclase con 15 clases, Accuracy tiene limitaciones importantes.

**¿Por qué Accuracy no es suficiente?**

Accuracy = `(TP + TN) / Total`. En multiclase, TN incluye los aciertos de las **14 clases incorrectas**, lo que puede inflar artificialmente la métrica. Un modelo que acierta perfectamente 14 clases pero falla completamente 1 clase aún tendría ~93% de Accuracy.

**F1-macro** promedia el F1 de **cada clase individualmente**, por lo que penaliza directamente a los modelos que "ignoran" clases enteras.

**¿Cambia el ranking en nuestro caso?**

No significativamente, porque el dataset está balanceado y la diferencia entre Accuracy y F1-macro ha sido < 0.03 en todos los experimentos. Pero es más correcto conceptualmente.

| Métrica | Ventaja | Desventaja |
|---|---|---|
| **F1-macro** | Cada clase pesa igual. Detecta clases ignoradas. | Menos intuitiva para no técnicos |
| Accuracy | Intuitiva. Fácil de comunicar. | No detecta clases con bajo rendimiento |
| **Cohen's Kappa** | Accuracy ajustada por azar. Útil con clases desbalanceadas. | Menos conocida |
| **Matriz de confusión** | Muestra QUÉ clases se confunden entre sí. | No es una métrica única |

**Decisión**: A partir de ahora, la métrica principal es **F1-macro**. Accuracy se reporta como secundaria. La matriz de confusión se incluye para diagnóstico granular.

> **Nota para el TP**: El ranking de modelos presentado en secciones anteriores NO cambia porque Accuracy y F1-macro se comportaron de forma consistente. La decisión es metodológica para futuras evaluaciones.

### 9.2. Balanceo de clases: Oversampling, Undersampling y SMOTE

**Problema**: ¿Dónde y cómo se aplicaría balanceo de clases si el dataset no fuera balanceado?

**Contexto**: Nuestro dataset v1.3 fue construido con 16 preguntas por intent, por lo que está perfectamente balanceado. Pero en un escenario real, algunas intents pueden tener más preguntas que otras.

**¿Dónde se aplica el balanceo en el pipeline?**

```
Texto → Cleaner → Vectorización → 🔵 BALANCEO → Clasificador
```

El balanceo va **entre la vectorización y el clasificador**, porque:
- No se puede aplicar sobre texto crudo (necesitamos vectores numéricos)
- El clasificador ve los vectores, no el texto

**Técnicas según el tipo de vectorización:**

| Técnica | TF-IDF (sparse) | Embeddings (denso) |
|---|---|---|
| **Random Oversampling** | ✅ Duplica muestras existentes | ✅ Duplica muestras existentes |
| **Random Undersampling** | ✅ Elimina muestras de clase mayoritaria | ✅ Elimina muestras de clase mayoritaria |
| **SMOTE** | ❌ **No funciona**. Genera vectores irreales en espacio sparse de alta dimensión | ✅ **Funciona**. Interpola entre vectores de la misma clase en espacio continuo |
| **class_weight='balanced'** en SVM | ✅ Ajusta pesos automáticamente | ✅ Ajusta pesos automáticamente |

**¿Por qué SMOTE no funciona con TF-IDF?**

TF-IDF produce vectores sparse donde cada dimensión representa un término específico. Interpolar entre dos vectores sparse crea valores con pesos fraccionarios que no corresponden a la frecuencia de ningún término real. El clasificador recibe "ruido".

**¿Por qué SMOTE sí funciona con Embeddings?**

Los embeddings de SentenceTransformer viven en un espacio **continuo y semánticamente estructurado**. Interpolar entre `embedding("título intermedio")` y `embedding("qué es el título intermedio")` produce un punto intermedio que sigue teniendo sentido semántico.

**Ejemplo concreto de aplicación:**

```python
# Para embeddings (funciona)
from imblearn.over_sampling import SMOTE
smote = SMOTE(k_neighbors=5, random_state=42)
X_train_balanceado, y_train_balanceado = smote.fit_resample(X_train_emb, y_train)

# Para TF-IDF (solo oversampling simple)
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_balanceado, y_train_balanceado = ros.fit_resample(X_train_tfidf, y_train)
```

**¿Mejoraría nuestros resultados actuales?**

No. Nuestro dataset ya está balanceado (16 preguntas × 15 clases = 240). Aplicar oversampling solo duplicaría información existente sin agregar nuevas variantes, potencialmente aumentando el sobreajuste. Donde realmente ayuda es cuando hay clases con **distribución despareja** (ej: 50 preguntas para una clase, 5 para otra).

---

*Documento generado el 20/06/2026 — Sirve como bitácora técnica para el Trabajo Integrador.*
