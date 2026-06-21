"""
pipeline.py — Orquestador del pipeline completo de clasificación de intenciones.

Ejecuta:
    1. Carga y aplanado del dataset
    2. Preprocesamiento (limpieza + tokenización + stopwords)
    3. Vectorización TF-IDF
    4. Entrenamiento y evaluación de clasificadores
    5. Reporte de métricas

Experimentos:
    - Exp. 1: Naive Bayes + TF-IDF (alpha=0.5 optimizado)
    - Exp. 2: SVM Lineal + TF-IDF (C=1.0)
    - Exp. 3: SVM RBF + Embeddings (C=10.0, gamma='scale')
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dataset import cargar_y_aplanar
from cleaner import preprocesar
from vectorizer import (
    crear_vectorizador_tfidf,
    vectorizar_tfidf,
    cargar_modelo_embeddings,
    vectorizar_embeddings,
)

# ============================================================
# Configuración
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80% entrenamiento, 20% prueba

RUTA_DATASET = (
    Path(__file__).parent.parent / "02-Dataset" / "02-Dataset.json"
)

# ============================================================
# Pipeline principal
# ============================================================
def ejecutar_pipeline_tfidf(
    df: pd.DataFrame,
    clasificador,
    nombre_clasificador: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Ejecuta el pipeline completo para un clasificador dado usando TF-IDF.

    1. Preprocesa los textos
    2. Separa en train/test (con stratify para平衡izar clases)
    3. Vectoriza con TF-IDF (fit en train, transform en test)
    4. Entrena y evalúa

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'texto' e 'intent'.
    clasificador : estimator
        Instancia del clasificador de scikit-learn.
    nombre_clasificador : str
        Nombre legible para los reportes.
    test_size : float
        Proporción del dataset para prueba (default: 0.2).
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    dict
        {
            'nombre': str,
            'accuracy': float,
            'reporte': str (classification_report),
            'matriz': np.ndarray (confusion_matrix),
            'y_true': np.ndarray,
            'y_pred': np.ndarray,
        }
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENTO: {nombre_clasificador} + TF-IDF")
    print(f"{'='*60}")

    # 1. Preprocesar textos
    print("\n[1/4] Preprocesando textos...")
    textos_limpios = [preprocesar(t) for t in df["texto"]]
    etiquetas = df["intent"].to_numpy().copy()
    print(f"      {len(textos_limpios)} textos procesados.")

    # 2. Separar train/test (con stratify para mantener proporción de clases)
    print("\n[2/4] Separando train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        textos_limpios,
        etiquetas,
        test_size=test_size,
        random_state=random_state,
        stratify=etiquetas,  # Mantiene misma proporción de clases
    )
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Vectorizar con TF-IDF (fit SOLO en train)
    print("\n[3/4] Vectorizando con TF-IDF...")
    X_train_vec, vectorizador = vectorizar_tfidf(X_train, entrenar=True)
    X_test_vec, _ = vectorizar_tfidf(X_test, vectorizador=vectorizador, entrenar=False)

    # 4. Entrenar y evaluar
    print(f"\n[4/4] Entrenando {nombre_clasificador}...")
    clasificador.fit(X_train_vec, y_train)
    y_pred = clasificador.predict(X_test_vec)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)

    print(f"\n>> Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n>> Reporte por clase:\n{reporte}")

    return {
        "nombre": nombre_clasificador,
        "accuracy": accuracy,
        "reporte": reporte,
        "matriz": matriz,
        "y_true": y_test,
        "y_pred": y_pred,
    }


def experimento_1_naive_bayes(df: pd.DataFrame) -> dict:
    """Experimento 1: Naive Bayes Multinomial + TF-IDF (alpha=0.5 optimo)."""
    clf = MultinomialNB(alpha=0.5)  # optimizado via GridSearchCV (Fase 4)
    return ejecutar_pipeline_tfidf(df, clf, "Naive Bayes Multinomial")


def experimento_2_svm(df: pd.DataFrame) -> dict:
    """Experimento 2: SVM Lineal + TF-IDF (C=1.0 optimo)."""
    clf = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
    return ejecutar_pipeline_tfidf(df, clf, "SVM Lineal")


def experimento_3_svm_embeddings(df: pd.DataFrame) -> dict:
    """Experimento 3: SVM RBF + Embeddings (C=10.0, gamma='scale' optimos)."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENTO: SVM RBF + Embeddings")
    print(f"{'='*60}")

    # 1. Preprocesar
    print("\n[1/4] Preprocesando textos...")
    textos_limpios = [preprocesar(t) for t in df["texto"]]
    etiquetas = df["intent"].to_numpy().copy()

    # 2. Separar train/test
    print("\n[2/4] Separando train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        textos_limpios, etiquetas,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=etiquetas,
    )

    # 3. Generar embeddings
    print("\n[3/4] Generando embeddings...")
    modelo_emb = cargar_modelo_embeddings()
    X_train_vec = vectorizar_embeddings(list(X_train), modelo_emb)
    X_test_vec = vectorizar_embeddings(list(X_test), modelo_emb)

    # 4. Entrenar SVM RBF con hiperparametros optimos
    print(f"\n[4/4] Entrenando SVM RBF (C=10.0, gamma='scale')...")
    clf = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=RANDOM_STATE)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    # Metricas
    accuracy = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)

    print(f"\n>> Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f">> F1 macro: {classification_report(y_test, y_pred, zero_division=0, output_dict=True)['macro avg']['f1-score']:.4f}")
    print(f"\n>> Reporte por clase:\n{reporte}")

    return {
        "nombre": "SVM RBF + Embeddings",
        "accuracy": accuracy,
        "reporte": reporte,
        "matriz": matriz,
        "y_true": y_test,
        "y_pred": y_pred,
    }


def crear_tabla_comparativa(resultados: list[dict]) -> pd.DataFrame:
    """
    Crea una tabla comparativa de resultados de experimentos.

    Parámetros
    ----------
    resultados : list[dict]
        Lista de diccionarios con resultados de cada experimento.

    Returns
    -------
    pd.DataFrame
        Tabla con Accuracy, Precision, Recall, F1 macro por experimento.
    """
    filas = []
    for r in resultados:
        # Extraer métricas macro del classification_report
        # Buscamos la última línea que contiene promedios
        lineas = r["reporte"].strip().split("\n")
        # La penúltima línea debería tener macro avg
        for linea in lineas:
            if "macro avg" in linea:
                partes = linea.split()
                precision = float(partes[2])
                recall = float(partes[3])
                f1 = float(partes[4])
                break
        else:
            precision = recall = f1 = 0.0

        filas.append({
            "Experimento": r["nombre"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision (macro)": round(precision, 4),
            "Recall (macro)": round(recall, 4),
            "F1 (macro)": round(f1, 4),
        })

    return pd.DataFrame(filas)


# ============================================================
# Ejecución principal
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  PIPELINE DE CLASIFICACIÓN DE INTENCIONES")
    print("  Dataset: Trámite Título Intermedio (FACENA-UNNE)")
    print("=" * 60)

    # 1. Cargar dataset
    print("\n[Paso 0] Cargando dataset...")
    df = cargar_y_aplanar(str(RUTA_DATASET))

    # 2. Experimentos con TF-IDF
    resultado_nb = experimento_1_naive_bayes(df)
    resultado_svm = experimento_2_svm(df)

    # 3. Experimento con Embeddings
    import gc
    gc.collect()  # liberar memoria antes de cargar el modelo de embeddings
    resultado_emb = experimento_3_svm_embeddings(df)

    # 4. Tabla comparativa
    print("\n" + "=" * 60)
    print("  TABLA COMPARATIVA DE RESULTADOS (FINAL)")
    print("=" * 60)
    tabla = crear_tabla_comparativa([resultado_nb, resultado_svm, resultado_emb])
    print(f"\n{tabla.to_string(index=False)}")

    print("\n[OK] Pipeline completado.")
    print(f"   Naive Bayes Accuracy:       {resultado_nb['accuracy']:.4f}")
    print(f"   SVM Lineal Accuracy:         {resultado_svm['accuracy']:.4f}")
    print(f"   SVM RBF + Embeddings Accuracy: {resultado_emb['accuracy']:.4f}")
