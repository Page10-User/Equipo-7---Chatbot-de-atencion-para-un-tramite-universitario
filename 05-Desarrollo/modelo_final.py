"""
modelo_final.py — Entrena y guarda los modelos finales con hiperparámetros óptimos.

Entrena 3 modelos sobre el dataset COMPLETO y los guarda en modelos/:
    1. NB + TF-IDF (alpha=0.5)              → rapido, liviano
    2. SVM Lineal + TF-IDF (C=1.0)          → alternativo
    3. SVM RBF + Embeddings (C=10.0)        ★ GANADOR — mejor accuracy

Uso: python modelo_final.py
Salida: modelos/*.joblib + modelos/metadata.json
"""

import gc
import json
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Para guardar/levantar modelos
try:
    import joblib
except ImportError:
    import pickle as joblib

from dataset import cargar_y_aplanar
from cleaner import preprocesar
from vectorizer import (
    crear_vectorizador_tfidf,
    vectorizar_tfidf,
    cargar_modelo_embeddings,
    vectorizar_embeddings,
)

RUTA_DATASET = Path(__file__).parent.parent / "02-Dataset" / "02-Dataset.json"
DIR_MODELOS = Path(__file__).parent / "modelos"
RANDOM_STATE = 42
NOMBRE_EMBEDDINGS = "distiluse-base-multilingual-cased-v2"


def entrenar_nb_tfidf(textos, etiquetas):
    """Entrena NB + TF-IDF y devuelve (modelo, vectorizador, metricas)."""
    print("\n>> Entrenando Naive Bayes + TF-IDF (alpha=0.5)...")
    X_vec, vec = vectorizar_tfidf(textos, entrenar=True)
    clf = MultinomialNB(alpha=0.5)
    clf.fit(X_vec, etiquetas)
    # Score sobre el mismo dataset (solo para referencia)
    acc = accuracy_score(etiquetas, clf.predict(X_vec))
    print(f"   Train accuracy: {acc:.4f}")
    return {"clf": clf, "vectorizador": vec, "nombre": "NB + TF-IDF (alpha=0.5)"}


def entrenar_svm_tfidf(textos, etiquetas):
    """Entrena SVM Lineal + TF-IDF y devuelve (modelo, vectorizador, metricas)."""
    print("\n>> Entrenando SVM Lineal + TF-IDF (C=1.0)...")
    X_vec, vec = vectorizar_tfidf(textos, entrenar=True)
    clf = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
    clf.fit(X_vec, etiquetas)
    acc = accuracy_score(etiquetas, clf.predict(X_vec))
    print(f"   Train accuracy: {acc:.4f}")
    return {"clf": clf, "vectorizador": vec, "nombre": "SVM Lineal + TF-IDF (C=1.0)"}


def entrenar_svm_embeddings(textos, etiquetas):
    """Entrena SVM RBF + Embeddings y devuelve (modelo, None, metricas)."""
    print(f"\n>> Entrenando SVM RBF + Embeddings (C=10.0)...")
    print(f"   Cargando modelo de embeddings: {NOMBRE_EMBEDDINGS}...")
    modelo_emb = cargar_modelo_embeddings(NOMBRE_EMBEDDINGS)
    print(f"   Generando embeddings para {len(textos)} textos...")
    X_emb = vectorizar_embeddings(textos, modelo_emb)
    clf = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=RANDOM_STATE)
    clf.fit(X_emb, etiquetas)
    acc = accuracy_score(etiquetas, clf.predict(X_emb))
    print(f"   Train accuracy: {acc:.4f}")
    return {"clf": clf, "vectorizador": modelo_emb, "nombre": "SVM RBF + Embeddings (C=10.0)"}


def guardar_modelo(objeto, nombre_archivo: str):
    """Guarda un modelo .joblib en el directorio modelos/."""
    ruta = DIR_MODELOS / nombre_archivo
    joblib.dump(objeto, ruta)
    print(f"   [OK] Guardado: {ruta.name}")
    return ruta


if __name__ == "__main__":
    print("=" * 60)
    print("  ENTRENAMIENTO DE MODELOS FINALES")
    print("  Dataset completo para produccion")
    print("=" * 60)

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)

    # Cargar y preprocesar TODO el dataset
    print("\n[1/3] Cargando y preprocesando dataset...")
    df = cargar_y_aplanar(str(RUTA_DATASET))
    textos = [preprocesar(t) for t in df["texto"]]
    etiquetas = df["intent"].to_numpy().copy()
    clases = list(df["intent"].unique())
    print(f"   {len(textos)} textos, {len(clases)} clases")

    metadata = {
        "dataset": "v1.3",
        "total_preguntas": len(textos),
        "clases": clases,
        "modelos": {},
    }

    # === 1. NB + TF-IDF ===
    print("\n[2/3] Entrenando modelos...")
    resultado = entrenar_nb_tfidf(textos, etiquetas)
    guardar_modelo(resultado["clf"], "nb_tfidf.joblib")
    guardar_modelo(resultado["vectorizador"], "vectorizador_tfidf.joblib")
    metadata["modelos"]["nb_tfidf"] = {
        "archivo": "nb_tfidf.joblib",
        "vectorizador": "vectorizador_tfidf.joblib",
        "params": {"alpha": 0.5},
    }

    gc.collect()

    # === 2. SVM + TF-IDF ===
    resultado = entrenar_svm_tfidf(textos, etiquetas)
    guardar_modelo(resultado["clf"], "svm_tfidf.joblib")
    # Reusa el mismo vectorizador TF-IDF (ya guardado)
    metadata["modelos"]["svm_tfidf"] = {
        "archivo": "svm_tfidf.joblib",
        "vectorizador": "vectorizador_tfidf.joblib",
        "params": {"kernel": "linear", "C": 1.0},
    }

    gc.collect()

    # === 3. SVM + Embeddings (GANADOR) ===
    resultado = entrenar_svm_embeddings(textos, etiquetas)
    guardar_modelo(resultado["clf"], "svm_embeddings.joblib")
    # NOTA: No guardamos el SentenceTransformer con joblib (no serializable).
    # Se guarda el nombre del modelo para recargarlo en inferencia.
    metadata["modelos"]["svm_embeddings"] = {
        "archivo": "svm_embeddings.joblib",
        "vectorizador": None,  # se carga por nombre en inferencia
        "params": {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        "embedding_model_name": NOMBRE_EMBEDDINGS,
        "es_ganador": True,
    }

    # === Guardar metadata ===
    ruta_meta = DIR_MODELOS / "metadata.json"
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n[3/3] Metadata guardada: {ruta_meta}")

    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"\n  Modelos guardados en: {DIR_MODELOS}")
    print(f"  Archivos:")
    for f in sorted(DIR_MODELOS.glob("*")):
        size = f.stat().st_size / 1024 / 1024
        print(f"    - {f.name} ({size:.1f} MB)")
    print(f"\n  [GANADOR] SVM RBF + Embeddings (C=10.0, gamma='scale')")
    print(f"  ★ Usar clasificar.py para inferencia")
