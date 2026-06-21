"""
prueba_resumen.py — Compila los mejores resultados.
Ejecuta la configuracion ganadora de cada prueba con semilla fija
y genera una tabla final comparable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import cargar_y_aplanar
from cleaner import preprocesar
from vectorizer import (
    crear_vectorizador_tfidf,
    vectorizar_tfidf,
    cargar_modelo_embeddings,
)

RUTA_DATASET = Path(__file__).parent.parent.parent / "02-Dataset" / "02-Dataset.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def probar_configuracion(nombre, X_train, X_test, y_train, y_test, vectorizador=None, modelo_emb=None):
    """
    Prueba una configuracion unica y devuelve resultados.
    """
    resultados = {}

    if vectorizador:
        # TF-IDF
        Xtr = vectorizador.fit_transform(X_train)
        Xte = vectorizador.transform(X_test)
        n_term = Xtr.shape[1]

        # NB
        nb = MultinomialNB(alpha=1.0)
        nb.fit(Xtr, y_train)
        yp = nb.predict(Xte)
        acc = accuracy_score(y_test, yp)
        r = classification_report(y_test, yp, zero_division=0, output_dict=True)
        resultados["NB acc"] = acc
        resultados["NB f1"] = r["macro avg"]["f1-score"]

        # SVM lineal
        svm = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
        svm.fit(Xtr, y_train)
        yp = svm.predict(Xte)
        acc = accuracy_score(y_test, yp)
        r = classification_report(y_test, yp, zero_division=0, output_dict=True)
        resultados["SVM acc"] = acc
        resultados["SVM f1"] = r["macro avg"]["f1-score"]

        resultados["terminos"] = n_term
        del Xtr, Xte, nb, svm
        gc.collect()

    if modelo_emb:
        # Embeddings
        Xtr_emb = modelo_emb.encode(X_train, show_progress_bar=True)
        Xte_emb = modelo_emb.encode(X_test, show_progress_bar=True)

        svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
        svm_rbf.fit(Xtr_emb, y_train)
        yp = svm_rbf.predict(Xte_emb)
        acc = accuracy_score(y_test, yp)
        r = classification_report(y_test, yp, zero_division=0, output_dict=True)
        resultados["Emb acc"] = acc
        resultados["Emb f1"] = r["macro avg"]["f1-score"]

    return resultados


def ejecutar():
    df = cargar_y_aplanar(str(RUTA_DATASET))
    textos = [preprocesar(t) for t in df["texto"]]
    etiquetas = df["intent"].to_numpy().copy()

    X_train, X_test, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=etiquetas
    )

    # Configuraciones
    configs = {
        "Baseline (1,2)": crear_vectorizador_tfidf(max_features=500, min_df=2, ngram_range=(1, 2)),
        "Unigrams (1,1)": crear_vectorizador_tfidf(max_features=500, min_df=2, ngram_range=(1, 1)),
    }

    filas = []
    for nombre, vec in configs.items():
        r = probar_configuracion(nombre, X_train, X_test, y_train, y_test, vectorizador=vec)
        filas.append({
            "Config": nombre,
            "Terminos": r["terminos"],
            "NB acc": f"{r['NB acc']:.2%}",
            "NB f1": round(r["NB f1"], 4),
            "SVM acc": f"{r['SVM acc']:.2%}",
            "SVM f1": round(r["SVM f1"], 4),
        })

    # Embeddings
    modelo = cargar_modelo_embeddings("distiluse-base-multilingual-cased-v2")
    r_emb = probar_configuracion(
        "Embeddings", X_train, X_test, y_train, y_test, modelo_emb=modelo
    )
    filas.append({
        "Config": "SVM+Emb (RBF)",
        "Terminos": 512,
        "NB acc": "-",
        "NB f1": "-",
        "SVM acc": f"{r_emb['Emb acc']:.2%}",
        "SVM f1": round(r_emb["Emb f1"], 4),
    })

    tabla = pd.DataFrame(filas)
    print("\n" + "=" * 65)
    print("  TABLA FINAL: Mejores configuraciones")
    print("=" * 65)
    print(f"\n{tabla.to_string(index=False)}")


if __name__ == "__main__":
    ejecutar()
