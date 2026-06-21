"""
clasificar.py — Inferencia: clasifica preguntas nuevas usando el modelo final.

Uso:
    python clasificar.py "texto de la pregunta"
    python clasificar.py                          (modo interactivo)
    python clasificar.py --modelo nb              (elige modelo: svm [default], nb, svm_tfidf)

Ejemplos:
    python clasificar.py "como saco el titulo intermedio"
    python clasificar.py "que documentos necesito"
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

try:
    import joblib
except ImportError:
    import pickle as joblib

# Para cargar modulos del proyecto
sys.path.insert(0, str(Path(__file__).parent))
from cleaner import preprocesar
from vectorizer import cargar_modelo_embeddings, vectorizar_embeddings

DIR_MODELOS = Path(__file__).parent / "modelos"
RUTA_METADATA = DIR_MODELOS / "metadata.json"


def cargar_metadata() -> dict:
    """Carga metadata de los modelos guardados."""
    if not RUTA_METADATA.exists():
        print(f"[ERROR] No se encuentra metadata.json en {DIR_MODELOS}")
        print("  Ejecuta primero: python modelo_final.py")
        sys.exit(1)
    with open(RUTA_METADATA, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_modelo(nombre: str, metadata: dict):
    """
    Carga un modelo guardado y su vectorizador asociado.

    Parametros
    ----------
    nombre : str
        'svm' (SVM+Embeddings, ganador),
        'nb'  (Naive Bayes + TF-IDF),
        'svm_tfidf' (SVM Lineal + TF-IDF)

    Returns
    -------
    tuple: (clf, vectorizador, info)
        clf: clasificador cargado
        vectorizador: objeto para vectorizar (TfidfVectorizer o SentenceTransformer)
        info: dict con metadatos del modelo
    """
    modelos = metadata["modelos"]
    if nombre == "svm":
        key = "svm_embeddings"
    elif nombre == "nb":
        key = "nb_tfidf"
    elif nombre == "svm_tfidf":
        key = "svm_tfidf"
    else:
        raise ValueError(f"Modelo desconocido: {nombre}. Opciones: svm, nb, svm_tfidf")

    info = modelos[key]
    ruta_clf = DIR_MODELOS / info["archivo"]

    if not ruta_clf.exists():
        print(f"[ERROR] No se encuentra: {ruta_clf}")
        print("  Ejecuta primero: python modelo_final.py")
        sys.exit(1)

    clf = joblib.load(ruta_clf)
    print(f"[OK] Modelo cargado: {info['archivo']}")

    # Cargar vectorizador segun el tipo de modelo
    if nombre == "svm":
        # Embeddings: cargar SentenceTransformer por nombre
        emb_name = info.get("embedding_model_name", "distiluse-base-multilingual-cased-v2")
        print(f"[OK] Cargando modelo de embeddings: {emb_name}...")
        vectorizador = cargar_modelo_embeddings(emb_name)
    else:
        # TF-IDF: cargar TfidfVectorizer guardado
        ruta_vec = DIR_MODELOS / info["vectorizador"]
        vectorizador = joblib.load(ruta_vec)
        print(f"[OK] Vectorizador TF-IDF cargado: {info['vectorizador']}")

    return clf, vectorizador, info


def clasificar(texto: str, clf, vectorizador, info: dict) -> dict:
    """
    Clasifica una pregunta y devuelve el intent detectado.

    Parameters
    ----------
    texto : str
        Pregunta del usuario en lenguaje natural.
    clf : estimator
        Clasificador cargado.
    vectorizador : TfidfVectorizer o SentenceTransformer
        Vectorizador correspondiente.
    info : dict
        Metadata del modelo (para saber si es embeddings o TF-IDF).

    Returns
    -------
    dict
        {
            'texto': str (original),
            'intent': str (clase predicha),
            'confianza': float (pseudo-probabilidad),
            'modelo': str (nombre del modelo),
        }
    """
    # Preprocesar
    texto_limpio = preprocesar(texto)
    print(f"  Texto limpio: '{texto_limpio}'")

    # Vectorizar segun el tipo
    es_embeddings = "embedding" in info.get("archivo", "") or info.get("vectorizador") is None

    if es_embeddings:
        X = vectorizar_embeddings([texto_limpio], vectorizador)
    else:
        X = vectorizador.transform([texto_limpio])

    # Predecir
    intent = clf.predict(X)[0]

    # Obtener confianza
    if hasattr(clf, "predict_proba"):
        # NB tiene predict_proba
        probas = clf.predict_proba(X)[0]
        confianza = float(np.max(probas))
    elif hasattr(clf, "decision_function"):
        # SVM: usar distancia al hiperplano como pseudo-confianza
        distancias = clf.decision_function(X)[0]
        # Normalizar a [0, 1] con sigmoide
        confianza = float(1.0 / (1.0 + np.exp(-np.max(distancias))))
    else:
        confianza = 1.0

    # Nombre del modelo
    nombre_modelo = info.get("archivo", "desconocido").replace(".joblib", "")

    return {
        "texto": texto,
        "intent": intent,
        "confianza": round(confianza, 4),
        "modelo": nombre_modelo,
    }


def mostrar_resultado(resultado: dict, ranking: bool = False):
    """Muestra el resultado formateado."""
    print("\n" + "=" * 50)
    print(f"  Pregunta:    {resultado['texto']}")
    print(f"  Intent:      {resultado['intent']}")
    print(f"  Confianza:   {resultado['confianza']:.2%}")
    print(f"  Modelo:      {resultado['modelo']}")
    print("=" * 50)


def modo_interactivo(clf, vectorizador, info):
    """Modo interactivo: clasifica preguntas una tras otra."""
    print("\n[OK] Modo interactivo. Escribi 'salir' para terminar.\n")
    while True:
        try:
            pregunta = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[OK] Salir.")
            break

        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            print("[OK] Salir.")
            break

        resultado = clasificar(pregunta, clf, vectorizador, info)
        mostrar_resultado(resultado)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificador de intenciones")
    parser.add_argument("pregunta", nargs="?", help="Pregunta a clasificar")
    parser.add_argument(
        "--modelo", "-m",
        choices=["svm", "nb", "svm_tfidf"],
        default="svm",
        help="Modelo a usar: svm (default, embeddings), nb (TF-IDF rapido), svm_tfidf",
    )
    args = parser.parse_args()

    # Cargar metadata y modelo
    metadata = cargar_metadata()
    clf, vectorizador, info = cargar_modelo(args.modelo, metadata)

    if args.pregunta:
        # Modo comando
        resultado = clasificar(args.pregunta, clf, vectorizador, info)
        mostrar_resultado(resultado)
    else:
        # Modo interactivo
        modo_interactivo(clf, vectorizador, info)
