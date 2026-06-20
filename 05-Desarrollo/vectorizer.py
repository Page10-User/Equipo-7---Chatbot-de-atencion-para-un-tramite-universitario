"""
vectorizer.py — Vectorización de texto: TF-IDF y Embeddings.

Funciones:
    - crear_vectorizador_tfidf(): configura y entrena un TfidfVectorizer
    - vectorizar_tfidf(textos, entrenar=True): vectoriza con TF-IDF
    - crear_modelo_embeddings(): carga SentenceTransformer
    - vectorizar_embeddings(textos): vectoriza con embeddings
    - obtener_dimensiones(matriz): info sobre dimensionalidad

Experimentos:
    Para Exp. 1 y 2 → TF-IDF (con cleaner pre-aplicado)
    Para Exp. 3 → SentenceTransformer multilingüe
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Parámetros del vectorizador TF-IDF
PARAMS_TFIDF = {
    "max_features": 500,       # Limitar vocabulario a las 500 palabras más frecuentes
    "min_df": 2,               # Ignorar términos que aparecen en menos de 2 documentos
    "max_df": 0.85,            # Ignorar términos que aparecen en más del 85% de documentos
    "ngram_range": (1, 2),     # Usar unigramas y bigramas (ej: "libre deuda")
    "sublinear_tf": True,      # Aplicar escala logarítmica a TF (1 + log(TF))
    "strip_accents": "unicode", # Normalizar acentos (doble seguridad)
    "lowercase": True,         # Minúsculas (doble seguridad)
}


def crear_vectorizador_tfidf(**kwargs) -> TfidfVectorizer:
    """
    Crea y configura un TfidfVectorizer con los parámetros del pipeline.

    Parámetros
    ----------
    **kwargs : dict
        Parámetros adicionales para sobreescribir los defaults.

    Returns
    -------
    TfidfVectorizer
        Vectorizador configurado (sin entrenar).
    """
    params = {**PARAMS_TFIDF, **kwargs}
    vectorizador = TfidfVectorizer(**params)

    print(f"[OK] TfidfVectorizer configurado:")
    print(f"      max_features={params['max_features']}, "
          f"min_df={params['min_df']}, "
          f"ngram_range={params['ngram_range']}")

    return vectorizador


def vectorizar_tfidf(
    textos: list[str], vectorizador: TfidfVectorizer = None, entrenar: bool = True
) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Vectoriza textos usando TF-IDF.

    Dos modos:
        1. entrenar=True:  entrena el vectorizador y transforma (fit_transform)
        2. entrenar=False: transforma con un vectorizador ya entrenado

    Parámetros
    ----------
    textos : list[str]
        Lista de textos preprocesados (o crudos si el vectorizador no entrena).
    vectorizador : TfidfVectorizer, opcional
        Vectorizador a usar. Si es None, crea uno nuevo.
    entrenar : bool
        Si True, entrena el vectorizador. Si False, solo transforma.

    Returns
    -------
    tuple[np.ndarray, TfidfVectorizer]
        (matriz TF-IDF, vectorizador entrenado)
    """
    if vectorizador is None:
        vectorizador = crear_vectorizador_tfidf()

    if entrenar:
        matriz = vectorizador.fit_transform(textos)
        print(f"[OK] TF-IDF entrenado y transformado.")
    else:
        matriz = vectorizador.transform(textos)
        print(f"[OK] TF-IDF transformado (vectorizador existente).")

    print(f"      Matriz: {matriz.shape[0]} documentos, "
          f"{matriz.shape[1]} términos")
    print(f"      Densidad: {matriz.nnz / (matriz.shape[0] * matriz.shape[1]):.4f} "
          f"({matriz.nnz} valores no-cero)")

    return matriz, vectorizador


def cargar_modelo_embeddings(nombre_modelo: str = None):
    """
    Carga un modelo de SentenceTransformer para generar embeddings.

    Parámetros
    ----------
    nombre_modelo : str, opcional
        Nombre del modelo en HuggingFace.
        Default: 'distiluse-base-multilingual-cased-v2' (384 dimensiones)
        Alternativa: 'paraphrase-multilingual-MiniLM-L12-v2'

    Returns
    -------
    SentenceTransformer
        Modelo cargado.
    """
    if nombre_modelo is None:
        nombre_modelo = "distiluse-base-multilingual-cased-v2"

    print(f"[...] Cargando modelo de embeddings: {nombre_modelo}...")
    print("      (puede tardar la primera vez que descarga el modelo)")

    # Import tardío: sentence-transformers no siempre está instalado
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers no está instalado. "
            "Ejecutá: pip install sentence-transformers"
        )

    modelo = SentenceTransformer(nombre_modelo)
    print(f"[OK] Modelo cargado: {nombre_modelo}")
    print(f"      Dimensiones: {modelo.get_embedding_dimension()}")

    return modelo


def vectorizar_embeddings(
    textos: list[str], modelo=None
) -> np.ndarray:
    """
    Genera embeddings densos para cada texto usando SentenceTransformer.

    Parámetros
    ----------
    textos : list[str]
        Lista de textos (crudos o preprocesados).
    modelo : SentenceTransformer, opcional
        Modelo a usar. Si es None, carga el default.

    Returns
    -------
    np.ndarray
        Matriz de embeddings: (n_documentos, n_dimensiones)
    """
    if modelo is None:
        modelo = cargar_modelo_embeddings()

    print(f"[...] Generando embeddings para {len(textos)} textos...")
    embeddings = modelo.encode(textos, show_progress_bar=True)
    print(f"[OK] Embeddings generados: {embeddings.shape}")

    return embeddings


def obtener_dimensiones(matriz) -> dict:
    """
    Devuelve información sobre dimensionalidad de una matriz.

    Parámetros
    ----------
    matriz : np.ndarray o scipy.sparse matrix
        Matriz de features.

    Returns
    -------
    dict
        documentosc, dimensiones, tipo, densidad
    """
    info = {
        "documentos": matriz.shape[0],
        "dimensiones": matriz.shape[1],
        "tipo": type(matriz).__name__,
    }

    # Intentar calcular densidad (si es sparse)
    if hasattr(matriz, "nnz"):
        info["densidad"] = matriz.nnz / (matriz.shape[0] * matriz.shape[1])
    else:
        info["densidad"] = 1.0  # densa

    return info


# ============================================================
# Ejecución directa (para pruebas)
# ============================================================
if __name__ == "__main__":
    from dataset import cargar_y_aplanar
    from cleaner import preprocesar
    from pathlib import Path

    # 1. Cargar dataset
    ruta_json = Path(__file__).parent.parent / "02-Dataset" / "02-Dataset.json"
    df = cargar_y_aplanar(str(ruta_json))

    # 2. Preprocesar
    print("\n--- Preprocesando textos ---")
    textos_limpios = [preprocesar(t) for t in df["texto"]]

    # 3. Probar TF-IDF
    print("\n--- TF-IDF ---")
    matriz_tfidf, vectorizador = vectorizar_tfidf(textos_limpios)
    print(f"\nPrimera fila (10 valores): {matriz_tfidf[0].toarray()[0][:10]}")

    # 4. Probar embeddings (solo si está disponible)
    print("\n--- Embeddings (breve prueba) ---")
    try:
        ejemplo = ["como saco libre deuda", "donde pago arancel titulo"]
        emb = vectorizar_embeddings(ejemplo)
        print(f"Embedding shape: {emb.shape}")
        print(f"Similitud coseno entre los dos textos: "
              f"{np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])):.4f}")
    except (ImportError, Exception) as e:
        print(f"[WARN] Embeddings no disponible: {e}")
