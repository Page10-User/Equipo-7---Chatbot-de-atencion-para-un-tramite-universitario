"""
util_stemming.py — Stemming en español SIN modificar cleaner.py.
Agrega stemming al pipeline de preprocesamiento como paso opcional.
"""

from nltk.stem.snowball import SnowballStemmer

# Stemmer para español
_STEMMER = SnowballStemmer("spanish")

# Prefijos que NO deben reducirse (palabras cortas o siglas)
_PALABRAS_CORTAS = {"siu", "dni", "pdf", "jpg", "rbf", "svm", "nb", "unne", "facena"}


def aplicar_stemming(tokens: list[str]) -> list[str]:
    """
    Aplica stemming a cada token, preservando palabras cortas y siglas.

    Parametros
    ----------
    tokens : list[str]
        Lista de tokens ya limpios (sin stopwords, sin puntuacion).

    Returns
    -------
    list[str]
        Tokens con stemming aplicado.
    """
    return [
        _STEMMER.stem(t) if t not in _PALABRAS_CORTAS and len(t) > 3 else t
        for t in tokens
    ]


def preprocesar_con_stemming(texto: str) -> str:
    """
    Preprocesa y aplica stemming: limpia, tokeniza, stem y une.

    Parametros
    ----------
    texto : str
        Texto crudo.

    Returns
    -------
    str
        Texto preprocesado con stemming, tokens separados por espacio.
    """
    from cleaner import tokenizar
    tokens = tokenizar(texto)
    tokens_stem = aplicar_stemming(tokens)
    return " ".join(tokens_stem)
