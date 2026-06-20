"""
cleaner.py — Limpieza, normalización y tokenización de texto.

Funciones:
    - limpiar_texto(texto): limpieza completa (minúsculas, tildes, puntuación, etc.)
    - tokenizar(texto): tokeniza y filtra stopwords en español
    - preprocesar(texto): pipeline completo (limpiar + tokenizar)
    - preprocesar_dataframe(df): aplica preprocesar a toda una columna

Dependencias:
    nltk: tokenización y stopwords en español
    re: expresiones regulares para limpieza
    unicodedata: normalización de tildes
"""

import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ============================================================
# Configuración de NLTK (descarga única de recursos)
# ============================================================
def _descargar_recursos_nltk():
    """Descarga los recursos de NLTK si no están disponibles."""
    recursos = [
        ("tokenizers/punkt_tab", "tokenizers/punkt_tab"),
        ("corpora/stopwords", "corpora/stopwords"),
    ]
    for recurso_id, _ in recursos:
        try:
            nltk.data.find(recurso_id)
        except LookupError:
            nltk.download(recurso_id.rsplit("/", 1)[-1], quiet=True)


_descargar_recursos_nltk()

# Cargar stopwords en español
STOPWORDS_ES = set(stopwords.words("spanish"))

# Palabras adicionales que podemos sacar porque no aportan al dominio
STOPWORDS_ADICIONALES = {
    "hacer", "puedo", "puede", "tiene", "tengo", "necesito",
    "quiero", "saber", "decir", "ser", "estar", "haber",
}
STOPWORDS_ES.update(STOPWORDS_ADICIONALES)


# ============================================================
# Limpieza y normalización
# ============================================================
def _quitar_tildes(texto: str) -> str:
    """
    Normaliza caracteres acentuados a su versión sin tilde.

    Ejemplo: "título" → "titulo", "cómo" → "como"
    """
    # NFKD descompone caracteres (ej: é → e +  ́)
    texto_normalizado = unicodedata.normalize("NFKD", texto)
    # Eliminamos los caracteres de acentuación (categoría "Mn")
    texto_sin_tildes = "".join(
        c for c in texto_normalizado if unicodedata.category(c) != "Mn"
    )
    return texto_sin_tildes


def _quitar_puntuacion(texto: str) -> str:
    """
    Elimina signos de puntuación y símbolos.

    Elimina: ¿ ? ¡ ! , . ; : ( ) [ ] { } " ' « » · — – …
    Conserva: letras, números y espacios.
    """
    return re.sub(r"[^\w\s]", " ", texto)


def _quitar_urls(texto: str) -> str:
    """Elimina URLs del texto."""
    patron_url = r"https?://\S+|www\.\S+"
    return re.sub(patron_url, "", texto)


def _quitar_emails(texto: str) -> str:
    """Elimina direcciones de correo electrónico."""
    patron_email = r"\S+@\S+"
    return re.sub(patron_email, "", texto)


def _quitar_numeros(texto: str) -> str:
    """Elimina dígitos numéricos."""
    return re.sub(r"\d+", "", texto)


def _normalizar_espacios(texto: str) -> str:
    """
    Normaliza espacios: elimina múltiples espacios y espacios al inicio/final.
    """
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def limpiar_texto(texto: str, quitar_numeros: bool = False) -> str:
    """
    Aplica todas las operaciones de limpieza y normalización.

    Pipeline completo:
        1. Minúsculas
        2. URLs → vacío
        3. Emails → vacío
        4. Quitar tildes
        5. Quitar puntuación
        6. Quitar números (opcional)
        7. Normalizar espacios

    Parámetros
    ----------
    texto : str
        Texto a limpiar.
    quitar_numeros : bool
        Si True, elimina también dígitos (default: False).

    Returns
    -------
    str
        Texto limpio y normalizado.
    """
    texto = texto.lower()
    texto = _quitar_urls(texto)
    texto = _quitar_emails(texto)
    texto = _quitar_tildes(texto)
    texto = _quitar_puntuacion(texto)
    if quitar_numeros:
        texto = _quitar_numeros(texto)
    texto = _normalizar_espacios(texto)

    return texto


# ============================================================
# Tokenización y stopwords
# ============================================================
def tokenizar(texto: str) -> list[str]:
    """
    Tokeniza el texto y filtra stopwords.

    1. Limpia el texto (sin números)
    2. Tokeniza con NLTK (word_tokenize)
    3. Filtra tokens vacíos y stopwords

    Parámetros
    ----------
    texto : str
        Texto pre-limpio o crudo.

    Returns
    -------
    list[str]
        Lista de tokens significativos.
    """
    # Aseguramos limpieza antes de tokenizar
    texto_limpio = limpiar_texto(texto, quitar_numeros=True)

    # Tokenización con NLTK
    tokens = word_tokenize(texto_limpio, language="spanish")

    # Filtro: sacamos tokens de 1 carácter, vacíos y stopwords
    tokens_filtrados = [
        t
        for t in tokens
        if len(t) > 1 and t not in STOPWORDS_ES
    ]

    return tokens_filtrados


def preprocesar(texto: str) -> str:
    """
    Pipeline completo: limpia y tokeniza, devuelve texto unido.

    Útil para generar el corpus listo para TF-IDF.

    Parámetros
    ----------
    texto : str
        Texto crudo.

    Returns
    -------
    str
        Texto preprocesado: tokens relevantes separados por espacio.
    """
    tokens = tokenizar(texto)
    return " ".join(tokens)


def preprocesar_dataframe(
    df, columna_texto: str = "texto", columna_destino: str = "texto_limpio"
):
    """
    Aplica preprocesar a toda una columna de un DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna de texto.
    columna_texto : str
        Nombre de la columna con texto crudo.
    columna_destino : str
        Nombre de la columna donde guardar el texto limpio.

    Returns
    -------
    pd.DataFrame
        DataFrame con la columna de texto preprocesado agregada.
    """
    df = df.copy()
    df[columna_destino] = df[columna_texto].apply(preprocesar)
    return df


# ============================================================
# Ejecución directa (para pruebas)
# ============================================================
if __name__ == "__main__":
    # Pruebas con ejemplos del dataset
    ejemplos = [
        "¿Cómo saco el libre deuda de la biblioteca de FACENA?",
        "¿Dónde pido el certificado de libre deuda de la facultad?",
        "¿Cuánto cuesta el trámite del título?",
        "El pago del arancel se hace en https://pagos.unne.edu.ar",
    ]

    print("=== Prueba de limpieza ===")
    for ej in ejemplos:
        print(f"\nOriginal:  {ej}")
        print(f"Limpio:    {limpiar_texto(ej)}")
        print(f"Tokens:    {tokenizar(ej)}")
        print(f"Preproc:   {preprocesar(ej)}")
