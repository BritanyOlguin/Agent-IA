import unicodedata
import re

def normalizar_texto(texto):
    """
    Normaliza un texto eliminando acentos, convirtiendo a minúsculas
    y removiendo caracteres especiales.
    """
    if not isinstance(texto, str):
        texto = str(texto)

    # Convertir a minúsculas
    texto = texto.lower()

    # Eliminar acentos
    texto = ''.join(
        char for char in unicodedata.normalize('NFKD', texto)
        if unicodedata.category(char) != 'Mn'
    )

    # Eliminar caracteres especiales y mantener solo letras, números y espacios
    texto = re.sub(r'[^a-z0-9\s,]', '', texto)

    # Eliminar espacios extra
    texto = ' '.join(texto.split())

    return texto.strip()
