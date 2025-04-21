import unicodedata
import re

def normalizar_texto(texto):
    """
    Normaliza un texto eliminando acentos, convirtiendo a minúsculas
    y removiendo caracteres especiales, pero preservando formatos específicos.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    
    # Si es texto vacío, retornar como está
    if not texto.strip():
        return ""
    
    # Paso 1: Detectar y preservar fechas y otros elementos especiales
    elementos_a_preservar = []
    
    # Detectar fechas en formatos comunes
    patrones_fecha = [
        r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})',  # dd/mm/yyyy o dd-mm-yyyy
        r'(\d{2,4})[/\-](\d{1,2})[/\-](\d{1,2})'   # yyyy/mm/dd o yyyy-mm-dd
    ]
    
    # Crear una copia de trabajo del texto
    texto_procesado = texto
    
    # Encontrar todas las fechas y guardarlas con su posición
    for patron in patrones_fecha:
        # Buscar todas las coincidencias del patrón
        for match in re.finditer(patron, texto):
            fecha_original = match.group(0)
            # Guardar la fecha original y su posición
            elementos_a_preservar.append(fecha_original)
            # Reemplazar la fecha con un marcador único
            texto_procesado = texto_procesado.replace(fecha_original, f"FECHA_{len(elementos_a_preservar)-1}_MARCA", 1)
    
    # Convertir a minúsculas
    texto_procesado = texto_procesado.lower()
    
    # Eliminar acentos
    texto_procesado = ''.join(
        c for c in unicodedata.normalize('NFKD', texto_procesado)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Eliminar caracteres especiales, pero mantener los marcadores de fecha
    texto_procesado_limpio = ""
    i = 0
    while i < len(texto_procesado):
        # Si encontramos un marcador de fecha, lo preservamos íntegro
        if i + 6 < len(texto_procesado) and texto_procesado[i:i+6] == "fecha_":
            # Buscar el final del marcador
            j = texto_procesado.find("_marca", i)
            if j != -1:
                texto_procesado_limpio += texto_procesado[i:j+6]
                i = j + 6
                continue
        
        # Para caracteres normales, solo conservamos letras, números, espacios y comas
        if texto_procesado[i].isalnum() or texto_procesado[i] in " ,":
            texto_procesado_limpio += texto_procesado[i]
        i += 1
    
    texto_procesado = texto_procesado_limpio
    
    # Restaurar los elementos originales
    for i, elemento in enumerate(elementos_a_preservar):
        texto_procesado = texto_procesado.replace(f"fecha_{i}_marca", elemento)
    
    # Eliminar espacios múltiples
    texto_procesado = ' '.join(texto_procesado.split())
    
    return texto_procesado.strip()