"""
Herramienta de búsqueda que utiliza Elasticsearch para consultas inteligentes.
Esta herramienta reemplaza todas las búsquedas tradicionales.
"""

from llama_index.core.tools import FunctionTool
from src.core.engine import buscar_con_elasticsearch, all_tools

def elasticsearch_search_tool_func(query: str) -> str:
    """
    Busca información usando Elasticsearch con capacidades avanzadas:
    - Tolerancia a errores ortográficos
    - Búsquedas en lenguaje natural
    - Búsquedas combinadas
    - Velocidad extrema
    
    Args:
        query: La consulta en lenguaje natural del usuario
        
    Returns:
        str: Resultados de la búsqueda formateados
    """
    return buscar_con_elasticsearch(query, max_results=10)

# Crear la herramienta
elasticsearch_search_tool = FunctionTool.from_defaults(
    fn=elasticsearch_search_tool_func,
    name="elasticsearch_search",
    description=(
        "Herramienta principal de búsqueda que utiliza Elasticsearch. "
        "Maneja CUALQUIER tipo de consulta con tolerancia a errores ortográficos. "
        "Ejemplos: 'Juan Peres' (con error), 'telefono 555123', 'vive en zapopan', "
        "'maria ingeniera guadalajara', 'doctor col centro'. "
        "Esta herramienta es la PRIMERA OPCIÓN para cualquier búsqueda."
    )
)

# Agregar como primera herramienta (prioridad máxima)
all_tools.insert(0, elasticsearch_search_tool)