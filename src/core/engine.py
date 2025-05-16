"""
Motor principal de procesamiento de consultas.
"""

import os
import re
import json
from difflib import SequenceMatcher
from .config import ruta_indices
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from src.utils.text_normalizer import normalizar_texto
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from transformers import pipeline
from .config import model, tokenizer

# --- 2) CARGAR TODOS LOS ÍNDICES ---
all_tools = []
indices = {}  # ALMACENAR LOS ÍNDICES CARGADOS

print(f"\nBuscando índices en: {ruta_indices}")
for nombre_dir in os.listdir(ruta_indices):
    ruta_indice = os.path.join(ruta_indices, nombre_dir)
    if not os.path.isdir(ruta_indice):
        continue
    if not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
        print(f"No hay indices.")
        continue

    fuente = nombre_dir.replace("index_", "")  # EXTRAER EL NOMBRE DE LA FUENTE

    try:
        print(f"Cargando índice para fuente: {fuente}")
        storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
        index = load_index_from_storage(storage_context)
        indices[fuente] = index

        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        query_engine = index.as_query_engine(streaming=False)

    except Exception as e:
        print(f"Error al cargar índice {ruta_indice}: {e}")


# POSIBLES CAMPOS A BUSCAR
def sugerir_campos(valor: str, campos_disponibles: list[str]) -> list[str]:
    """
    Dado un valor, sugiere los campos donde probablemente podría estar.
    Si es numérico largo, prioriza teléfono, tarjeta, etc.
    Si es texto, busca campos tipo dirección, municipio, etc.
    """
    # Protección contra valores None
    if valor is None:
        print("ADVERTENCIA: Valor None. Usando texto vacío.")
        valor = ""
        
    valor = valor.strip()
    campos_probables = []

    if valor.isdigit() and len(valor) >= 7:
        # TELÉFONO, TARJETA, NÚMERO
        claves = ['telefono', 'numero', 'tarjeta', 'fecha afiliacion', 'codigo postal', 'lada']
    elif any(c.isdigit() for c in valor) and any(c.isalpha() for c in valor):
        # ALFANUMÉRICO TIPO DIRECCIÓN
        claves = ['direccion', 'calle', 'colonia', 'cp', 'sector', 'entidad', 'clave ife', 'domicilio', 'numero']
    else:
        # SOLO TEXTO: MUNICIPIO, COLONIA, CIUDAD, ESTADO
        claves = ['municipio', 'colonia', 'ciudad', 'estado', 'localidad', 'edo de origen', 'sexo', 'ocupacion', ]

    for campo in campos_disponibles:
        campo_norm = normalizar_texto(campo)
        if any(clave in campo_norm for clave in claves):
            campos_probables.append(campo)

    if not campos_probables:
        campos_probables = campos_disponibles

    return campos_probables

def buscar_campos_inteligente(valor: str, carpeta_indices: str, campos_ordenados=None) -> str:
    print(f"\nBúsqueda para valor: '{valor}'")
    valor_normalizado = normalizar_texto(valor)
    resultados = []
    
    # DETECTAR SI ES UNA LOCALIDAD CONOCIDA
    localidades_conocidas = ["zapopan", "hidalgo", "san luis de la paz", "guanajuato", "aguascalientes", "lagos del country"]
    es_localidad = any(loc in valor_normalizado for loc in localidades_conocidas)
    
    if campos_ordenados is None:
        # PRIORIZAR CAMPOS DE LOCALIDAD
        if es_localidad:
            campos_ordenados = ['municipio', 'ciudad', 'sector', 'estado', 'colonia', 'direccion', 'calle', 'cp']
        else:
            campos_ordenados = ['municipio', 'colonia', 'direccion', 'estado', 'calle', 'ciudad', 'cp', 'sector']

    for campo in campos_ordenados:
        campo_variantes = campos_clave.get(campo, [campo])

        for variante in campo_variantes:
            key = normalizar_texto(variante)

            for nombre_dir in os.listdir(carpeta_indices):
                ruta_indice = os.path.join(carpeta_indices, nombre_dir)
                if not os.path.isdir(ruta_indice) or not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
                    continue

                fuente = nombre_dir.replace("index_", "")
                try:
                    storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
                    index = load_index_from_storage(storage_context)

                    # Búsqueda exacta
                    filters = MetadataFilters(filters=[
                        ExactMatchFilter(key=key, value=valor_normalizado)
                    ])
                    retriever = VectorIndexRetriever(index=index, similarity_top_k=10, filters=filters)
                    nodes = retriever.retrieve(f"{campo} es {valor}")

                    if nodes:
                        for node in nodes:
                            metadata = node.node.metadata
                            resumen = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                            resultados.append(f"Coincidencia en {fuente}:\n" + "\n".join(resumen))
                
                except Exception as e:
                    print(f"Error buscando en {fuente}: {e}")
                    continue
    
    # DEVOLVER TODOS LOS RESULTADOS ENCONTRADOS
    if es_localidad and resultados:
        return "\n\n".join(resultados)
    
    if resultados:
        return "\n\n".join(resultados)
        
    return f"No se encontraron coincidencias relevantes para el valor '{valor}'."

def extraer_valor(prompt: str) -> str:
    """
    Extrae un valor probable desde la pregunta simple, eliminando verbos
    como 'vive en', 'está en', etc.
    """
    prompt = prompt.strip().lower()

    # BUSCAR NÚMEROS LARGOS
    numeros = re.findall(r"\d{7,}", prompt)
    if numeros:
        return numeros[0]

    # PATRONES COMUNES PARA EXTRAER VALORES DESPUÉS DE CIERTAS FRASES
    frases_clave = [
        r"quien vive en\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)",
        r"vive en\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)",
        r"quien esta en\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)",
        r"en\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)$",
        r"de\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)$",
        r"quien\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)$"
    ]

    for frase in frases_clave:
        match = re.search(frase, prompt)
        if match:
            valor = match.group(1).strip()
            # SI TERMINA EN SIGNO DE INTERROGACIÓN, ELIMÍNALO
            valor = valor.rstrip('?')
            return valor

    # ELIMINAR PALABRAS COMUNES DE PREGUNTA AL INICIO
    palabras_pregunta = ["quien", "quién", "donde", "dónde", "cual", "cuál", "como", "cómo"]
    tokens = prompt.split()
    if tokens and tokens[0] in palabras_pregunta:
        return " ".join(tokens[1:])

    # SI TODO LO DEMÁS FALLA, DEVOLVER EL TEXTO SIN PALABRAS DE PREGUNTA
    palabras = prompt.split()
    palabras_filtradas = [p for p in palabras if p not in palabras_pregunta]
    if palabras_filtradas:
        return " ".join(palabras_filtradas)
    
    return prompt

def es_consulta_direccion(prompt: str) -> bool:
    """
    Determina si la consulta está relacionada con una dirección compleja.
    Devuelve False para consultas simples como "quien vive en zoquipan 1271"
    que deberían manejarse por la búsqueda por atributo.
    """
    prompt_lower = prompt.lower()
    
    # EXTRAER LA PARTE DESPUÉS PALABRAS CLAVE.
    patrones_extraccion = [
        r"quien vive en\s+(.+)$",
        r"vive en\s+(.+)$",
        r"quien esta en\s+(.+)$",
        r"donde esta\s+(.+)$",
        r"ubicado en\s+(.+)$"
    ]
    
    for patron in patrones_extraccion:
        match = re.search(patron, prompt_lower)
        if match:
            valor = match.group(1).strip()
            # SI EL VALOR CONTIENE COMA, ES UNA DIRECCIÓN COMPLEJA
            if ',' in valor:
                return True
            # SI EL VALOR TIENE POCAS PALABRAS, PROCESAR CON BÚSQUEDA POR ATRIBUTO
            palabras = valor.split()
            if len(palabras) < 4:
                return False
    
    # PATRONES COMUNES EN CONSULTAS DE DIRECCIÓN COMPLEJA
    patrones_consulta = [
        r"de\s+quien\s+es\s+la\s+direccion\s+",
        r"busca\s+la\s+direccion\s+",
        r"encuentra\s+la\s+direccion\s+"
    ]
    
    # SI CONTIENE ALGÚN PATRÓN ESPECÍFICO DE DIRECCIÓN COMPLEJA
    for patron in patrones_consulta:
        if re.search(patron, prompt_lower):
            return True
    
    # PALABRAS CLAVE COMUNES
    palabras_direccion = [
        "calle", "avenida", "av", "ave", "boulevard", "blvd", "calzada", "calz",
        "colonia", "col", "fraccionamiento", "fracc", 'calle', 'domicilio', 'numero', 'campo 14', 'colonia', 'cp', 'codigo postal', 'municipio', 'ciudad', 'sector', 'estado', 'edo de origen', 'entidad'
    ]
    
    # SI TIENE AL MENOS DOS PALABRAS CLAVE DE DIRECCIÓN ESPECÍFICAS, ES COMPLEJA
    palabras_encontradas = sum(1 for palabra in palabras_direccion if palabra in prompt_lower)
    if palabras_encontradas >= 2:
        return True
        
    # SI TIENE UNA COMA, ES UNA DIRECCIÓN COMPLEJA
    if ',' in prompt_lower:
        return True
    
    # DEJARLO PARA BÚSQUEDA POR ATRIBUTO
    return False

def extraer_texto_direccion(prompt: str) -> str:
    """
    Extrae el texto de dirección de una consulta del usuario.
    """
    prompt = prompt.strip()
    prompt_lower = prompt.lower()
    
    # PATRONES PARA EXTRAER DIRECCIONES DESPUÉS DE FRASES COMUNES
    patrones_extraccion = [
        r"quien vive en\s+(.+)$",
        r"de quien es la direccion\s+(.+)$",
        r"busca la direccion\s+(.+)$",
        r"encuentra\s+(.+)$",
        r"personas que viven en\s+(.+)$",
        r"domicilios? en\s+(.+)$",
        r"casas? en\s+(.+)$",
        r"habitantes de\s+(.+)$",
        r"vive en\s+(.+)$",
        r"donde esta\s+(.+)$",
        r"ubicado en\s+(.+)$"
    ]
    
    # INTENTAR EXTRAER CON PATRONES
    for patron in patrones_extraccion:
        match = re.search(patron, prompt_lower)
        if match:
            texto_direccion = match.group(1).strip()
            # LIMPIAR SIGNOS DE PUNTUACIÓN EXTRA
            texto_direccion = texto_direccion.strip('?!.,;:"\'')
            return texto_direccion
    
    # SI NO HAY PATRÓN ESPECÍFICO, VERIFICAR SI TODO EL TEXTO PARECE SER UNA DIRECCIÓN
    palabras_clave_direccion = ["calle", "avenida", "av", "colonia", "col", "fracc", 
                               "edificio", "número", "num", "#", "sector", "municipio",
                               "zoquipan", "lagos", "country", "hidalgo", "malva", "paseos"]
    
    palabras = prompt_lower.split()
    if any(palabra in palabras_clave_direccion for palabra in palabras):
        # SI HAY PALABRAS CLAVE DE DIRECCIÓN, ASUMIR QUE TODO EL TEXTO ES LA DIRECCIÓN
        return prompt
    
    # SI CONTIENE UN NÚMERO (PROBABLE NÚMERO DE CASA)
    if re.search(r"\d+", prompt) and len(palabras) > 1:
        return prompt
    
    # SI FALLA, DEVOLVER EL TEXTO COMPLETO
    return prompt

def detectar_campo_valor(prompt: str):
    prompt_lower = prompt.lower()

    aliases_ordenados = sorted([
        (campo_estandarizado, alias)
        for campo_estandarizado, alias_list in campos_clave.items()
        for alias in alias_list
    ], key=lambda x: -len(x[1]))

    for campo_estandarizado, alias in aliases_ordenados:
        if alias in prompt_lower:

            pattern = re.compile(rf"{alias}\s*(es|:)?\s*([\w\d\s\-.,]+)", re.IGNORECASE)
            match = pattern.search(prompt)
            if match:
                valor = match.group(2).strip()
                return campo_estandarizado, valor

            numeros = re.findall(r"\d{7,}", prompt)
            if numeros:
                return campo_estandarizado, numeros[0]

    return None, None

def extraer_componentes_nombre(query: str) -> list:
    """
    Extrae componentes de nombre de una consulta natural.
    Identifica nombres completos e iniciales/componentes parciales.
    """
    query_lower = query.lower()
    
    palabras_filtrar = [
        "quien", "quién", "cuantas", "cuántas", "personas", "nombre", "nombres", 
        "apellido", "apellidos", "con", "que", "qué", "tienen", "tiene", "hay",
        "se", "llama", "llaman", "empiezan", "empieza", "inicial", "iniciales",
        "primer", "primero", "primera", "segundo", "segunda", "de", "y", "o", "la", "el", "los", "las"
    ]
    
    patrones_extraccion = [
        r"nombre(?:s)?\s+([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+)(?:\s+(?:con|y|que|de|)\s+(?:iniciales?|apellidos?)?(?:\s+que\s+(?:empie(?:za|zan))?)?\s+([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])(?:\s+y\s+|\s+|\s*,\s*)([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]))?",
        r"([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{3,})\s+(?:con|y)\s+(?:iniciales?|apellidos?|letras?)?\s+([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])(?:\s+y\s+|\s+|\s*,\s*)([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])"
    ]
    
    for patron in patrones_extraccion:
        match = re.search(patron, query_lower)
        if match:
            grupos = match.groups()
            return [g for g in grupos if g]
    
    tokens = query_lower.split()
    componentes = []
    
    for token in tokens:
        if token not in palabras_filtrar and len(token) >= 1:
            token_limpio = token.strip('.,;:!?()"\'').lower()
            if token_limpio:
                componentes.append(token_limpio)
    
    componentes_ordenados = []
    iniciales = []
    
    for comp in componentes:
        if len(comp) >= 3:
            componentes_ordenados.append(comp)
        else:
            iniciales.append(comp)
    
    componentes_ordenados.extend(iniciales)
    
    return componentes_ordenados


def evaluar_coincidencia_componentes(metadata: dict, componentes: list) -> tuple:
    """
    Evalúa si un registro coincide con los componentes de nombre buscados.
    Retorna (categoría, puntuación, texto_resultado) o None si no coincide.
    """
    # OBTENER EL NOMBRE COMPLETO
    nombre_completo = (
        metadata.get("nombre_completo", "") or 
        metadata.get("nombre completo", "") or 
        metadata.get("nombre", "")
    ).strip()
    
    if not nombre_completo:
        return None
    
    nombre_norm = normalizar_texto(nombre_completo).lower()
    palabras_nombre = nombre_norm.split()
    iniciales_nombre = [p[0] for p in palabras_nombre]
    
    primer_componente = componentes[0].lower()
    if primer_componente not in nombre_norm and not any(palabra.startswith(primer_componente) for palabra in palabras_nombre):
        return None
    
    componentes_adicionales = [c.lower() for c in componentes[1:]]
    
    if not componentes_adicionales:
        resumen = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
        texto_resultado = "Encontrado:\n" + "\n".join(resumen)
        return "coincidencias_completas", 1.0, texto_resultado
    
    coincidencias_completas = 0
    coincidencias_iniciales = 0
    
    for componente in componentes_adicionales:
        if len(componente) == 1:
            # VERIFICAR SI COINCIDE CON ALGUNA INICIAL
            if componente in iniciales_nombre:
                coincidencias_iniciales += 1
            else:
                return None
        else:
            if componente in nombre_norm or any(palabra.startswith(componente) for palabra in palabras_nombre):
                coincidencias_completas += 1
            elif componente[0] in iniciales_nombre:
                coincidencias_iniciales += 1
            else:
                return None
    
    resumen = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
    texto_resultado = "Encontrado:\n" + "\n".join(resumen)
    
    if coincidencias_completas == len(componentes_adicionales):
        # TODAS LAS COINCIDENCIAS SON COMPLETAS
        return "coincidencias_completas", 1.0, texto_resultado
    elif coincidencias_iniciales + coincidencias_completas == len(componentes_adicionales):
        # TODAS COINCIDEN, ALGUNAS COMO INICIALES
        score = 0.8 + (0.2 * (coincidencias_completas / len(componentes_adicionales)))
        return "coincidencias_iniciales", score, texto_resultado
    else:
        # ALGUNAS COINCIDENCIAS PARCIALES
        score = 0.5 + (0.2 * ((coincidencias_completas + coincidencias_iniciales) / len(componentes_adicionales)))
        return "coincidencias_parciales", score, texto_resultado

def buscar_campos_similares(valor: str, campos: list[str], carpeta_indices: str) -> str:
    print(f"\nBuscando registros...\n")
    valor_normalizado = normalizar_texto(valor)
    resultados = []

    for campo in campos:
        campo_normalizado = normalizar_texto(campo)

        for nombre_dir in os.listdir(carpeta_indices):
            ruta_indice = os.path.join(carpeta_indices, nombre_dir)
            if not os.path.isdir(ruta_indice) or not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
                continue

            fuente = nombre_dir.replace("index_", "")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
                index = load_index_from_storage(storage_context)

                filters = MetadataFilters(filters=[
                    ExactMatchFilter(key=campo_normalizado, value=valor_normalizado)
                ])
                retriever = VectorIndexRetriever(index=index, similarity_top_k=5, filters=filters)
                nodes = retriever.retrieve(f"{campo} es {valor}")

                if nodes:
                    for node in nodes:
                        metadata = node.node.metadata
                        resumen = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                        resultados.append(f"Coincidencia en {fuente}:\n" + "\n".join(resumen))
            except Exception as e:
                print(f"Error buscando en {fuente}: {e}")
                continue

    if resultados:
        return "\n".join(resultados)
    else:
        return f"No se encontraron coincidencias para el valor '{valor}'."
    
def similitud(texto1, texto2):
    return SequenceMatcher(None, texto1, texto2).ratio()

# CAMPOS DISPONIBLES
campos_detectados = set()

for nombre_dir in os.listdir(ruta_indices):
    ruta_indice = os.path.join(ruta_indices, nombre_dir)
    if not os.path.isdir(ruta_indice) or not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
        continue

    try:
        storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
        index = load_index_from_storage(storage_context)
        for node_id, doc in index.docstore.docs.items():
            metadata = doc.metadata
            if metadata:
                campos_detectados.update(metadata.keys())
            break

    except Exception as e:
        print(f"Error al explorar metadatos en {nombre_dir}: {e}")
        continue

# ALIAS COMUNES
alias_comunes = {
    "telefono": ["telefono", "teléfono", "tel"],
    "tarjeta": ["tarjeta"],
    "direccion": ["direccion", "dirección", "calle"],
    "cp": ["cp", "código postal", "codigo postal"],
    "colonia": ["colonia"],
    "estado": ["estado"],
    "municipio": ["municipio"],
    "nombre_completo": ["nombre", "nombre completo"],
    "ocupacion": ["ocupacion", "profesion", "trabajo"],
    "sexo": ["sexo", "genero", "género"],
}

# CONSTRUIR `mapa_campos` Y `campos_clave` AUTOMÁTICAMENTE
mapa_campos = {}
campos_clave = {}

for campo in campos_detectados:
    coincidencia = None
    for base, variantes in alias_comunes.items():
        if campo in [normalizar_texto(alias) for alias in variantes] or base in campo.lower():
            coincidencia = base
            break
    if coincidencia:
        mapa_campos[campo] = coincidencia
        campos_clave.setdefault(coincidencia, []).append(campo)
    else:
        mapa_campos[campo] = campo
        campos_clave.setdefault(campo, []).append(campo)

llm_clasificador = pipeline("text-generation", model=model, tokenizer=tokenizer)

def interpretar_pregunta_llm(prompt: str, llm_clasificador=None) -> dict:
    """
    Analizador avanzado de intenciones que combina técnicas de NLP básicas con LLM
    para entender mejor la intención del usuario independientemente de la formulación.
    """
    # Si el pipeline está disponible, úsalo
    if llm_clasificador is not None:
        try:
            prompt_clasificacion = f"""
            Eres un sistema experto que clasifica consultas para una base de datos de personas. Necesito que clasifiques la siguiente consulta:
            
            "{prompt}"
            
            Debes determinar:
            1. El tipo de búsqueda: "nombre", "telefono", "direccion", "atributo" o "nombre_componentes"
            2. El campo específico (si aplica)
            3. El valor a buscar
            
            Responde con un objeto JSON que contenga exactamente "tipo_busqueda", "campo" y "valor".
            """
            
            salida_cruda = llm_clasificador(prompt_clasificacion, max_new_tokens=256, return_full_text=False)[0]['generated_text']
            
            match = re.search(r'\{[\s\S]*?\}', salida_cruda)
            if match:
                json_text = match.group(0)
                resultado = json.loads(json_text)
                
                # VERIFICACIÓN Y CORRECCIÓN DE VALORES
                if resultado.get("valor") is None or resultado.get("valor") == "":
                    # EXTRACCIÓN FALLBACK BASADA EN TIPO
                    if resultado.get("tipo_busqueda") == "nombre":
                        palabras = [p for p in prompt.split() if len(p) > 2 and p[0].isupper()]
                        if palabras:
                            resultado["valor"] = " ".join(palabras)
                        else:
                            palabras = prompt.split()
                            resultado["valor"] = " ".join(palabras[-min(3, len(palabras)):])
                    else:
                        resultado["valor"] = extraer_valor(prompt)
                
                return resultado
        except Exception as e:
            print(f"[⚠️ LLM] Error en el análisis LLM vía pipeline: {e}")
            print("Cambiando a método de respaldo...")

     # Si el pipeline falla o no está disponible, usar análisis de patrones
    print("[INFO] Usando análisis de patrones básico")
    
    prompt_lower = prompt.lower()

    # DETECTAR BÚSQUEDA POR COMPONENTES DE NOMBRE
    es_busqueda_componentes = (
        (re.search(r'cuantas?\s+personas\s+(?:de\s+)?nombre\s+([a-zA-ZáéíóúñÑ]+)\s+con', prompt_lower) or
        re.search(r'quien(?:es)?\s+(?:se\s+llaman?|tienen?|con)\s+(?:nombre\s+)?([a-zA-ZáéíóúñÑ]+)\s+(?:con|y|de)\s+(?:apellidos?|iniciales?)', prompt_lower) or
        re.search(r'([a-zA-ZáéíóúñÑ]+)\s+con\s+(?:iniciales?|apellidos?)\s+([a-zA-ZáéíóúñÑ])\s+y\s+([a-zA-ZáéíóúñÑ])', prompt_lower) or
        re.search(r'nombre\s+([a-zA-ZáéíóúñÑ]+)\s+(?:con\s+)?(?:iniciales?|letras?)\s+([a-zA-ZáéíóúñÑ])\s+y\s+([a-zA-ZáéíóúñÑ])', prompt_lower))
    )
    
    # SI ES UNA BÚSQUEDA CLARA POR COMPONENTES
    if es_busqueda_componentes:
        componentes = extraer_componentes_nombre(prompt)
        if componentes:
            return {
                "tipo_busqueda": "nombre_componentes",
                "campo": "nombre_completo",
                "valor": " ".join(componentes)
            }
    
    # PRE-PROCESAMIENTO Y ANÁLISIS RÁPIDO CON PATRONES COMUNES
    
    # DETECTORES RÁPIDOS POR CATEGORÍA
    es_consulta_telefono = any(palabra in prompt_lower for palabra in [
        "telefono", "teléfono", "tel", "numero", "número", "celular", "móvil", "movil", "contacto"
    ]) and re.search(r'\d{6,}', prompt_lower)
    
    es_consulta_nombre = (
        re.search(r'(?:quien|quién|quienes|quiénes) (?:es|son) ([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)', prompt_lower) or
        re.search(r'(?:busca|encuentra|dame info|información de|datos de) ([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)', prompt_lower) or
        re.search(r'(?:información|info|datos) (?:de|sobre) ([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)', prompt_lower)
    ) and not es_consulta_telefono  # PRIORIZAR TELÉFONO
    
    es_consulta_direccion = (
        ("dirección" in prompt_lower or "direccion" in prompt_lower or "domicilio" in prompt_lower or 
         "vive en" in prompt_lower or "casa" in prompt_lower or "calle" in prompt_lower) and
        (re.search(r'\d+', prompt_lower) or
         any(palabra in prompt_lower for palabra in [
             "colonia", "sector", "fraccionamiento", "fracc", "avenida", "ave", "av", "blvd", "boulevard"
         ]))
    )
    
    es_consulta_atributo = (
        re.search(r'(?:quien|quién) (?:tiene|posee|cuenta con) ([^?]+)', prompt_lower) or
        "hombres" in prompt_lower or "mujeres" in prompt_lower or "género" in prompt_lower or
        "genero" in prompt_lower or "profesión" in prompt_lower or "profesion" in prompt_lower or
        "ocupación" in prompt_lower or "ocupacion" in prompt_lower or "trabajo" in prompt_lower or
        "tarjeta" in prompt_lower or "curp" in prompt_lower or "rfc" in prompt_lower or
        "clave" in prompt_lower or "ife" in prompt_lower
    )
    
    # DECISIÓN RÁPIDA PARA CASOS CLAROS
    
    # SI ES UNA CLASIFICACIÓN CLARA, RETORNAR DIRECTAMENTE
    if es_consulta_telefono:
        numeros = re.findall(r'\d{6,}', prompt_lower)
        if numeros:
            return {
                "tipo_busqueda": "telefono",
                "campo": "telefono_completo",
                "valor": numeros[0]
            }
    
    if es_consulta_direccion:
        # EXTRAER LA DIRECCIÓN CON TÉCNICAS BÁSICAS
        texto_direccion = extraer_texto_direccion(prompt)
        return {
            "tipo_busqueda": "direccion",
            "campo": "direccion", 
            "valor": texto_direccion
        }
    
    if es_consulta_nombre and not es_consulta_direccion:
        # EXTRAER EL NOMBRE
        match = None
        for patron in [
            r'(?:quien|quién|quienes|quiénes) (?:es|son) ([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)',
            r'(?:busca|encuentra|dame info|información de|datos de) ([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)',
            r'(?:información|info|datos) (?:de|sobre) ([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)'
        ]:
            match = re.search(patron, prompt_lower)
            if match:
                break
        
        if match:
            nombre = match.group(1).strip()
            # FILTRAR PALABRAS COMUNES QUE NO FORMAN PARTE DEL NOMBRE
            palabras_filtrar = ["información", "info", "datos", "usuario", "persona", "registro", "la", "el", "de"]
            for palabra in palabras_filtrar:
                nombre = nombre.replace(f" {palabra} ", " ").strip()
            
            if nombre and len(nombre) > 2:  # EVITAR NOMBRES DEMASIADO CORTOS
                return {
                    "tipo_busqueda": "nombre",
                    "campo": "nombre_completo",
                    "valor": nombre
                }
    
    # PARA CASOS MÁS AMBIGUOS, USAR EL LLM
    
    system_prompt = (
        "Eres un clasificador de intenciones que analiza consultas de usuarios para un sistema de búsqueda de personas. "
        "Tu tarea es extraer:\n"
        "- 'tipo_busqueda': debe ser uno de estos valores exactamente: 'nombre', 'direccion', 'telefono' o 'atributo'.\n"
        "- 'campo': el campo específico relevante para la búsqueda como 'telefono', 'municipio', 'sexo', 'ocupacion', 'clave ife' etc.\n"
        "- 'valor': el dato específico mencionado en la consulta, sin palabras de pregunta.\n\n"
        "REGLAS IMPORTANTES:\n"
        "1. PRIORIZA TELEFONO si se menciona un número de varios dígitos junto con palabras como 'teléfono', 'tel', 'número', 'contacto'.\n"
        "2. PRIORIZA DIRECCION si se menciona 'vive en', 'domicilio', 'calle', 'colonia' o términos similares, especialmente con números.\n"
        "3. PRIORIZA NOMBRE si hay palabras que parecen nombres propios (capitalizados) o se busca información general sobre alguien.\n"
        "4. USA ATRIBUTO para consultas sobre características como 'sexo', 'ocupación', o identificadores (CURP, RFC, tarjeta).\n"
        "5. Si la consulta es ambigua, selecciona el tipo de búsqueda más probable según el contexto.\n"
        "6. Para VALOR, extrae SOLO la información relevante, sin palabras de pregunta ni verbos auxiliares.\n"
        f"Consulta: {prompt}\n"
        "Responde con un JSON válido que contenga tipo_busqueda, campo y valor."
    )
    
    try:
        salida_cruda = llm_clasificador(system_prompt, max_new_tokens=256, return_full_text=False)[0]['generated_text']
        match = re.search(r'\{[\s\S]*?\}', salida_cruda)
        if match:
            json_text = match.group(0)
            resultado = json.loads(json_text)
            
            # VERIFICACIÓN Y CORRECCIÓN DE VALORES
            if resultado.get("valor") is None or resultado.get("valor") == "":
                # EXTRACCIÓN FALLBACK BASADA EN TIPO
                if resultado.get("tipo_busqueda") == "nombre":
                    palabras = [p for p in prompt.split() if len(p) > 2 and p[0].isupper()]
                    if palabras:
                        resultado["valor"] = " ".join(palabras)
                    else:
                        palabras = prompt.split()
                        resultado["valor"] = " ".join(palabras[-min(3, len(palabras)):])
                else:
                    resultado["valor"] = extraer_valor(prompt)
            
            return resultado
        else:
            print("[⚠️ LLM] No se detectó JSON válido en la respuesta.")
    except Exception as e:
        print(f"[⚠️ LLM] Error en el análisis LLM: {e}")
    
    # FALLBACK FINAL: ANÁLISIS BÁSICO DE LA CONSULTA
    
    # EXTRAER NÚMEROS GRANDES (POSIBLE TELÉFONO)
    numeros = re.findall(r'\b\d{7,}\b', prompt)
    if numeros:
        return {"tipo_busqueda": "telefono", "campo": "telefono_completo", "valor": numeros[0]}
    
    # BUSCAR POSIBLES NOMBRES PROPIOS
    palabras = prompt.split()
    nombres_posibles = []
    for palabra in palabras:
        if len(palabra) > 2 and palabra[0].isupper() and palabra.isalpha():
            nombres_posibles.append(palabra)
    
    if nombres_posibles:
        return {"tipo_busqueda": "nombre", "campo": "nombre_completo", "valor": " ".join(nombres_posibles)}
    
    # EXTRAER PALABRAS SIGNIFICATIVAS
    palabras_filtradas = [p for p in palabras if len(p) > 3 and p.lower() not in ["quien", "quién", "como", "cómo"]]
    if palabras_filtradas:
        return {"tipo_busqueda": "desconocido", "campo": "", "valor": " ".join(palabras_filtradas)}
    
    return {"tipo_busqueda": "desconocido", "valor": prompt}

def desambiguar_consulta(analisis: dict, prompt: str, llm) -> dict:
    """
    Sistema para clarificar consultas ambiguas y determinar el tipo de búsqueda más adecuado.
    Intenta múltiples estrategias antes de pedir clarificación al usuario.
    
    Args:
        analisis: El resultado inicial del análisis de la consulta
        prompt: La consulta original del usuario
        llm: El modelo de lenguaje para análisis avanzado
    
    Returns:
        dict: El análisis refinado con tipo_busqueda, campo y valor
    """
    
    # Si no hay ejemplos relevantes, continuar con la desambiguación normal
    if analisis.get("tipo_busqueda") not in ["desconocido", None]:
        return analisis
    
    valor = analisis.get("valor", prompt)
    prompt_lower = prompt.lower()

    patrones_componentes = [
        r'nombre\s+([a-zA-ZáéíóúñÑ]+)\s+(?:con|y)\s+(?:inicial|letra|apellido)',
        r'([a-zA-ZáéíóúñÑ]{3,})\s+con\s+([a-zA-ZáéíóúñÑ])\s+y\s+([a-zA-ZáéíóúñÑ])',
        r'busca\s+(?:a\s+)?([a-zA-ZáéíóúñÑ]{3,})\s+que\s+(?:tenga|con)\s+([a-zA-ZáéíóúñÑ])\s+y\s+([a-zA-ZáéíóúñÑ])'
    ]

    for patron in patrones_componentes:
        if re.search(patron, prompt_lower):
            componentes = extraer_componentes_nombre(prompt)
            if componentes:
                return {
                    "tipo_busqueda": "nombre_componentes",
                    "campo": "nombre_completo",
                    "valor": " ".join(componentes)
                }
    
    # PASO 1: INTENTAR DESAMBIGUAR ANALIZANDO LAS CARACTERÍSTICAS DEL VALOR
    if re.match(r'^\d{7,}$', valor.strip()):
        return {
            "tipo_busqueda": "telefono",
            "campo": "telefono_completo",
            "valor": valor.strip()
        }
    
    # ¿TIENE FORMATO DE DIRECCIÓN?
    if any(palabra in prompt_lower for palabra in ["calle", "avenida", "av", "colonia", "domicilio"]) and re.search(r'\d+', prompt_lower):
        return {
            "tipo_busqueda": "direccion",
            "campo": "direccion",
            "valor": valor
        }
    
    # ¿CONTIENE PALABRAS QUE SUELEN ESTAR EN NOMBRES?
    palabras_nombre = ["apellido", "nombre", "llama", "persona", " sr ", " sra "]
    if any(palabra in prompt_lower for palabra in palabras_nombre):
        # Intentar extraer solo la parte que parece ser un nombre
        palabras = prompt.split()
        candidatos_nombre = []
        for palabra in palabras:
            if len(palabra) > 2 and palabra[0].isupper() and palabra.isalpha():
                candidatos_nombre.append(palabra)
        
        if candidatos_nombre:
            return {
                "tipo_busqueda": "nombre",
                "campo": "nombre_completo",
                "valor": " ".join(candidatos_nombre)
            }
    
    # ¿PARECE SER UNA CONSULTA SOBRE UN ATRIBUTO ESPECÍFICO?
    patrones_atributo = {
        "sexo": ["sexo", "genero", "género", "hombres", "mujeres", "masculino", "femenino"],
        "ocupacion": ["profesión", "profesion", "ocupación", "ocupacion", "trabajo", "empleo", "oficio"],
        "curp": ["curp"],
        "rfc": ["rfc"],
        "tarjeta": ["tarjeta", "credito", "crédito", "débito", "debito"],
        "ife": ["ife", "credencial", "electoral"],
    }
    
    for campo, palabras_clave in patrones_atributo.items():
        if any(palabra in prompt_lower for palabra in palabras_clave):
            # EXTRAER EL VALOR DEL ATRIBUTO DE LA CONSULTA
            valor_extraido = None
            for palabra in palabras_clave:
                if palabra in prompt_lower:
                    idx = prompt_lower.index(palabra)
                    resto = prompt_lower[idx + len(palabra):].strip()
                    resto = re.sub(r'^(?:es|son|con|de|del|la|el|los|las|que)\s+', '', resto)
                    if resto:
                        valor_extraido = resto
                        break
            
            if valor_extraido:
                return {
                    "tipo_busqueda": "atributo",
                    "campo": campo,
                    "valor": valor_extraido
                }
            else:
                return {
                    "tipo_busqueda": "atributo",
                    "campo": campo,
                    "valor": ""
                }
    
    # PASO 2: SI AÚN ES AMBIGUO, UTILIZAR ANÁLISIS LLM MÁS PROFUNDO
    
    system_prompt = (
        "Estás analizando una consulta ambigua para un sistema de búsqueda de personas. "
        "La consulta es ambigua y necesitamos determinar el tipo más probable de búsqueda. "
        "Analiza cuidadosamente el texto y decide entre estas opciones:\n"
        "1. Búsqueda por NOMBRE - si parece que se busca información general sobre una persona\n"
        "2. Búsqueda por TELÉFONO - si se menciona o parece referirse a un número telefónico\n"
        "3. Búsqueda por DIRECCIÓN - si se refiere a donde vive alguien o una ubicación física\n"
        "4. Búsqueda por ATRIBUTO - si busca personas con una característica específica\n\n"
        f"Consulta ambigua: '{prompt}'\n\n"
        "Responde SOLAMENTE con el tipo de búsqueda (NOMBRE, TELEFONO, DIRECCION, o ATRIBUTO) "
        "seguido de DOS PUNTOS y el valor específico que debería buscarse. Por ejemplo:\n"
        "NOMBRE: Juan Pérez\n"
        "o\n"
        "ATRIBUTO: médico"
    )
    
    try:
        # LLAMAR AL LLM PARA DESAMBIGUAR
        respuesta = llm(system_prompt, max_new_tokens=128, return_full_text=False)[0]['generated_text']
        respuesta = respuesta.strip()
        
        if ":" in respuesta:
            tipo, valor_extraido = respuesta.split(":", 1)
            tipo = tipo.strip().upper()
            valor_extraido = valor_extraido.strip()
            
            # MAPEAR EL TIPO DE RESPUESTA AL FORMATO ESPERADO
            tipo_mapeado = {
                "NOMBRE": "nombre",
                "TELEFONO": "telefono",
                "DIRECCION": "direccion",
                "ATRIBUTO": "atributo"
            }.get(tipo)
            
            if tipo_mapeado and valor_extraido:
                # DETERMINAR EL CAMPO SEGÚN EL TIPO
                campo = {"nombre": "nombre_completo", 
                         "telefono": "telefono_completo", 
                         "direccion": "direccion",
                         "atributo": ""}[tipo_mapeado]
                
                # PARA ATRIBUTOS, INTENTAR DETERMINAR EL CAMPO ESPECÍFICO
                if tipo_mapeado == "atributo":
                    if any(palabra in valor_extraido.lower() for palabra in ["hombre", "mujer", "masculino", "femenino"]):
                        campo = "sexo"
                    elif any(palabra in prompt_lower for palabra in ["profesión", "trabajo", "ocupación"]):
                        campo = "ocupacion"
                
                return {
                    "tipo_busqueda": tipo_mapeado,
                    "campo": campo,
                    "valor": valor_extraido
                }
    except Exception as e:
        print(f"[⚠️ LLM] Error en desambiguación LLM: {e}")
    
    # PASO 3: SI FALLA, INTENTAR FALLBACK
    palabras = prompt.split()
    palabras_capitalizadas = [p for p in palabras if len(p) > 2 and p[0].isupper()]
    
    if palabras_capitalizadas:
        return {
            "tipo_busqueda": "nombre",
            "campo": "nombre_completo",
            "valor": " ".join(palabras_capitalizadas)
        }
    
    # BÚSQUEDA GENÉRICA CON TODO EL TEXTO
    return {
        "tipo_busqueda": "nombre",
        "campo": "nombre_completo",
        "valor": valor
    }