import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.retrievers import VectorIndexRetriever
from difflib import SequenceMatcher
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import StorageContext, load_index_from_storage
from typing import Dict, Any, List
from transformers import pipeline
import json
import sys
# Importar la función normalizar_texto desde normalizar_texto.py
sys.path.append(r"C:\Users\Sistemas\Documents\OKIP\src")
from normalizar_texto import normalizar_texto
import re

# --- 1) CONFIGURACIÓN ---
ruta_modelo_embeddings = r"C:\Users\Sistemas\Documents\OKIP\models\models--intfloat--e5-large-v2"
ruta_indices = r"C:\Users\Sistemas\Documents\OKIP\llama_index_indices"
ruta_modelo_llama3 = r"C:\Users\Sistemas\Documents\OKIP\models\models--meta-llama--Meta-Llama-3-8B-Instruct"

# --- Constantes para buscar_direccion_combinada ---
CAMPOS_DIRECCION = ['domicilio', 'calle', 'numero', 'colonia', 'sector', 'municipio', 'ciudad', 'estado', 'cp', 'direccion', 'campo14', 'domicilio calle', 'codigo postal', 'edo de origen']
CAMPOS_BUSQUEDA_EXACTA = ['domicilio', 'direccion', 'calle']
STOP_WORDS = {'de', 'la', 'del', 'los', 'las', 'y', 'a', 'en', 'el', 'col', 'colonia', 'cp', 'sector', 'calzada', 'calz', 'boulevard', 'blvd', 'avenida', 'ave', 'av'}
UMBRAL_PUNTAJE_MINIMO = 0.55
MAX_RESULTADOS_FINALES = 10
TOLERANCIA_NUMERO_CERCANO = 50

# CONFIGURACIÓN DE DISPOSITIVO Y LLM
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("⚠️ Advertencia: Usando CPU para LLM. Las respuestas serán lentas.")
else:
    print(f"Usando dispositivo para LLM y Embeddings: {device}")

# --- CARGAR MODELO Y TOKENIZER CON TRANSFORMERS ---
print(f" Cargando Tokenizer y Modelo Llama 3 desde: {ruta_modelo_llama3}")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        ruta_modelo_llama3,
        local_files_only=True
    )
    print("Tokenizer cargado.")

    model = AutoModelForCausalLM.from_pretrained(
        ruta_modelo_llama3,
        torch_dtype=torch.float16,  # MENOR USO DE VRAM
        load_in_8bit=True,
        device_map="auto",  # QUE TRANSFORMERS DISTRIBUYA EN LA GPU
        local_files_only=True
    )
    print("Modelo LLM Llama 3 cargado en dispositivo.")

except Exception as e:
    print(f"Error al cargar Llama 3 desde {ruta_modelo_llama3}: {e}")
    exit()

# --- CONFIGURAR HuggingFaceLLM ---
try:
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=8000,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.1, "do_sample": False},
    )
    print("HuggingFaceLLM configurado con modelo Llama 3 cargado.")
except Exception as e:
    print(f"Error al configurar HuggingFaceLLM con modelo pre-cargado: {e}")
    exit()

# CONFIGURAR Modelo de Embeddings
print(f"Cargando modelo de embeddings: {os.path.basename(ruta_modelo_embeddings)}")
try:
    embed_model = HuggingFaceEmbedding(
        model_name=ruta_modelo_embeddings,
        device=device,
        normalize=True
    )
    print("Modelo de embeddings e5-large-v2 cargado.")
except Exception as e:
    print(f"Error cargando el modelo de embeddings desde {ruta_modelo_embeddings}: {e}")
    exit()

# APLICAR CONFIGURACIÓN LLAMAINDEX
Settings.llm = llm
Settings.embed_model = embed_model

# --- 2) CARGAR TODOS LOS ÍNDICES ---
all_tools = []
indices = {}  # ALMACENAR LOS ÍNDICES CARGADOS

print(f"\nBuscando índices en: {ruta_indices}")
for nombre_dir in os.listdir(ruta_indices):
    ruta_indice = os.path.join(ruta_indices, nombre_dir)
    if not os.path.isdir(ruta_indice):
        continue
    if not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
        print(f"No contiene índice válido.")
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
        print("[ADVERTENCIA] Valor None recibido en sugerir_campos. Usando texto vacío.")
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
    
    # Detectar si es una localidad conocida
    localidades_conocidas = ["zapopan", "hidalgo", "san luis de la paz", "guanajuato", "aguascalientes", "lagos del country"]
    es_localidad = any(loc in valor_normalizado for loc in localidades_conocidas)
    
    if campos_ordenados is None:
        # Priorizar campos de localidad si parece ser una localidad
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
    
    # Para localidades, devolver todos los resultados encontrados
    if es_localidad and resultados:
        return "\n\n".join(resultados)  # Limitar a 15 resultados para no sobrecargar
    
    # Para otras búsquedas, devolver los resultados normalmente
    if resultados:
        return "\n\n".join(resultados)  # Limitar a 5 resultados
        
    return f"No se encontraron coincidencias relevantes para el valor '{valor}'."

def extraer_valor(prompt: str) -> str:
    """
    Extrae un valor probable desde la pregunta simple, eliminando verbos
    como 'vive en', 'está en', etc.
    """
    prompt = prompt.strip().lower()

    # Buscar números largos (teléfonos, etc.)
    numeros = re.findall(r"\d{7,}", prompt)
    if numeros:
        return numeros[0]

    # Patrones comunes para extraer valores después de ciertas frases
    frases_clave = [
        r"quien vive en\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)",
        r"vive en\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)",
        r"quien esta en\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)",
        r"en\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)$",
        r"de\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)$",
        r"quien\s+([a-zA-ZáéíóúñÑ0-9\s\-]+)$"  # Para casos como "quien lomas de atemajac"
    ]

    for frase in frases_clave:
        match = re.search(frase, prompt)
        if match:
            valor = match.group(1).strip()
            # Si termina en signo de interrogación, elimínalo
            valor = valor.rstrip('?')
            return valor

    # Eliminar palabras comunes de pregunta al inicio
    palabras_pregunta = ["quien", "quién", "donde", "dónde", "cual", "cuál", "como", "cómo"]
    tokens = prompt.split()
    if tokens and tokens[0] in palabras_pregunta:
        return " ".join(tokens[1:])

    # Si todo lo demás falla, devolver el texto sin palabras de pregunta
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
    
    # Si es una consulta simple (pocos componentes), procesarla con búsqueda por atributo
    # Extraer la parte después de "quien vive en", "donde está", etc.
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
            # Si el valor contiene coma, es una dirección compleja
            if ',' in valor:
                return True
            # Si el valor tiene pocas palabras, procesar con búsqueda por atributo
            palabras = valor.split()
            if len(palabras) < 4:
                return False
    
    # Patrones comunes en consultas de dirección compleja
    patrones_consulta = [
        r"de\s+quien\s+es\s+la\s+direccion\s+",
        r"busca\s+la\s+direccion\s+",
        r"encuentra\s+la\s+direccion\s+"
    ]
    
    # Si contiene algún patrón específico de dirección compleja
    for patron in patrones_consulta:
        if re.search(patron, prompt_lower):
            return True
    
    # Palabras clave comunes en direcciones mexicanas
    palabras_direccion = [
        "calle", "avenida", "av", "ave", "boulevard", "blvd", "calzada", "calz",
        "colonia", "col", "fraccionamiento", "fracc", 'calle', 'domicilio', 'numero', 'campo 14', 'colonia', 'cp', 'codigo postal', 'municipio', 'ciudad', 'sector', 'estado', 'edo de origen', 'entidad'
    ]
    
    # Si tiene al menos dos palabras clave de dirección específicas, es compleja
    palabras_encontradas = sum(1 for palabra in palabras_direccion if palabra in prompt_lower)
    if palabras_encontradas >= 2:
        return True
        
    # Si tiene una coma, probablemente es una dirección compleja
    if ',' in prompt_lower:
        return True
    
    # En cualquier otro caso, dejarlo para búsqueda por atributo
    return False

def extraer_texto_direccion(prompt: str) -> str:
    """
    Extrae el texto de dirección de una consulta del usuario.
    """
    prompt = prompt.strip()
    prompt_lower = prompt.lower()
    
    # Patrones para extraer direcciones después de frases comunes
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
    
    # Intentar extraer con patrones
    for patron in patrones_extraccion:
        match = re.search(patron, prompt_lower)
        if match:
            texto_direccion = match.group(1).strip()
            # Limpiar signos de puntuación extra
            texto_direccion = texto_direccion.strip('?!.,;:"\'')
            return texto_direccion
    
    # Si no hay patrón específico, verificar si todo el texto parece ser una dirección
    palabras_clave_direccion = ["calle", "avenida", "av", "colonia", "col", "fracc", 
                               "edificio", "número", "num", "#", "sector", "municipio",
                               "zoquipan", "lagos", "country", "hidalgo", "malva", "paseos"]
    
    palabras = prompt_lower.split()
    if any(palabra in palabras_clave_direccion for palabra in palabras):
        # Si hay palabras clave de dirección, asumir que todo el texto es la dirección
        return prompt
    
    # Si contiene un número (probable número de casa) y otras palabras
    if re.search(r"\d+", prompt) and len(palabras) > 1:
        return prompt
    
    # Si todo lo demás falla, devolver el texto completo
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

def interpretar_pregunta_llm(prompt: str) -> dict:
    """
    Analiza la pregunta del usuario y extrae tipo de búsqueda, campo y valor.
    Mejorado para detectar mejor las consultas de nombres propios.
    """
    # Verificación rápida de patrones comunes para nombres
    prompt_lower = prompt.lower()
    
    # Patrones para detectar rápidamente consultas de nombre
    patrones_nombre = [
        r"(?:dame|muestra|busca|encuentra|quiero|necesito)?\s+(?:toda)?\s*(?:la)?\s+información\s+(?:de|sobre)\s+([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)",
        r"(?:qué|que)\s+(?:sabes|información\s+tienes)\s+(?:de|sobre)\s+([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)",
        r"busca(?:r|me)?\s+(?:a)?\s+([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)",
        r"encuentra\s+(?:a)?\s+([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)"
    ]
    
    # Verificar patrones de nombre primero (optimización)
    for patron in patrones_nombre:
        match = re.search(patron, prompt_lower)
        if match:
            nombre = match.group(1).strip()
            if len(nombre) > 0 and not nombre.isdigit():
                # Si parece un nombre, retornar directamente
                return {
                    "tipo_busqueda": "nombre",
                    "campo": "nombre_completo",
                    "valor": nombre
                }

    # Si no se detecta un patrón de nombre, continuar con LLM
    system_prompt = (
        "Eres un asistente que analiza preguntas del usuario. Tu tarea es extraer:\n"
        "- 'tipo_busqueda': puede ser 'nombre', 'direccion', 'telefono' o 'atributo'.\n"
        "- 'campo': si aplica, como 'telefono', 'municipio', 'sexo', 'ocupacion', 'clave ife' etc.\n"
        "- 'valor': el dato específico mencionado en la pregunta.\n\n"
        "REGLAS:\n"
        "- Si la pregunta es sobre un número telefónico o pregunta por teléfono, usa tipo_busqueda='telefono', campo='telefono_completo', valor='número mencionado', incluso si el número es corto o incompleto.\n"
        "- Si la pregunta contiene frases como 'de quién es este número', 'quién tiene este teléfono', o incluye la palabra 'teléfono' junto con algún número, siempre asigna tipo_busqueda='telefono'.\n" 
        "- Si solo se proporciona un valor alfanumérico sin especificar campo, usa campo='' (vacío).\n"
        "- Si la pregunta es sobre 'hombres' o 'mujeres', usa tipo_busqueda='atributo', campo='sexo', valor='M' o 'F'.\n"
        "- Si la pregunta es sobre ocupación/profesión, usa tipo_busqueda='atributo', campo='ocupacion', valor='profesión mencionada'.\n"
        "- Si la pregunta es sobre tarjeta, usa tipo_busqueda='atributo', campo='tarjeta', valor='numeros mencionados'.\n"
        "- Si la pregunta contiene nombres propios como 'Juan', 'María', 'González', asigna tipo_busqueda='nombre'.\n"
        f"Pregunta: {prompt}\n"
        "Responde solo con un JSON válido. No agregues explicaciones ni comentarios."
    )
    
    try:
        salida_cruda = llm_clasificador(system_prompt, max_new_tokens=256, return_full_text=False)[0]['generated_text']
        match = re.search(r'\{[\s\S]*?\}', salida_cruda)
        if match:
            json_text = match.group(0)
            resultado = json.loads(json_text)
            
            # Verificar si el valor es None o vacío
            if resultado.get("valor") is None or resultado.get("valor") == "":
                # Para búsquedas de nombre, extraer el valor usando heurísticas
                if resultado.get("tipo_busqueda") == "nombre":
                    # Intentar obtener el nombre de la consulta
                    palabras = prompt.split()
                    # Tomar las últimas 1-3 palabras como posible nombre
                    posible_nombre = " ".join(palabras[-min(3, len(palabras)):])
                    resultado["valor"] = posible_nombre
                else:
                    # Valor por defecto para evitar errores
                    resultado["valor"] = prompt
            
            return resultado
        else:
            print("[⚠️ LLM] No se detectó JSON válido.")
    except Exception as e:
        print(f"[⚠️ LLM] Error al decodificar JSON: {e}")

    # En caso de error, intentar detectar nombres propios como último recurso
    palabras = prompt.split()
    for palabra in palabras:
        if palabra[0].isupper() and len(palabra) > 2 and palabra.isalpha():
            return {"tipo_busqueda": "nombre", "campo": "nombre_completo", "valor": palabra}
    
    # Fallback final: devolver el prompt completo como valor
    return {"tipo_busqueda": "desconocido", "valor": prompt}

# --- 3) HERRAMIENTA 1: BUSCAR POR NOMBRE COMPLETO ---
def buscar_nombre(query: str) -> str:
    """
    Busca coincidencias de nombres completos o parciales y las retorna ordenadas por relevancia.
    Permite búsquedas por cualquier combinación de nombre/apellidos, en cualquier orden.
    """
    print(f"Ejecutando búsqueda de nombre: '{query}'")
    
    # Normalizar la consulta
    query = query.strip()
    query_norm = normalizar_texto(query)
    query_tokens = set(query_norm.split())
    
    # Estructura para almacenar resultados por categoría
    resultados_por_categoria = {
        "exactos": [],          # Coincidencia exacta o casi exacta 
        "completos": [],        # Todos los tokens de búsqueda están presentes
        "parciales_alta": [],   # Coincidencia significativa (múltiples tokens o apellido completo)
        "parciales_media": [],  # Coincidencia parcial básica
        "posibles": []          # Coincidencias de baja confianza pero potencialmente útiles
    }
    
    # Registros ya encontrados para evitar duplicados
    registros_encontrados = set()
    
    # Recorrer todos los índices
    for fuente, index in indices.items():
        try:
            # Usar búsqueda semántica para obtener candidatos iniciales
            retriever = VectorIndexRetriever(index=index, similarity_top_k=8)  # Aumentar para más candidatos
            nodes = retriever.retrieve(query)
            
            # Procesar cada nodo encontrado
            for node in nodes:
                metadata = node.node.metadata
                
                # Crear identificador único para este registro
                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                if not id_registro:
                    id_registro = node.node.node_id
                
                # Si ya procesamos este registro, saltarlo
                if id_registro in registros_encontrados:
                    continue
                
                # Obtener nombre completo de metadatos (probar diferentes campos)
                nombre_completo = (
                    metadata.get("nombre_completo", "") or 
                    metadata.get("nombre completo", "") or 
                    metadata.get("nombre", "")
                ).strip()
                
                if not nombre_completo:
                    continue  # Sin nombre, no podemos comparar
                
                # Normalizar el nombre para comparación
                nombre_norm = normalizar_texto(nombre_completo)
                
                # Dividir el nombre en tokens individuales
                nombre_tokens = nombre_norm.split()
                nombre_tokens_set = set(nombre_tokens)
                
                # Evaluar la coincidencia
                
                # 1. Detectar si hay coincidencia exacta (mismo nombre)
                sim_texto = similitud(query_norm, nombre_norm)
                
                # 2. Evaluar si todos los tokens de la consulta están en el nombre
                tokens_coincidentes = query_tokens.intersection(nombre_tokens_set)
                ratio_consulta = len(tokens_coincidentes) / len(query_tokens) if query_tokens else 0
                
                # 3. Evaluar qué porcentaje del nombre coincide con la consulta
                ratio_nombre = len(tokens_coincidentes) / len(nombre_tokens) if nombre_tokens else 0
                
                # 4. Verificar si hay coincidencia de apellidos
                # Suponemos que los apellidos son las últimas 1-2 palabras del nombre
                apellidos_nombre = set(nombre_tokens[-min(2, len(nombre_tokens)):])
                apellidos_query = set()
                if len(query_tokens) >= 2:
                    # Si la consulta tiene al menos 2 palabras, consideramos posibles apellidos
                    apellidos_query = set(list(query_tokens)[-min(2, len(query_tokens)):])
                coincidencia_apellidos = len(apellidos_nombre.intersection(apellidos_query))
                
                # 5. Verificar coincidencia de nombre de pila (primera palabra)
                nombre_pila_coincide = False
                if nombre_tokens and query_tokens:
                    nombre_pila = nombre_tokens[0]
                    if nombre_pila in query_tokens:
                        nombre_pila_coincide = True
                
                # Construir el resumen del registro
                resumen = [f"{k}: {v}" for k, v in metadata.items() 
                          if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                texto_resultado = f"Encontrado en {fuente}:\n" + "\n".join(resumen)
                
                # Clasificación de resultados basada en la calidad de coincidencia
                
                # Coincidencia exacta o casi exacta
                if sim_texto > 0.9 or (ratio_consulta > 0.9 and ratio_nombre > 0.9):
                    resultados_por_categoria["exactos"].append({
                        "texto": f"Coincidencia exacta: {texto_resultado}",
                        "score": sim_texto + 0.1,  # Bonificación para exactos
                        "fuente": fuente
                    })
                
                # Todos los tokens de la consulta están en el nombre
                elif ratio_consulta > 0.95:
                    resultados_por_categoria["completos"].append({
                        "texto": f"Coincidencia completa: {texto_resultado}",
                        "score": ratio_consulta * 0.9 + ratio_nombre * 0.1,
                        "fuente": fuente
                    })
                
                # Coincidencia de apellidos significativa (al menos un apellido completo)
                elif coincidencia_apellidos > 0 and ratio_consulta >= 0.5:
                    resultados_por_categoria["parciales_alta"].append({
                        "texto": f"Coincidencia parcial: {texto_resultado}",
                        "score": 0.7 + (coincidencia_apellidos * 0.15) + (ratio_consulta * 0.15),
                        "fuente": fuente
                    })
                
                # Coincidencia de nombre de pila y tokens significativos
                elif nombre_pila_coincide and len(tokens_coincidentes) >= 1:
                    resultados_por_categoria["parciales_alta"].append({
                        "texto": f"Coincidencia parcial: {texto_resultado}",
                        "score": 0.65 + (ratio_consulta * 0.35),
                        "fuente": fuente
                    })
                
                # Coincidencia parcial básica (al menos un token importante)
                elif len(tokens_coincidentes) >= 1 and any(token in nombre_tokens_set for token in query_tokens):
                    resultados_por_categoria["parciales_media"].append({
                        "texto": f"Coincidencia parcial: {texto_resultado}",
                        "score": 0.4 + (ratio_consulta * 0.6),
                        "fuente": fuente
                    })
                
                # Coincidencias de baja confianza pero potencialmente útiles
                elif tokens_coincidentes and sim_texto > 0.3:
                    resultados_por_categoria["posibles"].append({
                        "texto": f"Posible coincidencia: {texto_resultado}",
                        "score": sim_texto,
                        "fuente": fuente
                    })
                
                # Marcar como procesado
                registros_encontrados.add(id_registro)
                
            # Búsqueda adicional para apellidos específicos (si la consulta es corta)
            if len(query_tokens) <= 2 and len(query.strip()) > 3:
                # Intentar una segunda estrategia de búsqueda directa en los datos
                for node_id, doc in index.docstore.docs.items():
                    metadata = doc.metadata
                    if not metadata or "nombre" not in metadata and "nombre_completo" not in metadata:
                        continue
                    
                    id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                    if not id_registro:
                        id_registro = node_id
                    
                    # Si ya lo procesamos, saltar
                    if id_registro in registros_encontrados:
                        continue
                    
                    nombre_completo = (
                        metadata.get("nombre_completo", "") or 
                        metadata.get("nombre completo", "") or 
                        metadata.get("nombre", "")
                    ).strip()
                    
                    if not nombre_completo:
                        continue
                    
                    nombre_norm = normalizar_texto(nombre_completo)
                    
                    # Buscar directamente la aparición de la consulta como substring
                    if query_norm in nombre_norm:
                        resumen = [f"{k}: {v}" for k, v in metadata.items() 
                                if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                        texto_resultado = f"Encontrado en {fuente}:\n" + "\n".join(resumen)
                        
                        # Determinar el tipo de coincidencia basado en posición
                        if nombre_norm.startswith(query_norm + " "):
                            # Coincide con el nombre de pila
                            resultados_por_categoria["parciales_alta"].append({
                                "texto": f"Coincidencia parcial: {texto_resultado}",
                                "score": 0.75,
                                "fuente": fuente
                            })
                        elif nombre_norm.endswith(" " + query_norm):
                            # Coincide con el último apellido
                            resultados_por_categoria["parciales_alta"].append({
                                "texto": f"Coincidencia parcial: {texto_resultado}",
                                "score": 0.8,
                                "fuente": fuente
                            })
                        elif " " + query_norm + " " in nombre_norm:
                            # Coincide con una palabra interna (apellido o segundo nombre)
                            resultados_por_categoria["parciales_alta"].append({
                                "texto": f"Coincidencia parcial: {texto_resultado}",
                                "score": 0.7,
                                "fuente": fuente
                            })
                        else:
                            # Otra coincidencia de substring
                            resultados_por_categoria["parciales_media"].append({
                                "texto": f"Coincidencia parcial: {texto_resultado}",
                                "score": 0.5,
                                "fuente": fuente
                            })
                        
                        registros_encontrados.add(id_registro)
        
        except Exception as e:
            print(f"Error al buscar en índice {fuente}: {e}")
            continue
    
    # Compilar respuesta final, priorizando por categoría y luego por score
    todas_respuestas = []

    # Ver si hay exactos
    mostrar_solo_exactos = bool(resultados_por_categoria["exactos"])

    # Exactos
    if mostrar_solo_exactos:
        resultados_ordenados = sorted(resultados_por_categoria["exactos"], key=lambda x: x["score"], reverse=True)
        todas_respuestas.append("🔍 COINCIDENCIAS EXACTAS:")
        for res in resultados_ordenados:
            todas_respuestas.append(res["texto"])

    # Solo si NO hay exactos, mostrar lo demás
    if not mostrar_solo_exactos:
        if resultados_por_categoria["completos"]:
            resultados_ordenados = sorted(resultados_por_categoria["completos"], key=lambda x: x["score"], reverse=True)
            todas_respuestas.append("\n🔍 COINCIDENCIAS COMPLETAS:")
            for res in resultados_ordenados:
                todas_respuestas.append(res["texto"])

        if resultados_por_categoria["parciales_alta"]:
            resultados_ordenados = sorted(resultados_por_categoria["parciales_alta"], key=lambda x: x["score"], reverse=True)
            todas_respuestas.append("\n🔍 COINCIDENCIAS PARCIALES SIGNIFICATIVAS:")
            for res in resultados_ordenados:
                todas_respuestas.append(res["texto"])

        if resultados_por_categoria["parciales_media"]:
            resultados_ordenados = sorted(resultados_por_categoria["parciales_media"], key=lambda x: x["score"], reverse=True)
            todas_respuestas.append("\n🔍 COINCIDENCIAS PARCIALES:")
            for res in resultados_ordenados:
                todas_respuestas.append(res["texto"])

        if len(todas_respuestas) < 3 and resultados_por_categoria["posibles"]:
            resultados_ordenados = sorted(resultados_por_categoria["posibles"], key=lambda x: x["score"], reverse=True)
            todas_respuestas.append("\n🔍 POSIBLES COINCIDENCIAS (baja confianza):")
            for res in resultados_ordenados:
                todas_respuestas.append(res["texto"])

    # Si no hay nada de nada
    if not todas_respuestas:
        return f"No se encontraron coincidencias para '{query}' en ninguna fuente."

    # Componer respuesta final
    return "\n\n".join(todas_respuestas)

    
busqueda_global_tool = FunctionTool.from_defaults(
    fn=buscar_nombre,
    name="buscar_nombre",
    description=(
        "Usa esta herramienta para encontrar información completa de una persona en todas las bases, "
        "cuando el usuario da el nombre completo o parcial. Por ejemplo: 'Dame la información de Juan', "
        "'¿Qué sabes de Pérez?', 'Busca González', 'Encuentra a María Pérez'."
    )
)
all_tools.insert(0, busqueda_global_tool)

# --- 4) HERRAMIENTA 2: BUSCAR PERSONAS POR ATRIBUTO ---
def buscar_atributo(campo: str, valor: str, carpeta_indices: str) -> str:
    """
    Busca coincidencias por campo y valor en todos los índices.
    Versión mejorada con capacidad de búsqueda flexible y manejo de categorías.
    
    Características:
    - Búsqueda exacta con filtros de metadatos cuando campo es específico
    - Búsqueda en todos los campos cuando no se especifica campo o cuando no hay resultados
    - Manejo especial para categorías como sexo y ocupación
    - Normalización de valores para mejorar coincidencias
    """
    print(f"\nBuscando registros donde '{campo}' = '{valor}'\n")
    
    # Normalizar campo y valor
    campo_normalizado = normalizar_texto(campo) if campo else ""
    valor_normalizado = normalizar_texto(valor)
    
    # Casos especiales de normalización
    if campo_normalizado in ["sexo", "genero"]:
        if valor_normalizado in ["hombre", "hombres", "masculino", "varon", "varones", "m"]:
            valor_normalizado = "m"
        elif valor_normalizado in ["mujer", "mujeres", "femenino", "f"]:
            valor_normalizado = "f"
    
    # Preparar para resultados
    resultados = []
    registros_encontrados = set()  # Para evitar duplicados
    
    # Determinar si estamos buscando categorías específicas
    busqueda_categorica = campo_normalizado in ["sexo", "genero", "ocupacion", "profesion"]
    
    # Obtener campo final mapeado (para normalización de nombres de campo)
    campo_final = mapa_campos.get(campo_normalizado, campo_normalizado) if campo_normalizado else ""
    
    # Fase 1: Búsqueda exacta por filtros si tenemos un campo específico
    if campo_final and not busqueda_categorica:
        for nombre_dir in os.listdir(carpeta_indices):
            ruta_indice = os.path.join(carpeta_indices, nombre_dir)
            if not os.path.isdir(ruta_indice) or not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
                continue

            fuente = nombre_dir.replace("index_", "")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
                index = load_index_from_storage(storage_context)

                # Intentar búsqueda exacta con filtro primero
                try:
                    filters = MetadataFilters(filters=[
                        ExactMatchFilter(key=campo_final, value=valor_normalizado)
                    ])
                    retriever = VectorIndexRetriever(index=index, similarity_top_k=5, filters=filters)
                    nodes = retriever.retrieve(f"{campo_final} es {valor}")

                    if nodes:
                        for node in nodes:
                            metadata = node.node.metadata
                            id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                            if not id_registro:
                                id_registro = node.node.node_id
                                
                            if id_registro in registros_encontrados:
                                continue
                                
                            resumen = [f"{k}: {v}" for k, v in metadata.items() 
                                      if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                            resultados.append(f"Coincidencia exacta en {fuente}:\n" + "\n".join(resumen))
                            registros_encontrados.add(id_registro)
                except Exception as e_filter:
                    print(f"Error en filtro exacto para {fuente}: {e_filter}")
                    # Continuar si el filtro falla (campo no existente, etc.)
                    pass

            except Exception as e:
                print(f"Error al cargar índice {fuente}: {e}")
                continue
    
    # Fase 2: Si es búsqueda categórica o no tenemos resultados exactos, buscar en todos los documentos
    if busqueda_categorica or (not resultados and valor):
        print(f"Realizando búsqueda exhaustiva para '{valor_normalizado}'...")
        
        # Definir campos de búsqueda
        campos_a_buscar = []
        if campo_final:
            # Si tenemos un campo específico, buscar sus variantes
            if campo_final in campos_clave:
                campos_a_buscar = [normalizar_texto(c) for c in campos_clave[campo_final]]
            else:
                campos_a_buscar = [campo_final]
        
        # Recorrer todos los índices
        for fuente, index in indices.items():
            try:
                # Para cada documento en el índice
                for node_id, doc in index.docstore.docs.items():
                    metadata = doc.metadata
                    if not metadata:
                        continue
                    
                    # Crear ID único para el registro
                    id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                    if not id_registro:
                        id_registro = node_id
                    
                    # Evitar duplicados
                    if id_registro in registros_encontrados:
                        continue
                    
                    encontrado = False
                    coincidencia_campo = None
                    
                    # Si tenemos campos específicos a buscar
                    if campos_a_buscar:
                        for k, v in metadata.items():
                            k_norm = normalizar_texto(k)
                            
                            # Verificar si este campo nos interesa
                            if k_norm in campos_a_buscar:
                                v_str = str(v).strip().lower()
                                v_norm = normalizar_texto(v_str)
                                
                                # Verificar coincidencia exacta o parcial según el caso
                                if v_norm == valor_normalizado or (
                                    len(valor_normalizado) > 4 and (
                                        valor_normalizado in v_norm or 
                                        v_norm in valor_normalizado
                                    )
                                ):
                                    encontrado = True
                                    coincidencia_campo = k
                                    break
                    
                    # Si no encontramos en campos específicos o no los tenemos, 
                    # buscar en todos los campos si el valor es significativo
                    if not encontrado and len(valor_normalizado) >= 4:
                        for k, v in metadata.items():
                            if k in ['fuente', 'archivo', 'fila_excel']:
                                continue
                                
                            v_str = str(v).strip().lower()
                            v_norm = normalizar_texto(v_str)
                            
                            # Búsqueda exacta o como substring si es valor largo
                            if v_norm == valor_normalizado or (
                                len(valor_normalizado) > 6 and 
                                valor_normalizado in v_norm
                            ):
                                encontrado = True
                                coincidencia_campo = k
                                break
                    
                    # Si encontramos coincidencia, agregar a resultados
                    if encontrado:
                        tipo_coincidencia = "exacta" if coincidencia_campo else "en múltiples campos"
                        campo_texto = f" en campo '{coincidencia_campo}'" if coincidencia_campo else ""
                        
                        resumen = [f"{k}: {v}" for k, v in metadata.items() 
                                   if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                        resultados.append(f"Coincidencia {tipo_coincidencia}{campo_texto} en {fuente}:\n" + "\n".join(resumen))
                        registros_encontrados.add(id_registro)
                        
            except Exception as e:
                print(f"Error al buscar en índice {fuente}: {e}")
                continue
    
    # Formatear respuesta final
    if resultados:
        total_registros = len(resultados)
        num_mostrados = total_registros # Limitar a 15 resultados máximo
        
        # Mensaje introductorio según el tipo de búsqueda
        if campo:
            mensaje_intro = f"Se encontraron {total_registros} registros para {campo}='{valor}'."
        else:
            mensaje_intro = f"Se encontraron {total_registros} registros que contienen '{valor}'."
        
        # Agregar nota si estamos limitando resultados
        if total_registros > num_mostrados:
            mensaje_intro += f" Mostrando {num_mostrados} primeros resultados:"
        
        return mensaje_intro + "\n\n" + "\n\n".join(resultados[:num_mostrados])
    else:
        if campo:
            return f"No se encontraron coincidencias para '{campo}: {valor}'."
        else:
            return f"No se encontraron coincidencias para el valor '{valor}'."

# Actualizar la descripción de la herramienta buscar_por_atributo_tool
buscar_por_atributo_tool = FunctionTool.from_defaults(
    fn=lambda campo, valor: buscar_atributo(campo, valor, carpeta_indices=ruta_indices),
    name="buscar_atributo",
    description=(
        "Usa esta herramienta cuando el usuario busca por cualquier atributo específico o valor. "
        "Funciona tanto si se especifica el campo ('¿Quién tiene la clave IFE ABCDE?') "
        "como si solo se da un valor sin campo ('A quién pertenece ABCDE?'). "
        "También maneja categorías como sexo ('hombres', 'mujeres') y ocupación ('ingeniero', 'médico'). "
        "Por ejemplo: '¿Quién tiene el número 5544332211?', '¿A quién pertenece la CURP ABCD123?', "
        "'¿Qué personas viven en Querétaro?', 'Muestra a todas las mujeres', 'Busca ingenieros'."
    )
)
all_tools.insert(1, buscar_por_atributo_tool)

# --- 5) HERRAMIENTA 3: BUSCAR POR DIRECCION COMPLETA ---

def buscar_direccion_combinada(texto_direccion: str) -> str:
    """
    Busca coincidencias de dirección combinando búsqueda exacta por metadatos
    y búsqueda semántica con evaluación de componentes.

    Prioriza coincidencias exactas de "calle número", pero también busca
    coincidencias semánticas y evalúa relevancia basada en componentes
    y similitud numérica/textual en todos los índices.
    """
    print(f"\nBuscando dirección combinada: '{texto_direccion}'")

    # 1. Preprocesamiento y Extracción de Componentes
    texto_direccion_normalizado = normalizar_texto(texto_direccion)
    texto_direccion_normalizado = re.sub(r'([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])(\d+)', r'\1 \2', texto_direccion_normalizado)
    texto_direccion_normalizado = re.sub(r'(\d+)([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])', r'\1 \2', texto_direccion_normalizado)

    componentes_raw = re.split(r'[,\s]+', texto_direccion_normalizado)
    componentes = [c for c in componentes_raw if c and len(c) > 1]

    numeros_busqueda = [comp for comp in componentes if comp.isdigit()]
    calles_colonias_busqueda = [comp for comp in componentes if not comp.isdigit() and comp not in STOP_WORDS]
    componentes_clave = [comp for comp in componentes if comp not in STOP_WORDS]

    combinacion_principal = None
    combinacion_principal_norm = None
    if calles_colonias_busqueda and numeros_busqueda:
        combinacion_principal = f"{calles_colonias_busqueda[0]} {numeros_busqueda[0]}"
        combinacion_principal_norm = normalizar_texto(combinacion_principal)

    # 2. Almacenamiento de Resultados
    todos_resultados_detalle: Dict[str, Dict[str, Any]] = {}

    # 3. Búsqueda Iterativa en Todos los Índices
    # (Usamos los índices ya cargados en la variable global `indices`)
    for fuente, index in indices.items(): # Itera sobre los índices ya cargados
        try:
            # --- Estrategia 1: Búsqueda Exacta por Metadatos ---
            if combinacion_principal_norm:
                #print(f"[DEBUG] Intentando búsqueda exacta por metadatos para '{combinacion_principal_norm}' en {fuente}...")
                for campo_exacto in CAMPOS_BUSQUEDA_EXACTA:
                    campo_norm = normalizar_texto(campo_exacto)
                    try:
                        filters = MetadataFilters(filters=[
                            ExactMatchFilter(key=campo_norm, value=combinacion_principal_norm)
                        ])
                        retriever_exacto = VectorIndexRetriever(
                            index=index,
                            similarity_top_k=5,
                            filters=filters
                        )
                        nodes_exactos = retriever_exacto.retrieve(combinacion_principal) # Query simple, el filtro manda

                        if nodes_exactos:
                            #print(f"[DEBUG] Éxito en búsqueda exacta en campo '{campo_exacto}'. {len(nodes_exactos)} nodos.")
                            for node in nodes_exactos:
                                metadata = node.node.metadata
                                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                                if not id_registro: id_registro = node.node.node_id

                                puntaje_actual = todos_resultados_detalle.get(id_registro, {}).get('puntaje', -1.0)
                                puntaje_nuevo = 1.0 # Máxima prioridad

                                if puntaje_nuevo > puntaje_actual:
                                    resumen = [f"{k}: {v}" for k, v in metadata.items()
                                               if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                                    todos_resultados_detalle[id_registro] = {
                                        'texto_base': f"Coincidencia exacta directa en {fuente}:\n" + "\n".join(resumen),
                                        'puntaje': puntaje_nuevo,
                                        'fuente': fuente,
                                        'id': id_registro,
                                        'metadata': metadata,
                                        'tipo': 'exacta_directa'
                                    }


                    except Exception as e_filter:
                        # Silencioso si el campo no existe en los metadatos de este índice
                        if 'Metadata key' not in str(e_filter):
                            print(f"[WARN] Error en búsqueda exacta por filtro en campo '{campo_exacto}' en {fuente}: {e_filter}")

            # --- Estrategia 2: Búsqueda Semántica y Evaluación ---
            #print(f"[DEBUG] Realizando búsqueda semántica general en {fuente}...")
            top_k_dinamico = min(10000, len(index.docstore.docs))
            retriever_semantico = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k_dinamico
            )
            consulta_semantica = " ".join(componentes_clave)
            if not consulta_semantica: consulta_semantica = texto_direccion_normalizado

            nodes_semanticos = retriever_semantico.retrieve(consulta_semantica)
            #print(f"[DEBUG] Búsqueda semántica encontró {len(nodes_semanticos)} nodos iniciales.")

            for node in nodes_semanticos:
                metadata = node.node.metadata
                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                if not id_registro: id_registro = node.node.node_id

                puntaje_actual = todos_resultados_detalle.get(id_registro, {}).get('puntaje', -1.0)

                if puntaje_actual == 1.0: # Ya es perfecta, no evaluar más
                    continue

                texto_completo_registro = ""
                # Usar CAMPOS_DIRECCION definido globalmente
                for k, v in metadata.items():
                    if k in CAMPOS_DIRECCION:
                        texto_completo_registro += f" {v}"
                texto_completo_registro_norm = normalizar_texto(texto_completo_registro)

                if not texto_completo_registro_norm: # Si no hay campos de dirección, saltar
                    continue

                # Evaluación de Relevancia
                tiene_numero_exacto = False
                tiene_numero_cercano = False
                numeros_en_registro = re.findall(r'\b\d+\b', texto_completo_registro_norm)

                if numeros_busqueda:
                    num_b_str = numeros_busqueda[0] # Considerar el primer número
                    if num_b_str in numeros_en_registro:
                        tiene_numero_exacto = True
                    else:
                        try:
                            num_b_int = int(num_b_str)
                            for num_r_str in numeros_en_registro:
                                try:
                                    num_r_int = int(num_r_str)
                                    # Usar constante TOLERANCIA_NUMERO_CERCANO
                                    if abs(num_b_int - num_r_int) <= TOLERANCIA_NUMERO_CERCANO:
                                        tiene_numero_cercano = True
                                        break
                                except ValueError: continue
                        except ValueError: pass
                else: # Si no se busca número, no penalizar
                    tiene_numero_exacto = True
                    tiene_numero_cercano = True

                # Componentes clave
                componentes_clave_encontrados = sum(1 for comp in componentes_clave if comp in texto_completo_registro_norm)
                porcentaje_clave = componentes_clave_encontrados / len(componentes_clave) if componentes_clave else 1.0

                # Todos los componentes
                componentes_encontrados = sum(1 for comp in componentes if comp in texto_completo_registro_norm)
                calificacion_componentes = componentes_encontrados / len(componentes) if componentes else 1.0

                # Similitud textual
                similitud_textual = similitud(texto_direccion_normalizado, texto_completo_registro_norm)

                # Calcular Score Final
                score_final = 0.0
                # Pesos ajustados para dar más importancia al número y componentes clave
                peso_num_exacto = 0.50
                peso_num_cercano = 0.20
                peso_clave = 0.35 # Aumentado
                peso_componentes = 0.05 # Disminuido
                peso_similitud = 0.10

                if tiene_numero_exacto:
                    score_final = (peso_num_exacto * 1.0) + (peso_clave * porcentaje_clave) + \
                                  (peso_componentes * calificacion_componentes) + (peso_similitud * similitud_textual)
                    # Boost adicional si la primera calle/colonia coincide
                    if calles_colonias_busqueda and calles_colonias_busqueda[0] in texto_completo_registro_norm:
                        score_final = min(0.99, score_final * 1.1) # Pequeño boost sin llegar a 1.0
                elif tiene_numero_cercano:
                    score_final = (peso_num_cercano * 1.0) + (peso_clave * porcentaje_clave) + \
                                  (peso_componentes * calificacion_componentes) + (peso_similitud * similitud_textual)
                    score_final *= 0.85 # Penalizar un poco
                else: # Sin número o muy diferente
                    if porcentaje_clave > 0.6: # Solo si hay buena coincidencia de texto
                        score_final = (peso_clave * porcentaje_clave) + \
                                      (peso_componentes * calificacion_componentes) + (peso_similitud * similitud_textual)
                        score_final *= 0.65 # Penalizar más
                    else:
                        score_final = 0.0 # Relevancia muy baja

                # Filtrado Estricto
                if numeros_busqueda and not (tiene_numero_exacto or tiene_numero_cercano):
                    #print(f"[DEBUG] Descartado {id_registro} (semántico): Número buscado no encontrado.")
                    continue
                if porcentaje_clave < 0.4 and not tiene_numero_exacto: # Umbral bajo de componentes clave
                    #print(f"[DEBUG] Descartado {id_registro} (semántico): Pocos componentes clave ({porcentaje_clave:.2f}).")
                    continue
                # Usar constante UMBRAL_PUNTAJE_MINIMO
                if score_final < (UMBRAL_PUNTAJE_MINIMO - 0.1): # Un poco más permisivo aquí
                     #print(f"[DEBUG] Descartado {id_registro} (semántico): Score bajo ({score_final:.2f}).")
                     continue

                # Almacenar si es Mejor que el Existente
                if score_final > puntaje_actual:
                    tipo_resultado = "exacta_semantica" if tiene_numero_exacto and porcentaje_clave >= 0.8 else \
                                     "cercana_semantica" if tiene_numero_cercano else \
                                     "similar_semantica"

                    resumen = [f"{k}: {v}" for k, v in metadata.items()
                               if k not in ['fuente', 'archivo', 'fila_excel'] and v]

                    texto_display = f"Coincidencia ({tipo_resultado.replace('_', ' ')}) en {fuente} (Score: {score_final:.2f}):\n" + "\n".join(resumen)

                    todos_resultados_detalle[id_registro] = {
                        'texto_base': texto_display,
                        'puntaje': score_final,
                        'fuente': fuente,
                        'id': id_registro,
                        'metadata': metadata,
                        'tipo': tipo_resultado
                    }

        except Exception as e_index:
            print(f"[ERROR] Error procesando el índice {fuente}: {e_index}")
            # import traceback # Descomentar para depuración profunda
            # traceback.print_exc()
            continue

    # 4. Consolidación y Formateo Final
    if not todos_resultados_detalle:
        return f"No se encontraron coincidencias relevantes para la dirección '{texto_direccion}'."

    resultados_ordenados = sorted(todos_resultados_detalle.values(), key=lambda x: x['puntaje'], reverse=True)

    # Usar constante UMBRAL_PUNTAJE_MINIMO
    resultados_filtrados = [res for res in resultados_ordenados if res['puntaje'] >= UMBRAL_PUNTAJE_MINIMO]

    # Si el filtro estricto no dejó nada, mostrar los mejores aunque no lleguen al umbral
    if not resultados_filtrados and resultados_ordenados:
        resultados_finales = resultados_ordenados # Mostrar los 2 mejores
        mensaje_intro = "No se encontraron coincidencias muy relevantes. Mostrando los más cercanos:\n\n"
    elif not resultados_filtrados and not resultados_ordenados:
         return f"No se encontraron coincidencias relevantes para la dirección '{texto_direccion}'."
    else:
        # Usar constante MAX_RESULTADOS_FINALES
        resultados_finales = resultados_filtrados
        tipos_encontrados = {res['tipo'] for res in resultados_finales}
        if 'exacta_directa' in tipos_encontrados or 'exacta_semantica' in tipos_encontrados:
             mensaje_intro = "Se encontraron las siguientes coincidencias:\n\n"
        elif 'cercana_semantica' in tipos_encontrados:
             mensaje_intro = "No se encontraron coincidencias exactas. Mostrando direcciones con números/componentes similares:\n\n"
        else:
             mensaje_intro = "No se encontraron coincidencias muy precisas. Mostrando los resultados más similares:\n\n"


    textos_resultados = []
    for res in resultados_finales:
        texto_limpio = re.sub(r'\s*\(Score: \d+\.\d+\)', '', res['texto_base']).strip() # Limpiar score
        # Podrías añadir aquí lógica para reordenar los campos del resumen si quisieras
        textos_resultados.append(texto_limpio)

    return mensaje_intro + "\n\n".join(textos_resultados) # Separador más claro

# Crear herramienta para búsqueda de dirección combinada
buscar_direccion_tool = FunctionTool.from_defaults(
    fn=buscar_direccion_combinada, # Asegúrate que apunta a la nueva función
    name="buscar_direccion_combinada",
    description=(
        "Usa esta herramienta cuando el usuario busca una dirección completa o parcial que contenga calle, número y posiblemente colonia o ciudad. "
        "Es especialmente útil para direcciones combinadas como 'ZOQUIPAN 1260, LAGOS DEL COUNTRY'. "
        "Por ejemplo: '¿De quién es esta dirección: ZOQUIPAN 1260, LAGOS DEL COUNTRY, ZAPOPAN?', "
        "'Busca Malva 101, San Luis de la Paz', 'Quién vive en casa #63, colinas del rey, Zapopan', 'información de zoquipan 1260'. "
        "Esta herramienta realiza búsquedas semánticas y exactas en componentes de dirección."
    )
)

# Insertar la herramienta al inicio de la lista para darle prioridad
all_tools.insert(3, buscar_direccion_tool)

# --- 6) HERRAMIENTA 4: BUSCAR POR NUMERO TELEFONICO ---
def buscar_numero_telefono(valor: str) -> str:
    """
    Búsqueda tolerante para teléfonos, lada o combinaciones incompletas.
    Evita duplicados entre campos como 'telefono', 'lada' y 'telefono_completo'.
    """
    campos_telefono = ["telefono_completo", "telefono", "lada"]
    valor_norm = normalizar_texto(valor)
    resultados = {}
    
    for fuente, index in indices.items():
        for campo in campos_telefono:
            try:
                for node_id, doc in index.docstore.docs.items():
                    metadata = doc.metadata
                    if not metadata or campo not in metadata:
                        continue

                    valor_campo = str(metadata.get(campo, "")).strip()
                    if not valor_campo:
                        continue

                    valor_campo_norm = normalizar_texto(valor_campo)

                    if valor_norm in valor_campo_norm or valor_campo_norm.endswith(valor_norm) or valor_campo_norm.startswith(valor_norm):
                        score = 1.0 if valor_norm == valor_campo_norm else \
                                0.9 if valor_campo_norm.endswith(valor_norm) else \
                                0.8 if valor_campo_norm.startswith(valor_norm) else \
                                0.5
                        
                        id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", "")) + str(metadata.get("nombre_completo", "")).strip().lower()
                        if not id_registro:
                            id_registro = doc.node.node_id
                        
                        # Solo guardamos el mejor resultado por registro
                        if id_registro not in resultados or resultados[id_registro]['score'] < score:
                            resumen = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                            resultados[id_registro] = {
                                'score': score,
                                'fuente': fuente,
                                'campo': campo,
                                'texto': f"Coincidencia en {fuente} (campo '{campo}'):\n" + "\n".join(resumen)
                            }
            except Exception as e:
                print(f"[WARN] Error revisando {fuente} campo {campo}: {e}")
                continue

    if not resultados:
        return f"No se encontraron coincidencias relevantes para el número '{valor}'."

    resultados_ordenados = sorted(resultados.values(), key=lambda x: -x['score'])
    return "Se encontraron las siguientes coincidencias para número telefónico:\n\n" + "\n\n".join([r['texto'] for r in resultados_ordenados])

buscar_telefono_tool = FunctionTool.from_defaults(
    fn=buscar_numero_telefono,
    name="buscar_numero_telefono",
    description=(
        "Usa esta herramienta cuando el campo detectado sea 'telefono_completo' y el usuario consulta por un número telefónico completo. "
        "Ejemplo: '¿Quién tiene el número 5544332211?', pero NO para lada o partes de teléfonos."
    )
)

all_tools.insert(4, buscar_telefono_tool)


# --- 7) CREAR Y EJECUTAR EL AGENTE ---

# CREAR EL AGENTE REACT
try:
    agent = ReActAgent.from_tools(
        tools=all_tools,
        llm=llm,
        verbose=False
    )
    print("Agente creado correctamente.")
except Exception as e:
    print(f"Error al crear el agente: {e}")
    exit()

print("\n🤖 Agente listo. Escribe tu pregunta o 'salir' para terminar.")

while True:
    prompt = input("Pregunta: ")
    if prompt.lower() == 'salir':
        break
    if not prompt:
        continue

    try:
        # Analizar la consulta
        analisis = interpretar_pregunta_llm(prompt)
        tipo = analisis.get("tipo_busqueda")
        campo = analisis.get("campo")
        valor = analisis.get("valor")
        
        print(f"[INFO] Análisis: tipo={tipo}, campo={campo}, valor={valor}")
        
        # Verificar que valor no sea None antes de continuar
        if valor is None:
            print("[ERROR] No se pudo extraer un valor de la consulta. Usando texto completo.")
            valor = prompt

        # Procesar según el tipo de consulta
        if tipo == "direccion":
            respuesta_herramienta = buscar_direccion_combinada(valor)

        elif tipo == "telefono" and campo == "telefono_completo" and valor:
            print(f"[LLM] Búsqueda telefónica para valor: {valor}")
            respuesta_herramienta = buscar_numero_telefono(valor)

        elif tipo == "atributo" and campo and valor:
            print(f"[LLM] Campo detectado: {campo} = {valor}")
            respuesta_herramienta = buscar_atributo(campo, valor, carpeta_indices=ruta_indices)

        elif tipo == "nombre" and valor:
            print(f"[LLM] Búsqueda de nombre: {valor}")
            respuesta_herramienta = buscar_nombre(valor)

        else:
            print(f"[LLM] Sin análisis claro, intentando fallback con búsqueda de nombre")
            # Primero intentar búsqueda por nombre (más común y útil como fallback)
            respuesta_herramienta = buscar_nombre(valor)
            
            # Si no hay resultados, intentar búsqueda por campos
            if "No se encontraron coincidencias" in respuesta_herramienta:
                print(f"[LLM] Sin resultados de nombre, probando campos inteligentes")
                campos_disponibles = list(campos_detectados)
                campos_probables = sugerir_campos(valor, campos_disponibles)
                respuesta_herramienta = buscar_campos_inteligente(valor, carpeta_indices=ruta_indices, campos_ordenados=campos_probables)

        print(f"\n📄Resultado:\n{respuesta_herramienta}\n")

    except Exception as e:
        print(f"❌ Ocurrió un error durante la ejecución del agente: {e}")
        import traceback
        traceback.print_exc()

        try:
            # Intento de recuperación usando el agente React
            print("Intentando recuperación con agente React...")
            respuesta_agente = agent.query(prompt)
            print(f"\n📄Resultado (procesado por agente fallback):\n{respuesta_agente}\n")
        except Exception as e2:
            print(f"❌ También falló el agente fallback: {e2}")

# --- LIMPIEZA ---
del llm, embed_model, agent, all_tools, indices
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()
print("\n👋 ¡Hasta luego!")