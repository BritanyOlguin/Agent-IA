import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.retrievers import VectorIndexRetriever
from difflib import SequenceMatcher
from typing import Dict, Any, List
import json
import sys
from peft import PeftModel

sys.path.append(r"C:\Users\TEC-INT02\Documents\Agent-IA\src")
from normalizar_texto import normalizar_texto
import re

# --- 1) CONFIGURACIÓN ---
ruta_modelo_embeddings = r"C:\Users\TEC-INT02\Documents\Agent-IA\models\models--intfloat--e5-large-v2"
ruta_indices = r"C:\Users\TEC-INT02\Documents\Agent-IA\llama_index_indices"
ruta_modelo_llama3 = r"C:\Users\TEC-INT02\Documents\Agent-IA\models\models--meta-llama--Meta-Llama-3-8B-Instruct"
ruta_tus_adaptadores_lora = r"C:\Users\TEC-INT02\Documents\Agent-IA\fine_tuning\modelos\llama3-8b-agente-consulta-20250515_1007"

# --- CONSTANTES PARA BUSCAR_DIRECCION_COMBINADA ---
CAMPOS_DIRECCION = ['domicilio', 'calle', 'numero', 'colonia', 'sector', 'municipio', 'ciudad', 'estado', 'cp', 'direccion', 'campo14', 'domicilio calle', 'codigo postal', 'edo de origen']
CAMPOS_BUSQUEDA_EXACTA = ['domicilio', 'direccion', 'calle']
STOP_WORDS = {'de', 'la', 'del', 'los', 'las', 'y', 'a', 'en', 'el', 'col', 'colonia', 'cp', 'sector', 'calzada', 'calz', 'boulevard', 'blvd', 'avenida', 'ave', 'av'}
UMBRAL_PUNTAJE_MINIMO = 0.55
TOLERANCIA_NUMERO_CERCANO = 50

# CONFIGURACIÓN DE DISPOSITIVO Y LLM
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("ADVERTENCIA: Usando CPU para LLM. Las respuestas serán lentas.")
else:
    print(f"Usando dispositivo para LLM y Embeddings: {device}")

def cargar_modelo_con_lora(ruta_adaptadores: str, ruta_base: str, usar_4bit: bool = False):
    """Carga el modelo base y luego aplica los adaptadores LoRA entrenados."""
    print(f"Cargando modelo base original desde: {ruta_base}")

    load_in_8bit = not usar_4bit
    load_in_4bit_config = usar_4bit

    # CONFIGURACIÓN PARA CUANTIZACIÓN
    bnb_config = None
    if usar_4bit:
        from transformers import BitsAndBytesConfig
        print("   Configurando para carga en 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        load_in_8bit = False

    modelo_original = AutoModelForCausalLM.from_pretrained(
        ruta_base,
        torch_dtype=torch.float16,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    print("Modelo base original cargado.")

    print(f"Aplicando adaptadores LoRA desde: {ruta_adaptadores}")
    modelo_tuneado = PeftModel.from_pretrained(
        modelo_original,
        ruta_adaptadores,
        device_map="auto"
    )
    modelo_tuneado.eval() # PONER EN MODO EVALUACIÓN
    print("Adaptadores LoRA aplicados. Modelo fine-tuneado listo.")
    return modelo_tuneado

# --- CARGAR MODELO Y TOKENIZER CON TRANSFORMERS ---
print(f" Cargando Tokenizer y Modelo Llama 3 desde: {ruta_modelo_llama3}")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        ruta_modelo_llama3,
        local_files_only=True
    )
    print("Tokenizer cargado.")

    usar_carga_4_bits_para_agente = False

    model = cargar_modelo_con_lora(
        ruta_tus_adaptadores_lora,
        ruta_modelo_llama3,
        usar_4bit=usar_carga_4_bits_para_agente
    )
    print("Modelo LLM Llama 3 FINE-TUNEADO para agente cargado.")

except Exception as e:
    print(f"Error al cargar Llama 3 FINE-TUNEADO o su tokenizer: {e}")
    import traceback
    exit()

# --- CONFIGURAR HuggingFaceLLM ---
try:
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"temperature": 0.1, "do_sample": (0.1 > 0)},
    )
    print("HuggingFaceLLM configurado con TU modelo Llama 3 FINE-TUNEADO.")
except Exception as e:
    print(f"Error al configurar HuggingFaceLLM con TU modelo: {e}")
    exit()

# --- CONFIGURAR pipeline `llm_clasificador` CON MODELO FINE-TUNEADO ---
print("Configurando pipeline de clasificación de texto con modelo fine-tuneado...")
try:
    llm_clasificador = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16
    )
    print("Pipeline llm_clasificador configurado.")
except Exception as e:
    print(f"Error al configurar el pipeline llm_clasificador: {e}")
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

def convertir_a_mayusculas(texto: str) -> str:
    """
    Convierte a mayúsculas el texto de los resultados para presentación al usuario.
    Preserva formato específico como fechas y estructura.
    
    Args:
        texto: El texto a convertir
        
    Returns:
        str: Texto convertido a mayúsculas
    """
    if not texto:
        return ""
    
    lineas = texto.split('\n')
    lineas_mayusculas = []
    
    for linea in lineas:
        if linea.startswith('🔍') or linea.startswith('---') or 'COINCIDENCIAS' in linea:
            lineas_mayusculas.append(linea)
            continue
            
        if ':' in linea:
            clave, valor = linea.split(':', 1)
            lineas_mayusculas.append(f"{clave.strip().upper()}: {valor.strip().upper()}")
        else:
            lineas_mayusculas.append(linea.upper())
            
    return '\n'.join(lineas_mayusculas)

def interpretar_pregunta_llm(prompt: str, llm_clasificador) -> dict:
    """
    Analizador avanzado de intenciones que combina técnicas de NLP básicas con LLM
    para entender mejor la intención del usuario independientemente de la formulación.
    """

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

def ejecutar_consulta_inteligente(prompt: str, analisis, llm_clasificador):
    """
    Estrategia inteligente para ejecutar consultas que prueba múltiples herramientas
    cuando es necesario y devuelve los mejores resultados.
    
    Args:
        prompt: La consulta original del usuario
        analisis: Resultado del analizador de intenciones
        llm_clasificador: Modelo para análisis avanzado
        
    Returns:
        str: Resultado de la búsqueda más relevante
    """
    tipo = analisis.get("tipo_busqueda")
    campo = analisis.get("campo", "")
    valor = analisis.get("valor", "")
    
    print(f"[INFO] Ejecutando consulta inteligente - Tipo: {tipo}, Campo: {campo}, Valor: {valor}")
    
    # VALIDAR QUE TENGAMOS UN VALOR PARA BUSCAR
    if not valor:
        print("[ERROR] Valor de búsqueda vacío. Usando texto completo de la consulta.")
        valor = prompt
    
    resultados = {}  # ALMACENAR RESULTADOS
    herramientas_probadas = set()  # REGISTRO DE HERRAMIENTAS YA EJECUTADAS
    
    # ESTRATEGIA 1: EJECUCIÓN DIRECTA SEGÚN TIPO DE CONSULTA
    if tipo == "nombre_componentes":
        print("[HERRAMIENTA] Ejecutando búsqueda por componentes de nombre")
        resultados["nombre_componentes"] = buscar_nombre_componentes(valor)
        herramientas_probadas.add("nombre_componentes")
    
    if tipo == "direccion":
        print("[HERRAMIENTA] Ejecutando búsqueda de dirección")
        resultados["direccion"] = buscar_direccion_combinada(valor)
        herramientas_probadas.add("direccion")
    
    elif tipo == "telefono":
        print("[HERRAMIENTA] Ejecutando búsqueda de teléfono")
        resultados["telefono"] = buscar_numero_telefono(valor)
        herramientas_probadas.add("telefono")
    
    elif tipo == "atributo" and campo:
        print(f"[HERRAMIENTA] Ejecutando búsqueda por atributo: {campo}={valor}")
        resultados["atributo"] = buscar_atributo(campo, valor, carpeta_indices=ruta_indices)
        herramientas_probadas.add("atributo")
    
    elif tipo == "nombre":
        print(f"[HERRAMIENTA] Ejecutando búsqueda por nombre: {valor}")
        resultados["nombre"] = buscar_nombre(valor)
        herramientas_probadas.add("nombre")
    
    # ESTRATEGIA 2: PROBAR MÚLTIPLES HERRAMIENTAS
    
    necesita_mas_busquedas = (
        not resultados or
        all(("No se encontraron coincidencias" in res or not res) for res in resultados.values()) or
        tipo == "desconocido"
    )
    
    if necesita_mas_busquedas:
        print("[INFO] Estrategia de múltiples herramientas activada")
        
        # DETERMINAR QUÉ HERRAMIENTAS PROBAR, EN ORDEN DE PRIORIDAD
        herramientas_pendientes = []
        
        # SI PARECE UN NÚMERO, PRIORIZAR BÚSQUEDA POR TELÉFONO
        if re.search(r'\d{7,}', valor) and "telefono" not in herramientas_probadas:
            herramientas_pendientes.append(("telefono", None))
        
        # BÚSQUEDA POR NOMBRE
        if "nombre" not in herramientas_probadas:
            herramientas_pendientes.append(("nombre", None))
        
        # SI HAY INDICIOS DE DIRECCIÓN, AGREGAR A LA LISTA
        if (re.search(r'\d+', valor) or 
            any(palabra in prompt.lower() for palabra in ["calle", "colonia", "avenida"])) and "direccion" not in herramientas_probadas:
            herramientas_pendientes.append(("direccion", None))
        
        # BÚSQUEDA POR CAMPOS INTELIGENTES
        if "atributo" not in herramientas_probadas:
            campos_disponibles = list(campos_detectados)
            campos_probables = sugerir_campos(valor, campos_disponibles)
            herramientas_pendientes.append(("atributo", campos_probables))
        
        for tipo_herramienta, params in herramientas_pendientes:
            if tipo_herramienta == "telefono":
                resultados["telefono"] = buscar_numero_telefono(valor)
            elif tipo_herramienta == "nombre":
                resultados["nombre"] = buscar_nombre(valor)
            elif tipo_herramienta == "direccion":
                resultados["direccion"] = buscar_direccion_combinada(valor)
            elif tipo_herramienta == "atributo" and params:
                resultados["atributo"] = buscar_campos_inteligente(valor, carpeta_indices=ruta_indices, campos_ordenados=params)
    
    # ESTRATEGIA 3: ANÁLISIS Y SELECCIÓN DEL MEJOR RESULTADO
    
    resultados_positivos = {
        k: v for k, v in resultados.items() 
        if v and "No se encontraron coincidencias" not in v
    }
    
    if not resultados_positivos:
        return f"No se encontraron coincidencias para '{valor}' en ninguna de las herramientas. Por favor, intenta con otra consulta más específica."
    
    if len(resultados_positivos) == 1:
        tipo_busqueda, respuesta = list(resultados_positivos.items())[0]
        return respuesta
        
    # EVALUAR LA CALIDAD DE UN RESULTADO
    def evaluar_calidad(texto_resultado):
        num_coincidencias = texto_resultado.count("Coincidencia")
        calidad_coincidencias = texto_resultado.count("exacta") * 2 + texto_resultado.count("parcial")
        lineas_datos = len(texto_resultado.split("\n"))
        
        return num_coincidencias * 10 + calidad_coincidencias * 5 + lineas_datos
    
    # SELECCIONAR EL MEJOR
    calidades = {k: evaluar_calidad(v) for k, v in resultados_positivos.items()}
    mejor_herramienta = max(calidades.items(), key=lambda x: x[1])[0]
    
    valores_calidad = sorted(calidades.values(), reverse=True)
    diferencia_significativa = len(valores_calidad) < 2 or valores_calidad[0] > valores_calidad[1] * 1.5
    
    if diferencia_significativa:
        return resultados_positivos[mejor_herramienta]
    else:
        tipos_ordenados = sorted(resultados_positivos.keys(), key=lambda k: calidades.get(k, 0), reverse=True)
        mejores_tipos = tipos_ordenados[:2]
        
        respuesta_combinada = "Se encontraron varios tipos de coincidencias:\n\n"
        for tipo in mejores_tipos:
            respuesta_combinada += f"--- RESULTADOS DE BÚSQUEDA POR {tipo.upper()} ---\n"
            respuesta_combinada += resultados_positivos[tipo]
            respuesta_combinada += "\n\n"
        
        return convertir_a_mayusculas(respuesta_combinada)
    
def preprocesar_consulta(prompt: str) -> str:
    """
    Pre-procesa la consulta del usuario para hacerla más estandarizada
    y facilitar su posterior análisis.
    
    Args:
        prompt: Consulta original del usuario
    
    Returns:
        str: Consulta pre-procesada
    """
    # NORMALIZAR ESPACIOS Y PUNTUACIÓN
    prompt = prompt.strip()
    prompt = re.sub(r'\s+', ' ', prompt)
    prompt = re.sub(r'([.,;:!?])(\w)', r'\1 \2', prompt)
    
    # NORMALIZAR CARACTERES ESPECIALES
    prompt = prompt.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    prompt = prompt.replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U')
    prompt = prompt.replace('ñ', 'n').replace('Ñ', 'N')
    
    abreviaturas = {
        'tel': 'telefono',
        'tel.': 'telefono',
        'teléf': 'telefono',
        'teléf.': 'telefono',
        'núm': 'numero',
        'núm.': 'numero',
        'num': 'numero',
        'num.': 'numero',
        'dir': 'direccion',
        'dir.': 'direccion',
        'direc': 'direccion',
        'direc.': 'direccion',
        'col': 'colonia',
        'col.': 'colonia',
        'av': 'avenida',
        'av.': 'avenida',
        'ave': 'avenida',
        'ave.': 'avenida',
        'c.p.': 'codigo postal',
        'cp': 'codigo postal',
        'cp.': 'codigo postal',
        'fracc': 'fraccionamiento',
        'fracc.': 'fraccionamiento',
    }
    
    palabras = prompt.split()
    for i, palabra in enumerate(palabras):
        palabra_lower = palabra.lower()
        if palabra_lower in abreviaturas:
            palabras[i] = abreviaturas[palabra_lower]
    
    prompt = ' '.join(palabras)
    
    # ELIMINAR PALABRAS VACÍAS AL INICIO DE LA CONSULTA
    palabras_inicio = ['por favor', 'podrias', 'puedes', 'quisiera', 'quiero', 'necesito', 'dame']
    for palabra in palabras_inicio:
        if prompt.lower().startswith(palabra):
            prompt = prompt[len(palabra):].strip()
    
    # CONVERTIR PREGUNTAS IMPLÍCITAS EN EXPLÍCITAS
    prompt_lower = prompt.lower()
    
    # CONVERTIR "EL TELÉFONO 1234567" A "QUIÉN TIENE EL TELÉFONO 1234567"
    if (prompt_lower.startswith('el telefono') or prompt_lower.startswith('telefono')) and re.search(r'\d{7,}', prompt_lower):
        prompt = 'quien tiene ' + prompt
    
    # CONVERTIR "LA DIRECCIÓN CALLE X" A "QUIÉN VIVE EN CALLE X"
    if prompt_lower.startswith('la direccion') or prompt_lower.startswith('direccion'):
        prompt = 'quien vive en ' + prompt.split('direccion')[1].strip()
    
    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', prompt) and len(prompt.split()) <= 4:
        prompt = 'busca informacion de ' + prompt
    
    return prompt

def obtener_prompt_clasificacion_con_ejemplos(consulta):
    """
    Genera un prompt para clasificación con ejemplos específicos para mejorar la precisión.
    
    Args:
        consulta: La consulta a clasificar
    
    Returns:
        str: Prompt mejorado con ejemplos
    """
    return f"""
    Eres un sistema experto que clasifica consultas para una base de datos de personas. Necesito que clasifiques la siguiente consulta:
    
    "{consulta}"
    
    Debes determinar:
    1. El tipo de búsqueda: "nombre", "telefono", "direccion", "atributo" o "nombre_componentes"
    2. El campo específico (si aplica)
    3. El valor a buscar
    
    EJEMPLOS DE CLASIFICACIÓN CORRECTA:
    
    Consulta: "¿Quién es Juan Pérez?"
    Clasificación: {{"tipo_busqueda": "nombre", "campo": "nombre_completo", "valor": "Juan Pérez"}}
    
    Consulta: "Dame información de María González"
    Clasificación: {{"tipo_busqueda": "nombre", "campo": "nombre_completo", "valor": "María González"}}
    
    Consulta: "¿De quién es el teléfono 5544332211?"
    Clasificación: {{"tipo_busqueda": "telefono", "campo": "telefono_completo", "valor": "5544332211"}}
    
    Consulta: "¿A quién pertenece este número: 9988776655?"
    Clasificación: {{"tipo_busqueda": "telefono", "campo": "telefono_completo", "valor": "9988776655"}}
    
    Consulta: "¿Quién vive en Calle Principal 123, Colonia Centro?"
    Clasificación: {{"tipo_busqueda": "direccion", "campo": "direccion", "valor": "Calle Principal 123, Colonia Centro"}}
    
    Consulta: "Busca la dirección Zoquipan 1260, Lagos del Country"
    Clasificación: {{"tipo_busqueda": "direccion", "campo": "direccion", "valor": "Zoquipan 1260, Lagos del Country"}}
    
    Consulta: "¿Quiénes son médicos?"
    Clasificación: {{"tipo_busqueda": "atributo", "campo": "ocupacion", "valor": "médico"}}
    
    Consulta: "Busca mujeres en la base de datos"
    Clasificación: {{"tipo_busqueda": "atributo", "campo": "sexo", "valor": "F"}}
    
    Consulta: "Encuentra personas que vivan en Zapopan"
    Clasificación: {{"tipo_busqueda": "atributo", "campo": "municipio", "valor": "Zapopan"}}
    
    Consulta: "Información de Zoquipan 1271"
    Clasificación: {{"tipo_busqueda": "direccion", "campo": "direccion", "valor": "Zoquipan 1271"}}
    
    Consulta: "La persona en el teléfono 1234567"
    Clasificación: {{"tipo_busqueda": "telefono", "campo": "telefono_completo", "valor": "1234567"}}
    
    Consulta: "Quiero información del domicilio Hidalgo 123"
    Clasificación: {{"tipo_busqueda": "direccion", "campo": "direccion", "valor": "Hidalgo 123"}}

    Consulta: "¿Cuántas personas de nombre Carla con apellidos que empiecen con M y V hay?"
    Clasificación: {{"tipo_busqueda": "nombre_componentes", "campo": "nombre_completo", "valor": "Carla M V"}}
    
    Consulta: "Quién se llama Carla con M y V"
    Clasificación: {{"tipo_busqueda": "nombre_componentes", "campo": "nombre_completo", "valor": "Carla M V"}}
    
    Consulta: "Encuentra personas con nombre Juan y apellidos que inicien con L y P"
    Clasificación: {{"tipo_busqueda": "nombre_componentes", "campo": "nombre_completo", "valor": "Juan L P"}}
    
    Consulta: "Busca María con iniciales A y T"
    Clasificación: {{"tipo_busqueda": "nombre_componentes", "campo": "nombre_completo", "valor": "María A T"}}
    
    REGLAS IMPORTANTES:
    - Si la consulta tiene un número telefónico (7+ dígitos) junto a palabras como "teléfono", "número", "contacto", SIEMPRE es tipo "telefono".
    - Si la consulta menciona "vive en", "domicilio", "calle", "colonia" o términos similares, SIEMPRE es tipo "direccion".
    - Si la consulta busca información general sobre un nombre propio, es tipo "nombre".
    - Si busca personas con características específicas (sexo, ocupación, municipio, etc.), es tipo "atributo".
    - Para VALOR, extrae SOLO la información relevante, sin palabras de pregunta ni verbos auxiliares.
    - Si la consulta busca un nombre completo junto con iniciales o partes de apellidos (como "Carla con M y V"), es tipo "nombre_componentes".
    - Si la consulta busca personas con un nombre específico y filtradas por iniciales o letras, es "nombre_componentes".
    - Si la consulta pregunta cuántas personas cumplen con criterios parciales de nombre, es "nombre_componentes".
    
    Responde con un objeto JSON que contenga exactamente "tipo_busqueda", "campo" y "valor".
    """

# --- 3) HERRAMIENTA 1: BUSCAR POR NOMBRE COMPLETO ---
def buscar_nombre(query: str) -> str:
    """
    Busca coincidencias de nombres completos o parciales y las retorna ordenadas por relevancia.
    Permite búsquedas por cualquier combinación de nombre/apellidos, en cualquier orden.
    Ahora también permite búsquedas por subcadena (ej: 'Val' encontrará Valeria, Valentino, etc.)
    """
    print(f"Ejecutando búsqueda de nombre: '{query}'")
    
    query = query.strip()
    query_norm = normalizar_texto(query)
    query_tokens = set(query_norm.split())
    
    # DETECTAR SI ES BÚSQUEDA PARCIAL
    es_busqueda_parcial = len(query_tokens) <= 5
    
    resultados_por_categoria = {
        "exactos": [],          # COINCIDENCIA EXACTA
        "completos": [],        # TODOS LOS TOKENS DE BÚSQUEDA ESTÁN PRESENTES
        "parciales_alta": [],   # COINCIDENCIA SIGNIFICATIVA
        "parciales_media": [],  # COINCIDENCIA PARCIAL BÁSICA
        "substring": [],        # COINCIDENCIAS POR SUBCADENA
        "posibles": []          # COINCIDENCIAS DE BAJA CONFIANZA PERO ÚTILES
    }
    
    # EVITAR DUPLICADOS
    registros_encontrados = set()
    
    # RECORRER TODOS LOS ÍNDICES
    for fuente, index in indices.items():
        try:
            # OBTENER CANDIDATOS INICIALES
            retriever = VectorIndexRetriever(index=index, similarity_top_k=8)
            nodes = retriever.retrieve(query)
            
            # PROCESAR CADA NODO ENCONTRADO
            for node in nodes:
                metadata = node.node.metadata
                
                # CREAR IDENTIFICADOR ÚNICO
                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                if not id_registro:
                    id_registro = node.node.node_id
                
                if id_registro in registros_encontrados:
                    continue
                
                # OBTENER NOMBRE COMPLETO DE METADATOS
                nombre_completo = (
                    metadata.get("nombre_completo", "") or 
                    metadata.get("nombre completo", "") or 
                    metadata.get("nombre", "")
                ).strip()
                
                if not nombre_completo:
                    continue
                
                # NORMALIZAR EL NOMBRE
                nombre_norm = normalizar_texto(nombre_completo)
                
                # DIVIDIR EL NOMBRE EN TOKENS INDIVIDUALES
                nombre_tokens = nombre_norm.split()
                nombre_tokens_set = set(nombre_tokens)
                                
                # DETECTAR SI HAY COINCIDENCIA EXACTA (MISMO NOMBRE)
                sim_texto = similitud(query_norm, nombre_norm)
                
                # EVALUAR SI TODOS LOS TOKENS DE LA CONSULTA ESTÁN EN EL NOMBRE
                tokens_coincidentes = query_tokens.intersection(nombre_tokens_set)
                ratio_consulta = len(tokens_coincidentes) / len(query_tokens) if query_tokens else 0
                
                # EVALUAR QUÉ PORCENTAJE DEL NOMBRE COINCIDE CON LA CONSULTA
                ratio_nombre = len(tokens_coincidentes) / len(nombre_tokens) if nombre_tokens else 0
                
                # VERIFICAR SI HAY COINCIDENCIA DE APELLIDOS
                apellidos_nombre = set(nombre_tokens[-min(2, len(nombre_tokens)):])
                apellidos_query = set()
                if len(query_tokens) >= 2:
                    apellidos_query = set(list(query_tokens)[-min(2, len(query_tokens)):])
                coincidencia_apellidos = len(apellidos_nombre.intersection(apellidos_query))
                
                # VERIFICAR COINCIDENCIA DE NOMBRE DE PILA (PRIMERA PALABRA)
                nombre_pila_coincide = False
                if nombre_tokens and query_tokens:
                    nombre_pila = nombre_tokens[0]
                    if nombre_pila in query_tokens:
                        nombre_pila_coincide = True
                
                # VERIFICAR SI HAY COINCIDENCIA POR SUBCADENA
                coincidencia_substring = False
                token_con_substring = None
                
                # BUSCAR LA SUBCADENA EN CADA TOKEN DEL NOMBRE
                for token in nombre_tokens:
                    if query_norm in token:
                        coincidencia_substring = True
                        token_con_substring = token
                        break
                
                resumen = [f"{k}: {v}" for k, v in metadata.items() 
                          if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                texto_resultado = f"Encontrado en {fuente}:\n" + "\n".join(resumen)
                                
                # COINCIDENCIA EXACTA O CASI EXACTA
                if sim_texto > 0.9 or (ratio_consulta > 0.9 and ratio_nombre > 0.9):
                    resultados_por_categoria["exactos"].append({
                        "texto": f"Coincidencia exacta: {texto_resultado}",
                        "score": sim_texto + 0.1,  # Bonificación para exactos
                        "fuente": fuente
                    })
                
                # TODOS LOS TOKENS DE LA CONSULTA ESTÁN EN EL NOMBRE
                elif ratio_consulta > 0.95:
                    resultados_por_categoria["completos"].append({
                        "texto": f"Coincidencia completa: {texto_resultado}",
                        "score": ratio_consulta * 0.9 + ratio_nombre * 0.1,
                        "fuente": fuente
                    })
                
                # COINCIDENCIA DE APELLIDOS SIGNIFICATIVA (AL MENOS UN APELLIDO COMPLETO)
                elif coincidencia_apellidos > 0 and ratio_consulta >= 0.5:
                    resultados_por_categoria["parciales_alta"].append({
                        "texto": f"Coincidencia parcial: {texto_resultado}",
                        "score": 0.7 + (coincidencia_apellidos * 0.15) + (ratio_consulta * 0.15),
                        "fuente": fuente
                    })
                
                # COINCIDENCIA DE NOMBRE DE PILA Y TOKENS SIGNIFICATIVOS
                elif nombre_pila_coincide and len(tokens_coincidentes) >= 1:
                    resultados_por_categoria["parciales_alta"].append({
                        "texto": f"Coincidencia parcial: {texto_resultado}",
                        "score": 0.65 + (ratio_consulta * 0.35),
                        "fuente": fuente
                    })
                
                # COINCIDENCIA POR SUBCADENA
                elif coincidencia_substring:
                    resultados_por_categoria["substring"].append({
                        "texto": f"Coincidencia por subcadena '{query}' en '{token_con_substring}': {texto_resultado}",
                        "score": 0.6,  # Puntuación media-alta
                        "fuente": fuente
                    })
                
                # COINCIDENCIA PARCIAL BÁSICA
                elif len(tokens_coincidentes) >= 1 and any(token in nombre_tokens_set for token in query_tokens):
                    resultados_por_categoria["parciales_media"].append({
                        "texto": f"Coincidencia parcial: {texto_resultado}",
                        "score": 0.4 + (ratio_consulta * 0.6),
                        "fuente": fuente
                    })
                
                # COINCIDENCIAS DE BAJA CONFIANZA PERO ÚTILES
                elif tokens_coincidentes and sim_texto > 0.3:
                    resultados_por_categoria["posibles"].append({
                        "texto": f"Posible coincidencia: {texto_resultado}",
                        "score": sim_texto,
                        "fuente": fuente
                    })
                
                registros_encontrados.add(id_registro)
            
            # BÚSQUEDA EXHAUSTIVA ESPECÍFICA PARA SUBCADENAS
            if es_busqueda_parcial:
                print(f"Realizando búsqueda exhaustiva para subcadena '{query_norm}'...")
                
                for node_id, doc in index.docstore.docs.items():
                    metadata = doc.metadata
                    if not metadata:
                        continue
                    
                    id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                    if not id_registro:
                        id_registro = node_id
                    
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
                    
                    # VERIFICAR SI LA SUBCADENA ESTÁ EN CUALQUIER PARTE DEL NOMBRE
                    if query_norm in nombre_norm:
                        # DIVIDIR EL NOMBRE PARA ENCONTRAR QUÉ TOKEN CONTIENE LA SUBCADENA
                        tokens = nombre_norm.split()
                        token_con_subcadena = None
                        
                        for token in tokens:
                            if query_norm in token:
                                token_con_subcadena = token
                                break
                        
                        resumen = [f"{k}: {v}" for k, v in metadata.items() 
                                  if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                        texto_resultado = f"Encontrado en {fuente}:\n" + "\n".join(resumen)
                        
                        resultados_por_categoria["substring"].append({
                            "texto": f"Coincidencia por subcadena '{query}' en '{token_con_subcadena}': {texto_resultado}",
                            "score": 0.5,
                            "fuente": fuente
                        })
                        
                        registros_encontrados.add(id_registro)
        
        except Exception as e:
            print(f"Error al buscar en índice {fuente}: {e}")
            continue
    
    todas_respuestas = []

    mostrar_solo_exactos = bool(resultados_por_categoria["exactos"])

    if mostrar_solo_exactos:
        resultados_ordenados = sorted(resultados_por_categoria["exactos"], key=lambda x: x["score"], reverse=True)
        todas_respuestas.append("🔍 COINCIDENCIAS EXACTAS:")
        for res in resultados_ordenados:
            todas_respuestas.append(res["texto"])

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
                
        if resultados_por_categoria["substring"]:
            resultados_ordenados = sorted(resultados_por_categoria["substring"], key=lambda x: x["score"], reverse=True)
            todas_respuestas.append("\n🔍 COINCIDENCIAS POR SUBCADENA:")
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

    if not todas_respuestas:
        return f"No se encontraron coincidencias para '{query}' en ninguna fuente."

    return convertir_a_mayusculas("\n\n".join(todas_respuestas))

    
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
    
    # NORMALIZAR CAMPO Y VALOR
    campo_normalizado = normalizar_texto(campo) if campo else ""
    valor_normalizado = normalizar_texto(valor)
    
    # CASOS ESPECIALES DE NORMALIZACIÓN
    if campo_normalizado in ["sexo", "genero"]:
        if valor_normalizado in ["hombre", "hombres", "masculino", "varon", "varones", "m"]:
            valor_normalizado = "m"
        elif valor_normalizado in ["mujer", "mujeres", "femenino", "f"]:
            valor_normalizado = "f"
    
    resultados = []
    registros_encontrados = set()
    
    busqueda_categorica = campo_normalizado in ["sexo", "genero", "ocupacion", "profesion"]
    
    campo_final = mapa_campos.get(campo_normalizado, campo_normalizado) if campo_normalizado else ""
    
    # FASE 1: BÚSQUEDA EXACTA POR FILTROS SI TENEMOS UN CAMPO ESPECÍFICO
    if campo_final and not busqueda_categorica:
        for nombre_dir in os.listdir(carpeta_indices):
            ruta_indice = os.path.join(carpeta_indices, nombre_dir)
            if not os.path.isdir(ruta_indice) or not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
                continue

            fuente = nombre_dir.replace("index_", "")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
                index = load_index_from_storage(storage_context)

                # BÚSQUEDA EXACTA CON FILTRO
                try:
                    filters = MetadataFilters(filters=[
                        ExactMatchFilter(key=campo_final, value=valor_normalizado)
                    ])
                    top_k_dinamico = min(10000, len(index.docstore.docs))
                    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k_dinamico, filters=filters)
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
                    pass

            except Exception as e:
                print(f"Error al cargar índice {fuente}: {e}")
                continue
    
    # FASE 2: BUSCAR EN TODOS LOS DOCUMENTOS
    if busqueda_categorica or (not resultados and valor):
        print(f"Realizando búsqueda exhaustiva para '{valor_normalizado}'...")
        
        campos_a_buscar = []
        if campo_final:
            if campo_final in campos_clave:
                campos_a_buscar = [normalizar_texto(c) for c in campos_clave[campo_final]]
            else:
                campos_a_buscar = [campo_final]
        
        # RECORRER TODOS LOS ÍNDICES
        for fuente, index in indices.items():
            try:
                for node_id, doc in index.docstore.docs.items():
                    metadata = doc.metadata
                    if not metadata:
                        continue
                    
                    # CREAR ID ÚNICO
                    id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                    if not id_registro:
                        id_registro = node_id
                    
                    if id_registro in registros_encontrados:
                        continue
                    
                    encontrado = False
                    coincidencia_campo = None
                    
                    # CAMPOS ESPECÍFICOS A BUSCAR
                    if campos_a_buscar:
                        for k, v in metadata.items():
                            k_norm = normalizar_texto(k)
                            
                            if k_norm in campos_a_buscar:
                                v_str = str(v).strip().lower()
                                v_norm = normalizar_texto(v_str)
                                
                                if v_norm == valor_normalizado or (
                                    len(valor_normalizado) > 4 and (
                                        valor_normalizado in v_norm or 
                                        v_norm in valor_normalizado
                                    )
                                ):
                                    encontrado = True
                                    coincidencia_campo = k
                                    break
                    
                    if not encontrado and len(valor_normalizado) >= 4:
                        for k, v in metadata.items():
                            if k in ['fuente', 'archivo', 'fila_excel']:
                                continue
                                
                            v_str = str(v).strip().lower()
                            v_norm = normalizar_texto(v_str)
                            
                            if v_norm == valor_normalizado or (
                                len(valor_normalizado) > 6 and 
                                valor_normalizado in v_norm
                            ):
                                encontrado = True
                                coincidencia_campo = k
                                break
                    
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
    
    # FORMATEAR RESPUESTA FINAL
    if resultados:
        total_registros = len(resultados)
        num_mostrados = total_registros
        
        if campo:
            mensaje_intro = f"Se encontraron {total_registros} registros para {campo}='{valor}'."
        else:
            mensaje_intro = f"Se encontraron {total_registros} registros que contienen '{valor}'."
        
        if total_registros > num_mostrados:
            mensaje_intro += f" Mostrando {num_mostrados} primeros resultados:"
        
        return convertir_a_mayusculas(mensaje_intro + "\n\n" + "\n\n".join(resultados[:num_mostrados]))
    else:
        if campo:
            return convertir_a_mayusculas(f"No se encontraron coincidencias para '{campo}: {valor}'.")
        else:
            return convertir_a_mayusculas(f"No se encontraron coincidencias para el valor '{valor}'.")

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

    # PREPROCESAMIENTO Y EXTRACCIÓN DE COMPONENTES
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

    # ALMACENAMIENTO DE RESULTADOS
    todos_resultados_detalle: Dict[str, Dict[str, Any]] = {}

    # BÚSQUEDA ITERATIVA EN TODOS LOS ÍNDICES
    for fuente, index in indices.items():
        try:
            # BÚSQUEDA EXACTA POR METADATOS
            if combinacion_principal_norm:
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
                        nodes_exactos = retriever_exacto.retrieve(combinacion_principal)

                        if nodes_exactos:
                            for node in nodes_exactos:
                                metadata = node.node.metadata
                                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                                if not id_registro: id_registro = node.node.node_id

                                puntaje_actual = todos_resultados_detalle.get(id_registro, {}).get('puntaje', -1.0)
                                puntaje_nuevo = 1.0

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
                        if 'Metadata key' not in str(e_filter):
                            print(f"[WARN] Error en búsqueda exacta por filtro en campo '{campo_exacto}' en {fuente}: {e_filter}")

            # BÚSQUEDA SEMÁNTICA Y EVALUACIÓN
            top_k_dinamico = min(10000, len(index.docstore.docs))
            retriever_semantico = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k_dinamico
            )
            consulta_semantica = " ".join(componentes_clave)
            if not consulta_semantica: consulta_semantica = texto_direccion_normalizado

            nodes_semanticos = retriever_semantico.retrieve(consulta_semantica)

            for node in nodes_semanticos:
                metadata = node.node.metadata
                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                if not id_registro: id_registro = node.node.node_id

                puntaje_actual = todos_resultados_detalle.get(id_registro, {}).get('puntaje', -1.0)

                if puntaje_actual == 1.0:
                    continue

                texto_completo_registro = ""
                for k, v in metadata.items():
                    if k in CAMPOS_DIRECCION:
                        texto_completo_registro += f" {v}"
                texto_completo_registro_norm = normalizar_texto(texto_completo_registro)

                if not texto_completo_registro_norm:
                    continue

                tiene_numero_exacto = False
                tiene_numero_cercano = False
                numeros_en_registro = re.findall(r'\b\d+\b', texto_completo_registro_norm)

                if numeros_busqueda:
                    num_b_str = numeros_busqueda[0]
                    if num_b_str in numeros_en_registro:
                        tiene_numero_exacto = True
                    else:
                        try:
                            num_b_int = int(num_b_str)
                            for num_r_str in numeros_en_registro:
                                try:
                                    num_r_int = int(num_r_str)
                                    if abs(num_b_int - num_r_int) <= TOLERANCIA_NUMERO_CERCANO:
                                        tiene_numero_cercano = True
                                        break
                                except ValueError: continue
                        except ValueError: pass
                else:
                    tiene_numero_exacto = True
                    tiene_numero_cercano = True

                # COMPONENTES CLAVE
                componentes_clave_encontrados = sum(1 for comp in componentes_clave if comp in texto_completo_registro_norm)
                porcentaje_clave = componentes_clave_encontrados / len(componentes_clave) if componentes_clave else 1.0

                # TODOS LOS COMPONENTES
                componentes_encontrados = sum(1 for comp in componentes if comp in texto_completo_registro_norm)
                calificacion_componentes = componentes_encontrados / len(componentes) if componentes else 1.0

                # SIMILITUD TEXTUAL
                similitud_textual = similitud(texto_direccion_normalizado, texto_completo_registro_norm)

                # CALCULAR SCORE
                score_final = 0.0
                peso_num_exacto = 0.50
                peso_num_cercano = 0.20
                peso_clave = 0.35
                peso_componentes = 0.05
                peso_similitud = 0.10

                if tiene_numero_exacto:
                    score_final = (peso_num_exacto * 1.0) + (peso_clave * porcentaje_clave) + \
                                  (peso_componentes * calificacion_componentes) + (peso_similitud * similitud_textual)
                    # SI LA PRIMERA CALLE/COLONIA COINCIDE
                    if calles_colonias_busqueda and calles_colonias_busqueda[0] in texto_completo_registro_norm:
                        score_final = min(0.99, score_final * 1.1)
                elif tiene_numero_cercano:
                    score_final = (peso_num_cercano * 1.0) + (peso_clave * porcentaje_clave) + \
                                  (peso_componentes * calificacion_componentes) + (peso_similitud * similitud_textual)
                    score_final *= 0.85
                else:
                    if porcentaje_clave > 0.6:
                        score_final = (peso_clave * porcentaje_clave) + \
                                      (peso_componentes * calificacion_componentes) + (peso_similitud * similitud_textual)
                        score_final *= 0.65
                    else:
                        score_final = 0.0

                # FILTRADO ESTRICTO
                if numeros_busqueda and not (tiene_numero_exacto or tiene_numero_cercano):
                    continue
                if porcentaje_clave < 0.4 and not tiene_numero_exacto:
                    continue
                if score_final < (UMBRAL_PUNTAJE_MINIMO - 0.1):
                     continue

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
            continue

    # CONSOLIDACIÓN Y FORMATEO FINAL
    if not todos_resultados_detalle:
        return f"No se encontraron coincidencias relevantes para la dirección '{texto_direccion}'."

    resultados_ordenados = sorted(todos_resultados_detalle.values(), key=lambda x: x['puntaje'], reverse=True)

    resultados_filtrados = [res for res in resultados_ordenados if res['puntaje'] >= UMBRAL_PUNTAJE_MINIMO]

    if not resultados_filtrados and resultados_ordenados:
        resultados_finales = resultados_ordenados
        mensaje_intro = "No se encontraron coincidencias muy relevantes. Mostrando los más cercanos:\n\n"
    elif not resultados_filtrados and not resultados_ordenados:
         return f"No se encontraron coincidencias relevantes para la dirección '{texto_direccion}'."
    else:
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
        texto_limpio = re.sub(r'\s*\(Score: \d+\.\d+\)', '', res['texto_base']).strip() # LIMPIAR SCORE
        textos_resultados.append(texto_limpio)

    return convertir_a_mayusculas(mensaje_intro + "\n\n".join(textos_resultados))

buscar_direccion_tool = FunctionTool.from_defaults(
    fn=buscar_direccion_combinada,
    name="buscar_direccion_combinada",
    description=(
        "Usa esta herramienta cuando el usuario busca una dirección completa o parcial que contenga calle, número y posiblemente colonia o ciudad. "
        "Es especialmente útil para direcciones combinadas como 'ZOQUIPAN 1260, LAGOS DEL COUNTRY'. "
        "Por ejemplo: '¿De quién es esta dirección: ZOQUIPAN 1260, LAGOS DEL COUNTRY, ZAPOPAN?', "
        "'Busca Malva 101, San Luis de la Paz', 'Quién vive en casa #63, colinas del rey, Zapopan', 'información de zoquipan 1260'. "
        "Esta herramienta realiza búsquedas semánticas y exactas en componentes de dirección."
    )
)

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
                        
                        # SOLO GUARDAMOS EL MEJOR RESULTADO POR REGISTRO
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
    return convertir_a_mayusculas("Se encontraron las siguientes coincidencias para número telefónico:\n\n" + "\n\n".join([r['texto'] for r in resultados_ordenados]))

buscar_telefono_tool = FunctionTool.from_defaults(
    fn=buscar_numero_telefono,
    name="buscar_numero_telefono",
    description=(
        "Usa esta herramienta cuando el campo detectado sea 'telefono_completo' y el usuario consulta por un número telefónico completo. "
        "Ejemplo: '¿Quién tiene el número 5544332211?', pero NO para lada o partes de teléfonos."
    )
)

all_tools.insert(4, buscar_telefono_tool)

# --- 7) HERRAMIENTA 5: BUSCAR POR INICIALES DEL NOMBRE ---
def buscar_nombre_componentes(query: str) -> str:
    """
    Busca personas por componentes parciales de nombres.
    Permite buscar por nombre y filtrar por iniciales o partes de apellidos.
    
    Por ejemplo:
    - "Carla con apellidos M y V" 
    - "Carla M V"
    - "nombre Carla iniciales M V"
    
    Retorna coincidencias que contengan todos los componentes especificados.
    """
    print(f"Ejecutando búsqueda por componentes de nombre: '{query}'")
    
    # EXTRAER COMPONENTES DE LA CONSULTA
    componentes = extraer_componentes_nombre(query)
    if not componentes:
        return "No se pudieron identificar componentes de nombre en la consulta. Por favor especifica al menos un nombre o inicial."
    
    print(f"Componentes detectados: {componentes}")
    
    # BUSCAR CANDIDATOS INICIALES POR EL PRIMER COMPONENTE
    primer_componente = componentes[0]
    componentes_adicionales = componentes[1:] if len(componentes) > 1 else []
    
    resultados_por_categoria = {
        "coincidencias_completas": [], # COINCIDEN TODOS LOS COMPONENTES EXACTAMENTE
        "coincidencias_iniciales": [],  # ALGUNOS COMPONENTES COINCIDEN COMO INICIALES
        "coincidencias_parciales": []   # HAY COINCIDENCIAS PARCIALES CON TODOS LOS COMPONENTES
    }
    
    registros_encontrados = set()
    
    for fuente, index in indices.items():
        try:
            # BÚSQUEDA SEMÁNTICA PARA CANDIDATOS INICIALES
            retriever = VectorIndexRetriever(index=index, similarity_top_k=15)
            nodes = retriever.retrieve(primer_componente)
            
            # BÚSQUEDA EXHAUSTIVA
            todos_docs = index.docstore.docs.items()
            
            # COMBINAR RESULTADOS SEMÁNTICOS CON BÚSQUEDA COMPLETA
            nodos_procesados = set()
            
            # PROCESAR NODOS DE BÚSQUEDA SEMÁNTICA
            for node in nodes:
                metadata = node.node.metadata
                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                if not id_registro:
                    id_registro = node.node.node_id
                
                nodos_procesados.add(id_registro)
                
                if id_registro in registros_encontrados:
                    continue
                
                resultado = evaluar_coincidencia_componentes(metadata, componentes)
                if resultado:
                    categoria, score, texto = resultado
                    resultados_por_categoria[categoria].append({
                        "texto": texto,
                        "score": score,
                        "fuente": fuente
                    })
                    registros_encontrados.add(id_registro)
            
            if len(registros_encontrados) < 5:
                for node_id, doc in todos_docs:
                    metadata = doc.metadata
                    id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                    if not id_registro:
                        id_registro = node_id
                    
                    if id_registro in nodos_procesados or id_registro in registros_encontrados:
                        continue
                    
                    resultado = evaluar_coincidencia_componentes(metadata, componentes)
                    if resultado:
                        categoria, score, texto = resultado
                        resultados_por_categoria[categoria].append({
                            "texto": texto,
                            "score": score,
                            "fuente": fuente
                        })
                        registros_encontrados.add(id_registro)
            
        except Exception as e:
            print(f"Error al buscar en índice {fuente}: {e}")
            continue
    
    todas_respuestas = []
    total_coincidencias = sum(len(resultados_por_categoria[cat]) for cat in resultados_por_categoria)
    
    if total_coincidencias == 0:
        return f"No se encontraron personas que coincidan con todos los componentes: {', '.join(componentes)}"
    
    todas_respuestas.append(f"🔍 Se encontraron {total_coincidencias} personas que coinciden con los componentes: {', '.join(componentes)}")
    
    # COINCIDENCIAS COMPLETAS
    if resultados_por_categoria["coincidencias_completas"]:
        resultados_ordenados = sorted(resultados_por_categoria["coincidencias_completas"], 
                                      key=lambda x: x["score"], reverse=True)
        todas_respuestas.append("\n🔍 COINCIDENCIAS EXACTAS:")
        for res in resultados_ordenados:
            todas_respuestas.append(res["texto"])
    
    # COINCIDENCIAS POR INICIALES
    if resultados_por_categoria["coincidencias_iniciales"]:
        resultados_ordenados = sorted(resultados_por_categoria["coincidencias_iniciales"], 
                                      key=lambda x: x["score"], reverse=True)
        todas_respuestas.append("\n🔍 COINCIDENCIAS POR INICIALES:")
        for res in resultados_ordenados:
            todas_respuestas.append(res["texto"])
    
    # COINCIDENCIAS PARCIALES
    if resultados_por_categoria["coincidencias_parciales"]:
        resultados_ordenados = sorted(resultados_por_categoria["coincidencias_parciales"], 
                                      key=lambda x: x["score"], reverse=True)
        todas_respuestas.append("\n🔍 COINCIDENCIAS PARCIALES:")
        for res in resultados_ordenados:
            todas_respuestas.append(res["texto"])
    
    return convertir_a_mayusculas("\n\n".join(todas_respuestas))


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
    
buscar_nombre_componentes_tool = FunctionTool.from_defaults(
    fn=buscar_nombre_componentes,
    name="buscar_nombre_componentes",
    description=(
        "Usa esta herramienta cuando el usuario busca personas por componentes parciales de nombres. "
        "Es especialmente útil cuando se busca un nombre específico junto con iniciales o partes de apellidos. "
        "Por ejemplo: '¿Cuántas personas de nombre Carla con apellidos que empiecen con M y V hay?', "
        "'Quien se llama Carla con M y V', 'Nombre Juan iniciales L M', o 'María con P y G'."
    )
)

all_tools.insert(5, buscar_nombre_componentes_tool)

# --- 8) CREAR Y EJECUTAR EL AGENTE ---

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

print("\n🤖 Agente Inteligente Mejorado: Escribe tu pregunta en lenguaje natural o 'salir' para terminar.")

while True:
    prompt = input("\nPregunta: ")
    if prompt.lower() == 'salir':
        break
    if not prompt:
        continue

    herramienta_usada = None
    try:
        # PRE-PROCESAMIENTO DE LA CONSULTA
        prompt_procesado = preprocesar_consulta(prompt)
        if prompt_procesado != prompt:
            print(f"[DEBUG] Consulta procesada: {prompt_procesado}")
        
        prompt_clasificacion = obtener_prompt_clasificacion_con_ejemplos(prompt_procesado)
        
        salida_cruda = llm_clasificador(prompt_clasificacion, max_new_tokens=256, return_full_text=False)[0]['generated_text']
        
        match = re.search(r'\{[\s\S]*?\}', salida_cruda)
        if match:
            json_text = match.group(0)
            analisis = json.loads(json_text)
        else:
            print("[INFO] No se pudo extraer JSON del clasificador, usando analizador alternativo...")
            analisis = interpretar_pregunta_llm(prompt_procesado, llm_clasificador)
        
        print(f"[INFO] Análisis: tipo={analisis.get('tipo_busqueda')}, campo={analisis.get('campo')}, valor={analisis.get('valor')}")
        
        if analisis.get("tipo_busqueda") in ["desconocido", None] or not analisis.get("valor"):
            print("[INFO] Consulta ambigua, intentando desambiguar...")
            analisis = desambiguar_consulta(analisis, prompt_procesado, llm_clasificador)
            print(f"[INFO] Análisis post-desambiguación: tipo={analisis.get('tipo_busqueda')}, campo={analisis.get('campo')}, valor={analisis.get('valor')}")
        
        print(f"[INFO] Ejecutando búsqueda para '{analisis.get('valor')}' como {analisis.get('tipo_busqueda')}...")
        herramienta_usada = analisis.get('tipo_busqueda', 'desconocido')
        respuesta_final = ejecutar_consulta_inteligente(prompt_procesado, analisis, llm_clasificador)
        
        print(f"\n📄 Resultado:\n{respuesta_final}\n")
        
        if "No se encontraron coincidencias" in respuesta_final:
            print("\n[SUGERENCIA] Para mejorar los resultados, intenta:")
            if analisis.get("tipo_busqueda") == "nombre":
                print("- Usar nombre y apellido completos")
                print("- Verificar la ortografía del nombre")
            elif analisis.get("tipo_busqueda") == "direccion":
                print("- Incluir el número de la dirección")
                print("- Especificar la colonia o sector")
            elif analisis.get("tipo_busqueda") == "telefono":
                print("- Verificar que el número tenga el formato correcto")
                print("- Incluir el código de área o lada")

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

        try:
            # FALLBACK Y REGISTRO DE ERROR
            print("Intentando recuperación con agente fallback...")
            respuesta_agente = agent.query(prompt)
            print(f"\n📄 Resultado (procesado por agente fallback):\n{respuesta_agente}\n")
            
        except Exception as e2:
            print(f"❌ También falló el agente fallback: {e2}")
            print("Lo siento, no pude procesar tu consulta. Por favor, intenta reformularla.")

# --- LIMPIEZA ---
del llm, embed_model, agent, all_tools, indices
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()
print("\n👋 ¡Hasta luego!")