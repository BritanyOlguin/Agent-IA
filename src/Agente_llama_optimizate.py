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
SIMILARITY_TOP_K_DIRECCION = 15
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
        return "\n\n".join(resultados[:15])  # Limitar a 15 resultados para no sobrecargar
    
    # Para otras búsquedas, devolver los resultados normalmente
    if resultados:
        return "\n\n".join(resultados[:5])  # Limitar a 5 resultados
        
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
            retriever_semantico = VectorIndexRetriever(
                index=index,
                similarity_top_k=SIMILARITY_TOP_K_DIRECCION # Usar constante específica
            )
            consulta_semantica = " ".join(componentes_clave[:5])
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
        resultados_finales = resultados_ordenados[:2] # Mostrar los 2 mejores
        mensaje_intro = "No se encontraron coincidencias muy relevantes. Mostrando los más cercanos:\n\n"
    elif not resultados_filtrados and not resultados_ordenados:
         return f"No se encontraron coincidencias relevantes para la dirección '{texto_direccion}'."
    else:
        # Usar constante MAX_RESULTADOS_FINALES
        resultados_finales = resultados_filtrados[:MAX_RESULTADOS_FINALES]
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

# --- FIN DE LA FUNCIÓN COMBINADA ---

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

# --- 3) HERRAMIENTA 1: BUSCAR POR NOMBRE COMPLETO ---
def buscar_nombre(query: str) -> str:
    print(f"Ejecutando búsqueda de nombre: '{query}'")
    resultados_exactos = []
    resultados_top_1 = []
    query_upper = query.strip().upper()
    ya_guardados = set()

    for fuente, index in indices.items():
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        nodes = retriever.retrieve(query)

        for node in nodes:
            metadata = node.node.metadata
            nombre_metadata = (
                metadata.get("nombre_completo", "")
                or metadata.get("nombre completo", "")
                or metadata.get("nombre", "")
            ).strip().upper()

            def normalizar(nombre):
                return sorted(nombre.replace(",", "").replace("  ", " ").strip().upper().split())

            if normalizar(nombre_metadata) == normalizar(query_upper) and fuente not in ya_guardados:
                resumen = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                resultados_exactos.append(
                    f"Coincidencia exacta en {fuente}:\n" + "\n".join(resumen)
                )
                ya_guardados.add(fuente)

        # GUARDAR LAS MEJORES COINCIDENCIAS
        for node in nodes:
            metadata = node.node.metadata
            nombre_metadata = (
                metadata.get("nombre_completo", "")
                or metadata.get("nombre completo", "")
                or metadata.get("nombre", "")
            ).strip().upper()

            sim = similitud(nombre_metadata, query_upper)

            if sim >= 0.5 and fuente not in ya_guardados:
                resumen = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                resultados_top_1.append(
                    f"🔹 Coincidencia cercana en {fuente}:\n" + "\n".join(resumen)
                )
                ya_guardados.add(fuente)


    if resultados_exactos:
        respuesta_final = "Se encontraron estas coincidencias exactas en los archivos:\n\n" + "\n\n".join(resultados_exactos)
        return respuesta_final

    elif resultados_top_1:
        return "Se encontraron estas coincidencias:\n\n" + "\n\n".join(resultados_top_1)

    else:
        return "No se encontraron resultados relevantes en ninguna fuente."

    
busqueda_global_tool = FunctionTool.from_defaults(
    fn=buscar_nombre,
    name="buscar_nombre",
    description=(
        "Usa esta herramienta para encontrar información completa de una persona en todas las bases, "
        "cuando el usuario da el nombre completo. Por ejemplo: 'Dame la información de Juan Pérez', "
        "'¿Qué sabes de Adrian Lino Marmolejo?'."
    )
)
all_tools.insert(0, busqueda_global_tool)

# --- 4) HERRAMIENTA 2: BUSCAR PERSONAS POR ATRIBUTO ---
def buscar_atributo(campo: str, valor: str, carpeta_indices: str) -> str:
    """
    Busca coincidencias exactas por campo y valor en todos los índices dentro de la carpeta dada.
    Aplica normalización para coincidir con los metadatos indexados.
    """
    print(f"\nBuscando registros donde '{campo}' = '{valor}'\n")

    campo_normalizado = normalizar_texto(campo)
    campo_final = mapa_campos.get(campo_normalizado, campo_normalizado)

    campo_normalizado = normalizar_texto(campo)
    campo_final = mapa_campos.get(campo_normalizado, campo_normalizado)
    valor_final = normalizar_texto(valor)

    resultados = []

    for nombre_dir in os.listdir(carpeta_indices):
        ruta_indice = os.path.join(carpeta_indices, nombre_dir)
        if not os.path.isdir(ruta_indice) or not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
            continue

        fuente = nombre_dir.replace("index_", "")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
            index = load_index_from_storage(storage_context)

            filters = MetadataFilters(filters=[
                ExactMatchFilter(key=campo_final, value=valor_final)
            ])

            retriever = VectorIndexRetriever(index=index, similarity_top_k=5, filters=filters)
            nodes = retriever.retrieve(f"{campo_final} es {valor_final}")

            if nodes:
                for node in nodes:
                    metadata = node.node.metadata
                    resumen = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                    resultados.append(f"Coincidencia en {fuente}:\n" + "\n".join(resumen))
        except Exception as e:
            print(f"Error al buscar en {fuente}: {e}")
            continue

    if resultados:
        return "\n".join(resultados)
    else:
        return f"No se encontraron coincidencias para '{campo}: {valor}'"

buscar_por_atributo_tool = FunctionTool.from_defaults(
    fn=lambda campo, valor: buscar_atributo(campo, valor, carpeta_indices=ruta_indices),
    name="buscar_atributo",
    description=(
        "Usa esta herramienta cuando el usuario pregunta por un campo específico como teléfono, dirección, estado, tarjeta, etc. "
        "Por ejemplo: '¿Quién tiene el número 5544332211?', '¿Quién vive en Malva 101?', '¿Quién tiene la tarjeta terminación 8841?', "
        "'¿Qué personas viven en Querétaro?', '¿Quién vive en calle Reforma 123?'."
    )
)
all_tools.insert(1, buscar_por_atributo_tool)

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
all_tools.insert(0, buscar_direccion_tool)

# --- 5) CREAR Y EJECUTAR EL AGENTE ---

# CREAR EL AGENTE REACT
try:
    agent = ReActAgent.from_tools(
        tools=all_tools,
        llm=llm,
        verbose=False  # MUESTRA LOS PASOS DE PENSAMIENTO DEL AGENTE
    )
    print("Agente creado correctamente.")
except Exception as e:
    print(f"Error al crear el agente: {e}")
    exit()

print("\n🤖 Agente listo. Escribe tu pregunta o 'salir' para terminar.")

# CICLO DE CHAT MEJORADO
while True:
    prompt = input("Pregunta: ")
    if prompt.lower() == 'salir':
        break
    if not prompt:
        continue

    try:
        # 1. Verificar si es una consulta de dirección compleja
        if es_consulta_direccion(prompt):
            # Es una dirección compleja, usar búsqueda por dirección combinada
            texto_direccion = extraer_texto_direccion(prompt)
            
            respuesta_herramienta = buscar_direccion_combinada(texto_direccion)
        else:
            # 2. Si no es dirección compleja, verificar si es campo específico
            campo, valor = detectar_campo_valor(prompt)
            
            if campo and valor:
                print(f"[DEBUG] Campo y valor detectados: {campo}={valor}")
                respuesta_herramienta = buscar_atributo(campo, valor, carpeta_indices=ruta_indices)
            else:
                # 3. Para consultas simples, extraer el valor y buscar en múltiples campos
                valor_extraido = extraer_valor(prompt)
                print(f"[DEBUG] Consulta simple detectada, valor extraído: '{valor_extraido}'")
                
                campos_disponibles = list(campos_detectados)
                campos_probables = sugerir_campos(valor_extraido, campos_disponibles)
                
                respuesta_herramienta = buscar_campos_inteligente(valor_extraido, carpeta_indices=ruta_indices, campos_ordenados=campos_probables)
                
                # Si no hay resultados, intentar búsqueda por nombre
                if "No se encontraron coincidencias" in respuesta_herramienta:
                    print(f"[DEBUG] Intentando búsqueda por nombre")
                    respuesta_herramienta = buscar_nombre(prompt)

        print(f"\n📄Resultado:\n{respuesta_herramienta}\n")

    except Exception as e:
        print(f"❌ Ocurrió un error durante la ejecución del agente: {e}")
        import traceback
        traceback.print_exc()
        
        # Intentar usar el agente ReAct como fallback
        try:
            respuesta_agente = agent.query(prompt)
            print(f"\n📄Resultado (procesado por agente fallback):\n{respuesta_agente}\n")
        except Exception as e2:
            print(f"❌ También falló el agente fallback: {e2}")

# LIMPIAR MEMORIA AL SALIR
del llm, embed_model, agent, all_tools, indices
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()
print("\n👋 ¡Hasta luego!")