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
    Busca coincidencias semánticas de dirección en todos los índices.
    Maneja direcciones combinadas como "ZOQUIPAN 1260, LAGOS DEL COUNTRY"
    descomponiéndolas en partes para mejorar la búsqueda.
    """
    print(f"\nBuscando dirección combinada: '{texto_direccion}'")
    
    # Normalizar la dirección de búsqueda
    texto_direccion_normalizado = normalizar_texto(texto_direccion)
    
    # Preprocesar para separar números pegados a texto (como "malva101" -> "malva 101")
    texto_direccion_normalizado = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', texto_direccion_normalizado)
    
    # Dividir la dirección en componentes (calle, número, colonia, etc.)
    componentes = re.split(r'[,\s]+', texto_direccion_normalizado)
    componentes = [c for c in componentes if c and len(c) > 1]  # Eliminar componentes vacíos y muy cortos
    
    # Identificar posibles números de calle en la búsqueda
    numeros_busqueda = [comp for comp in componentes if comp.isdigit()]
    calles_busqueda = [comp for comp in componentes if not comp.isdigit() and not comp in ['de', 'la', 'del', 'los', 'las', 'y', 'a', 'en', 'el']]
    
    # Extraer componentes clave para el filtrado estricto posterior
    componentes_clave = [comp for comp in componentes if not comp in ['de', 'la', 'del', 'los', 'las', 'y', 'a', 'en', 'el']]
    
    # Imprimir los componentes para depuración
    print(f"[DEBUG] Componentes de búsqueda: {componentes}")
    if numeros_busqueda:
        print(f"[DEBUG] Números detectados: {numeros_busqueda}")
    if calles_busqueda:
        print(f"[DEBUG] Calles/colonias: {calles_busqueda}")
    
    # Almacenar todos los resultados encontrados
    todos_resultados = []
    resultados_exactos = []
    resultados_puntajes = {}  # Para rastrear la relevancia de cada resultado
    
    # Bandera para saber si ya se encontraron coincidencias exactas por número
    encontrado_por_numero = False
    
    # Buscar en todos los índices
    for nombre_dir in os.listdir(ruta_indices):
        ruta_indice = os.path.join(ruta_indices, nombre_dir)
        if not os.path.isdir(ruta_indice) or not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
            continue
            
        fuente = nombre_dir.replace("index_", "")
        
        try:
            # Cargar el índice
            storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
            index = load_index_from_storage(storage_context)
            
            # 1. Búsqueda ESPECÍFICA primero por número exacto si existe en la consulta
            if numeros_busqueda and calles_busqueda:
                # Buscar combinación de calle y número exacto - MUY ALTA PRIORIDAD
                for calle in calles_busqueda[:2]:  # Considerar solo hasta 2 primeras posibles calles
                    for numero in numeros_busqueda:
                        try:
                            # Búsqueda específica en metadatos
                            for metadata_key in ['domicilio', 'calle', 'direccion']:
                                # Buscar por componentes combinados (calle + número)
                                retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
                                consulta_especifica = f"{calle} {numero}"
                                nodes = retriever.retrieve(consulta_especifica)
                                
                                for node in nodes:
                                    metadata = node.node.metadata
                                    
                                    # Verificar coincidencia de número exacto en algún campo de dirección
                                    tiene_numero_exacto = False
                                    tiene_calle = False
                                    
                                    # Buscar el número exacto y la calle en cualquier campo de dirección
                                    for k, v in metadata.items():
                                        if k in ['domicilio', 'calle', 'numero', 'direccion']:
                                            # Extraer posibles números en este campo
                                            nums_en_campo = re.findall(r'\d+', str(v))
                                            if numero in nums_en_campo:
                                                tiene_numero_exacto = True
                                            if calle.lower() in str(v).lower():
                                                tiene_calle = True
                                    
                                    # Si contiene tanto el número exacto como la calle, es una coincidencia muy relevante
                                    if tiene_numero_exacto and tiene_calle:
                                        encontrado_por_numero = True
                                        id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                                        
                                        if id_registro not in resultados_puntajes or 1.0 > resultados_puntajes[id_registro]:
                                            resumen = [f"{k}: {v}" for k, v in metadata.items() 
                                                       if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                                            
                                            resultados_exactos.append({
                                                'texto': f"Coincidencia exacta en {fuente}:\n" + "\n".join(resumen),
                                                'id': id_registro,
                                                'puntaje': 1.0
                                            })
                                            resultados_puntajes[id_registro] = 1.0
                        except Exception as e:
                            print(f"Error en búsqueda de combinación exacta: {e}")
            
            # 2. Si no se encontraron coincidencias exactas por número, hacer búsqueda semántica
            if not encontrado_por_numero:
                # Búsqueda semántica amplia
                retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
                
                # Construir una consulta que incluya todos los componentes principales
                consulta = " ".join(componentes[:4]) if len(componentes) > 4 else texto_direccion_normalizado 
                nodes = retriever.retrieve(consulta)
                
                for node in nodes:
                    metadata = node.node.metadata
                    id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                    
                    # Construir un texto completo con todos los campos de dirección para búsqueda
                    texto_completo = ""
                    campos_direccion_valores = {}
                    
                    for k, v in metadata.items():
                        if k in ['domicilio', 'calle', 'numero', 'colonia', 'sector', 'municipio', 'ciudad', 'estado', 'cp', 'direccion']:
                            texto_completo += f" {v}"
                            campos_direccion_valores[k] = str(v)
                    
                    texto_completo = normalizar_texto(texto_completo)
                    
                    # Verificación de números exactos - ALTA PRIORIDAD
                    if numeros_busqueda:
                        tiene_numero_exacto = False
                        for num in numeros_busqueda:
                            nums_en_texto = re.findall(r'\b\d+\b', texto_completo)
                            if num in nums_en_texto:
                                tiene_numero_exacto = True
                                break
                        
                        # Verificación de números cercanos - MEDIA PRIORIDAD
                        if not tiene_numero_exacto:
                            tiene_numero_cercano = False
                            for num_busqueda in numeros_busqueda:
                                num_busqueda_int = int(num_busqueda)
                                for num_texto in nums_en_texto:
                                    try:
                                        num_texto_int = int(num_texto)
                                        if abs(num_busqueda_int - num_texto_int) <= 30:  # Se consideran cercanos si difieren en 30 o menos
                                            tiene_numero_cercano = True
                                            break
                                    except ValueError:
                                        continue
                    else:
                        tiene_numero_exacto = True  # Si no hay números en la búsqueda, no penalizamos
                        tiene_numero_cercano = True
                    
                    # Verificación estricta: todos los componentes clave deben estar presentes
                    componentes_clave_encontrados = sum(1 for comp in componentes_clave if comp in texto_completo)
                    porcentaje_clave = componentes_clave_encontrados / len(componentes_clave) if componentes_clave else 0
                    
                    # Contar cuántos componentes de la búsqueda están en los datos
                    componentes_encontrados = sum(1 for comp in componentes if comp in texto_completo)
                    calificacion_componentes = componentes_encontrados / len(componentes)
                    
                    # Calcular score final con mayor peso para números exactos
                    similitud_score = similitud(texto_direccion_normalizado, texto_completo)
                    
                    # Ajustar score según coincidencia de números
                    if tiene_numero_exacto:
                        score_final = max(0.9, (0.4 * porcentaje_clave) + (0.3 * calificacion_componentes) + (0.3 * similitud_score))
                    elif tiene_numero_cercano:
                        score_final = max(0.75, (0.5 * porcentaje_clave) + (0.3 * calificacion_componentes) + (0.2 * similitud_score))
                    else:
                        score_final = (0.6 * porcentaje_clave) + (0.2 * calificacion_componentes) + (0.2 * similitud_score)
                    
                    # Filtrado estricto para resultados irrelevantes
                    if (numeros_busqueda and not (tiene_numero_exacto or tiene_numero_cercano)) or porcentaje_clave < 0.5:
                        continue  # Descartar resultados poco relevantes
                    
                    # Verificar si este resultado coincide "perfectamente"
                    es_coincidencia_exacta = tiene_numero_exacto and porcentaje_clave >= 0.8
                    
                    # Almacenar todos los resultados con su puntuación para ordenarlos después
                    if id_registro not in resultados_puntajes or score_final > resultados_puntajes[id_registro]:
                        resumen = [f"{k}: {v}" for k, v in metadata.items() 
                                  if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                        
                        if es_coincidencia_exacta:
                            resultados_exactos.append({
                                'texto': f"Coincidencia exacta en {fuente}:\n" + "\n".join(resumen),
                                'id': id_registro,
                                'puntaje': score_final
                            })
                        
                        todos_resultados.append({
                            'texto': f"Coincidencia en {fuente} (relevancia: {score_final:.2f}):\n" + "\n".join(resumen),
                            'id': id_registro,
                            'puntaje': score_final
                        })
                        resultados_puntajes[id_registro] = score_final
                    
        except Exception as e:
            print(f"Error buscando en {fuente}: {e}")
            continue
    
    # Eliminar duplicados preservando el orden
    resultados_unicos = []
    ids_vistos = set()
    
    # Primero incluir los exactos
    if resultados_exactos:
        resultados_exactos = sorted(resultados_exactos, key=lambda x: x['puntaje'], reverse=True)
        for res in resultados_exactos:
            if res['id'] not in ids_vistos:
                resultados_unicos.append(res)
                ids_vistos.add(res['id'])
    
    # Luego incluir todos los demás ordenados por relevancia
    todos_ordenados = sorted(todos_resultados, key=lambda x: x['puntaje'], reverse=True)
    for res in todos_ordenados:
        if res['id'] not in ids_vistos and res['puntaje'] >= 0.6:  # Filtrar más los resultados adicionales
            resultados_unicos.append(res)
            ids_vistos.add(res['id'])
    
    # Limitar el número total de resultados (ahora usamos un umbral dinámico)
    umbral_minimo_puntaje = 0.6  # Reducido para capturar más resultados potencialmente relevantes
    resultados_finales = [res for res in resultados_unicos if res['puntaje'] >= umbral_minimo_puntaje]
    
    # Si no hay resultados con el umbral, mostrar al menos los mejores 2
    if not resultados_finales and resultados_unicos:
        resultados_finales = resultados_unicos[:2]
    
    # Formatear los resultados
    if resultados_finales:
        # Si hay resultados exactos, mostrarlos con un mensaje
        if any('Coincidencia exacta' in res['texto'] for res in resultados_finales[:3]):
            mensaje = "Se encontraron las siguientes coincidencias:\n\n"
        else:
            mensaje = "No se encontraron coincidencias exactas. Mostrando resultados similares:\n\n"
        
        # Eliminar la información de puntaje para presentación al usuario
        textos_resultados = []
        for res in resultados_finales:
            # Quitar la parte de relevancia del texto
            texto_limpio = re.sub(r'\(relevancia: \d+\.\d+\)', '', res['texto'])
            textos_resultados.append(texto_limpio)
        
        return mensaje + "\n\n".join(textos_resultados)
    else:
        return f"No se encontraron coincidencias relevantes para la dirección '{texto_direccion}'."

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
    fn=buscar_direccion_combinada,
    name="buscar_direccion_combinada",
    description=(
        "Usa esta herramienta cuando el usuario busca una dirección completa o parcial. "
        "Es especialmente útil para direcciones combinadas como 'ZOQUIPAN 1260, LAGOS DEL COUNTRY'. "
        "Por ejemplo: '¿De quién es esta dirección: ZOQUIPAN 1260, LAGOS DEL COUNTRY?', "
        "'Busca ZOQUIPAN 1260', 'Quién vive en casa #63, Zapopan', 'Quién vive en ZAPOPAN, DF', etc. "
        "Esta herramienta realiza búsquedas semánticas en componentes de dirección."
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
            print(f"[DEBUG] Consulta de dirección compleja detectada: '{texto_direccion}'")
            
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