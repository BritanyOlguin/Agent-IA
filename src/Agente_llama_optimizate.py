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
sys.path.append(r"C:\Users\Sistemas\Documents\OKIP\src")
from utils import normalizar_texto
import re

def detectar_campo_y_valor(prompt: str):
    prompt_lower = prompt.lower()

    # Ordena los campos por longitud de alias descendente para evitar capturar 'numero' antes que 'numero de tarjeta'
    aliases_ordenados = sorted([
        (campo_estandarizado, alias)
        for campo_estandarizado, alias_list in campos_clave.items()
        for alias in alias_list
    ], key=lambda x: -len(x[1]))  # ordenar por longitud del alias (más largo primero)

    for campo_estandarizado, alias in aliases_ordenados:
        if alias in prompt_lower:
            # buscar con regex si hay valor después del alias
            pattern = re.compile(rf"{alias}\s*(es|:)?\s*([\w\d\s\-.,]+)", re.IGNORECASE)
            match = pattern.search(prompt)
            if match:
                valor = match.group(2).strip()
                return campo_estandarizado, valor

            # si no encuentra valor explícito, busca un número largo (teléfono o tarjeta)
            numeros = re.findall(r"\d{7,}", prompt)
            if numeros:
                return campo_estandarizado, numeros[0]

    return None, None

def similitud(texto1, texto2):
    return SequenceMatcher(None, texto1, texto2).ratio()

# --- 1) CONFIGURACIÓN ---
ruta_modelo_embeddings = r"C:\Users\Sistemas\Documents\OKIP\models\models--intfloat--e5-large-v2"
ruta_indices = r"C:\Users\Sistemas\Documents\OKIP\llama_index_indices"
ruta_modelo_llama3 = r"C:\Users\Sistemas\Documents\OKIP\models\models--meta-llama--Meta-Llama-3-8B-Instruct"

# Configuración de Dispositivo y LLM
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("⚠️ Advertencia: Usando CPU para LLM. Las respuestas serán lentas.")
else:
    print(f"💻 Usando dispositivo para LLM y Embeddings: {device}")

# --- CARGAR MODELO Y TOKENIZER CON TRANSFORMERS ---
print(f" Cargando Tokenizer y Modelo Llama 3 desde: {ruta_modelo_llama3}")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        ruta_modelo_llama3,
        local_files_only=True  # Los archivos ya están descargados localmente
    )
    print("✅ Tokenizer cargado.")

    model = AutoModelForCausalLM.from_pretrained(
        ruta_modelo_llama3,
        torch_dtype=torch.float16,  # Menor uso de VRAM
        load_in_8bit=True,  # Cargar en 8 bits
        device_map="auto",  # Dejar que transformers distribuya en la GPU
        local_files_only=True
    )
    print("✅ Modelo LLM Llama 3 cargado en dispositivo.")

except Exception as e:
    print(f"❌ Error al cargar Llama 3 desde {ruta_modelo_llama3}: {e}")
    print(
        " Asegúrate de que la ruta es correcta, el modelo está descargado, aceptaste los términos y tienes bitsandbytes instalado si usas load_in_8bit=True."
    )
    exit()

# --- CONFIGURAR HuggingFaceLLM USANDO OBJETOS CARGADOS ---
try:
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=8000,  # Llama 3 tiene ventana de 8k, ajusta si es necesario
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.1, "do_sample": False},
    )
    print("✅ HuggingFaceLLM configurado con modelo Llama 3 cargado.")
except Exception as e:
    print(f"❌ Error al configurar HuggingFaceLLM con modelo pre-cargado: {e}")
    exit()

# Configurar Modelo de Embeddings (sin cambios)
print(f"⚙️ Cargando modelo de embeddings: {os.path.basename(ruta_modelo_embeddings)}")
try:
    # Asegúrate que coincida con la configuración del script de indexación
    embed_model = HuggingFaceEmbedding(
        model_name=ruta_modelo_embeddings,
        device=device,
        normalize=True  # Mantener True si lo usaste al indexar
    )
    print("✅ Modelo de embeddings e5-large-v2 cargado.")
except Exception as e:
    print(f"❌ Error cargando el modelo de embeddings desde {ruta_modelo_embeddings}: {e}")
    exit()

# Aplicar configuración global a LlamaIndex (sin cambios)
Settings.llm = llm
Settings.embed_model = embed_model

# --- 2) CARGAR TODOS LOS ÍNDICES DE LA CARPETA DE ÍNDICES ---
all_tools = []
indices = {}  # Diccionario para almacenar los índices cargados

print(f"\n🔎 Buscando índices en: {ruta_indices}")
for nombre_dir in os.listdir(ruta_indices):
    ruta_indice = os.path.join(ruta_indices, nombre_dir)
    if not os.path.isdir(ruta_indice):
        continue
    if not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
        print(f"⚠️ Saltando {ruta_indice}, no contiene índice válido.")
        continue

    fuente = nombre_dir.replace("index_", "")  # Extraer el nombre de la fuente

    try:
        print(f"📂 Cargando índice para fuente: {fuente}")
        storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
        index = load_index_from_storage(storage_context)
        indices[fuente] = index  # Almacenar el índice en el diccionario

        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        query_engine = index.as_query_engine(streaming=False)

    except Exception as e:
        print(f"❌ Error al cargar índice {ruta_indice}: {e}")

# -----------------------------
# 🔍 DETECTAR CAMPOS DISPONIBLES DESDE LOS ÍNDICES
# -----------------------------
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
            break  # solo necesitamos uno por índice

    except Exception as e:
        print(f"⚠️ Error al explorar metadatos en {nombre_dir}: {e}")
        continue

# 🔧 Alias comunes para mapear variaciones a campos clave
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

# 🧠 Construir `mapa_campos` y `campos_clave` automáticamente
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

        
# Herramienta 1: Búsqueda Semántica Global Mejorada
def buscar_en_todos_los_indices(query: str) -> str:
    print(f"⚙️ Ejecutando búsqueda semántica global mejorada: '{query}'")
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
                    f"✅ Coincidencia exacta en {fuente}:\n" + "\n".join(resumen)
                )
                ya_guardados.add(fuente)

        # Si no hay exacto, guardar las mejores coincidencias con similitud >= 0.9
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
        respuesta_final = "✅ Se encontraron estas coincidencias exactas en los archivos:\n\n" + "\n\n".join(resultados_exactos)
        return respuesta_final

    elif resultados_top_1:
        return "✅ Se encontraron estas coincidencias:\n\n" + "\n\n".join(resultados_top_1)

    else:
        return "❌ No se encontraron resultados relevantes en ninguna fuente."

    
busqueda_global_tool = FunctionTool.from_defaults(
    fn=buscar_en_todos_los_indices,
    name="busqueda_semantica_en_todos_los_indices",
    description=(
        "Usa esta herramienta para encontrar información completa de una persona en todas las bases, "
        "cuando el usuario da el nombre completo. Por ejemplo: 'Dame la información de Juan Pérez', "
        "'¿Qué sabes de Adrian Lino Marmolejo?'."
    )
)
all_tools.insert(0, busqueda_global_tool)

# Herramienta 3: Buscar personas por atributo específico (Campo y Valor)
def buscar_por_atributo_en_indices(campo: str, valor: str, carpeta_indices: str) -> str:
    """
    Busca coincidencias exactas por campo y valor en todos los índices dentro de la carpeta dada.
    Aplica normalización para coincidir con los metadatos indexados.
    """
    print(f"\n🔍 Buscando registros donde '{campo}' = '{valor}'\n")

    # Mapa para alias de campos comunes
    # usa mapa_campos ya generado dinámicamente arriba
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
                    resultados.append(f"✅ Coincidencia en {fuente}:\n" + "\n".join(resumen))
        except Exception as e:
            print(f"⚠️ Error al buscar en {fuente}: {e}")
            continue

    if resultados:
        return "\n".join(resultados)
    else:
        return f"❌ No se encontraron coincidencias para '{campo}: {valor}' en los índices."
    
# Envolver la función con la ruta real de índices
buscar_por_atributo_tool = FunctionTool.from_defaults(
    fn=lambda campo, valor: buscar_por_atributo_en_indices(campo, valor, carpeta_indices=ruta_indices),
    name="buscar_por_atributo_en_indices",
    description=(
        "Usa esta herramienta cuando el usuario pregunta por un campo específico como teléfono, dirección, estado, tarjeta, etc. "
        "Por ejemplo: '¿Quién tiene el número 5544332211?', '¿Quién vive en Malva 101?', '¿Quién tiene la tarjeta terminación 8841?', "
        "'¿Qué personas viven en Querétaro?', '¿Quién vive en calle Reforma 123?'."
    )
)
all_tools.insert(1, buscar_por_atributo_tool)



# --- 4) CREAR Y EJECUTAR EL AGENTE ---

# Crear el agente ReAct (que razona y actúa)
try:
    agent = ReActAgent.from_tools(
        tools=all_tools,
        llm=llm,
        verbose=True  # Muestra los pasos de pensamiento del agente
    )
    print("✅ Agente creado correctamente.")
except Exception as e:
    print(f"❌ Error al crear el agente: {e}")
    exit()

print("\n🤖 Agente listo. Escribe tu pregunta o 'salir' para terminar.")

# Ciclo de chat
while True:
    prompt = input("Pregunta: ")
    if prompt.lower() == 'salir':
        break
    if not prompt:
        continue

    try:
        campo, valor = detectar_campo_y_valor(prompt)

        if campo and valor:
            respuesta_herramienta = buscar_por_atributo_en_indices(campo, valor, carpeta_indices=ruta_indices)
        else:
            respuesta_herramienta = buscar_en_todos_los_indices(prompt)

        print(f"\nSalida de la herramienta:\n{respuesta_herramienta}\n")

    except Exception as e:
        print(f"❌ Ocurrió un error durante la ejecución del agente: {e}")


# Limpiar memoria al salir
del llm, embed_model, agent, all_tools, indices
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()
print("\n👋 ¡Hasta luego!")