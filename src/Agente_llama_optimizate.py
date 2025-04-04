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
                or metadata.get("Nombre Completo", "")
                or metadata.get("NOMBRE", "")
                or metadata.get("Nombre", "")
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
                or metadata.get("Nombre Completo", "")
                or metadata.get("NOMBRE", "")
                or metadata.get("Nombre", "")
            ).strip().upper()

            sim = similitud(nombre_metadata, query_upper)

            if sim >= 0.5 and fuente not in ya_guardados:
                resumen = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                resultados_top_1.append(
                    f"🔹 Coincidencia cercana en {fuente} (Similitud {sim:.2f}):\n" + "\n".join(resumen)
                )
                ya_guardados.add(fuente)


    if resultados_exactos:
        respuesta_final = "✅ Se encontraron estas coincidencias exactas en los archivos:\n\n" + "\n\n".join(resultados_exactos)
        return respuesta_final

    elif resultados_top_1:
        return "⚠️ No se encontraron coincidencias exactas, pero se encontraron coincidencias cercanas (≥60% de similitud):\n\n" + "\n\n".join(resultados_top_1)

    else:
        return "❌ No se encontraron resultados relevantes en ninguna fuente."

    
busqueda_global_tool = FunctionTool.from_defaults(
    fn=buscar_en_todos_los_indices,
    name="busqueda_semantica_en_todos_los_indices",
    description=(
        "Usa esta herramienta cuando necesites encontrar información completa de una persona, dirección, tarjeta o dato específico "
        "en todas las bases de datos al mismo tiempo. Por ejemplo: 'Dame toda la información de Juan Pérez', '¿Quién vive en malva 101?', "
        "'¿Qué sabes de Adrian Lino Marmolejo?'. Prioriza coincidencias exactas de nombre completo."
    )
)
all_tools.insert(0, busqueda_global_tool)

# Herramienta 3: Buscar personas por atributo específico (Campo y Valor)
def buscar_personas_por_atributo(campo: str, valor: str) -> str:
    """
    Busca personas que coincidan exactamente con un valor en un campo específico en todos los índices disponibles.
    (ej., Telefono, Ciudad, Calle, etc.). Devuelve un resumen o lista de nombres.
    Utiliza filtros de metadatos.
    """
    print(f"⚙️ Ejecutando herramienta: buscar_personas_por_atributo con campo='{campo}', valor='{valor}'")
    resultados = []
    # Normalizar nombres de campos comunes
    campo_map = {
        "telefono": "Telefono", "teléfono": "Telefono",
        "ciudad": "Ciudad", "calle": "Calle",
        "estado": "Estado", "cp": "CP", "código postal": "CP",
        "tarjeta": "Tarjeta", "nombre": "Nombre",
        "direccion": "Dirección", "dirección": "Dirección", "DIRECCION": "Dirección",
        "colonia": "Colonia", "COLONIA": "Colonia",
        "sector": "Sector", "SECTOR": "Sector",
        "municipio": "Municipio", "MUNICIPIO": "Municipio",
        # Añade más mapeos si es necesario
    }
    campo_normalizado = campo_map.get(campo.lower(), campo.capitalize())  # Intentar normalizar

    for fuente, index in indices.items():
        filters = MetadataFilters(filters=[
            ExactMatchFilter(key=campo_normalizado, value=valor.strip()),
            ExactMatchFilter(key="fuente", value=fuente)  # Filtrar por fuente
        ])
        # Recuperar más nodos para ver si hay múltiples coincidencias
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3, filters=filters)
        nodes = retriever.retrieve(f"{campo_normalizado} es {valor}")  # Query semántica relacionada

        if nodes:
            nombres_encontrados = set()
            for node in nodes:
                nombre = node.node.metadata.get("Nombre Completo", "Nombre Desconocido")
                if nombre != "Nombre Desconocido":
                    # Verificar si realmente coincide el valor en los metadatos (doble check)
                    if str(node.node.metadata.get(campo_normalizado, "")).strip().upper() == valor.strip().upper():
                        nombres_encontrados.add(nombre)

            if nombres_encontrados:
                if len(nombres_encontrados) == 1:
                    resultados.append(f"En {fuente}: Se encontró 1 persona con {campo_normalizado} = '{valor}': {list(nombres_encontrados)[0]}")
                else:
                    lista_nombres = ", ".join(list(nombres_encontrados)[:5])  # Mostrar hasta 5
                    resultados.append(f"En {fuente}: Se encontraron {len(nombres_encontrados)} personas con {campo_normalizado} = '{valor}'. Algunas son: {lista_nombres}...")
            else:
                # La búsqueda vectorial encontró algo, pero el filtro exacto falló el doble check
                resultados.append(f"En {fuente}: No se encontró una coincidencia exacta verificada para {campo_normalizado} = '{valor}' en los metadatos recuperados.")

        else:
            resultados.append(f"En {fuente}: No se encontraron personas con {campo_normalizado} = '{valor}'.")

    if resultados:
        return "\n\n".join(resultados)
    else:
        return f"No se encontraron personas con {campo_normalizado} = '{valor}' en ninguna fuente."

atributo_tool = FunctionTool.from_defaults(
    fn=buscar_personas_por_atributo,
    name="buscar_personas_por_atributo_especifico",
    description=(
        "Utiliza esta herramienta cuando necesites encontrar personas que tengan un VALOR específico en un CAMPO conocido. "
        "Por ejemplo: '¿Quién tiene el teléfono 5512345678?', '¿Cuántas personas viven en la Ciudad Mexico?', "
        "'Busca personas cuya calle sea Reforma'. Debes especificar el CAMPO y el VALOR."

        "Por ejemplo: '¿Quién tiene el teléfono 5512345678?', '¿Cuántas personas viven en la Ciudad Mexico?', "
        "'Busca personas cuya calle sea Reforma'. Debes especificar el CAMPO y el VALOR."
    )
)
all_tools.insert(2, atributo_tool)


# --- 4) CREAR Y EJECUTAR EL AGENTE ---

# Crear el agente ReAct (que razona y actúa)
try:
    agent = ReActAgent.from_tools(
        tools=all_tools,
        llm=llm,
        verbose=False  # Muestra los pasos de pensamiento del agente
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
        # Ejecutamos la herramienta y obtenemos la salida
        respuesta_herramienta = buscar_en_todos_los_indices(prompt)

        # Guardamos la salida en una variable
        salida_azul = respuesta_herramienta  # <-- ESTO ES NUEVO

        # Imprimimos la variable "salida_azul"
        print(f"\nSalida de la herramienta:\n{salida_azul}\n")
    except Exception as e:
        print(f"❌ Ocurrió un error durante la ejecución del agente: {e}")
        # Podrías intentar resetear el agente si los errores son persistentes
        # agent.reset()

# Limpiar memoria al salir
del llm, embed_model, agent, all_tools, indices
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()
print("\n👋 ¡Hasta luego!")