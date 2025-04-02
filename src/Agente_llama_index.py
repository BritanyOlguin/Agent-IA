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

# --- 1) CONFIGURACIÓN ---
ruta_modelo_embeddings = r"C:\Users\Sistemas\Documents\OKIP\models\models--intfloat--e5-large-v2"
ruta_indice = r"C:\Users\Sistemas\Documents\OKIP\llama_index_banco_bancomer_e5_large"
ruta_modelo_llama3 = r"C:\Users\Sistemas\Documents\OKIP\models\models--meta-llama--Meta-Llama-3-8B-Instruct"
NOMBRE_FUENTE = "banco_bancomer" # Nombre usado al indexar

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
        local_files_only=True # Los archivos ya están descargados localmente
    )
    print("✅ Tokenizer cargado.")

    model = AutoModelForCausalLM.from_pretrained(
        ruta_modelo_llama3,
        torch_dtype=torch.float16, # Menor uso de VRAM
        load_in_8bit=True,        # Cargar en 8 bits
        device_map="auto",        # Dejar que transformers distribuya en la GPU
        local_files_only=True
    )
    print("✅ Modelo LLM Llama 3 cargado en dispositivo.")

except Exception as e:
    print(f"❌ Error al cargar Llama 3 desde {ruta_modelo_llama3}: {e}")
    print(" Asegúrate de que la ruta es correcta, el modelo está descargado, aceptaste los términos y tienes bitsandbytes instalado si usas load_in_8bit=True.")
    exit()

# --- CONFIGURAR HuggingFaceLLM USANDO OBJETOS CARGADOS ---
try:
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=8000, # Llama 3 tiene ventana de 8k, ajusta si es necesario
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
        normalize=True # Mantener True si lo usaste al indexar
        )
    print("✅ Modelo de embeddings e5-large-v2 cargado.")
except Exception as e:
    print(f"❌ Error cargando el modelo de embeddings desde {ruta_modelo_embeddings}: {e}")
    exit()

# Aplicar configuración global a LlamaIndex (sin cambios)
Settings.llm = llm
Settings.embed_model = embed_model

# --- 2) CARGAR ÍNDICE ---
print(f"Cargando índice desde: {ruta_indice}")
if not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
    print(f"❌ Error: No se encontró el índice en {ruta_indice}.")
    print("   Asegúrate que la ruta sea correcta y que el índice (creado con e5-large-v2) exista.")
    exit()

try:
    # Carga el índice desde la ruta correcta
    storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
    index = load_index_from_storage(storage_context)
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    query_engine = index.as_query_engine(streaming=False)
    print("✅ Índice (e5-large-v2) cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el índice: {e}")
    exit()

# --- 3) DEFINIR HERRAMIENTAS (TOOLS) ---

# Herramienta 1: Búsqueda Semántica General (Usa el retriever del índice)
semantic_search_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="busqueda_semantica_general",
    description=(
        "Utiliza esta herramienta para responder preguntas generales sobre los datos "
        "o cuando necesites encontrar información basada en el significado o contexto, "
        "no solo coincidencias exactas. Es buena para preguntas abiertas como 'Háblame de...', "
        "'¿Qué información hay sobre...?', o para encontrar relaciones indirectas."
    )
)

# Herramienta 2: Obtener información completa por Nombre Completo (Usa Metadatos)
def obtener_info_por_nombre(nombre_completo: str) -> str:
    """
    Busca toda la información asociada a una persona dado su nombre completo exacto.
    Utiliza filtros de metadatos para una búsqueda precisa.
    """
    print(f"⚙️ Ejecutando herramienta: obtener_info_por_nombre con nombre='{nombre_completo}'")
    filters = MetadataFilters(filters=[
        ExactMatchFilter(key="Nombre Completo", value=nombre_completo.strip()),
        ExactMatchFilter(key="fuente", value=NOMBRE_FUENTE) # Filtrar por fuente
        ]
    )
    retriever = VectorIndexRetriever(index=index, similarity_top_k=1, filters=filters) # Solo queremos 1 si es exacto
    nodes = retriever.retrieve(nombre_completo) # Usamos el nombre como query también

    if nodes:
        # Devolver la información formateada de los metadatos del primer nodo encontrado
        metadata = nodes[0].node.metadata
        info = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
        return f"Información encontrada para {nombre_completo}:\n" + "\n".join(info)
    else:
        return f"No se encontró información exacta para la persona con nombre completo: {nombre_completo}"

info_nombre_tool = FunctionTool.from_defaults(
    fn=obtener_info_por_nombre,
    name="obtener_info_por_nombre_completo",
    description=(
        "Utiliza esta herramienta EXCLUSIVAMENTE cuando necesites obtener TODA la información "
        "disponible de una persona y tengas su NOMBRE COMPLETO exacto. "
        "Por ejemplo: 'Dame la información de Juan Perez Garcia' o 'Datos de Maria Lopez Aguilar'."
    )
)

# Herramienta 3: Buscar personas por atributo específico (Campo y Valor)
def buscar_personas_por_atributo(campo: str, valor: str) -> str:
    """
    Busca personas que coincidan exactamente con un valor en un campo específico
    (ej., Telefono, Ciudad, Calle, etc.). Devuelve un resumen o lista de nombres.
    Utiliza filtros de metadatos.
    """
    print(f"⚙️ Ejecutando herramienta: buscar_personas_por_atributo con campo='{campo}', valor='{valor}'")
    # Normalizar nombres de campos comunes (opcional pero útil)
    campo_map = {
        "telefono": "Telefono", "teléfono": "Telefono",
        "ciudad": "Ciudad",
        "calle": "Calle",
        "estado": "Estado",
        "cp": "CP", "código postal": "CP",
        "tarjeta": "Tarjeta"
        # Añade más mapeos si es necesario
    }
    campo_normalizado = campo_map.get(campo.lower(), campo.capitalize()) # Intentar normalizar

    filters = MetadataFilters(filters=[
        ExactMatchFilter(key=campo_normalizado, value=valor.strip()),
        ExactMatchFilter(key="fuente", value=NOMBRE_FUENTE) # Filtrar por fuente
        ]
    )
    # Recuperar más nodos para ver si hay múltiples coincidencias
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10, filters=filters)
    nodes = retriever.retrieve(f"{campo_normalizado} es {valor}") # Query semántica relacionada

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
                 return f"Se encontró 1 persona con {campo_normalizado} = '{valor}': {list(nombres_encontrados)[0]}"
            else:
                 lista_nombres = ", ".join(list(nombres_encontrados)[:5]) # Mostrar hasta 5
                 return (f"Se encontraron {len(nombres_encontrados)} personas con {campo_normalizado} = '{valor}'. "
                         f"Algunas son: {lista_nombres}...")
        else:
            # La búsqueda vectorial encontró algo, pero el filtro exacto falló el doble check
             return f"No se encontró una coincidencia exacta verificada para {campo_normalizado} = '{valor}' en los metadatos recuperados."

    else:
        return f"No se encontraron personas con {campo_normalizado} = '{valor}'."

atributo_tool = FunctionTool.from_defaults(
    fn=buscar_personas_por_atributo,
    name="buscar_personas_por_atributo_especifico",
    description=(
        "Utiliza esta herramienta cuando necesites encontrar personas que tengan un VALOR específico en un CAMPO conocido. "
        "Por ejemplo: '¿Quién tiene el teléfono 5512345678?', '¿Cuántas personas viven en la Ciudad Mexico?', "
        "'Busca personas cuya calle sea Reforma'. Debes especificar el CAMPO y el VALOR."
    )
)


# Lista de todas las herramientas disponibles para el agente
all_tools = [semantic_search_tool, info_nombre_tool, atributo_tool]

# --- 4) CREAR Y EJECUTAR EL AGENTE ---

# Crear el agente ReAct (que razona y actúa)
try:
    agent = ReActAgent.from_tools(
        tools=all_tools,
        llm=llm,
        verbose=True # Muestra los pasos de pensamiento del agente
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
        response = agent.chat(prompt)
        print(f"\nRespuesta: {response}\n")
    except Exception as e:
        print(f"❌ Ocurrió un error durante la ejecución del agente: {e}")
        # Podrías intentar resetear el agente si los errores son persistentes
        # agent.reset()

# Limpiar memoria al salir
del llm, embed_model, index, agent, all_tools
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()
print("\n👋 ¡Hasta luego!")