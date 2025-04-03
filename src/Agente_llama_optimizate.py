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
ruta_indices = r"C:\Users\Sistemas\Documents\OKIP\llama_index_indices"  # Carpeta de índices
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

        retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        query_engine = index.as_query_engine(streaming=False)

        # Herramienta de búsqueda general para esta fuente
        tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=f"busqueda_semantica_{fuente}",
            description=f"Consulta semántica para la fuente '{fuente}'. Úsala para preguntas generales sobre esta base."
        )
        all_tools.append(tool)

    except Exception as e:
        print(f"❌ Error al cargar índice {ruta_indice}: {e}")


# Herramienta 2: Obtener información completa por Nombre Completo (Usa Metadatos)
def obtener_info_por_nombre(nombre_completo: str) -> str:
    """
    Busca toda la información asociada a una persona dado su nombre completo,
    intentando diferentes campos de nombre en todos los índices disponibles.
    También maneja el caso de nombres separados en campos PATERNO, MATERNO y NOMBRE.
    """
    print(f"⚙️ Ejecutando herramienta: obtener_info_por_nombre con nombre='{nombre_completo}'")
    resultados = []
    
    # Lista de posibles campos de nombre en los diferentes índices
    campos_nombre = ["Nombre Completo", "NOMBRE", "Nombre"]
    
    for fuente, index in indices.items():
        encontrado = False
        
        # Probar cada campo de nombre posible
        for campo_nombre in campos_nombre:
            if encontrado:
                break
                
            filters = MetadataFilters(filters=[
                ExactMatchFilter(key=campo_nombre, value=nombre_completo.strip()),
                ExactMatchFilter(key="fuente", value=fuente)
            ])
            
            retriever = VectorIndexRetriever(index=index, similarity_top_k=1, filters=filters)
            nodes = retriever.retrieve(nombre_completo)
            
            if nodes:
                # Encontramos coincidencia con este campo
                encontrado = True
                metadata = nodes[0].node.metadata
                info = [f"{k}: {v}" for k, v in metadata.items() 
                       if k not in ['fuente', 'archivo', 'fila_excel', 'fila_origen'] and v]
                resultados.append(f"Información encontrada en {fuente} para {nombre_completo}:\n" + "\n".join(info))
        
        # Buscar nombre en formato separado (PATERNO, MATERNO, NOMBRE)
        if not encontrado:
            # Intentar formar un nombre completo a partir de los componentes
            retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
            nodes = retriever.retrieve(nombre_completo)
            
            for node in nodes:
                metadata = node.node.metadata
                
                # Verificar si tenemos los campos separados
                paterno = metadata.get("PATERNO", "")
                materno = metadata.get("MATERNO", "")
                primer_nombre = metadata.get("NOMBRE", "")
                
                # Intentar formar nombre completo de diferentes maneras
                posibles_nombres = []
                if paterno and materno and primer_nombre:
                    posibles_nombres.append(f"{paterno} {materno} {primer_nombre}")
                    posibles_nombres.append(f"{primer_nombre} {paterno} {materno}")
                
                # Verificar si alguna versión coincide con el nombre buscado
                if any(nombre.upper().strip() == nombre_completo.upper().strip() for nombre in posibles_nombres):
                    encontrado = True
                    info = [f"{k}: {v}" for k, v in metadata.items() 
                           if k not in ['fuente', 'archivo', 'fila_excel', 'fila_origen'] and v]
                    resultados.append(f"Información encontrada en {fuente} para {nombre_completo}:\n" + "\n".join(info))
                    break
                
                # Si no hay coincidencia exacta, buscar coincidencia parcial
                palabras_buscadas = [p.upper() for p in nombre_completo.split() if len(p) > 2]
                
                # Combinar todos los campos de nombre para buscar coincidencia parcial
                nombre_compuesto = f"{paterno} {materno} {primer_nombre}".upper()
                
                if nombre_compuesto and all(p in nombre_compuesto for p in palabras_buscadas):
                    info = [f"{k}: {v}" for k, v in metadata.items() 
                           if k not in ['fuente', 'archivo', 'fila_excel', 'fila_origen'] and v]
                    resultados.append(f"Posible coincidencia en {fuente} para {nombre_completo}:\n" + "\n".join(info))
                    break  # Solo mostrar la primera coincidencia parcial

    if resultados:
        return "\n\n".join(resultados)
    else:
        return f"No se encontró información para la persona con nombre: {nombre_completo}"

# 2. Agregar una nueva herramienta para buscar por nombre separado

def buscar_por_nombre_compuesto(paterno: str, materno: str, nombre: str) -> str:
    """
    Busca una persona usando sus apellidos y nombre separados.
    """
    print(f"⚙️ Ejecutando herramienta: buscar_por_nombre_compuesto con P:{paterno}, M:{materno}, N:{nombre}")
    
    # Formar diferentes variantes del nombre completo
    nombre_completo_1 = f"{paterno} {materno} {nombre}".strip()
    nombre_completo_2 = f"{nombre} {paterno} {materno}".strip()
    
    # Buscar con la primera variante
    resultado_1 = obtener_info_por_nombre(nombre_completo_1)
    
    # Si no se encontró con la primera variante, intentar con la segunda
    if "No se encontró información" in resultado_1:
        resultado_2 = obtener_info_por_nombre(nombre_completo_2)
        if "No se encontró información" not in resultado_2:
            return resultado_2
    
    return resultado_1

nombre_compuesto_tool = FunctionTool.from_defaults(
    fn=buscar_por_nombre_compuesto,
    name="buscar_por_nombre_compuesto",
    description=(
        "Utiliza esta herramienta cuando tengas los componentes separados del nombre "
        "(apellido paterno, apellido materno y nombre). "
        "Por ejemplo: 'Busca información de la persona con apellido paterno Acevedo, "
        "apellido materno Perez y nombre Guadalupe'."
    )
)
all_tools.append(nombre_compuesto_tool)

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
        "clave ife": "CLAVE IFE", "ife": "CLAVE IFE",
        "paterno": "PATERNO", "apellido paterno": "PATERNO",
        "materno": "MATERNO", "apellido materno": "MATERNO",
        "edo de origen": "EDO DE ORIGEN", "estado de origen": "EDO DE ORIGEN",
        "sexo": "SEXO", "género": "SEXO",
        "ocupacion": "OCUPACION", "ocupación": "OCUPACION",
        "domicilio: calle": "DOMICILIO: CALLE", "calle": "DOMICILIO: CALLE",
        "número": "NÚMERO", "num": "NÚMERO", "numero": "NÚMERO",
        "código postal": "CODIGO POSTAL", "codigo postal": "CODIGO POSTAL", "cp": "CODIGO POSTAL",
        "fecha afiliacion": "FECHA_AFILIACION", "fecha_afiliacion": "FECHA_AFILIACION", 
        "fecha de afiliacion": "FECHA_AFILIACION", "fecha de afiliación": "FECHA_AFILIACION",
        "entidad": "ENTIDAD", "estado": "ENTIDAD", "entidad federativa": "ENTIDAD",
        # Añade más mapeos si es necesario
    }
    campo_normalizado = campo_map.get(campo.lower(), campo.capitalize())  # Intentar normalizar

    for fuente, index in indices.items():
        filters = MetadataFilters(filters=[
            ExactMatchFilter(key=campo_normalizado, value=valor.strip()),
            ExactMatchFilter(key="fuente", value=fuente)  # Filtrar por fuente
        ])
        # Recuperar más nodos para ver si hay múltiples coincidencias
        retriever = VectorIndexRetriever(index=index, similarity_top_k=10, filters=filters)
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
all_tools.append(atributo_tool)

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
        response = agent.chat(prompt)
        print(f"\nRespuesta: {response}\n")
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
