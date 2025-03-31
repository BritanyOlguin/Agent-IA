import os
import pandas as pd
import tqdm
import torch
import time
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
import gc

# --- 1) CONFIGURACIÓN ---
ruta_modelo_embeddings = r"C:\Users\Sistemas\Documents\OKIP\src\models--intfloat--e5-large-v2"

ruta_datos_excel = r"C:\Users\Sistemas\Documents\OKIP\banco_bancomer.xlsx"
ruta_llama_index = r"C:\Users\Sistemas\Documents\OKIP\llama_index_banco_bancomer_e5_large"
NOMBRE_FUENTE = "banco_bancomer" # Identificador único para esta fuente

os.makedirs(ruta_llama_index, exist_ok=True)

# Forzar uso de GPU si está disponible (Opcional, pero recomendado para velocidad)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("⚠️ Advertencia: No se detectó GPU. La indexación será más lenta, especialmente con e5-large-v2.")
else:
    print(f"💻 Usando dispositivo para embeddings: {device}")

# Configurar LlamaIndex Settings
print(f"⚙️ Cargando modelo de embeddings: {os.path.basename(ruta_modelo_embeddings)}")
try:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=ruta_modelo_embeddings,
        device=device,
        normalize=True # Añadido: Normalizar embeddings suele mejorar el rendimiento de E5
        )
    print("✅ Modelo de embeddings cargado.")
except Exception as e:
    print(f"❌ Error cargando el modelo de embeddings desde {ruta_modelo_embeddings}: {e}")
    print("   Asegúrate que la ruta sea correcta o que el identificador exista en Hugging Face.")
    exit()

# Settings.llm = None # No necesitamos LLM para indexar
# Settings.chunk_size = 512 # Ajustar si es necesario

# --- 2) CARGAR DATOS ---
print(f"Cargando datos desde: {ruta_datos_excel}")
try:
    # Asegurarse de leer todo como texto inicialmente
    df = pd.read_excel(ruta_datos_excel, dtype=str, engine="openpyxl").fillna("")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo Excel en {ruta_datos_excel}")
    exit()
except Exception as e:
    print(f"❌ Error al leer el archivo Excel: {e}")
    exit()

total_filas = len(df)
# e5-large-v2 es más pesado, podrías necesitar reducir el batch_size si tienes errores de memoria GPU
batch_size = 100 # Reducido (ajusta según tu VRAM)
total_lotes = (total_filas + batch_size - 1) // batch_size
print(f"📄 Total de filas a procesar: {total_filas} en {total_lotes} lotes de {batch_size}")

nombre_archivo = os.path.basename(ruta_datos_excel)

# --- 3) INDEXAR POR LOTES ---

# Comprobar si ya existe un índice y preguntar si se quiere sobreescribir o añadir
index = None
if os.path.exists(os.path.join(ruta_llama_index, "docstore.json")):
    print(f"ℹ️ Índice existente encontrado en {ruta_llama_index}.")
    print("\n‼️ IMPORTANTE: Como cambiaste el modelo de embeddings, DEBES sobreescribir el índice.")
    respuesta = input("¿Desea [s]obreescribir el índice existente? (S/n): ").lower()
    # Forzar 's' si no se responde 'n', ya que es necesario sobreescribir
    if respuesta == 'n':
        print("❌ Cancelando. No se puede añadir a un índice creado con un modelo de embedding diferente.")
        exit()
    else:
         print("⚠️ Sobreescribiendo índice existente (requerido por cambio de modelo).")
         # No es necesario borrar manualmente, LlamaIndex sobreescribirá al persistir un nuevo índice.

# Lista para almacenar todos los documentos del DataFrame
documentos_totales = []

start_time_total = time.time()

# Preparar documentos y metadatos
print("⚙️ Preparando documentos desde el DataFrame...")
with tqdm.tqdm(total=total_filas, desc="📄 Preparando documentos", unit=" filas") as pbar:
    for i, row in df.iterrows():
        datos_fila = {col: str(row[col]).strip() for col in df.columns}

        # Limpiar valores no deseados o vacíos
        datos_fila_limpios = {}
        for col, valor in datos_fila.items():
             # Convertir 'nan' o similares a string vacío
            if pd.isna(valor) or str(valor).lower() == 'nan':
                datos_fila_limpios[col] = ""
            # Eliminar números específicos si son considerados inválidos/ruido
            elif valor == "3586127":
                datos_fila_limpios[col] = ""
            else:
                datos_fila_limpios[col] = valor

        # Crear el texto principal para indexar (más limpio)
        # Solo incluimos columnas con valor
        texto_base_items = [f"{col}: {valor}" for col, valor in datos_fila_limpios.items() if valor]
        texto_base = "\n".join(texto_base_items)

        # --- Opcional: Añadir prefijo "passage:" para E5 ---
        # texto_a_incrustar = "passage: " + texto_base
        # Descomenta la línea anterior y usa texto_a_incrustar en Document()
        # si quieres probar con el prefijo recomendado por E5. Empieza sin él.
        # ---------------------------------------------------
        texto_a_incrustar = texto_base # Usar sin prefijo inicialmente

        # Crear metadatos (importante para filtros)
        metadata = {
            "fuente": NOMBRE_FUENTE,
            "archivo": nombre_archivo,
            "fila_excel": i,
            **datos_fila_limpios # Añadir todos los datos limpios como metadatos
        }

        # Crear documento LlamaIndex (sin embedding aquí, se hará al indexar)
        documentos_totales.append(Document(text=texto_a_incrustar, metadata=metadata))
        pbar.update(1)

# Indexar todos los documentos a la vez o actualizar índice
if documentos_totales:
    print(f"\n⚙️ Indexando {len(documentos_totales)} documentos con e5-large-v2...")
    print("   (Esto puede tardar más que con MiniLM)")
    start_time_indexing = time.time()

    # Siempre creamos uno nuevo ya que forzamos sobreescribir si existía
    print(" Creando un nuevo índice...")

    embed_batch_size = 32
    print(f"   Usando embed_batch_size = {embed_batch_size}")

    # Pasar embed_model explícitamente puede ser más robusto que depender solo de Settings
    index = VectorStoreIndex(
        documentos_totales,
        embed_model=Settings.embed_model, # Pasar el modelo configurado
        embed_batch_size=embed_batch_size,
        show_progress=True
        )

    print("💾 Guardando índice en disco...")
    # Asegurarse de que el directorio esté vacío si se va a sobreescribir
    # (LlamaIndex debería manejarlo, pero limpiar manualmente podría ser más seguro)
    # import shutil
    # if os.path.exists(ruta_llama_index):
    #     shutil.rmtree(ruta_llama_index)
    # os.makedirs(ruta_llama_index)
    index.storage_context.persist(persist_dir=ruta_llama_index)

    duracion_indexing = time.time() - start_time_indexing
    print(f"✅ Indexación completada en {duracion_indexing:.2f} segundos.")
else:
    print("⚠️ No se prepararon documentos para indexar.")

# Limpiar memoria
del df, documentos_totales, index
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()

duracion_total = time.time() - start_time_total
print(f"\n🎉 Proceso de indexación finalizado en {duracion_total:.2f} segundos.")
print(f"📂 Índice guardado en: {ruta_llama_index}")