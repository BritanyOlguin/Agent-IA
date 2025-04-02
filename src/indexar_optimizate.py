import os
import pandas as pd
import tqdm
import torch
import time
import gc
import pyodbc
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# --- CONFIGURACIÓN GLOBAL ---
ruta_modelo_embeddings = r"C:\Users\Sistemas\Documents\OKIP\models\models--intfloat--e5-large-v2"
carpeta_bd = r"C:\Users\Sistemas\Documents\OKIP\archivos"
carpeta_indices = r"C:\Users\Sistemas\Documents\OKIP\llama_index_indices"

os.makedirs(carpeta_indices, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Dispositivo de embeddings: {device}")

Settings.embed_model = HuggingFaceEmbedding(
    model_name=ruta_modelo_embeddings,
    device=device,
    normalize=True
)

def cargar_dataframe(archivo):
    ext = os.path.splitext(archivo)[1].lower()
    print(f"📁 Detectado archivo: {archivo} (tipo: {ext})")
    try:
        if ext in ['.xlsx', '.xls']:
            return pd.read_excel(archivo, dtype=str, engine="openpyxl").fillna("")
        elif ext in ['.accdb', '.mdb']:
            conn_str = (
                r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
                f"DBQ={archivo};"
            )
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            tablas = [row.table_name for row in cursor.tables(tableType='TABLE')]
            if not tablas:
                raise ValueError("No se encontraron tablas en la base de datos Access.")
            print(f"📌 Usando tabla: {tablas[0]}")
            df = pd.read_sql(f"SELECT * FROM {tablas[0]}", conn).fillna("")
            conn.close()
            return df
        elif ext == '.dbf':
            from dbfread import DBF
            table = DBF(archivo, load=True)
            df = pd.DataFrame(iter(table)).fillna("")
            return df
        else:
            print(f"⚠️ Formato no soportado: {ext}")
            return None
    except Exception as e:
        print(f"❌ Error al cargar archivo {archivo}: {e}")
        return None

# --- RECORRER CARPETA Y PROCESAR ARCHIVOS ---
for archivo_nombre in os.listdir(carpeta_bd):
    ruta_archivo = os.path.join(carpeta_bd, archivo_nombre)
    if not os.path.isfile(ruta_archivo):
        continue

    nombre_archivo = os.path.basename(ruta_archivo)
    nombre_fuente = os.path.splitext(nombre_archivo)[0]
    ruta_llama_index = os.path.join(carpeta_indices, f"index_{nombre_fuente}")
    os.makedirs(ruta_llama_index, exist_ok=True)

    print(f"\n============================")
    print(f"📦 Procesando: {nombre_archivo}")
    print(f"📌 Fuente: {nombre_fuente}")
    print(f"📂 Índice: {ruta_llama_index}")

    df = cargar_dataframe(ruta_archivo)
    if df is None or df.empty:
        print("⚠️ Archivo vacío o no se pudo procesar. Saltando...")
        continue

    total_filas = len(df)
    batch_size = 100
    print(f"📄 Filas a procesar: {total_filas}")

    documentos_totales = []
    with tqdm.tqdm(total=total_filas, desc="📄 Preparando documentos", unit=" filas") as pbar:
        for i, row in df.iterrows():
            datos_fila = {col: str(row[col]).strip() for col in df.columns}
            datos_fila_limpios = {col: "" if pd.isna(v) or str(v).lower() in ["nan", "3586127"] else str(v) for col, v in datos_fila.items()}
            texto_base = "\n".join([f"{col}: {v}" for col, v in datos_fila_limpios.items() if v])
            texto_a_incrustar = texto_base
            metadata = {
                "fuente": nombre_fuente,
                "archivo": nombre_archivo,
                "fila_origen": i,
                **datos_fila_limpios
            }
            documentos_totales.append(Document(text=texto_a_incrustar, metadata=metadata))
            pbar.update(1)

    if documentos_totales:
        print(f"⚙️ Indexando {len(documentos_totales)} documentos...")
        start_time_indexing = time.time()
        index = VectorStoreIndex(
            documentos_totales,
            embed_model=Settings.embed_model,
            embed_batch_size=32,
            show_progress=True
        )
        index.storage_context.persist(persist_dir=ruta_llama_index)
        print(f"✅ Indexación completada en {time.time() - start_time_indexing:.2f} segundos.")
    else:
        print("⚠️ No se generaron documentos.")

    del df, documentos_totales, index
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
