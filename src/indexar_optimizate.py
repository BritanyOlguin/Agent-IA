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
        elif ext == '.csv':
            # Intentar con diferentes codificaciones y delimitadores comunes
            try:
                # Primero intentar con UTF-8 y delimitador ',' (estándar)
                return pd.read_csv(archivo, dtype=str, encoding='utf-8', delimiter=',').fillna("")
            except UnicodeDecodeError:
                # Si falla, intentar con latin-1 (común para datos en español)
                return pd.read_csv(archivo, dtype=str, encoding='latin-1', delimiter=',').fillna("")
            except:
                # Si aún falla, intentar con otros delimitadores (punto y coma es común en países europeos)
                try:
                    return pd.read_csv(archivo, dtype=str, encoding='utf-8', delimiter=';').fillna("")
                except:
                    return pd.read_csv(archivo, dtype=str, encoding='latin-1', delimiter=';').fillna("")
        # Añadir soporte para TXT
        elif ext == '.txt':
            print("📄 Procesando archivo TXT...")
            
            # Intentar detectar la estructura del archivo
            try:
                # Primero intentar leer como CSV por si es un archivo delimitado
                for encoding in ['utf-8', 'latin-1']:
                    for delimiter in [',', ';', '\t', '|']:
                        try:
                            df = pd.read_csv(archivo, dtype=str, encoding=encoding, delimiter=delimiter)
                            if len(df.columns) > 1:  # Si tiene más de una columna, probablemente es un CSV
                                print(f"✅ Archivo TXT leído como CSV con delimitador '{delimiter}'")
                                return df.fillna("")
                        except:
                            continue
                
                # Si no se pudo leer como CSV, intentar detectar formato de líneas clave-valor
                with open(archivo, 'r', encoding='utf-8') as f:
                    contenido = f.readlines()
                
                # Ver si las líneas tienen formato "clave: valor" o "clave=valor"
                pares_clave_valor = []
                for linea in contenido:
                    linea = linea.strip()
                    if not linea:
                        continue
                    
                    # Intentar varios separadores comunes
                    for sep in [':', '=', '\t']:
                        if sep in linea:
                            partes = linea.split(sep, 1)
                            if len(partes) == 2:
                                clave = partes[0].strip()
                                valor = partes[1].strip()
                                pares_clave_valor.append((clave, valor))
                                break
                
                if pares_clave_valor:
                    # Convertir a DataFrame
                    print("✅ Archivo TXT interpretado como pares clave-valor")
                    df_dict = {}
                    for clave, valor in pares_clave_valor:
                        df_dict[clave] = [valor]
                    return pd.DataFrame(df_dict).fillna("")
                
                # Si todo lo anterior falla, leer como texto plano y crear un DataFrame simple
                print("ℹ️ No se detectó estructura. Procesando como texto plano.")
                with open(archivo, 'r', encoding='utf-8') as f:
                    texto = f.read()
                
                # Crear un DataFrame con una sola fila y una columna "texto"
                return pd.DataFrame({"texto": [texto]}).fillna("")
                
            except UnicodeDecodeError:
                # Si falla con UTF-8, intentar con latin-1
                try:
                    with open(archivo, 'r', encoding='latin-1') as f:
                        texto = f.read()
                    return pd.DataFrame({"texto": [texto]}).fillna("")
                except Exception as e:
                    print(f"❌ Error al procesar archivo TXT: {e}")
                    return None
            except Exception as e:
                print(f"❌ Error al procesar archivo TXT: {e}")
                return None
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
