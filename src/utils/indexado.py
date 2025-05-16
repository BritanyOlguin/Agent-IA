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
from src.utils.text_normalizer import normalizar_texto

catalogo_inegi_estados = {
    "01": "Aguascalientes", "02": "Baja California", "03": "Baja California Sur", "04": "Campeche",
    "05": "Coahuila", "06": "Colima", "07": "Chiapas", "08": "Chihuahua", "09": "Ciudad de México",
    "10": "Durango", "11": "Guanajuato", "12": "Guerrero", "13": "Hidalgo", "14": "Jalisco",
    "15": "Estado de México", "16": "Michoacán", "17": "Morelos", "18": "Nayarit", "19": "Nuevo León",
    "20": "Oaxaca", "21": "Puebla", "22": "Querétaro", "23": "Quintana Roo", "24": "San Luis Potosí",
    "25": "Sinaloa", "26": "Sonora", "27": "Tabasco", "28": "Tamaulipas", "29": "Tlaxcala",
    "30": "Veracruz", "31": "Yucatán", "32": "Zacatecas"
}

# Diccionario de mapeo de abreviaturas de estados a nombres completos
ESTADOS_ABREVIATURAS = {
    "DF": "Ciudad de México",
    "CDMX": "Ciudad de México",
    "MEX": "Estado de México",
    "AGS": "Aguascalientes",
    "BC": "Baja California",
    "BCS": "Baja California Sur",
    "CAMP": "Campeche",
    "COAH": "Coahuila",
    "COL": "Colima",
    "CHIS": "Chiapas",
    "CHIH": "Chihuahua",
    "DGO": "Durango",
    "GTO": "Guanajuato",
    "GRO": "Guerrero",
    "HGO": "Hidalgo",
    "JAL": "Jalisco",
    "MICH": "Michoacán",
    "MOR": "Morelos",
    "NAY": "Nayarit",
    "NL": "Nuevo León",
    "OAX": "Oaxaca",
    "PUE": "Puebla",
    "QRO": "Querétaro",
    "QROO": "Quintana Roo",
    "SLP": "San Luis Potosí",
    "SIN": "Sinaloa",
    "SON": "Sonora",
    "TAB": "Tabasco",
    "TAMPS": "Tamaulipas",
    "TLAX": "Tlaxcala",
    "VER": "Veracruz",
    "YUC": "Yucatán",
    "ZAC": "Zacatecas"
}

def traducir_codigo_inegi_a_estado(codigo: str) -> str:
    codigo = codigo.zfill(2)  # Asegura formato "01", "09", etc.
    return catalogo_inegi_estados.get(codigo, codigo)

def crear_nombre_completo(row):
    """
    Crea un nombre completo a partir de columnas si existen nombre, apellido paterno y materno.
    """
    columnas_nombre = ['nombre', 'NOMBRE']
    columnas_apellido_paterno = ['apellido_paterno', 'apellido_p', 'a_paterno', 'PATERNO']
    columnas_apellido_materno = ['apellido_materno', 'apellido_m', 'a_materno', 'MATERNO']

    columnas_df = [col.lower() for col in row.index]
    col_nombre = col_apellido_paterno = col_apellido_materno = None

    for nombre_pos in columnas_nombre:
        for col in columnas_df:
            if nombre_pos.lower() in col:
                col_nombre = row.index[columnas_df.index(col)]
                break
        if col_nombre:
            break

    for apellido_pos in columnas_apellido_paterno:
        for col in columnas_df:
            if apellido_pos.lower() in col:
                col_apellido_paterno = row.index[columnas_df.index(col)]
                break
        if col_apellido_paterno:
            break

    for apellido_pos in columnas_apellido_materno:
        for col in columnas_df:
            if apellido_pos.lower() in col:
                col_apellido_materno = row.index[columnas_df.index(col)]
                break
        if col_apellido_materno:
            break

    if col_nombre and col_apellido_paterno and col_apellido_materno:
        partes = [row[col_nombre], row[col_apellido_paterno], row[col_apellido_materno]]
        if all(str(p).strip() for p in partes):
            return " ".join(str(p).strip() for p in partes)
    
    return None

# --- CONFIGURACIÓN GLOBAL ---
ruta_modelo_embeddings = r"C:\Users\TEC-INT02\Documents\Agent-IA\models\base\models--intfloat--e5-large-v2"
carpeta_bd = r"C:\Users\TEC-INT02\Documents\Agent-IA\data\source"
carpeta_indices = r"C:\Users\TEC-INT02\Documents\Agent-IA\data\indices"

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
        # Modificar la preparación de documentos
        for i, row in df.iterrows():
            # 1. Generar nombre completo si es posible
            nombre_completo_raw = crear_nombre_completo(row)
            nombre_completo = normalizar_texto(nombre_completo_raw) if nombre_completo_raw else None

            # 2. Convertir todo a texto limpio y normalizado
            datos_fila = {col: str(row[col]).strip() for col in df.columns}
            datos_fila_limpios = {
                normalizar_texto(col): normalizar_texto(v) 
                for col, v in datos_fila.items() 
                if not pd.isna(v) and str(v).lower() not in ["nan", "3586127"]
            }

            # 4.1 Generar campo telefono_completo si existen lada y telefono
            telefono_completo = None
            col_lada = None
            col_telefono = None

            # Buscar columnas candidatas (normalizadas) para lada y telefono
            for col in datos_fila_limpios:
                if 'lada' in col:
                    col_lada = col
                if 'telefono' in col:
                    col_telefono = col

            if col_telefono:
                telefono = datos_fila_limpios.get(col_telefono, "")
                lada = datos_fila_limpios.get(col_lada, "") if col_lada else ""
                
                # Si existe lada, concatenarla
                telefono_completo = f"{lada}{telefono}" if lada else telefono
                datos_fila_limpios["telefono_completo"] = telefono_completo
                texto_a_incrustar += f"\ntelefono_completo: {telefono_completo}"


            # Agregar después de traducir códigos INEGI
            campos_estado = ["estado", "estado de origen", "edo registro", "entidad", "edo", "entidad federativa"]

            for campo in list(datos_fila_limpios.keys()):
                campo_normalizado = normalizar_texto(campo)
                # Verificar si este es un campo de estado
                if any(normalizar_texto(c) in campo_normalizado for c in campos_estado):
                    valor = datos_fila_limpios[campo]
                    
                    # Caso 1: Es un código numérico INEGI
                    if valor.isdigit() and len(valor) <= 2:
                        estado_nombre = traducir_codigo_inegi_a_estado(valor)
                        datos_fila_limpios[campo] = normalizar_texto(estado_nombre)
                    
                    # Caso 2: Es una abreviatura conocida
                    elif valor.upper() in ESTADOS_ABREVIATURAS:
                        estado_nombre = ESTADOS_ABREVIATURAS[valor.upper()]
                        datos_fila_limpios[campo] = normalizar_texto(estado_nombre)


            # 3. Si se generó nombre completo, eliminar nombre, paterno, materno
            if nombre_completo:
                for key in list(datos_fila_limpios.keys()):
                    if any(x in key for x in ["nombre", "apellido_p", "apellido_m", "a_paterno", "a_materno", "paterno", "materno"]):
                        del datos_fila_limpios[key]

            # 4. Generar texto e incluir nombre_completo si aplica
            texto_base = "\n".join([f"{col}: {v}" for col, v in datos_fila_limpios.items() if v])
            texto_a_incrustar = texto_base
            if nombre_completo:
                texto_a_incrustar += f"\nnombre_completo: {nombre_completo}"

            # Generar campo direccion_completa
            componentes_direccion = []
            componentes_encontrados = {}
            campos_direccion = ['calle', 'domicilio', 'numero', 'campo14', 'colonia', 'cp', 'codigo postal', 'municipio', 'ciudad', 'sector', 'estado', 'entidad']

            for campo in campos_direccion:
                for col in datos_fila_limpios.keys():
                    if campo in col:
                        valor = datos_fila_limpios[col]
                        if valor and valor.lower() != "nan":
                            componentes_encontrados[campo] = valor

            direccion_partes = []

            if 'domicilio' in componentes_encontrados:
                domicilio = componentes_encontrados['domicilio']
                # Si hay un número, añadirlo con espacio
                if 'numero' in componentes_encontrados:
                    domicilio += " " + componentes_encontrados['numero']
                    componentes_encontrados.pop('numero')  # Eliminar número para no agregarlo dos veces
                direccion_partes.append(domicilio)
            elif 'calle' in componentes_encontrados:
                calle = componentes_encontrados['calle']
                # Si hay un número, añadirlo con espacio
                if 'numero' in componentes_encontrados:
                    calle += " " + componentes_encontrados['numero']
                    componentes_encontrados.pop('numero')  # Eliminar número para no agregarlo dos veces
                direccion_partes.append(calle)

            for campo in campos_direccion:
                if campo not in ['domicilio', 'calle', 'numero'] and campo in componentes_encontrados:
                    direccion_partes.append(componentes_encontrados[campo])

            # Unir todos los componentes con comas
            direccion_completa = ", ".join(direccion_partes) if direccion_partes else None

            if direccion_completa:
                direccion_completa = normalizar_texto(direccion_completa)
                datos_fila_limpios["direccion"] = direccion_completa
                texto_a_incrustar += f"\ndireccion: {direccion_completa}"


            # 5. Generar metadata, también condicional
            metadata = {
                "fuente": nombre_fuente,
                "archivo": nombre_archivo,
                "fila_origen": i,
                **datos_fila_limpios
            }
            if nombre_completo:
                metadata["nombre_completo"] = nombre_completo

            # 6. Guardar el documento
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
