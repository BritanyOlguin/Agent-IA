import os
import pandas as pd
import json
import random
from tqdm import tqdm
import sys
import re

# Añadir ruta para normalizar_texto si es necesario
sys.path.append(r"C:\Users\TEC-INT02\Documents\Agent-IA\src")
try:
    from normalizar_texto import normalizar_texto
except ImportError:
    print("Advertencia: No se pudo importar normalizar_texto. Usando función simplificada.")
    def normalizar_texto(texto):
        if not isinstance(texto, str):
            texto = str(texto)
        return texto.lower().strip()

# Configuración de rutas
CARPETA_BD = r"C:\Users\TEC-INT02\Documents\Agent-IA\archivos"
CARPETA_SALIDA = r"C:\Users\TEC-INT02\Documents\Agent-IA\fine_tuning\datos"
CARPETA_FEEDBACK = r"C:\Users\TEC-INT02\Documents\Agent-IA\feedback"

# Formatos para las conversaciones
FORMATO_INSTRUCCION = """<instrucción>
Eres un asistente que consulta información personal en bases de datos. Responde la siguiente pregunta con precisión usando los datos disponibles.

{pregunta}
</instrucción>

<respuesta>
{respuesta}
</respuesta>"""

def cargar_dataframe(archivo):
    """Carga un dataframe desde diferentes formatos de archivo"""
    ext = os.path.splitext(archivo)[1].lower()
    print(f"📄 Cargando: {archivo}")
    
    try:
        if ext in ['.xlsx', '.xls']:
            return pd.read_excel(archivo, dtype=str).fillna("")
        elif ext == '.csv':
            try:
                return pd.read_csv(archivo, dtype=str, encoding='utf-8').fillna("")
            except UnicodeDecodeError:
                return pd.read_csv(archivo, dtype=str, encoding='latin-1').fillna("")
        elif ext in ['.mdb', '.accdb']:
            try:
                import pyodbc
                conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={archivo};'
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                tables = [table_info.table_name for table_info in cursor.tables(tableType='TABLE') 
                         if not table_info.table_name.startswith('MSys')]
                if tables:
                    print(f"    -> Tablas encontradas: {tables}. Usando '{tables[0]}'")
                    df = pd.read_sql(f"SELECT * FROM [{tables[0]}]", conn)
                    conn.close()
                    return df.astype(str).fillna("")
                else:
                    print(f"⚠️ No se encontraron tablas en: {archivo}")
                    return None
            except Exception as e:
                print(f"❌ Error al cargar {archivo}: {e}")
                return None
        elif ext == '.txt':
            print("📄 Procesando archivo TXT...")
            try:
                # Intentar detectar el delimitador
                with open(archivo, 'r', encoding='utf-8') as f:
                    primeras_lineas = [next(f) for _ in range(min(5, sum(1 for _ in open(archivo))))]
                
                # Detectar delimitador más probable
                delimitador = '|' if '|' in primeras_lineas[0] else ('\t' if '\t' in primeras_lineas[0] else ',')
                print(f"    Delimitador detectado: '{delimitador}'")
                
                # Leer el archivo con el delimitador detectado
                return pd.read_csv(archivo, dtype=str, encoding='utf-8', sep=delimitador).fillna("")
            except UnicodeDecodeError:
                try:
                    # Si falla con UTF-8, probar con latin-1
                    with open(archivo, 'r', encoding='latin-1') as f:
                        primeras_lineas = [next(f) for _ in range(min(5, sum(1 for _ in open(archivo, encoding='latin-1'))))]
                    
                    delimitador = '|' if '|' in primeras_lineas[0] else ('\t' if '\t' in primeras_lineas[0] else ',')
                    print(f"    Delimitador detectado (latin-1): '{delimitador}'")
                    
                    return pd.read_csv(archivo, dtype=str, encoding='latin-1', sep=delimitador).fillna("")
                except Exception as e:
                    print(f"❌ Error al procesar archivo TXT con latin-1: {e}")
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

def detectar_y_normalizar_campos(df):
    """Detecta y normaliza campos comunes en diferentes formatos"""
    # Mapeo básico de nombres de campos
    mapeo_campos = {
        'nombre_completo': ['nombre_completo', 'nombre completo', 'nombre y apellidos'],
        'direccion_completa': ['direccion', 'dirección', 'domicilio', 'direccion completa'],
        'telefono_completo': ['telefono', 'teléfono', 'tel', 'celular', 'movil'],
        'ocupacion': ['ocupacion', 'ocupación', 'profesion', 'profesión', 'trabajo'],
        'estado': ['estado', 'entidad', 'estado de origen'],
        'municipio': ['municipio', 'ciudad', 'localidad'],
        'cp': ['cp', 'codigo postal', 'código postal'],
        'colonia': ['colonia', 'col', 'barrio', 'fraccionamiento'],
        'sexo': ['sexo', 'genero', 'género'],
        'curp': ['curp'],
        'rfc': ['rfc'],
        'clave_ife': ['clave_ife', 'ife', 'clave de elector', 'credencial electoral']
    }
    
    campos = {}
    df_columnas_norm = {normalizar_texto(col): col for col in df.columns}
    
    for campo_std, posibles_nombres in mapeo_campos.items():
        for nombre_pos in posibles_nombres:
            nombre_norm = normalizar_texto(nombre_pos)
            if nombre_norm in df_columnas_norm:
                campos[campo_std] = df_columnas_norm[nombre_norm]
                break
    
    # Construcción de campos compuestos si es necesario (nombre completo, dirección, etc.)
    # Código de construcción aquí si es necesario...
    
    print(f"    Campos detectados: {list(campos.keys())}")
    return df, campos

def generar_respuesta_para_feedback(tipo_busqueda, campo, valor, persona_ejemplo=None):
    """
    Genera una respuesta completa de ejemplo similar a las respuestas reales del sistema
    para los ejemplos de feedback.
    """
    # Nombres ficticios para usar en ejemplos
    nombres = ["JUAN PÉREZ GARCÍA", "MARÍA LÓPEZ SÁNCHEZ", "ROBERTO GONZÁLEZ TORRES", 
               "ALEJANDRA MARTÍNEZ RUIZ", "CARLOS RODRÍGUEZ CAMPOS", "PATRICIA FLORES VEGA"]
    
    # Direcciones ficticias
    direcciones = ["AV. UNIVERSIDAD 3000, COL. COPILCO, CP 04360",
                  "CALLE REVOLUCIÓN 123, COL. ESCANDÓN, CP 11800",
                  "PASEO DE LA REFORMA 222, COL. JUÁREZ, CP 06600",
                  "AV. INSURGENTES SUR 1602, COL. CRÉDITO CONSTRUCTOR, CP 03940"]
    
    # Teléfonos ficticios
    telefonos = ["5512345678", "5587654321", "5599887766", "5566778899", "5543215678"]
    
    # Ocupaciones comunes
    ocupaciones = ["INGENIERO", "MÉDICO", "ABOGADO", "PROFESOR", "CONTADOR", "ESTUDIANTE", "COMERCIANTE"]
    
    # Si no se proporciona una persona de ejemplo, crear datos aleatorios
    if not persona_ejemplo:
        persona_ejemplo = {
            'nombre_completo': random.choice(nombres),
            'direccion': random.choice(direcciones),
            'telefono': random.choice(telefonos),
            'ocupacion': random.choice(ocupaciones)
        }
    
    # Generar respuesta según el tipo de búsqueda
    if tipo_busqueda == "atributo" and campo == "ocupacion":
        return (
            f"SE ENCONTRARON LOS SIGUIENTES REGISTROS DE PERSONAS CON OCUPACIÓN: {valor.upper()}\n\n"
            f"COINCIDENCIA EN BANCO_DATOS:\n"
            f"NOMBRE_COMPLETO: {random.choice(nombres)}\n"
            f"OCUPACION: {valor.upper()}\n"
            f"DIRECCION: {random.choice(direcciones)}\n"
            f"TELEFONO: {random.choice(telefonos)}\n\n"
            f"COINCIDENCIA EN CLIENTES:\n"
            f"NOMBRE_COMPLETO: {random.choice(nombres)}\n"
            f"OCUPACION: {valor.upper()}\n"
            f"DIRECCION: {random.choice(direcciones)}\n"
            f"TELEFONO: {random.choice(telefonos)}"
        )
    elif tipo_busqueda == "nombre":
        return (
            f"COINCIDENCIA EXACTA:\n"
            f"NOMBRE_COMPLETO: {valor.upper()}\n"
            f"DIRECCION: {random.choice(direcciones)}\n"
            f"TELEFONO: {random.choice(telefonos)}\n"
            f"OCUPACION: {random.choice(ocupaciones)}"
        )
    elif tipo_busqueda == "telefono":
        return (
            f"SE ENCONTRARON LAS SIGUIENTES COINCIDENCIAS PARA NÚMERO TELEFÓNICO:\n\n"
            f"COINCIDENCIA EN BASE_USUARIOS:\n"
            f"NOMBRE_COMPLETO: {random.choice(nombres)}\n"
            f"TELEFONO: {valor}\n"
            f"DIRECCION: {random.choice(direcciones)}\n"
            f"OCUPACION: {random.choice(ocupaciones)}"
        )
    elif tipo_busqueda == "atributo" and campo == "clave ife":
        return (
            f"COINCIDENCIA EXACTA EN CAMPO 'CLAVE_IFE':\n"
            f"NOMBRE_COMPLETO: {random.choice(nombres)}\n"
            f"CLAVE_IFE: {valor}\n"
            f"DIRECCION: {random.choice(direcciones)}\n"
            f"TELEFONO: {random.choice(telefonos)}\n"
            f"OCUPACION: {random.choice(ocupaciones)}"
        )
    elif tipo_busqueda == "atributo" and campo == "cp":
        return (
            f"SE ENCONTRARON {random.randint(3, 10)} REGISTROS PARA CP={valor}.\n\n"
            f"COINCIDENCIA EN BANCO_DATOS:\n"
            f"NOMBRE_COMPLETO: {random.choice(nombres)}\n"
            f"DIRECCION: {random.choice(['CALLE', 'AV.', 'BLVD.'])} {random.choice(['PRINCIPAL', 'CENTRAL', 'JUÁREZ', 'HIDALGO'])} {random.randint(100, 999)}, COL. {random.choice(['CENTRO', 'REFORMA', 'JUÁREZ', 'MODERNA'])}, CP: {valor}\n"
            f"TELEFONO: {random.choice(telefonos)}\n\n"
            f"COINCIDENCIA EN CLIENTES:\n"
            f"NOMBRE_COMPLETO: {random.choice(nombres)}\n"
            f"DIRECCION: {random.choice(['CALLE', 'AV.', 'BLVD.'])} {random.choice(['REVOLUCIÓN', 'CONSTITUCIÓN', 'INDEPENDENCIA', 'REFORMA'])} {random.randint(100, 999)}, COL. {random.choice(['DOCTORES', 'NARVARTE', 'DEL VALLE', 'CONDESA'])}, CP: {valor}"
        )
    elif tipo_busqueda == "direccion":
        return (
            f"SE ENCONTRARON COINCIDENCIAS PARA LA DIRECCIÓN '{valor.upper()}':\n\n"
            f"COINCIDENCIA EN REGISTRO_DOMICILIOS:\n"
            f"DIRECCION COMPLETA: {valor.upper()}\n"
            f"RESIDENTE: {random.choice(nombres)}\n"
            f"TELEFONO: {random.choice(telefonos)}\n"
            f"OCUPACION: {random.choice(ocupaciones)}"
        )
    elif tipo_busqueda == "nombre_componentes":
        nombre_completo = valor.upper()
        if len(nombre_completo.split()) < 3:
            nombre_completo = f"{valor.upper()} {random.choice(['GARCÍA', 'LÓPEZ', 'MARTÍNEZ', 'RODRÍGUEZ'])}"
        
        return (
            f"COINCIDENCIA POR COMPONENTES DE NOMBRE:\n"
            f"NOMBRE COMPLETO: {nombre_completo}\n"
            f"DIRECCION: {random.choice(direcciones)}\n"
            f"TELEFONO: {random.choice(telefonos)}\n"
            f"OCUPACION: {random.choice(ocupaciones)}"
        )
    else:
        # Respuesta genérica para otros casos
        return (
            f"SE ENCONTRARON COINCIDENCIAS PARA {campo.upper()}: {valor.upper()}\n\n"
            f"COINCIDENCIA EN BASE_DATOS:\n"
            f"NOMBRE_COMPLETO: {random.choice(nombres)}\n"
            f"{campo.upper()}: {valor.upper()}\n"
            f"DIRECCION: {random.choice(direcciones)}\n"
            f"TELEFONO: {random.choice(telefonos)}\n"
            f"OCUPACION: {random.choice(ocupaciones)}"
        )

def generar_ejemplos_qa(df, campos, nombre_archivo, num_ejemplos=50):
    """Genera ejemplos de preguntas y respuestas basados en los datos"""
    # Plantillas de preguntas por tipos
    plantillas = {
        'info_general': [
            '¿Quién es {nombre_completo}?',
            'Dame todos los datos que tengas de {nombre_completo}.',
            'Busca información sobre {nombre_completo}',
            'Información completa de {nombre_completo}'
        ],
        'atributo_persona': [
            '¿Cuál es la dirección completa de {nombre_completo}?',
            '¿Cuál es el teléfono de {nombre_completo}?',
            '¿Cuál es la ocupación de {nombre_completo}?',
            '¿En qué municipio vive {nombre_completo}?',
            '¿En qué estado vive {nombre_completo}?'
        ],
        'busqueda_inversa': [
            '¿A quién pertenece el teléfono {telefono_completo}?',
            '¿De quién es la dirección {direccion_completa}?',
            '¿Quién vive en {direccion_completa}?',
            '¿Quién tiene el número {telefono_completo}?'
        ]
    }
    
    ejemplos = []
    
    # Filtrar filas con datos suficientes
    filas_validas = []
    for idx, row in df.iterrows():
        datos_validos = True
        for campo_req in ['nombre_completo', 'direccion_completa', 'telefono_completo']:
            if campo_req in campos:
                valor = str(row[campos[campo_req]]).strip()
                if not valor or valor.lower() in ['nan', 'none', '']:
                    datos_validos = False
                    break
        if datos_validos:
            filas_validas.append((idx, row))
    
    if not filas_validas:
        print(f"⚠️ No hay filas con datos suficientes en {nombre_archivo}")
        return []
    
    # Generar ejemplos limitados por el número solicitado
    ejemplos_gen = 0
    pbar = tqdm(total=min(num_ejemplos, len(filas_validas)), desc=f"Generando para {nombre_archivo}", unit=" ej")
    
    for idx, row in filas_validas:
        if ejemplos_gen >= num_ejemplos:
            break
        
        # Datos disponibles para esta fila
        datos_fila = {}
        for campo_std, col in campos.items():
            valor = str(row[col]).strip()
            if valor and valor.lower() not in ['nan', 'none', '']:
                datos_fila[campo_std] = valor
        
        # Solo procesar si tiene nombre_completo
        if 'nombre_completo' not in datos_fila:
            continue
        
        # Generar respuesta tipo resumen de datos
        resumen_datos = []
        for campo, valor in datos_fila.items():
            campo_display = campo.replace('_', ' ').title()
            resumen_datos.append(f"{campo_display}: {valor}")
        
        resumen_texto = "\n".join(resumen_datos)
        
        # Generar ejemplos de diferentes tipos
        tipos = list(plantillas.keys())
        tipo_elegido = random.choice(tipos)
        
        # Plantillas disponibles según los datos
        plantillas_disponibles = []
        
        if tipo_elegido == 'info_general':
            plantillas_disponibles = plantillas['info_general']
        
        elif tipo_elegido == 'atributo_persona':
            for p in plantillas['atributo_persona']:
                if '{direccion_completa}' in p and 'direccion_completa' in datos_fila:
                    plantillas_disponibles.append(p)
                elif '{telefono_completo}' in p and 'telefono_completo' in datos_fila:
                    plantillas_disponibles.append(p)
                elif '{ocupacion}' in p and 'ocupacion' in datos_fila:
                    plantillas_disponibles.append(p)
                elif '{municipio}' in p and 'municipio' in datos_fila:
                    plantillas_disponibles.append(p)
                elif '{estado}' in p and 'estado' in datos_fila:
                    plantillas_disponibles.append(p)
                elif '{nombre_completo}' in p and 'nombre_completo' in datos_fila:
                    plantillas_disponibles.append(p)
        
        elif tipo_elegido == 'busqueda_inversa':
            for p in plantillas['busqueda_inversa']:
                if '{telefono_completo}' in p and 'telefono_completo' in datos_fila:
                    plantillas_disponibles.append(p)
                elif '{direccion_completa}' in p and 'direccion_completa' in datos_fila:
                    plantillas_disponibles.append(p)
        
        # Si no hay plantillas disponibles para este tipo, continuar con la siguiente fila
        if not plantillas_disponibles:
            continue
        
        # Elegir una plantilla aleatoria
        plantilla = random.choice(plantillas_disponibles)
        
        # Formatear pregunta y respuesta
        try:
            pregunta = plantilla.format(**datos_fila)
            
            # Generar una respuesta adecuada según el tipo de plantilla
            if "¿Quién es" in plantilla or "Dame todos los datos" in plantilla:
                respuesta = f"{datos_fila['nombre_completo']}:\n{resumen_texto}"
            elif "dirección" in plantilla.lower():
                respuesta = f"La dirección de {datos_fila['nombre_completo']} es: {datos_fila.get('direccion_completa', 'No disponible')}.\nDetalles adicionales:\n{resumen_texto}"
            elif "teléfono" in plantilla.lower():
                respuesta = f"El teléfono de {datos_fila['nombre_completo']} es: {datos_fila.get('telefono_completo', 'No disponible')}.\nDetalles adicionales:\n{resumen_texto}"
            elif "ocupación" in plantilla.lower():
                respuesta = f"La ocupación de {datos_fila['nombre_completo']} es: {datos_fila.get('ocupacion', 'No disponible')}.\nDetalles adicionales:\n{resumen_texto}"
            elif "municipio" in plantilla.lower():
                respuesta = f"{datos_fila['nombre_completo']} vive en el municipio: {datos_fila.get('municipio', 'No disponible')}.\nDetalles adicionales:\n{resumen_texto}"
            elif "estado" in plantilla.lower():
                respuesta = f"{datos_fila['nombre_completo']} vive en el estado: {datos_fila.get('estado', 'No disponible')}.\nDetalles adicionales:\n{resumen_texto}"
            elif "A quién pertenece el teléfono" in plantilla or "Quién tiene el número" in plantilla:
                respuesta = f"El teléfono {datos_fila['telefono_completo']} pertenece a {datos_fila['nombre_completo']}.\nDetalles adicionales:\n{resumen_texto}"
            elif "De quién es la dirección" in plantilla or "Quién vive en" in plantilla:
                respuesta = f"En la dirección {datos_fila['direccion_completa']} vive {datos_fila['nombre_completo']}.\nDetalles adicionales:\n{resumen_texto}"
            else:
                respuesta = resumen_texto
            
            # Convertir a formato de instrucción
            texto_final = FORMATO_INSTRUCCION.format(
                pregunta=pregunta,
                respuesta=respuesta
            )
            
            # Guardar ejemplo
            ejemplos.append({
                "text": texto_final,
                "pregunta_original": pregunta,
                "respuesta_original": respuesta,
                "tipo_plantilla": tipo_elegido,
                "fuente_archivo": nombre_archivo,
                "fila_indice_original": int(idx)
            })
            
            ejemplos_gen += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"    ❌ Error al generar ejemplo: {e}")
            continue
    
    pbar.close()
    return ejemplos

def main():
    """Función principal mejorada para generar los datos de entrenamiento"""
    print("=== GENERADOR DE DATOS DE ENTRENAMIENTO MEJORADO ===")
    
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)

    todos_ejemplos = []
    
    # PARTE 1: CARGAR Y PROCESAR EJEMPLOS DE FEEDBACK (PRIORIDAD ALTA)
    print("\n=== PROCESANDO EJEMPLOS DE RETROALIMENTACIÓN ===")
    ruta_feedback = os.path.join(CARPETA_FEEDBACK, "ejemplos_entrenamiento.json")
    
    ejemplos_feedback_cargados = False
    
    if os.path.exists(ruta_feedback):
        try:
            with open(ruta_feedback, 'r', encoding='utf-8') as f:
                ejemplos_feedback = json.load(f)
                if ejemplos_feedback:
                    print(f"✓ Cargados {len(ejemplos_feedback)} ejemplos de retroalimentación")
                    
                    # Verificar estructura de los ejemplos
                    for idx, ejemplo in enumerate(ejemplos_feedback):
                        if not isinstance(ejemplo, dict) or 'prompt' not in ejemplo:
                            print(f"⚠️ Ejemplo #{idx} tiene formato incorrecto: {ejemplo}")
                            continue
                    
                    # Convertir ejemplos de feedback al formato de entrenamiento
                    ejemplos_convertidos = []
                    for ejemplo in ejemplos_feedback:
                        prompt = ejemplo.get('prompt', '')
                        tipo_busqueda = ejemplo.get('tipo_busqueda_correcta', '')
                        campo = ejemplo.get('campo_correcto', '')
                        valor = ejemplo.get('valor_correcto', '')
                        
                        if prompt and tipo_busqueda:
                            # Generar respuesta realista similar a los ejemplos normales
                            respuesta_simulada = generar_respuesta_para_feedback(
                                tipo_busqueda, campo, valor
                            )
                            
                            formato_instruccion = FORMATO_INSTRUCCION.format(
                                pregunta=prompt,
                                respuesta=respuesta_simulada
                            )
                            
                            ejemplo_formateado = {
                                "text": formato_instruccion,
                                "pregunta_original": prompt,
                                "respuesta_original": respuesta_simulada,
                                "tipo_plantilla": f"feedback_{tipo_busqueda}",
                                "fuente_archivo": "feedback_usuario",
                                "fila_indice_original": 0
                            }
                            ejemplos_convertidos.append(ejemplo_formateado)
                    
                    # Añadir los ejemplos convertidos con un peso extra
                    if ejemplos_convertidos:
                        peso_feedback = 10  # Aumentamos el peso a 10 para dar más importancia
                        for _ in range(peso_feedback):
                            todos_ejemplos.extend(ejemplos_convertidos)
                        print(f"✓ Agregados {len(ejemplos_convertidos)} ejemplos de feedback al conjunto (peso x{peso_feedback})")
                        ejemplos_feedback_cargados = True
                    else:
                        print("⚠️ No se pudieron convertir ejemplos de feedback al formato de entrenamiento.")
        except Exception as e:
            print(f"❌ Error al cargar ejemplos de retroalimentación: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ No se encontró el archivo de feedback en {ruta_feedback}")
    
    # PARTE 2: GENERAR EJEMPLOS DE LOS ARCHIVOS DE DATOS
    print("\n=== PROCESANDO ARCHIVOS DE DATOS ===")
    
    archivos_en_carpeta = [f for f in os.listdir(CARPETA_BD) if os.path.isfile(os.path.join(CARPETA_BD, f))]
    print(f"Archivos encontrados: {len(archivos_en_carpeta)}")
    
    for nombre_fichero in archivos_en_carpeta:
        ruta_completa = os.path.join(CARPETA_BD, nombre_fichero)
        df_actual = cargar_dataframe(ruta_completa)
        
        if df_actual is None or df_actual.empty:
            print(f"⚠️ No se pudo cargar o está vacío: {nombre_fichero}. Saltando...")
            continue
        
        print(f"🔍 Procesando: {nombre_fichero} ({len(df_actual)} filas)")
        df_procesado, campos_detectados = detectar_y_normalizar_campos(df_actual.copy())
        
        if not campos_detectados:
            print(f"⚠️ No se detectaron campos utilizables. Saltando...")
            continue
            
        # Generar ejemplos de este archivo
        num_ej_por_archivo = 30  # Reducimos si tenemos ejemplos de feedback
        if ejemplos_feedback_cargados:
            num_ej_por_archivo = 10  # Menos ejemplos si ya tenemos feedback
            
        ejemplos_del_archivo = generar_ejemplos_qa(df_procesado, campos_detectados, 
                                                nombre_fichero, num_ejemplos=num_ej_por_archivo)
        
        todos_ejemplos.extend(ejemplos_del_archivo)
        print(f"Total ejemplos acumulados: {len(todos_ejemplos)}")
    
    # PARTE 3: GUARDAR EJEMPLOS PARA ENTRENAMIENTO
    if todos_ejemplos:
        random.shuffle(todos_ejemplos)
        
        # Dividir en entrenamiento y validación (90% - 10%)
        idx_division = int(len(todos_ejemplos) * 0.9)
        ejemplos_entrenamiento = todos_ejemplos[:idx_division]
        ejemplos_validacion = todos_ejemplos[idx_division:]
        
        # Guardar en formato JSONL
        ruta_entrenamiento = os.path.join(CARPETA_SALIDA, "train_data.jsonl")
        with open(ruta_entrenamiento, 'w', encoding='utf-8') as f_train:
            for ejemplo in ejemplos_entrenamiento:
                f_train.write(json.dumps(ejemplo, ensure_ascii=False) + '\n')
        
        ruta_validacion = os.path.join(CARPETA_SALIDA, "val_data.jsonl")
        with open(ruta_validacion, 'w', encoding='utf-8') as f_val:
            for ejemplo in ejemplos_validacion:
                f_val.write(json.dumps(ejemplo, ensure_ascii=False) + '\n')
        
        # Guardar muestra para inspección
        num_muestras = min(20, len(ejemplos_entrenamiento))
        ejemplos_muestra = random.sample(ejemplos_entrenamiento, num_muestras)
        ruta_muestra = os.path.join(CARPETA_SALIDA, "ejemplos_muestra.txt")
        
        with open(ruta_muestra, 'w', encoding='utf-8') as f_muestra:
            for i, ej in enumerate(ejemplos_muestra):
                f_muestra.write(f"--- EJEMPLO {i+1} ---\n")
                f_muestra.write(f"Tipo: {ej.get('tipo_plantilla', 'N/A')}, Fuente: {ej.get('fuente_archivo', 'N/A')}\n\n")
                f_muestra.write(ej.get("text", "FORMATO TEXT NO ENCONTRADO"))
                f_muestra.write("\n\n" + "="*60 + "\n\n")
        
        print(f"\n✅ ¡Proceso completado! Datos generados:")
        print(f"  Total de ejemplos: {len(todos_ejemplos)}")
        print(f"  Ejemplos para entrenamiento: {len(ejemplos_entrenamiento)} (en {ruta_entrenamiento})")
        print(f"  Ejemplos para validación: {len(ejemplos_validacion)} (en {ruta_validacion})")
        print(f"  Muestra para inspección: {num_muestras} ejemplos (en {ruta_muestra})")
        
        if ejemplos_feedback_cargados:
            peso = 10
            num_ejemplos_feedback = len([e for e in ejemplos_entrenamiento if e.get('fuente_archivo') == 'feedback_usuario'])
            print(f"  Ejemplos de feedback incluidos: {num_ejemplos_feedback//peso} (repetidos x{peso} = {num_ejemplos_feedback})")
    else:
        print("❌ No se pudieron generar ejemplos.")

if __name__ == "__main__":
    main()