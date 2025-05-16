import os
import torch
import gc
import json
import re
from datetime import datetime
import pickle
from src.core.config import device, llm, llm_clasificador, embed_model
from src.core.engine import all_tools, interpretar_pregunta_llm, desambiguar_consulta, indices
from src.tools.interpret import preprocesar_consulta, obtener_prompt_clasificacion_con_ejemplos, ejecutar_consulta_inteligente
from llama_index.core.agent import ReActAgent

# --- CONFIGURACIÓN PARA RETROALIMENTACIÓN ---
RUTA_FEEDBACK = r"C:\Users\TEC-INT02\Documents\Agent-IA\data\feedback"
ARCHIVO_FEEDBACK = os.path.join(RUTA_FEEDBACK, "feedback_registro.pkl")
ARCHIVO_FEEDBACK_CSV = os.path.join(RUTA_FEEDBACK, "feedback_registro.csv")
UMBRAL_EJEMPLOS_NUEVOS = 100  # CUÁNTOS EJEMPLOS NUEVOS PARA VOLVER A ENTRENAR

def cargar_registro_feedback():
    """Carga el registro de retroalimentación desde un archivo pickle."""
    if os.path.exists(ARCHIVO_FEEDBACK):
        try:
            with open(ARCHIVO_FEEDBACK, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error al cargar registro de feedback: {e}")
            return []
    else:
        os.makedirs(os.path.dirname(ARCHIVO_FEEDBACK), exist_ok=True)
        return []

def guardar_registro_feedback(registro):
    """Guarda el registro de retroalimentación en un archivo pickle y CSV."""
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(ARCHIVO_FEEDBACK), exist_ok=True)
    
    # Guardar en formato pickle para uso del programa
    with open(ARCHIVO_FEEDBACK, 'wb') as f:
        pickle.dump(registro, f)
    
    # Guardar en CSV para revisión humana
    try:
        import pandas as pd
        df = pd.DataFrame(registro)
        df.to_csv(ARCHIVO_FEEDBACK_CSV, index=False, encoding='utf-8')
        print(f"✓ Registro de feedback guardado. Total: {len(registro)} entradas.")
    except ImportError:
        print("Pandas no está instalado. No se pudo generar archivo CSV.")
        print(f"✓ Registro de feedback guardado en formato pickle. Total: {len(registro)} entradas.")

def registrar_feedback(prompt, analisis, respuesta, herramienta_usada, es_correcto, tipo_error=None, sugerencia=None):
    """
    Registra la retroalimentación del usuario sobre una respuesta.
    
    Args:
        prompt: Consulta original del usuario
        analisis: El análisis de la consulta (tipo, campo, valor)
        respuesta: La respuesta proporcionada
        herramienta_usada: Herramienta que generó la respuesta
        es_correcto: Booleano indicando si la respuesta fue correcta
        tipo_error: Tipo de error (opcional, solo si es_correcto=False)
        sugerencia: Sugerencia de corrección (opcional)
    """
    registro = cargar_registro_feedback()
    
    nueva_entrada = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prompt': prompt,
        'tipo_busqueda': analisis.get('tipo_busqueda'),
        'campo': analisis.get('campo'),
        'valor': analisis.get('valor'),
        'herramienta_usada': herramienta_usada,
        'respuesta': respuesta,
        'es_correcto': es_correcto,
        'tipo_error': tipo_error,
        'sugerencia': sugerencia
    }
    
    registro.append(nueva_entrada)
    guardar_registro_feedback(registro)
    
    # Si hay suficientes ejemplos nuevos, evaluar si debemos reentrenar
    if len([r for r in registro if not r['es_correcto']]) >= UMBRAL_EJEMPLOS_NUEVOS:
        print(f"\n⚠️ Se han acumulado {UMBRAL_EJEMPLOS_NUEVOS} o más ejemplos con feedback negativo.")
        print("Considere ejecutar el script de generación de datos para reentrenar al modelo.")
    
    return len(registro)

def generar_ejemplos_entrenamiento():
    """
    Genera archivos de ejemplos para entrenamiento a partir del feedback.
    Estos ejemplos pueden ser usados por generar_datos.py para reentrenar el modelo.
    """
    from src.feedback.collector import generar_ejemplos_entrenamiento as gen_ejemplos
    
    print("\n=== GENERANDO EJEMPLOS DE ENTRENAMIENTO DESDE FEEDBACK ===")
    resultado = gen_ejemplos()
    
    if resultado:
        print("\n¿Qué deseas hacer ahora?")
        print("1. Iniciar entrenamiento con estos ejemplos")
        print("2. Solo generar ejemplos y salir")
        
        opcion = input("Opción (1/2): ")
        
        if opcion == "1":
            print("\nIniciando proceso de entrenamiento...")
            try:
                from src.training.model_trainer import entrenar_modelo
                entrenar_modelo()
            except Exception as e:
                print(f"❌ Error al iniciar entrenamiento: {e}")
                import traceback
                traceback.print_exc()
    
    return resultado

# --- PRINCIPAL: AGENTE CON ENTRENAMIENTO ---

# CREAR EL AGENTE REACT
try:
    agent = ReActAgent.from_tools(
        tools=all_tools,
        llm=llm,
        verbose=False
    )
    print("Agente creado correctamente.")
except Exception as e:
    print(f"Error al crear el agente: {e}")
    exit()
    
print("\n🤖 Agente Inteligente con Entrenamiento: Escribe tu pregunta en lenguaje natural, 'entrenar' para generar ejemplos de entrenamiento o 'salir' para terminar.")
print("    Ahora puedes preguntar de cualquier forma y el agente entenderá tu intención.")
print("    Además, podrás proporcionar retroalimentación para que el agente aprenda.")

while True:
    prompt = input("\nPregunta: ")
    
    # Comandos especiales
    if prompt.lower() == 'salir':
        break
    elif prompt.lower() == 'entrenar':
        generar_ejemplos_entrenamiento()
        continue
    if not prompt:
        continue

    herramienta_usada = None
    try:
        # PRE-PROCESAMIENTO DE LA CONSULTA
        prompt_procesado = preprocesar_consulta(prompt)
        if prompt_procesado != prompt:
            print(f"[DEBUG] Consulta procesada: {prompt_procesado}")
        
        prompt_clasificacion = obtener_prompt_clasificacion_con_ejemplos(prompt_procesado)
        
        salida_cruda = llm_clasificador(prompt_clasificacion, max_new_tokens=256, return_full_text=False)[0]['generated_text']
        
        match = re.search(r'\{[\s\S]*?\}', salida_cruda)
        if match:
            json_text = match.group(0)
            analisis = json.loads(json_text)
        else:
            print("[INFO] No se pudo extraer JSON del clasificador, usando analizador alternativo...")
            analisis = interpretar_pregunta_llm(prompt_procesado, llm_clasificador)
        
        print(f"[INFO] Análisis: tipo={analisis.get('tipo_busqueda')}, campo={analisis.get('campo')}, valor={analisis.get('valor')}")
        
        if analisis.get("tipo_busqueda") in ["desconocido", None] or not analisis.get("valor"):
            print("[INFO] Consulta ambigua, intentando desambiguar...")
            analisis = desambiguar_consulta(analisis, prompt_procesado, llm_clasificador)
            print(f"[INFO] Análisis post-desambiguación: tipo={analisis.get('tipo_busqueda')}, campo={analisis.get('campo')}, valor={analisis.get('valor')}")
        
        print(f"[INFO] Ejecutando búsqueda para '{analisis.get('valor')}' como {analisis.get('tipo_busqueda')}...")
        herramienta_usada = analisis.get('tipo_busqueda', 'desconocido')
        respuesta_final = ejecutar_consulta_inteligente(prompt_procesado, analisis, llm_clasificador)
        
        print(f"\n📄 Resultado:\n{respuesta_final}\n")
        
        # SOLICITAR RETROALIMENTACIÓN AL USUARIO
        es_correcto = input("\n¿Es correcta la respuesta? (s/n): ").lower().startswith('s')
        
        if not es_correcto:
            print("\n--- Información para mejorar el modelo ---")
            print("1. Tipo de búsqueda incorrecta")
            print("2. Campo incorrecto")
            print("3. Valor extraído incorrecto")
            print("4. Respuesta incompleta o falta de resultados")
            print("5. Otro error")
            
            tipo_error = input("Tipo de error (1-5): ")
            sugerencia = None
            
            if tipo_error in ["1", "2", "3"]:
                print("\nIndica cómo debería haberse analizado la consulta:")
                tipo_correcto = input("Tipo de búsqueda correcto ('nombre', 'telefono', 'direccion', 'atributo', etc.): ").strip()
                campo_correcto = input("Campo correcto: ").strip()
                valor_correcto = input("Valor correcto: ").strip()
                
                sugerencia = {
                    'tipo_busqueda': tipo_correcto if tipo_correcto else analisis.get('tipo_busqueda'),
                    'campo': campo_correcto if campo_correcto else analisis.get('campo'),
                    'valor': valor_correcto if valor_correcto else analisis.get('valor')
                }
            
            registrar_feedback(prompt, analisis, respuesta_final, herramienta_usada, 
                            es_correcto=False, tipo_error=tipo_error, sugerencia=sugerencia)
            print("Gracias por tu retroalimentación. El agente mejorará.")
        else:
            registrar_feedback(prompt, analisis, respuesta_final, herramienta_usada, es_correcto=True)
            print("¡Excelente! Seguimos aprendiendo.")
        
        if "No se encontraron coincidencias" in respuesta_final:
            print("\n[SUGERENCIA] Para mejorar los resultados, intenta:")
            if analisis.get("tipo_busqueda") == "nombre":
                print("- Usar nombre y apellido completos")
                print("- Verificar la ortografía del nombre")
            elif analisis.get("tipo_busqueda") == "direccion":
                print("- Incluir el número de la dirección")
                print("- Especificar la colonia o sector")
            elif analisis.get("tipo_busqueda") == "telefono":
                print("- Verificar que el número tenga el formato correcto")
                print("- Incluir el código de área o lada")

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

        try:
            # FALLBACK Y REGISTRO DE ERROR
            print("Intentando recuperación con agente fallback...")
            respuesta_agente = agent.query(prompt)
            print(f"\n📄 Resultado (procesado por agente fallback):\n{respuesta_agente}\n")
            
            # Registrar el error con el agente fallback
            registrar_feedback(prompt, {'tipo_busqueda': 'desconocido', 'campo': '', 'valor': ''}, 
                            str(e), herramienta_usada, es_correcto=False, 
                            tipo_error="error_ejecucion", sugerencia=None)
            
            # Preguntar si la respuesta del fallback fue correcta
            fallback_correcto = input("¿La respuesta del agente fallback fue correcta? (s/n): ").lower().startswith('s')
            if fallback_correcto:
                print("Gracias. Registraremos este tipo de consulta para mejorar el agente principal.")
            
        except Exception as e2:
            print(f"❌ También falló el agente fallback: {e2}")
            print("Lo siento, no pude procesar tu consulta. Por favor, intenta reformularla.")

# --- LIMPIEZA ---
del llm, embed_model, agent, all_tools, indices
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()
print("\n👋 ¡Hasta luego!")