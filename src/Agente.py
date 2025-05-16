import torch
import gc
from llama_index.core.agent import ReActAgent
import json
import re
from src.core.engine import indices, all_tools, llm_clasificador, interpretar_pregunta_llm, desambiguar_consulta
from src.tools.interpret import preprocesar_consulta, obtener_prompt_clasificacion_con_ejemplos, ejecutar_consulta_inteligente
from src.core.config import device, embed_model, llm

# --- 8) CREAR Y EJECUTAR EL AGENTE ---

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

print("\n🤖 Agente Inteligente Mejorado: Escribe tu pregunta en lenguaje natural o 'salir' para terminar.")

while True:
    prompt = input("\nPregunta: ")
    if prompt.lower() == 'salir':
        break
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
            
        except Exception as e2:
            print(f"❌ También falló el agente fallback: {e2}")
            print("Lo siento, no pude procesar tu consulta. Por favor, intenta reformularla.")

# --- LIMPIEZA ---
del llm, embed_model, agent, all_tools, indices
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()
print("\n👋 ¡Hasta luego!")