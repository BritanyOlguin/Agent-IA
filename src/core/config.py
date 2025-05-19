"""
Configuración central del sistema.
"""

import os
import torch
from transformers import pipeline, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from src.core.llm import cargar_modelo_con_lora

# --- 1) CONFIGURACIÓN ---
ruta_modelo_embeddings = r"C:\Users\TEC-INT02\Documents\Agent-IA\models\base\models--intfloat--e5-large-v2"
ruta_indices = r"C:\Users\TEC-INT02\Documents\Agent-IA\data\indices"
ruta_modelo_llama3 = r"C:\Users\TEC-INT02\Documents\Agent-IA\models\base\models--meta-llama--Meta-Llama-3-8B-Instruct"
ruta_tus_adaptadores_lora = r"C:\Users\TEC-INT02\Documents\Agent-IA\models\fine_tuned\llama3-8b-agente-consulta-20250519_1307"

# --- CONSTANTES PARA BUSCAR_DIRECCION_COMBINADA ---
CAMPOS_DIRECCION = ['domicilio', 'calle', 'numero', 'colonia', 'sector', 'municipio', 'ciudad', 'estado', 'cp', 'direccion', 'campo14', 'domicilio calle', 'codigo postal', 'edo de origen']
CAMPOS_BUSQUEDA_EXACTA = ['domicilio', 'direccion', 'calle']
STOP_WORDS = {'de', 'la', 'del', 'los', 'las', 'y', 'a', 'en', 'el', 'col', 'colonia', 'cp', 'sector', 'calzada', 'calz', 'boulevard', 'blvd', 'avenida', 'ave', 'av'}
UMBRAL_PUNTAJE_MINIMO = 0.55
TOLERANCIA_NUMERO_CERCANO = 50

# --- CARGAR MODELO Y TOKENIZER CON TRANSFORMERS ---
print(f" Cargando Tokenizer y Modelo Llama 3 desde: {ruta_modelo_llama3}")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        ruta_modelo_llama3,
        local_files_only=True
    )
    print("Tokenizer cargado.")

    usar_carga_4_bits_para_agente = False

    model = cargar_modelo_con_lora(
        ruta_tus_adaptadores_lora,
        ruta_modelo_llama3,
        usar_4bit=usar_carga_4_bits_para_agente
    )
    print("Modelo LLM Llama 3 FINE-TUNEADO para agente cargado.")

    # VERIFICAR QUE EL MODELO TENGA LORA
    def verificar_modelo_cargado(modelo):
        """Verifica que el modelo LoRA se ha cargado correctamente"""
        modulos_lora = []
        for nombre, modulo in modelo.named_modules():
            if 'lora' in nombre.lower() or hasattr(modulo, 'lora_A') or hasattr(modulo, 'lora_B'):
                modulos_lora.append(nombre)
        
        tiene_lora = len(modulos_lora) > 0
        
        if tiene_lora:
            print(f"✅ Modelo LoRA cargado correctamente. Encontrados {len(modulos_lora)} módulos LoRA.")
            return True
        else:
            print("❌ ADVERTENCIA: No se detectaron adaptadores LoRA en el modelo.")
            return False

    # Verificar el modelo después de cargarlo
    tiene_adaptadores = verificar_modelo_cargado(model)
    
    # FUSIONAR EL MODELO PARA HACERLO COMPATIBLE CON PIPELINE
    def fusionar_modelo_lora(modelo_peft):
        """Fusiona los adaptadores LoRA con el modelo base para hacerlo compatible con pipelines"""
        print("Fusionando adaptadores LoRA con el modelo base...")
        modelo_fusionado = modelo_peft.merge_and_unload()
        print("✅ Modelo fusionado correctamente")
        return modelo_fusionado
    
    # Fusionar el modelo si tiene adaptadores
    if tiene_adaptadores:
        model_fusionado = fusionar_modelo_lora(model)
        # Usar el modelo fusionado para el pipeline
        print("Configurando pipeline de clasificación de texto con modelo fusionado...")
        llm_clasificador = pipeline(
            "text-generation",
            model=model_fusionado,  # Usar el modelo fusionado
            tokenizer=tokenizer,
            torch_dtype=torch.float16
        )
    else:
        # Si no tiene adaptadores, usar el modelo directamente
        model_para_pipeline = model.base_model  # Usar el modelo base si no se detectan adaptadores
        print("Configurando pipeline de clasificación de texto con modelo base...")
        llm_clasificador = pipeline(
            "text-generation",
            model=model_para_pipeline,
            tokenizer=tokenizer,
            torch_dtype=torch.float16
        )
    
    print("Pipeline llm_clasificador configurado.")

except Exception as e:
    print(f"Error al cargar Llama 3 FINE-TUNEADO o su tokenizer: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- CONFIGURAR HuggingFaceLLM ---
try:
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"temperature": 0.1, "do_sample": (0.1 > 0)},
    )
    print("HuggingFaceLLM configurado con TU modelo Llama 3 FINE-TUNEADO.")
except Exception as e:
    print(f"Error al configurar HuggingFaceLLM con TU modelo: {e}")
    exit()

# CONFIGURACIÓN DE DISPOSITIVO Y LLM
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("ADVERTENCIA: Usando CPU para LLM. Las respuestas serán lentas.")
else:
    print(f"Usando dispositivo para LLM y Embeddings: {device}")

    # --- CONFIGURAR pipeline `llm_clasificador` CON MODELO FINE-TUNEADO ---
print("Configurando pipeline de clasificación de texto con modelo fine-tuneado...")
try:
    llm_clasificador = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16
    )
    print("Pipeline llm_clasificador configurado.")
except Exception as e:
    print(f"Error al configurar el pipeline llm_clasificador: {e}")
    exit()

# CONFIGURAR Modelo de Embeddings
print(f"Cargando modelo de embeddings: {os.path.basename(ruta_modelo_embeddings)}")
try:
    embed_model = HuggingFaceEmbedding(
        model_name=ruta_modelo_embeddings,
        device=device,
        normalize=True
    )
    print("Modelo de embeddings e5-large-v2 cargado.")
except Exception as e:
    print(f"Error cargando el modelo de embeddings desde {ruta_modelo_embeddings}: {e}")
    exit()

# APLICAR CONFIGURACIÓN LLAMAINDEX
Settings.llm = llm
Settings.embed_model = embed_model