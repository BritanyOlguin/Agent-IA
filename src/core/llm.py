"""
Funciones para cargar y gestionar modelos de lenguaje.
"""

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

def cargar_modelo_con_lora(ruta_adaptadores: str, ruta_base: str, usar_4bit: bool = False):
    """Carga el modelo base y luego aplica los adaptadores LoRA entrenados."""
    print(f"Cargando modelo base original desde: {ruta_base}")

    load_in_8bit = not usar_4bit
    load_in_4bit_config = usar_4bit

    # CONFIGURACIÓN PARA CUANTIZACIÓN
    bnb_config = None
    if usar_4bit:
        from transformers import BitsAndBytesConfig
        print("   Configurando para carga en 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        load_in_8bit = False

    modelo_original = AutoModelForCausalLM.from_pretrained(
        ruta_base,
        torch_dtype=torch.float16,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    print("Modelo base original cargado.")

    print(f"Aplicando adaptadores LoRA desde: {ruta_adaptadores}")
    modelo_tuneado = PeftModel.from_pretrained(
        modelo_original,
        ruta_adaptadores,
        device_map="auto"
    )
    modelo_tuneado.eval() # PONER EN MODO EVALUACIÓN
    print("Adaptadores LoRA aplicados. Modelo fine-tuneado listo.")
    return modelo_tuneado