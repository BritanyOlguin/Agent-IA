import os
import torch
import json
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
import argparse
from datetime import datetime

# Configuración de rutas
CARPETA_DATOS = r"C:\Users\TEC-INT02\Documents\Agent-IA\fine_tuning\datos"
CARPETA_MODELOS = r"C:\Users\TEC-INT02\Documents\Agent-IA\fine_tuning\modelos"
MODELO_BASE = r"C:\Users\TEC-INT02\Documents\Agent-IA\models\models--meta-llama--Meta-Llama-3-8B-Instruct"

def parse_args():
    parser = argparse.ArgumentParser(description="Script para entrenar un modelo con QLoRA")
    parser.add_argument("--epochs", type=int, default=3, help="Número de epochs de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=2, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Longitud máxima de secuencia")
    return parser.parse_args()

def cargar_datos(ruta_train, ruta_val=None):
    """Carga los datos para entrenamiento y validación desde archivos JSON Lines"""
    print(f"Intentando cargar datos de entrenamiento (JSONL) desde: {ruta_train}") # Mensaje de depuración
    if not os.path.exists(ruta_train): # Comprobación explícita
        print(f"¡ALERTA! El archivo de entrenamiento NO EXISTE en: {ruta_train}")
        # Podrías lanzar un error aquí o retornar None para manejarlo más arriba
        raise FileNotFoundError(f"El archivo de entrenamiento no se encontró en la ruta: {ruta_train}")

    datos_train = []
    with open(ruta_train, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                datos_train.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decodificando JSON en la línea: '{line.strip()}' del archivo {ruta_train}. Error: {e}")
                # Decide si quieres saltar la línea o detener el proceso
                continue # Salta la línea con error

    train_dataset = Dataset.from_list(datos_train)

    val_dataset = None
    if ruta_val: # Solo intentar cargar si ruta_val no es None
        print(f"Intentando cargar datos de validación (JSONL) desde: {ruta_val}") # Mensaje de depuración
        if os.path.exists(ruta_val):
            datos_val = []
            with open(ruta_val, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        datos_val.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decodificando JSON en la línea: '{line.strip()}' del archivo {ruta_val}. Error: {e}")
                        continue # Salta la línea con error
            if datos_val: # Solo crear el dataset si se cargaron datos
                val_dataset = Dataset.from_list(datos_val)
            else:
                print(f"Advertencia: El archivo de validación {ruta_val} está vacío o todas sus líneas tuvieron errores.")
        else:
            print(f"Advertencia: El archivo de validación NO EXISTE en: {ruta_val}. Se continuará sin datos de validación.")

    print(f"Ejemplos de entrenamiento cargados: {len(train_dataset)}")
    if val_dataset:
        print(f"Ejemplos de validación cargados: {len(val_dataset)}")
    else:
        print("No se cargaron datos de validación.")

    return train_dataset, val_dataset
    
def crear_nombre_modelo(args):
    """Crea un nombre para el modelo entrenado basado en parámetros"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"llama3-8b-agente-consulta-{timestamp}"

def entrenar_modelo(args):
    """Función principal para entrenar el modelo"""
    print("=== ENTRENAMIENTO DE MODELO CON QLORA ===")
    
    # Crear carpeta para el modelo
    nombre_modelo = crear_nombre_modelo(args)
    ruta_salida = os.path.join(CARPETA_MODELOS, nombre_modelo)
    os.makedirs(ruta_salida, exist_ok=True)
    
    # Guardar los argumentos de entrenamiento
    with open(os.path.join(ruta_salida, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    
    # Cargar los datos
    ruta_train = os.path.join(CARPETA_DATOS, "train_data.jsonl")
    ruta_val = os.path.join(CARPETA_DATOS, "val_data.jsonl")
    train_dataset, val_dataset = cargar_datos(ruta_train, ruta_val)
    
    # Configuración para cuantización a 4-bits
    print(f"Configurando BitsAndBytes para cuantización 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Cargar modelo base con cuantización
    print(f"Cargando modelo base: {os.path.basename(MODELO_BASE)}")
    model = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Preparar modelo para entrenamiento
    print("Preparando modelo para entrenamiento...")
    model = prepare_model_for_kbit_training(model)
    
    # Cargar tokenizer
    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE, local_files_only=True)
    
    # Asegurarse de que el tokenizer tenga token de padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configurar LoRA
    print("Configurando LoRA...")
    lora_config = LoraConfig(
        r=16,                   # Rango de adaptación
        lora_alpha=32,          # Parámetro de escala
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,      # Dropout para regularización
        bias="none",            # No modificar bias
        task_type=TaskType.CAUSAL_LM  # Tipo de tarea
    )
    
    # Aplicar LoRA al modelo
    model = get_peft_model(model, lora_config)
    
    print("Estructura de parámetros del modelo:")
    model.print_trainable_parameters()
    
    # Configurar entrenamiento
    print("Configurando argumentos de entrenamiento...")
    training_args = TrainingArguments(
        output_dir=ruta_salida,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # Acumular gradientes para simular batches más grandes
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        report_to="none",
        save_total_limit=3  # Guardar solo los 3 mejores checkpoints
    )
    
    # Inicializar trainer
    print("Inicializando SFTTrainer (adaptado a la firma detectada)...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    
    # Entrenar modelo
    print("\n=== Iniciando entrenamiento ===")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Secuencia máxima: {args.max_seq_length}")
    print(f"  - Salida: {ruta_salida}")
    
    trainer.train()
    
    # Guardar modelo
    print("\n=== Guardando modelo entrenado ===")
    trainer.save_model(ruta_salida)
    tokenizer.save_pretrained(ruta_salida)
    
    # Crear un archivo README con información del modelo
    with open(os.path.join(ruta_salida, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# Modelo {nombre_modelo}\n\n")
        f.write("## Información de entrenamiento\n\n")
        f.write(f"- Modelo base: Llama-3-8B-Instruct\n")
        f.write(f"- Fecha de entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Learning rate: {args.lr}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Ejemplos de entrenamiento: {len(train_dataset)}\n")
        if val_dataset:
            f.write(f"- Ejemplos de validación: {len(val_dataset)}\n")
        f.write("\n## Descripción\n\n")
        f.write("Este modelo ha sido entrenado específicamente para consultas en bases de datos personales, ")
        f.write("permitiendo responder preguntas sobre personas, direcciones, teléfonos y otros atributos.\n")
    
    print(f"\n✅ Entrenamiento completado con éxito")
    print(f"  - Modelo guardado en: {ruta_salida}")
    
    return ruta_salida

if __name__ == "__main__":
    args = parse_args()
    ruta_modelo = entrenar_modelo(args)