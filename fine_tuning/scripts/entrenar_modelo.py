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
    """Función principal para entrenar el modelo con ejemplos de feedback"""
    print("=== ENTRENAMIENTO DE MODELO CON QLORA (OPTIMIZADO PARA FEEDBACK) ===")
    
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
    
    if train_dataset is None:
        print("❌ No se pudieron cargar datos de entrenamiento. Verifique la ruta y formato.")
        return None
    
    # Analizar distribución de ejemplos (para depuración)
    tipos_ejemplos = {}
    for ejemplo in train_dataset:
        tipo = ejemplo.get('tipo_plantilla', 'desconocido')
        tipos_ejemplos[tipo] = tipos_ejemplos.get(tipo, 0) + 1
    
    print("\n📊 Distribución de ejemplos de entrenamiento:")
    for tipo, count in tipos_ejemplos.items():
        print(f"  - {tipo}: {count} ejemplos ({count/len(train_dataset)*100:.1f}%)")
    
    # Ajustar hiperparámetros según la composición del dataset
    ejemplos_feedback = sum(count for tipo, count in tipos_ejemplos.items() if 'feedback' in tipo)
    if ejemplos_feedback > 0:
        print(f"\n🔍 Detectados {ejemplos_feedback} ejemplos de feedback.")
        print("   Ajustando hiperparámetros para priorizar aprendizaje de feedback...")
        
        # Ajustar learning rate y epochs para mejor aprendizaje de ejemplos de feedback
        if ejemplos_feedback > len(train_dataset) * 0.3:  # Si más del 30% son de feedback
            args.lr = max(args.lr * 0.8, 1e-5)  # Reducir learning rate
            args.epochs = max(args.epochs, 5)   # Asegurar suficientes epochs
            print(f"   - Learning rate ajustado a: {args.lr}")
            print(f"   - Epochs ajustados a: {args.epochs}")
    
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
    
    # Configurar LoRA con parámetros optimizados para feedback
    print("Configurando LoRA optimizado para feedback...")
    lora_config = LoraConfig(
        r=16,  # Rango de adaptadores (mayor = más capacidad, pero más parámetros)
        lora_alpha=32,  # Peso de LoRA vs modelo base
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,  # Importante para evitar sobreajuste en ejemplos de feedback
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Aplicar LoRA al modelo
    model = get_peft_model(model, lora_config)
    
    print("Estructura de parámetros del modelo:")
    model.print_trainable_parameters()
    
    # Configurar entrenamiento con ajustes para feedback
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
        save_total_limit=3,  # Guardar solo los 3 mejores checkpoints
        # Evitar sobreajuste con ejemplos de feedback
        gradient_checkpointing=True,  # Reduce memoria y puede ayudar con sobreajuste
        max_grad_norm=0.3,            # Clipping de gradientes para estabilidad
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
    
    try:
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
            
            # Añadir información de feedback
            if ejemplos_feedback > 0:
                f.write(f"- Ejemplos de feedback: {ejemplos_feedback//10} (repetidos x10)\n")
                
            f.write("\n## Distribución de ejemplos\n\n")
            for tipo, count in tipos_ejemplos.items():
                f.write(f"- {tipo}: {count} ({count/len(train_dataset)*100:.1f}%)\n")
                
            f.write("\n## Descripción\n\n")
            f.write("Este modelo ha sido entrenado específicamente para consultas en bases de datos personales, ")
            f.write("permitiendo responder preguntas sobre personas, direcciones, teléfonos y otros atributos.\n")
        
        print(f"\n✅ Entrenamiento completado con éxito")
        print(f"  - Modelo guardado en: {ruta_salida}")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        
        # Intentar guardar el modelo parcial si es posible
        try:
            print("\n⚠️ Intentando guardar modelo parcial...")
            trainer.save_model(os.path.join(ruta_salida, "parcial"))
            print(f"  - Modelo parcial guardado en: {os.path.join(ruta_salida, 'parcial')}")
        except:
            print("  - No se pudo guardar modelo parcial")
    
    return ruta_salida

if __name__ == "__main__":
    args = parse_args()
    ruta_modelo = entrenar_modelo(args)