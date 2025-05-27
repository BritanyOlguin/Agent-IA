"""
Configuración central del sistema.
"""

import torch
from transformers import pipeline, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from src.core.llm import cargar_modelo_con_lora
import os

# --- CONFIGURACIÓN DE RUTAS ---
ruta_modelo_embeddings = r"C:\Users\TEC-INT02\Documents\Agent-IA\models\base\models--intfloat--e5-large-v2"
ruta_indices = r"C:\Users\TEC-INT02\Documents\Agent-IA\data\indices"
ruta_modelo_llama3 = r"C:\Users\TEC-INT02\Documents\Agent-IA\models\base\models--meta-llama--Meta-Llama-3-8B-Instruct"

# --- CONFIGURACIÓN DE ELASTICSEARCH ---
try:
    from .elasticsearch_engine import ElasticsearchEngine
    
    # Inicializar motor de Elasticsearch
    print("🔄 Inicializando Elasticsearch...")
    elasticsearch_engine = ElasticsearchEngine()
    
    # Verificar si hay datos indexados
    stats = elasticsearch_engine.get_stats()
    if stats.get('total_documents', 0) == 0:
        print("⚠️ No hay datos en Elasticsearch. Ejecuta el script de indexación.")
    else:
        print(f"✅ Elasticsearch listo con {stats['total_documents']:,} documentos")
        
except Exception as e:
    print(f"❌ Error inicializando Elasticsearch: {e}")
    print("   Verifique que Elasticsearch esté ejecutándose.")
    elasticsearch_engine = None

# --- CONFIGURACIÓN PARA FUTURO FINE-TUNING (OPCIONAL) ---
# Estas variables se usarán cuando implementes feedback + entrenamiento
USAR_MODELO_ENTRENADO = False  # Cambiar a True cuando tengas modelo entrenado

if USAR_MODELO_ENTRENADO:
    # Esta sección se activará en el futuro
    print("🤖 Modo híbrido: Elasticsearch + Modelo entrenado")
    # TODO: Cargar modelo fine-tuneado cuando esté disponible
else:
    print("⚡ Modo Elasticsearch puro: Máxima velocidad")

# --- VARIABLES PARA MANTENER COMPATIBILIDAD ---
device = "cuda" if torch.cuda.is_available() else "cpu"
llm = None              # Se cargará cuando implementes fine-tuning
embed_model = None      # Se cargará cuando implementes fine-tuning
llm_clasificador = None # Se cargará cuando implementes fine-tuning