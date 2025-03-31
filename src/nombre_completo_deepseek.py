import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# CONFIGURACIÓN
ruta_modelo_embeddings = r"C:\Users\Sistemas\Documents\OKIP\src\models--sentence-transformers--all-MiniLM-L6-v2"
ruta_indice = r"C:\Users\Sistemas\Documents\OKIP\llama_index_banco_bancomer_mpnet\index_part_0"
ruta_modelo_llm = r"C:\Users\Sistemas\Documents\OKIP\src\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
pregunta = "Dame la informacion de GLORIA ESQUIVEL COLIN"

# Inicializar embeddings
modelo_embeddings = SentenceTransformer(ruta_modelo_embeddings, device="cuda" if torch.cuda.is_available() else "cpu")
embedding_pregunta = modelo_embeddings.encode(pregunta, convert_to_tensor=True)
Settings.embed_model = HuggingFaceEmbedding(model_name=ruta_modelo_embeddings, device="cuda")
Settings.llm = None

# Cargar índice
ctx = StorageContext.from_defaults(persist_dir=ruta_indice)
index = load_index_from_storage(ctx)

# -------------------------------
# 🧠 1. EXTRAER NOMBRE COMPLETO (si aplica) CON DEEPSEEK
# -------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained(ruta_modelo_llm, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(ruta_modelo_llm, local_files_only=True)
extractor = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

prompt_nombre = f"""
Pregunta: {pregunta}

👉 Si la pregunta menciona claramente un nombre completo, respóndelo sin agregar nada más. Solo responde el nombre exacto. 
Si no hay un nombre completo claro, responde: "NINGUNO".
"""

salida_nombre = extractor(prompt_nombre, max_new_tokens=15, do_sample=False, temperature=0.1)[0]["generated_text"]
# Extraer solo nombre y limpiar etiquetas o respuestas incorrectas
nombre_detectado = salida_nombre.replace(prompt_nombre, "").strip()

# Filtrar cualquier línea que no parezca un nombre completo
nombre_detectado = re.search(r"[A-ZÑÁÉÍÓÚ][A-ZÑÁÉÍÓÚ\s]{5,}", nombre_detectado.upper())
nombre_detectado = nombre_detectado.group(0).strip() if nombre_detectado else None


contexto_manual = None
metadata_encontrada = None

if "NINGUNO" not in nombre_detectado.upper() and len(nombre_detectado.split()) >= 2:
    print(f"\n🧠 Nombre completo detectado por DeepSeek: {nombre_detectado}")

    for doc in index.docstore.docs.values():
        nombre_doc = doc.metadata.get("Nombre Completo", "").upper().strip()
        if nombre_doc == nombre_detectado.upper():
            contexto_manual = doc.text
            metadata_encontrada = doc.metadata
            print(f"\n🎯 Coincidencia exacta en 'Nombre Completo': {nombre_detectado}")
            break

# -------------------------------
# 🔍 2. COINCIDENCIA EXACTA POR OTRO METADATO (si no hubo nombre completo)
# -------------------------------
if not contexto_manual:
    valores_clave = set(re.findall(r'\d{4,}|\b[A-ZÑÁÉÍÓÚ]{3,}\b', pregunta.upper()))
    for doc in index.docstore.docs.values():
        for k, v in doc.metadata.items():
            if isinstance(v, str) and v.upper().strip() in valores_clave:
                contexto_manual = doc.text
                metadata_encontrada = doc.metadata
                print(f"\n📎 Coincidencia exacta encontrada en metadato '{k}': {v}")
                break
        if contexto_manual:
            break

# -------------------------------
# 💡 3. SI NO HAY MATCH EXACTO, USAR EMBEDDINGS
# -------------------------------
if not contexto_manual:
    print("⚠️ No se encontró coincidencia exacta. Consultando por embeddings...")
    query_engine = index.as_query_engine()
    respuesta = query_engine.query(pregunta)
    contexto_manual = respuesta.response

    emb_resp = modelo_embeddings.encode(contexto_manual, convert_to_tensor=True)
    similitud = util.cos_sim(embedding_pregunta, emb_resp).item()
    print(f"\n🔁 Similitud semántica: {similitud:.4f}")
else:
    print("📄 Documento relacionado (texto completo):")
    print(contexto_manual)

# -------------------------------
# 🧩 4. CONSTRUIR CONTEXTO PARA DEEPSEEK
# -------------------------------
fragmentos = []

if metadata_encontrada:
    claves_utiles = ["Nombre Completo", "Telefono", "Tarjeta", "Calle", "Numero", "Colonia", "Ciudad", "Estado", "CP"]
    fragmentos.append("🗂️ Datos estructurados encontrados:")
    for clave in claves_utiles:
        valor = metadata_encontrada.get(clave)
        if valor:
            fragmentos.append(f"{clave}: {valor}")

fragmentos.append("\n📄 Texto del documento:")
fragmentos.append(contexto_manual)
contexto_final = "\n".join(fragmentos)

# -------------------------------
# 🤖 5. RESPUESTA FINAL CON DEEPSEEK
# -------------------------------
print("\n🤖 Generando respuesta natural con DeepSeek...\n")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

prompt = f"""
Pregunta del usuario:
{pregunta}

Contexto encontrado en la base de datos:
{contexto_final}

👉 Responde de forma clara, directa y basada solamente en el contexto. No inventes ni repitas la pregunta.
"""

salida = generator(prompt, max_new_tokens=100, do_sample=False, temperature=0.1)[0]["generated_text"]

print("✅ Respuesta generada por DeepSeek:")
print(salida.strip())
