import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# CONFIGURACIÓN
ruta_modelo_embeddings = r"C:\Users\Sistemas\Documents\OKIP\src\models--sentence-transformers--all-MiniLM-L6-v2"
ruta_indice = r"C:\Users\Sistemas\Documents\OKIP\llama_index_banco_bancomer_mpnet\index_part_0"
ruta_modelo_llm = r"C:\Users\Sistemas\Documents\OKIP\src\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
pregunta = "Cuantas personas viven en el DF"

# Inicializar embeddings
modelo_embeddings = SentenceTransformer(ruta_modelo_embeddings, device="cuda" if torch.cuda.is_available() else "cpu")
embedding_pregunta = modelo_embeddings.encode(pregunta, convert_to_tensor=True)
Settings.embed_model = HuggingFaceEmbedding(model_name=ruta_modelo_embeddings, device="cuda")
Settings.llm = None

# Cargar índice
ctx = StorageContext.from_defaults(persist_dir=ruta_indice)
index = load_index_from_storage(ctx)

# Inicializar modelo LLM DeepSeek
tokenizer = AutoTokenizer.from_pretrained(ruta_modelo_llm, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(ruta_modelo_llm, local_files_only=True)
extractor = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# FUNCIONES REUTILIZABLES
def buscar_en_metadatos(dato):
    coincidencias = []
    for doc in index.docstore.docs.values():
        for k, v in doc.metadata.items():
            if isinstance(v, str) and dato.upper() == v.strip().upper():
                coincidencias.append((doc, k, v))
    return coincidencias

def buscar_por_embeddings(dato):
    emb_dato = modelo_embeddings.encode(dato, convert_to_tensor=True)
    mejor_similitud = -1
    mejor_doc = None
    for doc in index.docstore.docs.values():
        emb_doc = modelo_embeddings.encode(doc.text, convert_to_tensor=True)
        similitud = util.cos_sim(emb_dato, emb_doc).item()
        if similitud > mejor_similitud:
            mejor_similitud = similitud
            mejor_doc = doc
    return mejor_doc, mejor_similitud

def contar_dato(dato):
    coincidencias = buscar_en_metadatos(dato)
    if coincidencias:
        print(f"🔢 Se encontró el dato '{dato}' en {len(coincidencias)} documento(s):")
        for doc, k, v in coincidencias:
            nombre = doc.metadata.get("Nombre Completo", "SIN NOMBRE")
            print(f" - {k}: {v} (👤 {nombre})")
    else:
        print(f"⚠️ No hay coincidencia exacta para '{dato}', buscando por similitud...")
        doc, similitud = buscar_por_embeddings(dato)
        if doc and similitud >= 0.80:
            nombre = doc.metadata.get("Nombre Completo", "SIN NOMBRE")
            print(f"🔁 Coincidencia semántica encontrada (similitud {similitud:.2f})")
            print(f"👤 {nombre}\n📄 {doc.text[:500]}...")
        else:
            print("❌ No se encontró ninguna coincidencia confiable.")

def buscar_dato(dato):
    coincidencias = buscar_en_metadatos(dato)
    if coincidencias:
        doc, k, v = coincidencias[0]
        nombre = doc.metadata.get("Nombre Completo", "SIN NOMBRE")
        print(f"🎯 Coincidencia exacta por '{k}': {v}")
        print(f"👤 {nombre}\n📄 {doc.text}")
    else:
        doc, similitud = buscar_por_embeddings(dato)
        if doc and similitud >= 0.80:
            nombre = doc.metadata.get("Nombre Completo", "SIN NOMBRE")
            print(f"🔁 Coincidencia semántica encontrada (similitud {similitud:.2f})")
            print(f"👤 {nombre}\n📄 {doc.text}")
        else:
            print("❌ No se encontró ninguna coincidencia confiable.")

def buscar_por_nombre_completo(nombre):
    for doc in index.docstore.docs.values():
        nombre_doc = doc.metadata.get("Nombre Completo", "").strip().upper()
        if nombre_doc == nombre.strip().upper():
            print(f"👤 Información completa de {nombre}:\n")
            for k, v in doc.metadata.items():
                print(f"{k}: {v}")
            print("\n📄 Texto:\n", doc.text)
            return
    print(f"⚠️ No se encontró a '{nombre}' exactamente.")
    buscar_dato(nombre)

def buscar_por_metadata_exacta(campo, valor):
    for doc in index.docstore.docs.values():
        valor_doc = doc.metadata.get(campo, "").strip().upper()
        if valor_doc == valor.strip().upper():
            print(f"🎯 Coincidencia en {campo}: {valor}")
            for k, v in doc.metadata.items():
                print(f"{k}: {v}")
            print("\n📄 Texto:\n", doc.text)
            return
    print(f"⚠️ No hay coincidencia exacta para {campo} = {valor}.")
    buscar_dato(valor)

def buscar_persona_por_valor(campo, valor):
    for doc in index.docstore.docs.values():
        valor_doc = doc.metadata.get(campo, "").strip().upper()
        if valor_doc == valor.strip().upper():
            print(f"👤 Persona con {campo}: {valor}")
            for k, v in doc.metadata.items():
                print(f"{k}: {v}")
            print("\n📄 Texto:\n", doc.text)
            return
    print(f"⚠️ No se encontró persona con {campo} = {valor}.")
    buscar_dato(valor)

def detectar_nombre(pregunta: str) -> str:
    prompt = f"""
Pregunta: {pregunta}
👉 Si contiene nombre completo, responde solo el nombre. Si no, responde: "NINGUNO"
"""
    salida = extractor(prompt, max_new_tokens=15, do_sample=False, temperature=0.1)[0]["generated_text"]
    nombre = salida.replace(prompt, "").strip()
    nombre = re.search(r"[A-ZÑÁÉÍÓÚ][A-ZÑÁÉÍÓÚ\s]{5,}", nombre.upper())
    return nombre.group(0).strip() if nombre else None


def detectar_intencion_llm(pregunta: str) -> str:
    prompt = f"""
Clasifica la intención de la siguiente pregunta. Solo responde con una de estas etiquetas exactas:

nombre_completo → cuando se pide información de una persona completa por nombre.
contar_coincidencias → cuando se pregunta cuántas personas tienen un dato.
buscar_por_direccion → cuando se menciona calle, ciudad, colonia o estado.
buscar_dato_de_persona → cuando se quiere saber un dato de alguien (teléfono, tarjeta, etc).
buscar_dato_exactitud → cuando se quiere buscar un dato específico sin saber el campo.

Pregunta: {pregunta}
Intención:
"""
    salida = extractor(prompt, max_new_tokens=5, do_sample=False, temperature=0.0)[0]["generated_text"]
    respuesta = salida.split("Intención:")[-1].strip().split("\n")[0].strip().lower()

    opciones = {
        "nombre_completo",
        "contar_coincidencias",
        "buscar_por_direccion",
        "buscar_dato_de_persona",
        "buscar_dato_exactitud"
    }

    if respuesta not in opciones:
        print(f"⚠️ Intención no reconocida: {respuesta}. Usando 'buscar_dato_exactitud'")
        return "buscar_dato_exactitud"

    return respuesta


def detectar_campo_llm(pregunta: str) -> str:
    prompt = f"""Detecta qué campo de persona se menciona en la pregunta. Solo responde con una de estas opciones:

Nombre, Paterno, Materno, Tarjeta, Telefono, Calle, Ciudad, Estado, CP, Lada, Sexo

Pregunta: {pregunta}
Campo:"""
    salida = extractor(prompt, max_new_tokens=3, do_sample=False, temperature=0.0)[0]["generated_text"]
    campo = salida.split("Campo:")[-1].strip().split("\n")[0].strip()
    return campo

# -------------------------------
# FLUJO PRINCIPAL
# -------------------------------
intencion = detectar_intencion_llm(pregunta)
print(f"\n🧠 Intención detectada: {intencion}")

nombre_detectado = detectar_nombre(pregunta)

if intencion == "nombre_completo":
    if nombre_detectado:
        buscar_por_nombre_completo(nombre_detectado)
    else:
        print("❌ No se detectó nombre completo válido en la pregunta.")

elif intencion == "contar_coincidencias":
    contar_dato(pregunta)

elif intencion in {"buscar_por_direccion", "buscar_dato_de_persona", "buscar_dato_exactitud"}:
    campo_detectado = detectar_campo_llm(pregunta)
    campo_normalizado = campo_detectado.capitalize()
    print(f"🔍 Campo detectado automáticamente: {campo_normalizado}")
    buscar_persona_por_valor(campo_normalizado, pregunta)

else:
    buscar_dato(pregunta)
