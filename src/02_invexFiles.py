import os
import pandas as pd
from tqdm import tqdm

# Versión reciente: importamos desde langchain_community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# 1) Leer el Excel
df = pd.read_excel(
    r"C:\Users\Sistemas\Documents\OKIP\datos_1000_personas.xlsx",
    header=0
)

# 2) Crear objetos Document
documents = []
for idx, row in df.iterrows():
    content_lines = []
    for col in df.columns:
        valor = row[col]
        content_lines.append(f"{col}: {valor}")
    content_str = "\n".join(content_lines)
    documents.append(Document(page_content=content_str, metadata={"id": idx})) # Asignar un ID único

# 3) Crear función de embeddings local
embedding_function = HuggingFaceEmbeddings(
    #model_name=r"C:\Users\Monitoreo\Desktop\oki\src\DeepSeek-R1-Distill-Qwen-1.5B",
    model_name=r"C:\Users\Sistemas\Documents\OKIP\src\models--sentence-transformers--all-MiniLM-L6-v2",
    #model_name=r"C:\Users\Monitoreo\Desktop\oki\src\DeepSeek-R1-Distill-Qwen-14B",

    model_kwargs = {'device': 'cuda'}
)

# 4) Crear la base vectorial *vacía* primero
vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory="chroma_db"
)

# 5) Agregar documentos con barra de progreso
print("Indexando documentos en Chroma con progreso...")
for doc in tqdm(documents, desc="Indexing", unit="doc"):
    # `add_texts` se encarga internamente de generar embeddings del texto y guardarlo
    vectorstore.add_texts(
        texts=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents]
    )

# 6) Persistir los cambios en disco
vectorstore.persist()

print("¡Indexación completada!")

# Exportar `documents`
def get_documents():
    return documents