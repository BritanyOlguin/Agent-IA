import gc
import psutil
import os
import torch

def liberar_memoria_cpu():
    print("🧹 Liberando memoria RAM (CPU)...")
    gc.collect()

def liberar_memoria_gpu():
    if torch.cuda.is_available():
        print("🧹 Liberando caché de memoria GPU (CUDA)...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    else:
        print("⚠️ CUDA no disponible. No se liberó memoria de GPU.")

def mostrar_estado_memoria():
    mem = psutil.virtual_memory()
    print(f"\n📊 Uso actual de RAM:")
    print(f"  Total: {mem.total // (1024 ** 2)} MB")
    print(f"  Usada: {mem.used // (1024 ** 2)} MB")
    print(f"  Libre: {mem.available // (1024 ** 2)} MB\n")

if __name__ == "__main__":
    mostrar_estado_memoria()
    liberar_memoria_cpu()
    liberar_memoria_gpu()
    mostrar_estado_memoria()
