"""Script de prueba para diagnosticar el guardado de imágenes"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import librosa
import seaborn as sns
from scipy.spatial.distance import cdist
import os

# Mismas constantes que en ojo_bd.py
FS = 200  
NPERSEG = 256
N_MFCC = 13

DEFAULT_WORK = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\auditory-evoked-potential-eeg-biometric-dataset-1.0.0\Filtered_Data"
DEFAULT_SAVE = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\Imagenes"

print("=" * 60)
print("PRUEBA DE GUARDADO DE IMÁGENES")
print("=" * 60)

# 1. Verificar carpetas
print(f"\n1. Verificando rutas:")
print(f"   DEFAULT_WORK existe: {os.path.exists(DEFAULT_WORK)}")
print(f"   DEFAULT_SAVE existe: {os.path.exists(DEFAULT_SAVE)}")

# 2. Cargar señal de prueba
print(f"\n2. Cargando archivo de prueba...")
csv_files = [f for f in os.listdir(DEFAULT_WORK) if f.endswith('.csv')]
print(f"   Archivos CSV encontrados: {len(csv_files)}")

if not csv_files:
    print("   ERROR: No hay archivos CSV!")
    exit(1)

test_file = os.path.join(DEFAULT_WORK, csv_files[0])
print(f"   Usando: {csv_files[0]}")

try:
    df = pd.read_csv(test_file)
    channel = "Cz"
    signal = df[channel].values.astype(float)
    print(f"   ✓ Señal cargada: {len(signal)} muestras")
except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

# 3. Calcular componentes
print(f"\n3. Calculando componentes...")
try:
    # Espectrograma
    f, t, Sxx = spectrogram(signal, FS, nperseg=NPERSEG, noverlap=int(NPERSEG * 0.75), window="hann")
    print(f"   ✓ Espectrograma: {Sxx.shape}")
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=FS, n_mfcc=N_MFCC, n_fft=NPERSEG, hop_length=64)
    print(f"   ✓ MFCC: {mfcc.shape}")
    
    # Distancia Coseno
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    dist_matrix = cdist(Sxx_log.T, Sxx_log.T, metric='cosine')
    print(f"   ✓ Distancia Coseno: {dist_matrix.shape}")
    
except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

# 4. INTENTAR GUARDAR IMÁGENES
print(f"\n4. Guardando imágenes...")
base_name = os.path.basename(test_file).replace(".csv", f"_{channel}")

# Espectrograma
print(f"\n   4a. Espectrograma")
try:
    fig = plt.Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    out_folder = os.path.join(DEFAULT_SAVE, "espectrogramas")
    print(f"       Carpeta destino: {out_folder}")
    os.makedirs(out_folder, exist_ok=True)
    print(f"       Carpeta existe: {os.path.exists(out_folder)}")
    
    img_path = os.path.join(out_folder, f"espectrograma_{base_name}.png")
    print(f"       Ruta completa: {img_path}")
    
    fig.savefig(img_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    file_exists = os.path.exists(img_path)
    file_size = os.path.getsize(img_path) if file_exists else 0
    print(f"       ✓ Guardado: {file_exists} | Tamaño: {file_size} bytes")
    
except Exception as e:
    print(f"       ✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# MFCC
print(f"\n   4b. MFCC")
try:
    fig = plt.Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    sns.heatmap(mfcc, ax=ax, cmap="magma", cbar=False)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    out_folder = os.path.join(DEFAULT_SAVE, "mfcc")
    os.makedirs(out_folder, exist_ok=True)
    
    img_path = os.path.join(out_folder, f"mfcc_{base_name}.png")
    fig.savefig(img_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    file_exists = os.path.exists(img_path)
    file_size = os.path.getsize(img_path) if file_exists else 0
    print(f"       ✓ Guardado: {file_exists} | Tamaño: {file_size} bytes")
    
except Exception as e:
    print(f"       ✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Distancia Coseno
print(f"\n   4c. Distancia Coseno")
try:
    fig = plt.Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    dist_masked = dist_matrix.copy()
    np.fill_diagonal(dist_masked, np.nan)
    sns.heatmap(dist_masked, ax=ax, cmap="mako", cbar=False)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    out_folder = os.path.join(DEFAULT_SAVE, "distancia_coseno")
    os.makedirs(out_folder, exist_ok=True)
    
    img_path = os.path.join(out_folder, f"cosine_{base_name}.png")
    fig.savefig(img_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    file_exists = os.path.exists(img_path)
    file_size = os.path.getsize(img_path) if file_exists else 0
    print(f"       ✓ Guardado: {file_exists} | Tamaño: {file_size} bytes")
    
except Exception as e:
    print(f"       ✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("PRUEBA COMPLETADA")
print("=" * 60)
