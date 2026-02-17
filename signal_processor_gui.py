import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import librosa
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from scipy.spatial.distance import cdist
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import threading
from queue import Queue
import sqlite3
from datetime import datetime

# --- Constantes ---
FS = 200  
NPERSEG = 256
N_MFCC = 13
CHANNELS = ["T7", "F8", "Cz", "P4"]
WINDOW_TYPES = ["hann", "hamming", "blackman", "boxcar"]
OVERLAP_OPTIONS = [60, 70, 75, 80, 85, 90, 95]

DEFAULT_WORK = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\auditory-evoked-potential-eeg-biometric-dataset-1.0.0\Filtered_Data"
DEFAULT_SAVE = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\Imagenes"
DB_PATH = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\eeg_database.db"


class DatabaseManager:
    """Gestor de base de datos SQLite."""
    
    def __init__(self, db_path=DB_PATH):
        """
        Inicializar gestor de base de datos.
        
        Args:
            db_path (str): Ruta de la base de datos
        """
        self.db_path = db_path
        self.create_tables()
    
    def create_tables(self):
        """Crear tablas en la base de datos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla para almacenar análisis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                csv_file TEXT NOT NULL,
                channel TEXT NOT NULL,
                window_type TEXT NOT NULL,
                overlap_percentage REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                spectrogram_path TEXT,
                mfcc_path TEXT,
                cosine_distance_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, csv_file, channel, window_type, overlap_percentage, 
                     spectrogram_path=None, mfcc_path=None, cosine_distance_path=None):
        """
        Guardar análisis en la base de datos.
        
        Args:
            csv_file (str): Nombre del archivo CSV
            channel (str): Canal EEG
            window_type (str): Tipo de ventana
            overlap_percentage (float): Porcentaje de traslape
            spectrogram_path (str): Ruta del espectrograma
            mfcc_path (str): Ruta del MFCC
            cosine_distance_path (str): Ruta de la distancia coseno
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analyses (csv_file, channel, window_type, overlap_percentage,
                                 spectrogram_path, mfcc_path, cosine_distance_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (csv_file, channel, window_type, overlap_percentage,
              spectrogram_path, mfcc_path, cosine_distance_path))
        
        conn.commit()
        conn.close()
    
    def get_analyses(self, csv_file=None, channel=None):
        """
        Obtener análisis de la base de datos.
        
        Args:
            csv_file (str): Filtrar por archivo CSV
            channel (str): Filtrar por canal
            
        Returns:
            list: Lista de análisis
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM analyses WHERE 1=1"
        params = []
        
        if csv_file:
            query += " AND csv_file = ?"
            params.append(csv_file)
        if channel:
            query += " AND channel = ?"
            params.append(channel)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return results


class EEGProcessor:
    """Clase para procesar señales EEG."""
    
    def __init__(self, fs=FS, nperseg=NPERSEG, n_mfcc=N_MFCC):
        """
        Inicializar procesador EEG.
        
        Args:
            fs (int): Frecuencia de muestreo
            nperseg (int): Tamaño de la ventana
            n_mfcc (int): Número de coeficientes MFCC
        """
        self.fs = fs
        self.nperseg = nperseg
        self.n_mfcc = n_mfcc
        self.signal = None
        self.channel = None
    
    def load_signal(self, filepath, channel):
        """
        Cargar señal desde archivo CSV.
        
        Args:
            filepath (str): Ruta del archivo CSV
            channel (str): Nombre del canal
            
        Returns:
            bool: True si se cargó correctamente
        """
        try:
            df = pd.read_csv(filepath)
            if channel not in df.columns:
                raise ValueError(f"Canal '{channel}' no encontrado en el archivo")
            self.signal = df[channel].values.astype(float)
            self.channel = channel
            if len(self.signal) == 0:
                raise ValueError("La señal está vacía")
            return True
        except Exception as e:
            raise Exception(f"Error al cargar archivo: {str(e)}")
    
    def get_spectrogram(self, window_type="hann", overlap_percentage=75.0):
        """
        Calcular espectrograma.
        
        Args:
            window_type (str): Tipo de ventana
            overlap_percentage (float): Porcentaje de traslape
            
        Returns:
            tuple: (f, t, Sxx) - Frecuencias, tiempos, potencia
        """
        if self.signal is None:
            raise ValueError("No hay señal cargada")
        
        noverlap = int(self.nperseg * (overlap_percentage / 100.0))
        f, t, Sxx = spectrogram(
            self.signal, 
            self.fs, 
            nperseg=self.nperseg, 
            noverlap=noverlap, 
            window=window_type
        )
        return f, t, Sxx
    
    def get_mfcc(self, window_type="hann", overlap_percentage=75.0):
        """
        Calcular coeficientes MFCC.
        
        Args:
            window_type (str): Tipo de ventana
            overlap_percentage (float): Porcentaje de traslape
            
        Returns:
            np.ndarray: Matriz MFCC
        """
        if self.signal is None:
            raise ValueError("No hay señal cargada")
        
        hop_length = int(self.nperseg * (1 - overlap_percentage / 100.0))
        mfcc = librosa.feature.mfcc(
            y=self.signal, 
            sr=self.fs, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.nperseg, 
            hop_length=max(1, hop_length)
        )
        return mfcc
    
    def get_cosine_distance_matrix(self, window_type="hann", overlap_percentage=75.0):
        """
        Calcular matriz de distancia del coseno.
        
        Args:
            window_type (str): Tipo de ventana
            overlap_percentage (float): Porcentaje de traslape
            
        Returns:
            np.ndarray: Matriz de distancia del coseno
        """
        f, t, Sxx = self.get_spectrogram(window_type, overlap_percentage)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        dist_matrix = cdist(Sxx_log.T, Sxx_log.T, metric='cosine')
        return dist_matrix
    
    def get_signal_stats(self):
        """Obtener estadísticas de la señal."""
        if self.signal is None:
            return {}
        
        return {
            'mean': np.mean(self.signal),
            'std': np.std(self.signal),
            'min': np.min(self.signal),
            'max': np.max(self.signal),
            'length': len(self.signal),
            'duration_seconds': len(self.signal) / self.fs
        }


class SpectrogramApp:
    """Aplicación GUI para visualización EEG con base de datos."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador EEG - Base de Datos")
        self.root.geometry("1600x1000")
        
        # Variables
        self.window_type = tk.StringVar(value="hann")
        self.overlap_percentage = tk.DoubleVar(value=75.0)
        self.channel_selected = tk.StringVar(value="Cz")
        self.current_file = None
        
        # Managers
        self.processor = EEGProcessor()
        self.db = DatabaseManager()
        
        # Cola para threading
        self.queue = Queue()
        
        self.setup_ui()
        self.load_csv_files()
        self.check_queue()
    
    def setup_ui(self):
        """Configurar interfaz de usuario."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # --- Panel de opciones ---
        options_frame = ttk.LabelFrame(main_frame, text="Opciones de Análisis", padding="10")
        options_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(options_frame, text="Archivo CSV:").pack(side=tk.LEFT, padx=5)
        self.file_combobox = ttk.Combobox(options_frame, state="readonly", width=50)
        self.file_combobox.pack(side=tk.LEFT, padx=5)
        self.file_combobox.bind("<<ComboboxSelected>>", self.on_file_selected)
        
        ttk.Label(options_frame, text="Canal EEG:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(options_frame, textvariable=self.channel_selected, 
                     values=CHANNELS, state="readonly", width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(options_frame, text="Ventana:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(options_frame, textvariable=self.window_type, 
                     values=WINDOW_TYPES, state="readonly", width=12).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(options_frame, text="Traslape (%):").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(options_frame, textvariable=self.overlap_percentage, 
                     values=OVERLAP_OPTIONS, state="readonly", width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(options_frame, text="Generar Análisis", 
                   command=self.generate_analysis_threaded).pack(side=tk.LEFT, padx=10)
        
        # --- Barra de estado ---
        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # --- Área de gráficas (3 columnas) ---
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.columnconfigure(1, weight=1)
        canvas_frame.columnconfigure(2, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Espectrograma
        spec_frame = ttk.LabelFrame(canvas_frame, text="Espectrograma", padding="5")
        spec_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        spec_frame.columnconfigure(0, weight=1)
        spec_frame.rowconfigure(0, weight=1)
        self.spec_canvas_frame = spec_frame
        
        # MFCC
        mfcc_frame = ttk.LabelFrame(canvas_frame, text="MFCC", padding="5")
        mfcc_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        mfcc_frame.columnconfigure(0, weight=1)
        mfcc_frame.rowconfigure(0, weight=1)
        self.mfcc_canvas_frame = mfcc_frame
        
        # Distancia Coseno
        cosine_frame = ttk.LabelFrame(canvas_frame, text="Distancia Coseno", padding="5")
        cosine_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        cosine_frame.columnconfigure(0, weight=1)
        cosine_frame.rowconfigure(0, weight=1)
        self.cosine_canvas_frame = cosine_frame
        
        # Almacenar referencias
        self.canvases = {
            'spec': None,
            'mfcc': None,
            'cosine': None
        }
    
    def load_csv_files(self):
        """Cargar lista de archivos CSV."""
        try:
            csv_files = sorted([f for f in os.listdir(DEFAULT_WORK) if f.endswith('.csv')])
            self.file_combobox['values'] = csv_files
            if csv_files:
                self.file_combobox.set(csv_files[0])
                self.current_file = os.path.join(DEFAULT_WORK, csv_files[0])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer la carpeta: {str(e)}")
    
    def on_file_selected(self, event):
        """Actualizar archivo actual cuando se selecciona uno en el combobox."""
        selected = self.file_combobox.get()
        if selected:
            self.current_file = os.path.join(DEFAULT_WORK, selected)
            print(f"Archivo seleccionado: {self.current_file}")
    
    def generate_analysis_threaded(self):
        """Generar análisis en thread separado."""
        if not self.current_file:
            messagebox.showwarning("Advertencia", "Seleccione un archivo CSV.")
            return
        
        self.status_var.set("Procesando...")
        self.root.update()
        
        thread = threading.Thread(target=self._analysis_worker)
        thread.daemon = True
        thread.start()
    
    def _analysis_worker(self):
        """Worker para generar análisis."""
        try:
            channel = self.channel_selected.get()
            window_type = self.window_type.get()
            overlap = self.overlap_percentage.get()
            
            # Verificar que DEFAULT_SAVE existe
            os.makedirs(DEFAULT_SAVE, exist_ok=True)
            
            # Cargar señal
            self.processor.load_signal(self.current_file, channel)
            
            # Calcular componentes
            f, t, Sxx = self.processor.get_spectrogram(window_type, overlap)
            mfcc = self.processor.get_mfcc(window_type, overlap)
            dist_matrix = self.processor.get_cosine_distance_matrix(window_type, overlap)
            stats = self.processor.get_signal_stats()
            
            # Guardar imágenes
            base_name = os.path.basename(self.current_file).replace(".csv", f"_{channel}")
            
            # Espectrograma
            spec_img = self._save_spectrogram_image(f, t, Sxx, channel, base_name)
            
            # MFCC
            mfcc_img = self._save_mfcc_image(mfcc, channel, base_name)
            
            # Distancia Coseno
            cosine_img = self._save_cosine_image(dist_matrix, channel, base_name)
            
            # Guardar en BD
            csv_name = os.path.basename(self.current_file)
            self.db.save_analysis(csv_name, channel, window_type, overlap,
                                 spec_img, mfcc_img, cosine_img)
            
            self.queue.put(('success', {
                'f': f, 't': t, 'Sxx': Sxx,
                'mfcc': mfcc,
                'dist_matrix': dist_matrix,
                'stats': stats,
                'spec_img': spec_img,
                'mfcc_img': mfcc_img,
                'cosine_img': cosine_img
            }))
        except Exception as e:
            self.queue.put(('error', str(e)))
    
    def _save_spectrogram_image(self, f, t, Sxx, channel, base_name):
        """Guardar imagen del espectrograma."""
        try:
            fig = plt.Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            out_folder = os.path.join(DEFAULT_SAVE, "espectrogramas")
            os.makedirs(out_folder, exist_ok=True)
            
            img_path = os.path.join(out_folder, f"espectrograma_{base_name}.png")
            fig.savefig(img_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            if not os.path.exists(img_path):
                raise Exception(f"No se guardó el archivo: {img_path}")
            
            return img_path
        except Exception as e:
            raise Exception(f"Error al guardar espectrograma: {str(e)}")
    
    def _save_mfcc_image(self, mfcc, channel, base_name):
        """Guardar imagen del MFCC."""
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
            
            if not os.path.exists(img_path):
                raise Exception(f"No se guardó el archivo: {img_path}")
            
            return img_path
        except Exception as e:
            raise Exception(f"Error al guardar MFCC: {str(e)}")
    
    def _save_cosine_image(self, dist_matrix, channel, base_name):
        """Guardar imagen de distancia coseno."""
        try:
            fig = plt.Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Enmascarar diagonal
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
            
            if not os.path.exists(img_path):
                raise Exception(f"No se guardó el archivo: {img_path}")
            
            return img_path
        except Exception as e:
            raise Exception(f"Error al guardar distancia coseno: {str(e)}")
    
    def _display_results(self, data):
        """Mostrar resultados en la GUI."""
        # Limpiar canvases anteriores
        for key in self.canvases:
            if self.canvases[key]:
                self.canvases[key].get_tk_widget().destroy()
        
        channel = self.channel_selected.get()
        window_type = self.window_type.get()
        overlap = self.overlap_percentage.get()
        
        # --- Espectrograma ---
        fig_spec = plt.Figure(figsize=(6, 4), dpi=100)
        ax_spec = fig_spec.add_subplot(111)
        
        f, t, Sxx = data['f'], data['t'], data['Sxx']
        im = ax_spec.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        ax_spec.set_title(f"Espectrograma - {channel}")
        ax_spec.set_xlabel("Tiempo [s]")
        ax_spec.set_ylabel("Frecuencia [Hz]")
        fig_spec.colorbar(im, ax=ax_spec, label="Potencia (dB)")
        fig_spec.tight_layout()
        
        self.canvases['spec'] = FigureCanvasTkAgg(fig_spec, self.spec_canvas_frame)
        self.canvases['spec'].draw()
        self.canvases['spec'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- MFCC ---
        fig_mfcc = plt.Figure(figsize=(6, 4), dpi=100)
        ax_mfcc = fig_mfcc.add_subplot(111)
        
        mfcc = data['mfcc']
        sns.heatmap(mfcc, ax=ax_mfcc, cmap="magma", cbar=True)
        ax_mfcc.set_title(f"MFCC - {channel}")
        ax_mfcc.set_xlabel("Ventanas temporales")
        ax_mfcc.set_ylabel("Coeficientes MFCC")
        fig_mfcc.tight_layout()
        
        self.canvases['mfcc'] = FigureCanvasTkAgg(fig_mfcc, self.mfcc_canvas_frame)
        self.canvases['mfcc'].draw()
        self.canvases['mfcc'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- Distancia Coseno ---
        fig_cosine = plt.Figure(figsize=(6, 4), dpi=100)
        ax_cosine = fig_cosine.add_subplot(111)
        
        dist_matrix = data['dist_matrix']
        dist_masked = dist_matrix.copy()
        np.fill_diagonal(dist_masked, np.nan)
        
        sns.heatmap(dist_masked, ax=ax_cosine, cmap="mako", cbar=True)
        ax_cosine.set_title(f"Distancia Coseno - {channel}")
        ax_cosine.set_xlabel("Ventanas temporales")
        ax_cosine.set_ylabel("Ventanas temporales")
        fig_cosine.tight_layout()
        
        self.canvases['cosine'] = FigureCanvasTkAgg(fig_cosine, self.cosine_canvas_frame)
        self.canvases['cosine'].draw()
        self.canvases['cosine'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set(
            f"✓ Análisis completado | {channel} | {window_type} | {overlap}% | "
            f"Duración: {data['stats']['duration_seconds']:.2f}s"
        )
    
    def check_queue(self):
        """Verificar cola de mensajes desde threads."""
        try:
            msg_type, data = self.queue.get_nowait()
            
            if msg_type == 'success':
                self._display_results(data)
            elif msg_type == 'error':
                messagebox.showerror("Error", f"Error al procesar: {data}")
                self.status_var.set("Error")
        except:
            pass
        
        self.root.after(100, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrogramApp(root)
    
    # Cargar archivos al iniciar
    root.after(100, app.load_csv_files)
    
    root.mainloop()
