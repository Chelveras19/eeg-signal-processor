
import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
import sqlite3
from datetime import datetime
import json

# --- Constantes ---
FS = 200  # Frecuencia de muestreo (Hz)
CHANNELS = ["T7", "F8", "Cz", "P4"]
DATA_PATH = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\auditory-evoked-potential-eeg-biometric-dataset-1.0.0\Filtered_Data"
OUTPUT_PATH = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\outputs\features"
DB_PATH = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\eeg_database.db"


class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado para tipos NumPy."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class EEGFeatureExtractor:
    """Extractor de features de señales EEG."""
    
    def __init__(self, fs=FS, data_path=DATA_PATH, output_path=OUTPUT_PATH):
        """
        Inicializar extractor.
        
        Args:
            fs (int): Frecuencia de muestreo
            data_path (str): Ruta de los datos
            output_path (str): Ruta de salida
        """
        self.fs = fs
        self.data_path = data_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        self.features_db = {}
        self.feature_matrix = None
    
    def load_signal(self, filepath):
        """Cargar señal desde CSV."""
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise Exception(f"Error cargando {filepath}: {str(e)}")
    
    # ==================== FEATURES TEMPORALES ====================
    
    def extract_temporal_features(self, signal_data):
        """
        Extraer features del dominio temporal.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            
        Returns:
            dict: Features temporales
        """
        features = {
            # Estadísticas básicas
            'mean': float(np.mean(signal_data)),
            'median': float(np.median(signal_data)),
            'std': float(np.std(signal_data)),
            'var': float(np.var(signal_data)),
            'min': float(np.min(signal_data)),
            'max': float(np.max(signal_data)),
            'peak_to_peak': float(np.max(signal_data) - np.min(signal_data)),
            
            # Amplitud y energía
            'rms': float(np.sqrt(np.mean(signal_data**2))),
            'mean_absolute': float(np.mean(np.abs(signal_data))),
            'energy': float(np.sum(signal_data**2)),
            'power': float(np.mean(signal_data**2)),
            
            # Forma de distribución
            'skewness': float(skew(signal_data)),
            'kurtosis': float(kurtosis(signal_data)),
            
            # Información de la señal
            'length': int(len(signal_data)),
            'duration_seconds': float(len(signal_data) / self.fs),
            'sampling_rate': int(self.fs),
        }
        
        return features
    
    def extract_statistical_features(self, signal_data):
        """
        Extraer features estadísticas adicionales.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            
        Returns:
            dict: Features estadísticos
        """
        features = {
            # Percentiles
            'q25': float(np.percentile(signal_data, 25)),
            'q50': float(np.percentile(signal_data, 50)),
            'q75': float(np.percentile(signal_data, 75)),
            'iqr': float(np.percentile(signal_data, 75) - np.percentile(signal_data, 25)),
            
            # Rangos normalizados
            'range': float(np.max(signal_data) - np.min(signal_data)),
            'cv': float(np.std(signal_data) / (np.mean(signal_data) + 1e-10)),  # Coeficiente variación
            
            # Valores de amplitud específicos
            'max_absolute': float(np.max(np.abs(signal_data))),
            'mean_absolute_deviation': float(np.mean(np.abs(signal_data - np.mean(signal_data)))),
            
            # Momentos estadísticos
            'moment_3': float(np.mean((signal_data - np.mean(signal_data))**3)),
            'moment_4': float(np.mean((signal_data - np.mean(signal_data))**4)),
        }
        
        return features
    
    # ==================== FEATURES DE VARIABILIDAD ====================
    
    def extract_variability_features(self, signal_data):
        """
        Extraer features de variabilidad temporal.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            
        Returns:
            dict: Features de variabilidad
        """
        # Diferencias de primer orden
        diff = np.diff(signal_data)
        
        features = {
            # Variabilidad de cambios
            'diff_mean': float(np.mean(np.abs(diff))),
            'diff_std': float(np.std(diff)),
            'diff_max': float(np.max(np.abs(diff))),
            'diff_energy': float(np.sum(diff**2)),
            
            # Cambios relativos
            'velocity_mean': float(np.mean(np.abs(diff)) / (np.max(signal_data) - np.min(signal_data) + 1e-10)),
            
            # Zero-crossing rate (número de cambios de signo)
            'zero_crossings': int(np.sum(np.abs(np.diff(np.sign(signal_data)))) / 2),
            'zero_crossing_rate': float(np.sum(np.abs(np.diff(np.sign(signal_data)))) / (2 * len(signal_data))),
            
            # Turning points (cambios de dirección)
            'turning_points': int(np.sum(np.abs(np.diff(np.sign(diff))))),
            
            # Autocorrelación en lag 1
            'autocorr_lag1': float(np.correlate(signal_data - np.mean(signal_data), 
                                               signal_data - np.mean(signal_data), mode='same')[len(signal_data)//2] / np.var(signal_data) / len(signal_data)),
        }
        
        return features
    
    # ==================== FEATURES DE COMPLEJIDAD ====================
    
    def extract_complexity_features(self, signal_data):
        """
        Extraer features de complejidad de la señal.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            
        Returns:
            dict: Features de complejidad
        """
        features = {
            # Entropía espectral
            'entropy': float(entropy(np.abs(np.fft.fft(signal_data))[:len(signal_data)//2 + 1])),
            
            # Entropía de Shannon (usando histograma)
            'shannon_entropy': self._calculate_shannon_entropy(signal_data),
            
            # Aproximación de dimensión fractal
            'fractal_dimension': self._calculate_fractal_dimension(signal_data),
        }
        
        return features
    
    def _calculate_shannon_entropy(self, signal_data):
        """Calcular entropía de Shannon."""
        # Normalizar y crear histograma
        hist, _ = np.histogram(signal_data, bins=32)
        hist = hist / hist.sum()  # Normalizar
        hist = hist[hist > 0]  # Remover ceros
        entropy_value = -np.sum(hist * np.log2(hist))
        return float(entropy_value)
    
    def _calculate_fractal_dimension(self, signal_data, max_scale=100):
        """Calcular dimensión fractal usando Higuchi."""
        scales = np.arange(1, min(max_scale, len(signal_data)//10))
        lk = []
        
        for k in scales:
            Lmk = []
            for m in range(k):
                indices = np.arange(m, len(signal_data), k)
                if len(indices) > 1:
                    Lm = np.sum(np.abs(np.diff(signal_data[indices]))) * (len(signal_data) - 1) / ((len(signal_data) - 1) // k * k) ** 2
                    Lmk.append(Lm)
            if Lmk:
                lk.append(np.log(np.mean(Lmk)))
        
        if len(scales) > 1 and len(lk) > 1:
            coeffs = np.polyfit(np.log(scales), lk, 1)
            return float(-coeffs[0])  # Dimensión fractal
        return 0.0
    
    # ==================== FEATURES FRECUENCIALES ====================
    
    def extract_frequency_features(self, signal_data):
        """
        Extraer features del dominio frecuencial.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            
        Returns:
            dict: Features frecuenciales
        """
        # Calcular PSD
        freqs, psd = signal.welch(signal_data, fs=self.fs, nperseg=256)
        
        # Bandas EEG
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 50)
        }
        
        features = {
            'peak_frequency': float(freqs[np.argmax(psd)]),
            'peak_power': float(np.max(psd)),
            'total_power': float(np.sum(psd)),
            'mean_frequency': float(np.sum(freqs * psd) / np.sum(psd)),
        }
        
        # Potencia por banda
        for band_name, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs <= f_high)
            band_power = np.sum(psd[mask])
            features[f'{band_name}_power'] = float(band_power)
            features[f'{band_name}_percentage'] = float(band_power / np.sum(psd) * 100)
        
        # Ratios de bandas
        delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 12)])
        theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
        beta_power = np.sum(psd[(freqs >= 12) & (freqs <= 30)])
        
        features['delta_theta_ratio'] = float(delta_power / (theta_power + 1e-10))
        features['alpha_beta_ratio'] = float(alpha_power / (beta_power + 1e-10))
        features['theta_alpha_ratio'] = float(theta_power / (alpha_power + 1e-10))
        features['delta_alpha_ratio'] = float(delta_power / (alpha_power + 1e-10))
        
        return features
    
    # ==================== FEATURES DE ACTIVIDAD ====================
    
    def extract_activity_features(self, signal_data):
        """
        Extraer features de actividad y amplitud.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            
        Returns:
            dict: Features de actividad
        """
        features = {
            # Actividad (variance)
            'activity': float(np.var(signal_data)),
            
            # Movilidad (variancia de primera derivada)
            'mobility': float(np.sqrt(np.var(np.diff(signal_data)) / np.var(signal_data))),
            
            # Complejidad (relación entre movilidad)
            'complexity': float(np.sqrt(np.var(np.diff(np.diff(signal_data))) / np.var(np.diff(signal_data))) / (np.sqrt(np.var(np.diff(signal_data)) / np.var(signal_data)))),
        }
        
        return features
    
    # ==================== DETECCIÓN DE PROBLEMAS ====================
    
    def detect_quality_issues(self, signal_data):
        """
        Detectar problemas de calidad.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            
        Returns:
            dict: Problemas detectados
        """
        issues = {
            'has_flat_segments': False,
            'has_high_artifacts': False,
            'has_saturation': False,
            'has_extreme_outliers': False,
            'quality_score': 100.0,  # De 0 a 100
        }
        
        # Detectar segmentos planos
        window_size = int(0.1 * self.fs)
        flat_count = 0
        for i in range(0, len(signal_data) - window_size, window_size):
            if np.std(signal_data[i:i+window_size]) < 0.1:
                flat_count += 1
        if flat_count > 5:
            issues['has_flat_segments'] = True
            issues['quality_score'] -= 20
        
        # Detectar artefactos (amplitud > 100 µV)
        artifact_pct = (np.sum(np.abs(signal_data) > 100) / len(signal_data)) * 100
        if artifact_pct > 5:
            issues['has_high_artifacts'] = True
            issues['quality_score'] -= 25
        issues['artifact_percentage'] = float(artifact_pct)
        
        # Detectar saturación
        saturation_pct = (np.sum(np.abs(signal_data) > 90) / len(signal_data)) * 100
        if saturation_pct > 1:
            issues['has_saturation'] = True
            issues['quality_score'] -= 10
        issues['saturation_percentage'] = float(saturation_pct)
        
        # Detectar outliers extremos
        z_scores = np.abs((signal_data - np.mean(signal_data)) / np.std(signal_data))
        outlier_pct = (np.sum(z_scores > 3) / len(signal_data)) * 100
        if outlier_pct > 2:
            issues['has_extreme_outliers'] = True
            issues['quality_score'] -= 15
        issues['outlier_percentage'] = float(outlier_pct)
        
        # Asegurar que quality_score esté entre 0 y 100
        issues['quality_score'] = max(0, min(100, issues['quality_score']))
        
        return issues
    
    # ==================== EXTRACCIÓN COMPLETA ====================
    
    def extract_all_features(self, filename):
        """
        Extraer todos los features de un archivo.
        
        Args:
            filename (str): Nombre del archivo CSV
            
        Returns:
            dict: Todos los features extraídos
        """
        filepath = os.path.join(self.data_path, filename)
        
        print(f"Extrayendo features: {filename}")
        
        try:
            df = self.load_signal(filepath)
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return None
        
        # Extraer información del nombre
        parts = filename.replace('.csv', '').split('_')
        subject = parts[0]
        experiment = parts[1]
        session = parts[2] if len(parts) > 2 else "N/A"
        
        file_features = {
            'filename': filename,
            'subject': subject,
            'experiment': experiment,
            'session': session,
            'timestamp': datetime.now().isoformat(),
            'channels': {}
        }
        
        # Procesar cada canal
        for channel in CHANNELS:
            if channel not in df.columns:
                continue
            
            signal_data = df[channel].values.astype(float)
            
            channel_features = {}
            
            # Extraer todos los grupos de features
            channel_features.update(self.extract_temporal_features(signal_data))
            channel_features.update(self.extract_statistical_features(signal_data))
            channel_features.update(self.extract_variability_features(signal_data))
            channel_features.update(self.extract_complexity_features(signal_data))
            channel_features.update(self.extract_frequency_features(signal_data))
            channel_features.update(self.extract_activity_features(signal_data))
            channel_features.update(self.detect_quality_issues(signal_data))
            
            file_features['channels'][channel] = channel_features
        
        self.features_db[filename] = file_features
        return file_features
    
    def extract_all_files(self):
        """Extraer features de todos los archivos."""
        csv_files = sorted([f for f in os.listdir(self.data_path) if f.endswith('.csv')])
        print(f"\nEncontrados {len(csv_files)} archivos CSV\n")
        
        for filename in csv_files:
            self.extract_all_features(filename)
        
        print(f"\nExtracción completada: {len(self.features_db)} archivos procesados\n")
    
    # ==================== GUARDADO DE RESULTADOS ====================
    
    def save_results(self):
        """Guardar resultados en múltiples formatos."""
        
        # 1. JSON detallado
        with open(os.path.join(self.output_path, 'features_detailed.json'), 'w') as f:
            json.dump(self.features_db, f, indent=2, cls=NumpyEncoder)
        print("✓ Características detalladas guardadas: features_detailed.json")
        
        # 2. CSV con todas las características por archivo y canal
        rows = []
        for filename, file_data in self.features_db.items():
            subject = file_data['subject']
            experiment = file_data['experiment']
            session = file_data['session']
            
            for channel, channel_features in file_data['channels'].items():
                row = {
                    'filename': filename,
                    'subject': subject,
                    'experiment': experiment,
                    'session': session,
                    'channel': channel,
                }
                row.update(channel_features)
                rows.append(row)
        
        df_features = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_path, 'features_table.csv')
        df_features.to_csv(csv_path, index=False)
        print(f"✓ Tabla de características guardada: features_table.csv ({len(rows)} filas, {len(df_features.columns)} columnas)")
        
        # 3. Matriz de características por archivo (promedio de canales)
        channel_avg_rows = []
        for filename, file_data in self.features_db.items():
            subject = file_data['subject']
            experiment = file_data['experiment']
            session = file_data['session']
            
            # Promediar características de los 4 canales
            all_channel_features = []
            for channel, channel_features in file_data['channels'].items():
                all_channel_features.append(channel_features)
            
            if all_channel_features:
                # Calcular promedio para features numéricas
                avg_row = {
                    'filename': filename,
                    'subject': subject,
                    'experiment': experiment,
                    'session': session,
                }
                
                # Obtener nombres de features
                feature_keys = all_channel_features[0].keys()
                
                for key in feature_keys:
                    values = [f[key] for f in all_channel_features if isinstance(f.get(key), (int, float))]
                    if values:
                        avg_row[f'{key}_avg'] = float(np.mean(values))
                        avg_row[f'{key}_std'] = float(np.std(values))
                
                channel_avg_rows.append(avg_row)
        
        df_avg = pd.DataFrame(channel_avg_rows)
        avg_path = os.path.join(self.output_path, 'features_by_file.csv')
        df_avg.to_csv(avg_path, index=False)
        print(f"✓ Características promediadas por archivo: features_by_file.csv ({len(channel_avg_rows)} filas)")
        
        # 4. Resumen de calidad
        quality_rows = []
        for filename, file_data in self.features_db.items():
            subject = file_data['subject']
            experiment = file_data['experiment']
            session = file_data['session']
            
            for channel, channel_features in file_data['channels'].items():
                quality_rows.append({
                    'filename': filename,
                    'subject': subject,
                    'experiment': experiment,
                    'session': session,
                    'channel': channel,
                    'quality_score': channel_features.get('quality_score', 0),
                    'artifact_percentage': channel_features.get('artifact_percentage', 0),
                    'saturation_percentage': channel_features.get('saturation_percentage', 0),
                    'outlier_percentage': channel_features.get('outlier_percentage', 0),
                    'has_flat_segments': channel_features.get('has_flat_segments', False),
                    'has_high_artifacts': channel_features.get('has_high_artifacts', False),
                    'has_saturation': channel_features.get('has_saturation', False),
                    'has_extreme_outliers': channel_features.get('has_extreme_outliers', False),
                })
        
        df_quality = pd.DataFrame(quality_rows)
        quality_path = os.path.join(self.output_path, 'quality_assessment.csv')
        df_quality.to_csv(quality_path, index=False)
        print(f"✓ Evaluación de calidad guardada: quality_assessment.csv")
    
    def create_database_table(self):
        """Crear tabla en SQLite para características."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS eeg_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                subject TEXT,
                experiment TEXT,
                session TEXT,
                channel TEXT NOT NULL,
                mean REAL, median REAL, std REAL, var REAL,
                min REAL, max REAL, peak_to_peak REAL,
                rms REAL, mean_absolute REAL, energy REAL, power REAL,
                skewness REAL, kurtosis REAL,
                length INTEGER, duration_seconds REAL, sampling_rate INTEGER,
                q25 REAL, q50 REAL, q75 REAL, iqr REAL,
                range REAL, cv REAL, max_absolute REAL, mean_absolute_deviation REAL,
                moment_3 REAL, moment_4 REAL,
                diff_mean REAL, diff_std REAL, diff_max REAL, diff_energy REAL,
                velocity_mean REAL, zero_crossings INTEGER, zero_crossing_rate REAL,
                turning_points INTEGER, autocorr_lag1 REAL,
                entropy REAL, shannon_entropy REAL, fractal_dimension REAL,
                peak_frequency REAL, peak_power REAL, total_power REAL, mean_frequency REAL,
                delta_power REAL, delta_percentage REAL,
                theta_power REAL, theta_percentage REAL,
                alpha_power REAL, alpha_percentage REAL,
                beta_power REAL, beta_percentage REAL,
                gamma_power REAL, gamma_percentage REAL,
                delta_theta_ratio REAL, alpha_beta_ratio REAL, theta_alpha_ratio REAL, delta_alpha_ratio REAL,
                activity REAL, mobility REAL, complexity REAL,
                quality_score REAL, artifact_percentage REAL, saturation_percentage REAL,
                outlier_percentage REAL,
                has_flat_segments INTEGER, has_high_artifacts INTEGER,
                has_saturation INTEGER, has_extreme_outliers INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(filename, channel)
            )
        ''')
        
        # Insertar datos
        for filename, file_data in self.features_db.items():
            for channel, channel_features in file_data['channels'].items():
                try:
                    # Preparar valores, usando None para claves faltantes
                    values = [filename, file_data['subject'], file_data['experiment'], file_data['session'], channel]
                    
                    # Lista de todas las columnas de features en orden
                    feature_cols = [
                        'mean', 'median', 'std', 'var', 'min', 'max', 'peak_to_peak',
                        'rms', 'mean_absolute', 'energy', 'power', 'skewness', 'kurtosis',
                        'length', 'duration_seconds', 'sampling_rate',
                        'q25', 'q50', 'q75', 'iqr', 'range', 'cv', 'max_absolute', 'mean_absolute_deviation',
                        'moment_3', 'moment_4',
                        'diff_mean', 'diff_std', 'diff_max', 'diff_energy', 'velocity_mean',
                        'zero_crossings', 'zero_crossing_rate', 'turning_points', 'autocorr_lag1',
                        'entropy', 'shannon_entropy', 'fractal_dimension',
                        'peak_frequency', 'peak_power', 'total_power', 'mean_frequency',
                        'delta_power', 'delta_percentage', 'theta_power', 'theta_percentage',
                        'alpha_power', 'alpha_percentage', 'beta_power', 'beta_percentage',
                        'gamma_power', 'gamma_percentage',
                        'delta_theta_ratio', 'alpha_beta_ratio', 'theta_alpha_ratio', 'delta_alpha_ratio',
                        'activity', 'mobility', 'complexity',
                        'quality_score', 'artifact_percentage', 'saturation_percentage', 'outlier_percentage',
                        'has_flat_segments', 'has_high_artifacts', 'has_saturation', 'has_extreme_outliers'
                    ]
                    
                    for col in feature_cols:
                        val = channel_features.get(col)
                        if isinstance(val, bool):
                            val = int(val)
                        values.append(val)
                    
                    placeholders = ', '.join(['?' for _ in values])
                    cursor.execute(f'''
                        INSERT OR REPLACE INTO eeg_features
                        (filename, subject, experiment, session, channel, {', '.join(feature_cols)})
                        VALUES ({placeholders})
                    ''', values)
                except Exception as e:
                    print(f"  ⚠ Error insertando {filename}, {channel}: {str(e)}")
        
        conn.commit()
        conn.close()
        print("✓ Características guardadas en base de datos SQLite")
    
    def print_summary(self):
        """Imprimir resumen de features extraídos."""
        if not self.features_db:
            print("No hay features para mostrar")
            return
        
        # Obtener primer archivo como ejemplo
        first_file = list(self.features_db.values())[0]
        first_channel = first_file['channels']['Cz']
        
        print("\n" + "="*70)
        print("RESUMEN DE FEATURES EXTRAÍDOS")
        print("="*70)
        
        print(f"\nArchivos procesados: {len(self.features_db)}")
        print(f"Canales por archivo: {len(CHANNELS)}")
        print(f"Total de análisis (archivo × canal): {len(self.features_db) * len(CHANNELS)}")
        
        print(f"\nNúmero total de features por canal: {len(first_channel)}")
        
        print("\nGrupos de features:")
        print(f"  - Temporales: 19 features")
        print(f"  - Estadísticos: 9 features")
        print(f"  - Variabilidad: 9 features")
        print(f"  - Complejidad: 3 features")
        print(f"  - Frecuenciales: 21 features")
        print(f"  - Actividad: 3 features")
        print(f"  - Calidad: 8 features")
        print(f"  - TOTAL: ~72 features por canal")
        
        print("\n" + "="*70)


def main():
    """Función principal."""
    print("="*70)
    print("EEG FEATURE EXTRACTION - EXTRACCIÓN AUTOMÁTICA DE CARACTERÍSTICAS")
    print("="*70)
    
    extractor = EEGFeatureExtractor()
    
    # Extraer features de todos los archivos
    print("\nExtrayendo características...")
    extractor.extract_all_files()
    
    # Guardar resultados
    print("\nGuardando resultados...")
    extractor.save_results()
    
    # Guardar en base de datos
    print("\nGuardando en base de datos...")
    extractor.create_database_table()
    
    # Mostrar resumen
    extractor.print_summary()
    
    print("\n" + "="*70)
    print("EXTRACCIÓN COMPLETADA")
    print("="*70)
    print(f"Resultados guardados en: {extractor.output_path}")


if __name__ == "__main__":
    main()
