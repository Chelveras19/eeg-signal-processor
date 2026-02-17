import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import signal
from scipy.stats import skew, kurtosis
import json
from datetime import datetime
import sqlite3


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

# --- Constantes ---
FS = 200  # Frecuencia de muestreo (Hz)
CHANNELS = ["T7", "F8", "Cz", "P4"]
DATA_PATH = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\auditory-evoked-potential-eeg-biometric-dataset-1.0.0\Filtered_Data"
OUTPUT_PATH = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\outputs\preprocessing"
DB_PATH = r"C:\Users\salva\Documents\Licenciatura\Servicio Social\eeg_database.db"


class EEGStatsAnalyzer:
    """Analizador de estadísticas y detección de problemas en señales EEG."""
    
    def __init__(self, fs=FS, data_path=DATA_PATH, output_path=OUTPUT_PATH):
        """
        Inicializar analizador.
        
        Args:
            fs (int): Frecuencia de muestreo
            data_path (str): Ruta de los datos
            output_path (str): Ruta de salida para reportes
        """
        self.fs = fs
        self.data_path = data_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        self.stats_db = {}
        self.problems = {}
    
    def load_signal(self, filepath):
        """
        Cargar señal desde CSV.
        
        Args:
            filepath (str): Ruta del archivo CSV
            
        Returns:
            pd.DataFrame: DataFrame con la señal
        """
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise Exception(f"Error cargando {filepath}: {str(e)}")
    
    def calculate_basic_stats(self, signal_data):
        """
        Calcular estadísticas básicas.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            
        Returns:
            dict: Diccionario con estadísticas
        """
        return {
            'mean': np.mean(signal_data),
            'median': np.median(signal_data),
            'std': np.std(signal_data),
            'var': np.var(signal_data),
            'min': np.min(signal_data),
            'max': np.max(signal_data),
            'peak_to_peak': np.max(signal_data) - np.min(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'skewness': skew(signal_data),
            'kurtosis': kurtosis(signal_data),
            'length': len(signal_data),
            'duration_seconds': len(signal_data) / self.fs
        }
    
    def detect_artifacts(self, signal_data, method='amplitude'):
        """
        Detectar artefactos en la señal.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            method (str): Método de detección ('amplitude', 'zscore', 'iqr')
            
        Returns:
            dict: Información sobre artefactos detectados
        """
        artifacts = {
            'method': method,
            'num_artifacts': 0,
            'artifact_indices': [],
            'artifact_percentage': 0.0,
            'details': {}
        }
        
        if method == 'amplitude':
            # Detectar valores extremos (típicamente > 100 µV en EEG)
            threshold = 100.0  # µV
            artifact_mask = np.abs(signal_data) > threshold
            artifacts['num_artifacts'] = np.sum(artifact_mask)
            artifacts['artifact_indices'] = np.where(artifact_mask)[0].tolist()
            artifacts['threshold'] = threshold
            artifacts['details']['max_value'] = float(np.max(np.abs(signal_data)))
            
        elif method == 'zscore':
            # Z-score: valores > 3 desviaciones estándar
            z_scores = np.abs((signal_data - np.mean(signal_data)) / np.std(signal_data))
            artifact_mask = z_scores > 3
            artifacts['num_artifacts'] = np.sum(artifact_mask)
            artifacts['artifact_indices'] = np.where(artifact_mask)[0].tolist()
            artifacts['threshold'] = 3
            artifacts['details']['max_zscore'] = float(np.max(z_scores))
            
        elif method == 'iqr':
            # IQR: rango intercuartil
            Q1 = np.percentile(signal_data, 25)
            Q3 = np.percentile(signal_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            artifact_mask = (signal_data < lower_bound) | (signal_data > upper_bound)
            artifacts['num_artifacts'] = np.sum(artifact_mask)
            artifacts['artifact_indices'] = np.where(artifact_mask)[0].tolist()
            artifacts['threshold'] = (lower_bound, upper_bound)
            artifacts['details']['outliers_detected'] = artifacts['num_artifacts']
        
        artifacts['artifact_percentage'] = (artifacts['num_artifacts'] / len(signal_data)) * 100
        
        return artifacts
    
    def detect_noise_issues(self, signal_data):
        """
        Detectar problemas de ruido y saturación.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            
        Returns:
            dict: Información sobre problemas de ruido
        """
        issues = {
            'flat_segments': 0,
            'noisy_periods': 0,
            'near_saturation': 0,
            'details': {}
        }
        
        # Detectar segmentos planos (sin variación)
        window_size = int(0.1 * self.fs)  # 100ms
        for i in range(0, len(signal_data) - window_size, window_size):
            window = signal_data[i:i+window_size]
            if np.std(window) < 0.1:  # Muy baja variación
                issues['flat_segments'] += 1
        
        # Detectar períodos ruidosos (alta energía de alta frecuencia)
        # Usar diferencias para detectar cambios abruptos
        diff = np.diff(signal_data)
        high_freq_energy = np.mean(np.abs(diff))
        issues['details']['high_freq_energy'] = float(high_freq_energy)
        if high_freq_energy > np.std(signal_data) * 2:
            issues['noisy_periods'] = 1
        
        # Detectar saturación (valores pegados a límites)
        near_sat = np.sum(np.abs(signal_data) > 90) / len(signal_data)
        issues['near_saturation'] = float(near_sat) * 100
        
        return issues
    
    def calculate_frequency_content(self, signal_data):
        """
        Analizar contenido en frecuencia de la señal.
        
        Args:
            signal_data (np.ndarray): Datos de la señal
            
        Returns:
            dict: Información de frecuencia
        """
        # Calcular PSD (Power Spectral Density)
        freqs, psd = signal.welch(signal_data, fs=self.fs, nperseg=256)
        
        # Bandas de EEG estándar
        bands = {
            'delta': (0.5, 4),      # Sueño profundo
            'theta': (4, 8),        # Meditación, relajación
            'alpha': (8, 12),       # Relajación con ojos cerrados
            'beta': (12, 30),       # Actividad mental
            'gamma': (30, 50)       # Procesamiento cognitivo
        }
        
        freq_content = {
            'peak_frequency': float(freqs[np.argmax(psd)]),
            'peak_power': float(np.max(psd)),
            'total_power': float(np.sum(psd)),
            'bands': {}
        }
        
        for band_name, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs <= f_high)
            band_power = np.sum(psd[mask])
            freq_content['bands'][band_name] = {
                'power': float(band_power),
                'percentage': float(band_power / np.sum(psd) * 100)
            }
        
        return freq_content
    
    def analyze_file(self, filename):
        """
        Analizar un archivo completo.
        
        Args:
            filename (str): Nombre del archivo CSV
            
        Returns:
            dict: Análisis completo del archivo
        """
        filepath = os.path.join(self.data_path, filename)
        
        print(f"Analizando: {filename}")
        
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
        
        analysis = {
            'filename': filename,
            'subject': subject,
            'experiment': experiment,
            'session': session,
            'timestamp': datetime.now().isoformat(),
            'channels': {}
        }
        
        # Analizar cada canal
        for channel in CHANNELS:
            if channel not in df.columns:
                print(f"  ⚠ Canal {channel} no encontrado")
                continue
            
            signal_data = df[channel].values.astype(float)
            
            channel_analysis = {
                'basic_stats': self.calculate_basic_stats(signal_data),
                'artifacts_amplitude': self.detect_artifacts(signal_data, method='amplitude'),
                'artifacts_zscore': self.detect_artifacts(signal_data, method='zscore'),
                'artifacts_iqr': self.detect_artifacts(signal_data, method='iqr'),
                'noise_issues': self.detect_noise_issues(signal_data),
                'frequency_content': self.calculate_frequency_content(signal_data)
            }
            
            analysis['channels'][channel] = channel_analysis
            
            # Detectar problemas
            problems = []
            
            # Problema 1: Muchos artefactos
            if channel_analysis['artifacts_amplitude']['artifact_percentage'] > 5:
                problems.append(f"Muchos artefactos por amplitud ({channel_analysis['artifacts_amplitude']['artifact_percentage']:.2f}%)")
            
            # Problema 2: Segmentos planos
            if channel_analysis['noise_issues']['flat_segments'] > 5:
                problems.append(f"Múltiples segmentos planos ({channel_analysis['noise_issues']['flat_segments']})")
            
            # Problema 3: Ruido alto
            if channel_analysis['noise_issues']['noisy_periods'] == 1:
                problems.append("Señal muy ruidosa")
            
            # Problema 4: Saturación
            if channel_analysis['noise_issues']['near_saturation'] > 10:
                problems.append(f"Saturación detectada ({channel_analysis['noise_issues']['near_saturation']:.2f}%)")
            
            if problems:
                if filename not in self.problems:
                    self.problems[filename] = {}
                self.problems[filename][channel] = problems
        
        self.stats_db[filename] = analysis
        
        return analysis
    
    def analyze_all_files(self):
        """Analizar todos los archivos CSV en el directorio."""
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        print(f"\nEncontrados {len(csv_files)} archivos CSV\n")
        
        for filename in csv_files:
            self.analyze_file(filename)
        
        print(f"\nAnálisis completado: {len(self.stats_db)} archivos procesados\n")
    
    def generate_summary_report(self):
        """Generar reporte resumido."""
        report = {
            'total_files': len(self.stats_db),
            'timestamp': datetime.now().isoformat(),
            'files_with_problems': len(self.problems),
            'summary_by_channel': {},
            'issues': {}
        }
        
        # Resumen por canal
        for channel in CHANNELS:
            channel_stats = []
            for filename, analysis in self.stats_db.items():
                if channel in analysis['channels']:
                    channel_stats.append(analysis['channels'][channel])
            
            if channel_stats:
                all_artifacts = [s['artifacts_amplitude']['artifact_percentage'] for s in channel_stats]
                report['summary_by_channel'][channel] = {
                    'files_analyzed': len(channel_stats),
                    'avg_artifact_percentage': float(np.mean(all_artifacts)),
                    'max_artifact_percentage': float(np.max(all_artifacts)),
                    'min_artifact_percentage': float(np.min(all_artifacts))
                }
        
        # Problemas encontrados
        for filename, channels_problems in self.problems.items():
            report['issues'][filename] = channels_problems
        
        return report
    
    def save_reports(self):
        """Guardar reportes en JSON y CSV."""
        
        # Reporte JSON detallado
        with open(os.path.join(self.output_path, 'detailed_stats.json'), 'w') as f:
            json.dump(self.stats_db, f, indent=2, cls=NumpyEncoder)
        print("✓ Reporte detallado guardado: detailed_stats.json")
        
        # Reporte resumido
        summary = self.generate_summary_report()
        with open(os.path.join(self.output_path, 'summary_report.json'), 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        print("✓ Reporte resumido guardado: summary_report.json")
        
        # Tabla CSV con estadísticas por archivo y canal
        rows = []
        for filename, analysis in self.stats_db.items():
            subject = analysis['subject']
            experiment = analysis['experiment']
            session = analysis['session']
            
            for channel, channel_data in analysis['channels'].items():
                stats = channel_data['basic_stats']
                artifacts = channel_data['artifacts_amplitude']
                noise = channel_data['noise_issues']
                
                rows.append({
                    'filename': filename,
                    'subject': subject,
                    'experiment': experiment,
                    'session': session,
                    'channel': channel,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'rms': stats['rms'],
                    'skewness': stats['skewness'],
                    'kurtosis': stats['kurtosis'],
                    'artifact_percentage': artifacts['artifact_percentage'],
                    'num_artifacts': artifacts['num_artifacts'],
                    'flat_segments': noise['flat_segments'],
                    'saturation_percentage': noise['near_saturation']
                })
        
        df_stats = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_path, 'statistics_table.csv')
        df_stats.to_csv(csv_path, index=False)
        print(f"✓ Tabla de estadísticas guardada: statistics_table.csv ({len(rows)} filas)")
        
        # Archivos problemáticos
        with open(os.path.join(self.output_path, 'problematic_files.json'), 'w') as f:
            json.dump(self.problems, f, indent=2, cls=NumpyEncoder)
        print(f"✓ Archivos problemáticos guardados: {len(self.problems)} archivos con problemas")
    
    def plot_statistics(self):
        """Generar visualizaciones de estadísticas."""
        
        # Recolectar datos
        data_by_channel = {ch: [] for ch in CHANNELS}
        artifacts_by_channel = {ch: [] for ch in CHANNELS}
        
        for filename, analysis in self.stats_db.items():
            for channel in CHANNELS:
                if channel in analysis['channels']:
                    rms = analysis['channels'][channel]['basic_stats']['rms']
                    artifact_pct = analysis['channels'][channel]['artifacts_amplitude']['artifact_percentage']
                    data_by_channel[channel].append(rms)
                    artifacts_by_channel[channel].append(artifact_pct)
        
        # Crear figuras
        fig = plt.figure(figsize=(16, 10))
        
        # Distribución RMS por canal
        ax1 = plt.subplot(2, 3, 1)
        for channel in CHANNELS:
            ax1.hist(data_by_channel[channel], alpha=0.6, label=channel, bins=20)
        ax1.set_xlabel('RMS (µV)')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución RMS por Canal')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot RMS
        ax2 = plt.subplot(2, 3, 2)
        bp_data = [data_by_channel[ch] for ch in CHANNELS]
        ax2.boxplot(bp_data, labels=CHANNELS)
        ax2.set_ylabel('RMS (µV)')
        ax2.set_title('RMS por Canal (Box Plot)')
        ax2.grid(alpha=0.3, axis='y')
        
        # Porcentaje de artefactos
        ax3 = plt.subplot(2, 3, 3)
        for channel in CHANNELS:
            ax3.hist(artifacts_by_channel[channel], alpha=0.6, label=channel, bins=20)
        ax3.set_xlabel('Porcentaje de Artefactos (%)')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Artefactos por Canal')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Promedio RMS por canal
        ax4 = plt.subplot(2, 3, 4)
        avg_rms = [np.mean(data_by_channel[ch]) for ch in CHANNELS]
        bars = ax4.bar(CHANNELS, avg_rms, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax4.set_ylabel('RMS Promedio (µV)')
        ax4.set_title('RMS Promedio por Canal')
        ax4.grid(alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Promedio artefactos por canal
        ax5 = plt.subplot(2, 3, 5)
        avg_artifacts = [np.mean(artifacts_by_channel[ch]) for ch in CHANNELS]
        bars = ax5.bar(CHANNELS, avg_artifacts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax5.set_ylabel('Artefactos Promedio (%)')
        ax5.set_title('Artefactos Promedio por Canal')
        ax5.grid(alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Resumen de problemas
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        summary_text = f"""
        RESUMEN DE ANÁLISIS
        ─────────────────────
        Total de archivos: {len(self.stats_db)}
        Archivos problemáticos: {len(self.problems)}
        
        Artefactos por canal (promedio):
        """
        for channel in CHANNELS:
            if channel in artifacts_by_channel and artifacts_by_channel[channel]:
                avg = np.mean(artifacts_by_channel[channel])
                summary_text += f"\n  {channel}: {avg:.2f}%"
        
        ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_path, 'statistics_plots.png'), dpi=300, bbox_inches='tight')
        print("✓ Gráficos de estadísticas guardados: statistics_plots.png")
        plt.close()
    
    def create_database_table(self):
        """Crear tabla en SQLite para estadísticas."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                subject TEXT,
                experiment TEXT,
                session TEXT,
                channel TEXT NOT NULL,
                mean REAL,
                std REAL,
                min REAL,
                max REAL,
                rms REAL,
                skewness REAL,
                kurtosis REAL,
                artifact_percentage REAL,
                num_artifacts INTEGER,
                flat_segments INTEGER,
                saturation_percentage REAL,
                peak_frequency REAL,
                delta_power REAL,
                theta_power REAL,
                alpha_power REAL,
                beta_power REAL,
                gamma_power REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(filename, channel)
            )
        ''')
        
        # Insertar datos
        for filename, analysis in self.stats_db.items():
            subject = analysis['subject']
            experiment = analysis['experiment']
            session = analysis['session']
            
            for channel, channel_data in analysis['channels'].items():
                stats = channel_data['basic_stats']
                artifacts = channel_data['artifacts_amplitude']
                noise = channel_data['noise_issues']
                freq = channel_data['frequency_content']
                
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO signal_statistics
                        (filename, subject, experiment, session, channel, mean, std, min, max, rms,
                         skewness, kurtosis, artifact_percentage, num_artifacts, flat_segments,
                         saturation_percentage, peak_frequency, delta_power, theta_power, alpha_power,
                         beta_power, gamma_power)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        filename, subject, experiment, session, channel,
                        stats['mean'], stats['std'], stats['min'], stats['max'], stats['rms'],
                        stats['skewness'], stats['kurtosis'],
                        artifacts['artifact_percentage'], artifacts['num_artifacts'],
                        noise['flat_segments'], noise['near_saturation'],
                        freq['peak_frequency'],
                        freq['bands']['delta']['power'],
                        freq['bands']['theta']['power'],
                        freq['bands']['alpha']['power'],
                        freq['bands']['beta']['power'],
                        freq['bands']['gamma']['power']
                    ))
                except Exception as e:
                    print(f"  ⚠ Error insertando {filename}, {channel}: {str(e)}")
        
        conn.commit()
        conn.close()
        print("✓ Datos guardados en base de datos SQLite")


def main():
    """Función principal."""
    print("="*70)
    print("EEG PREPROCESSING - ANÁLISIS ESTADÍSTICO Y DETECCIÓN DE PROBLEMAS")
    print("="*70)
    
    analyzer = EEGStatsAnalyzer()
    
    # Analizar todos los archivos
    analyzer.analyze_all_files()
    
    # Generar reportes
    print("\nGenerando reportes...")
    analyzer.save_reports()
    
    # Crear gráficos
    print("\nGenerando visualizaciones...")
    analyzer.plot_statistics()
    
    # Guardar en base de datos
    print("\nGuardando en base de datos...")
    analyzer.create_database_table()
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)
    print(f"Reportes guardados en: {analyzer.output_path}")


if __name__ == "__main__":
    main()
