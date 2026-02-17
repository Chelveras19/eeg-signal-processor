# EEG Signal Processor GUI

Una aplicación de escritorio para analizar y visualizar señales EEG (Electroencefalograma) con almacenamiento en base de datos SQLite.

## Características

- **Visualización de Espectrogramas**: Análisis frecuencial de señales EEG
- **Análisis MFCC**: Extracción de coeficientes cepstrales en escala de Mel
- **Matriz de Distancia Coseno**: Cálculo de similitud entre ventanas temporales
- **Base de Datos**: Almacenamiento de análisis en SQLite con historial completo
- **Interfaz Gráfica**: GUI intuitiva desarrollada con Tkinter
- **Procesamiento Multihilo**: Análisis sin bloquear la interfaz
- **Parámetros Ajustables**: Control de ventanas, traslape y canales EEG

## Requisitos

- Python 3.7+
- Las dependencias están listadas en `requirements.txt`

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/eeg-signal-processor.git
cd eeg-signal-processor
```

2. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Ejecuta la aplicación:
```bash
python signal_processor_gui.py
```

### Pasos:

1. **Selecciona un archivo CSV** con datos EEG
2. **Elige el canal EEG** a analizar (T7, F8, Cz, P4)
3. **Configura los parámetros**:
   - Tipo de ventana (hann, hamming, blackman, boxcar)
   - Porcentaje de traslape (60-95%)
4. **Haz clic en "Generar Análisis"**
5. Visualiza los resultados en la interfaz

## Archivos de Salida

Las imágenes generadas se guardan en:
- `Imagenes/espectrogramas/` - Espectrogramas
- `Imagenes/mfcc/` - Análisis MFCC
- `Imagenes/distancia_coseno/` - Matrices de distancia coseno

Los análisis se registran en `eeg_database.db`

## Estructura del Proyecto

```
eeg-signal-processor/
├── signal_processor_gui.py    # Aplicación principal
├── requirements.txt           # Dependencias Python
├── .gitignore                # Archivos ignorados por Git
├── README.md                 # Este archivo
├── Imagenes/                 # Salida de gráficas
│   ├── espectrogramas/
│   ├── mfcc/
│   └── distancia_coseno/
└── eeg_database.db           # Base de datos SQLite
```

## Clase Principal

### SpectrogramApp
Aplicación GUI con las siguientes funcionalidades:
- Carga de archivos CSV
- Selección de canales EEG
- Generación de análisis (espectrograma, MFCC, distancia coseno)
- Almacenamiento en base de datos
- Visualización en tiempo real

### EEGProcessor
Procesador de señales con métodos para:
- `load_signal()` - Cargar datos de archivo CSV
- `get_spectrogram()` - Calcular espectrograma
- `get_mfcc()` - Calcular coeficientes MFCC
- `get_cosine_distance_matrix()` - Matriz de distancia coseno
- `get_signal_stats()` - Estadísticas de la señal

### DatabaseManager
Gestor de conexión a SQLite para:
- Crear tablas de análisis
- Guardar resultados de procesamiento
- Consultar análisis previos

## Constantes Configurables

```python
FS = 200           # Frecuencia de muestreo (Hz)
NPERSEG = 256      # Tamaño de ventana
N_MFCC = 13        # Número de coeficientes MFCC
CHANNELS = ["T7", "F8", "Cz", "P4"]  # Canales disponibles
```

## Requerimientos de Base de Datos

La tabla `analyses` almacena:
- Archivo CSV de origen
- Canal EEG analizado
- Tipo de ventana utilizada
- Porcentaje de traslape
- Rutas de las imágenes generadas
- Timestamp del análisis

## Licencia

[Especifica tu licencia aquí - ej: MIT, GPL, Apache 2.0]

## Autor

Desarrollado como parte del proyecto de Servicio Social.

## Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama con tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Notas Técnicas

- **Threading**: El procesamiento se ejecuta en un thread separado para no bloquear la GUI
- **Matplotlib**: Se utiliza el backend Tkinter para integración con la interfaz
- **Librosa**: Procesamiento de audio/señales con MFCC
- **SciPy**: Cálculo de espectrogramas y matrices de distancia
