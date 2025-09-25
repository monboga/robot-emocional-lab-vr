# Proyecto de Reconocimiento de Emociones Faciales en Tiempo Real 🤖

Este proyecto es una aplicación desarrollada en Python que utiliza una Red Neuronal para detectar rostros en tiempo real a través de una cámara web y clasificar su expresión facial en una de siete emociones: **enojo, disgusto, miedo, felicidad, tristeza, sorpresa y neutral**.

El modelo de IA ha sido entrenado desde cero utilizando el dataset FER-2013 y aplicando técnicas avanzadas para mejorar la precisión en emociones difíciles de distinguir.

## ✨ Características Principales

* **Detección de Rostros en Tiempo Real:** Utiliza OpenCV para localizar rostros en el feed de la cámara.
* **Clasificación de 7 Emociones:** Implementa un modelo de Keras/TensorFlow para identificar la expresión facial.
* **Modelo Optimizado:** Entrenado con técnicas avanzadas para mejorar la precisión:
    * **Aumento de Datos (Data Augmentation):** Para crear más ejemplos de entrenamiento.
    * **Ponderación de Clases (Class Weights):** Para combatir el desbalance del dataset.
    * **Normalización por Lotes (Batch Normalization):** Para un entrenamiento más rápido y estable.
    * **Tasa de Aprendizaje Adaptativa:** Para un ajuste fino del modelo.
* **Interfaz Visual Simple:** Muestra el resultado directamente en la ventana de video.

## 🛠️ Tecnologías Utilizadas

* **Python 3.10+**
* **TensorFlow / Keras:** Para la construcción y entrenamiento de la red neuronal.
* **OpenCV:** Para la captura de video y detección de rostros.
* **NumPy y Pandas:** Para la manipulación de datos.
* **Scikit-learn:** Para el cálculo de la ponderación de clases.
* **Dataset:** [FER-2013 de Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

## 📁 Estructura del Proyecto

```
ROBOT-EMOCIONAL-LAB-VR/
├── venv/                     # Entorno virtual de Python
├── .gitignore                # Archivos ignorados por Git
├── modelo_emociones_final.keras # Modelo entrenado y listo para usar
├── fer2013.csv               # Dataset (debe descargarse por separado)
├── entrenar_modelo_final.py  # Script para entrenar el modelo (optimizado para Colab)
├── reconocimiento_en_vivo.py # Script principal para ejecutar la aplicación
└── README.md                 # La documentación del proyecto
```

## ⚙️ Configuración e Instalación

Sigue estos pasos para poner en marcha el proyecto en tu máquina local.

**1. Clona el Repositorio**
```bash
git clone [git@github.com:monboga/robot-emocional-lab-vr.git](git@github.com:monboga/robot-emocional-lab-vr.git)
cd ROBOT-EMOCIONAL-LAB-VR
```

**2. Crea y Activa el Entorno Virtual**
```bash
# Crear el entorno
python -m venv venv

# Activar en Windows
venv\Scripts\activate.ps1

# Activar en Mac/Linux
source venv/bin/activate
```

**3. Instala las Dependencias**
Crea un archivo `requirements.txt` con el siguiente comando y luego instala las librerías.
```bash
# Este comando crea el archivo (hazlo una sola vez)
pip freeze > requirements.txt

# Instala las librerías desde el archivo
pip install -r requirements.txt
```

**4. Descarga los Archivos Necesarios**
* **Dataset:** Descarga el archivo `fer2013.csv` desde [este enlace de Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) y colócalo en la raíz del proyecto.
* **Modelo Pre-entrenado:** Asegúrate de tener el archivo `modelo_emociones_final.keras` en la raíz del proyecto.

## 🚀 Uso del Proyecto

### Detección en Tiempo Real

Para ejecutar la aplicación principal, simplemente corre el siguiente comando en tu terminal (con el entorno virtual activado):
```bash
python reconocimiento_en_vivo.py
```
Presiona la tecla **'q'** para cerrar la aplicación.

### Re-entrenar el Modelo

El entrenamiento es un proceso que consume muchos recursos. Se recomienda encarecidamente realizarlo en **Google Colab** utilizando una GPU.
1.  Sube el archivo `entrenar_modelo_final.py` y el dataset `fer2013.csv` a tu Google Drive.
2.  Abre el script en un cuaderno de Colab.
3.  Activa el acelerador por hardware (GPU).
4.  Ejecuta el cuaderno para entrenar y guardar un nuevo archivo `.keras`.

## 🧠 Proceso de Entrenamiento del Modelo

El modelo es una **Red Neuronal** diseñada para la clasificación de imágenes. Para lograr una mayor precisión en emociones poco representadas en el dataset (como "miedo" o "enojo"), se implementaron las siguientes estrategias durante el entrenamiento:

* **Ponderación de Clases:** Se asignó un "peso" mayor a las clases con menos imágenes para que el modelo les prestara más atención durante el aprendizaje.
* **Aumento de Datos:** Se generaron imágenes sintéticas con variaciones (rotación, zoom, etc.) para aumentar la diversidad del dataset.
* **Batch Normalization:** Se incluyeron capas de normalización para acelerar la convergencia y estabilizar el entrenamiento.
* **ReduceLROnPlateau:** Se utilizó un callback para ajustar dinámicamente la tasa de aprendizaje, permitiendo un ajuste más fino en las etapas finales.

## 🔮 Mejoras Futuras

* **Usar un detector de rostros más moderno** como MediaPipe para mayor velocidad y precisión.
* **Entrenar con un dataset de mayor calidad** como AffectNet o CK+ para mejorar aún más el reconocimiento de emociones sutiles.
* **Optimizar la inferencia con ONNX Runtime** para reducir el consumo de CPU (una vez que las dependencias de `tf2onnx` se estabilicen con las versiones más recientes de TensorFlow).