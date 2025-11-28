# Proyecto de Reconocimiento de Emociones Faciales en Tiempo Real 🤖

Este proyecto es una aplicación desarrollada en Python que utiliza una Red Neuronal para detectar rostros en tiempo real a través de una cámara web y clasificar su expresión facial en una de siete emociones: **enojo, disgusto, miedo, felicidad, tristeza, sorpresa y neutral**.

El modelo de IA ha sido entrenado desde cero utilizando el dataset FER-2013 y aplicando técnicas avanzadas para mejorar la precisión en emociones difíciles de distinguir.

## ✨ Características Principales

* **Detección de Rostros en Tiempo Real:** Utiliza OpenCV para localizar rostros en el feed de la cámara.
* **Clasificación de 7 Emociones:** Implementa un modelo de Keras/TensorFlow para identificar la expresión facial.
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
robot-emocional-lab-vr/
├── venv/                     # Entorno virtual de Python
├── .gitignore                # Archivos ignorados por Git
├── modelo_emociones_final.keras # Modelo entrenado y listo para usar
├── entrenar_modelo_final.py  # Script para entrenar el modelo (optimizado para Colab)
├── reconocimiento_con_api_local.py # Script principal para ejecutar la aplicación
└── README.md                 # La documentación del proyecto
```

## ⚙️ Configuración e Instalación

Sigue estos pasos para poner en marcha el proyecto en tu máquina local.

**1. Clona el Repositorio**
```bash
git clone https://github.com/monboga/robot-emocional-lab-vr.git
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
* **Modelo Pre-entrenado:** Asegúrate de tener el archivo `modelo_emociones_final.keras` en la raíz del proyecto que se encuentra en el siguiente drive: [Drive de Descarga](https://drive.google.com/drive/folders/1dvm5o8bP28coVf3IU2nVYKm89ZIn70l5?usp=sharing).

## 🚀 Uso del Proyecto

### Detección en Tiempo Real

Para ejecutar la aplicación principal, simplemente corre el siguiente comando en tu terminal (con el entorno virtual activado):
```bash
python reconocimiento_con_api_local.py
```
Presiona la tecla **'q'** para cerrar la aplicación.

### Re-entrenar el Modelo

El entrenamiento es un proceso que consume muchos recursos. Se recomienda encarecidamente realizarlo en **Google Colab** utilizando una GPU.
1.  Sube el archivo `entrenar_modelo_final.py` y el dataset `fer2013.csv` a tu Google Drive.
2.  Abre el script en un cuaderno de Colab.
3.  Activa el acelerador por hardware (GPU).
4.  Ejecuta el cuaderno para entrenar y guardar un nuevo archivo `.keras`.

## 🔮 Posibles Mejoras
* Remover la GUI (Interfaz Gráfica de Usuario) del robot emocional, para solamente ver y manejar los estados de la aplicación a través de logs del sistema por consola.
* Ver la posibilidad de generar un script ya sea en python o en bash para que la aplicación pueda iniciar nada más al encender el raspberry pi. 


