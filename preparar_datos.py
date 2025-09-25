# Importamos las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Documentación del Código ---
# Este script se encarga de procesar el dataset FER-2013.
# 1. Carga el archivo fer2013.csv usando Pandas.
# 2. Muestra un resumen de las emociones y cuántas imágenes hay por cada una.
# 3. Convierte la columna de píxeles (que es un string) en una matriz de NumPy.
# 4. Redimensiona cada matriz de píxeles a su forma original de 48x48.
# 5. Normaliza los valores de los píxeles (de 0-255 a 0-1) para un mejor entrenamiento.
# 6. Muestra una imagen de ejemplo de cada emoción para verificar que todo está correcto.

print("Iniciando el preprocesamiento de datos...")

# 1. Cargar el dataset con Pandas
try:
    data = pd.read_csv('fer2013.csv')
    print("Dataset cargado correctamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'fer2013.csv'.")
    print("Por favor, asegúrate de haberlo descargado de Kaggle y colocado en la carpeta del proyecto.")
    exit()

# Mapeo de las etiquetas numéricas a emociones
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# 2. Mostrar un resumen de los datos
print("\nResumen de datos por emoción:")
print(data['emotion'].value_counts().rename(index=emotion_map))

# 3. Preparar los datos de las imágenes
# La columna 'pixels' es un string de números separados por espacios.
# Necesitamos convertirla en una matriz de números.
pixels = data['pixels'].tolist() # Convertir la columna a una lista de strings
faces = []

for pixel_sequence in pixels:
    # Dividir el string en números individuales y convertirlos a enteros
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    
    # 4. Redimensionar a 48x48
    face = np.asarray(face).reshape(48, 48)
    faces.append(face)

# Convertir la lista de caras a un array de NumPy
faces = np.asarray(faces)
# Añadir una dimensión para el canal (ya que son imágenes en escala de grises)
faces = np.expand_dims(faces, -1)

# 5. Normalizar los datos
# Los valores de los píxeles van de 0 a 255. Normalizarlos a un rango de 0 a 1
# ayuda a que la red neuronal aprenda de manera más eficiente.
faces = faces.astype('float32') / 255.0

print(f"\nProcesamiento completado. Se han preparado {len(faces)} imágenes.")
print(f"La forma del array de imágenes es: {faces.shape}")

# 6. Visualizar una imagen de ejemplo por cada emoción
print("\nMostrando una imagen de ejemplo para cada categoría...")

plt.figure(figsize=(10, 10))
for emotion_num, emotion_name in emotion_map.items():
    # Encontrar el índice de la primera imagen con esta emoción
    idx = data[data['emotion'] == emotion_num].index[0]
    
    # Crear un subplot para cada emoción
    ax = plt.subplot(3, 3, emotion_num + 1)
    
    # Mostrar la imagen
    # Usamos np.squeeze para quitar la dimensión del canal y poder mostrarla
    plt.imshow(np.squeeze(faces[idx]), cmap='gray')
    
    plt.title(f"{emotion_name} ({emotion_num})")
    plt.axis('off')

plt.suptitle("Ejemplos del Dataset FER-2013")
plt.show()

print("\n¡Fase 2 completada! Tus datos están cargados y listos para ser usados en la Fase 3.")