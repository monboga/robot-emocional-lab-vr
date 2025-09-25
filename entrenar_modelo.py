import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# --- Documentación del Código ---
# Este script construye y entrena una Red Neuronal Convolucional (CNN)
# para clasificar las emociones faciales del dataset FER-2013.
#
# 1. Carga los datos desde 'fer2013.csv' usando la lógica de la Fase 2.
# 2. Procesa y normaliza los píxeles de las imágenes.
# 3. Divide los datos en conjuntos de entrenamiento y prueba según la columna 'Usage'.
# 4. Convierte las etiquetas de emoción a un formato categórico (one-hot encoding).
# 5. Define la arquitectura de la CNN con varias capas.
# 6. Compila el modelo, especificando el optimizador y la función de pérdida.
# 7. Entrena el modelo con los datos de entrenamiento y lo valida con los de prueba.
# 8. Guarda el modelo entrenado en un archivo para su uso posterior.

# --- 1. Cargar y procesar los datos ---
print("Cargando y procesando datos...")
try:
    data = pd.read_csv('fer2013.csv')
except FileNotFoundError:
    print("Error: No se encontró 'fer2013.csv'. Descárgalo de Kaggle y ponlo en esta carpeta.")
    exit()

pixels = data['pixels'].tolist()
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(48, 48)
    faces.append(face)

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)
faces = faces.astype('float32') / 255.0

emotions = data['emotion'].values

# 4. Convertir etiquetas a formato one-hot encoding
# Ejemplo: la etiqueta '3' (Happy) se convierte en [0, 0, 0, 1, 0, 0, 0]
emotions_categorical = to_categorical(emotions, num_classes=7)

# --- 3. Dividir datos en entrenamiento y prueba ---
# El dataset ya tiene una columna 'Usage' para esto.
X_train, y_train = faces[data['Usage'] == 'Training'], emotions_categorical[data['Usage'] == 'Training']
X_test, y_test = faces[data['Usage'] == 'PublicTest'], emotions_categorical[data['Usage'] == 'PublicTest']

print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
print(f"Datos de prueba: {X_test.shape[0]} muestras")

# --- 5. Definir la arquitectura del modelo CNN ---
print("Construyendo el modelo de la Red Neuronal Convolucional...")
model = Sequential([
    # Bloque 1
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Bloque 2
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Bloque 3
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Aplanar los mapas de características para las capas densas
    Flatten(),

    # Capa Densa
    Dense(256, activation='relu'),
    Dropout(0.5),

    # Capa de Salida
    # 7 neuronas, una para cada emoción. Softmax convierte la salida en probabilidades.
    Dense(7, activation='softmax')
])

# --- 6. Compilar el modelo ---
# 'adam' es un buen optimizador para empezar.
# 'categorical_crossentropy' es la función de pérdida para clasificación multiclase.
# 'accuracy' es la métrica que queremos monitorear.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define el callback
# monitor='val_loss': Vigila la pérdida en el set de validación.
# patience=5: Si no hay mejora en 5 épocas consecutivas, detiene el entrenamiento.
# restore_best_weights=True: Se asegura de que el modelo final tenga los mejores pesos que encontró, no los de la última época.
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.summary() # Muestra un resumen de la arquitectura del modelo

# --- 7. Entrenar el modelo ---
print("\nIniciando el entrenamiento del modelo...")
# epochs: Número de veces que el modelo verá todo el dataset.
# batch_size: Número de imágenes a procesar antes de actualizar el modelo.
# validation_data: Datos que no se usan para entrenar, solo para evaluar el rendimiento.
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,  # Puedes empezar con menos épocas (ej. 20) si tarda mucho
    validation_data=(X_test, y_test),
    callbacks=[early_stopping] # ¡Aquí está la magia!
)

# --- 8. Guardar el modelo entrenado ---
# Guardamos el modelo para poder usarlo en la Fase 4 sin tener que reentrenar.
model_save_path = 'modelo_emociones.keras'
model.save(model_save_path)

print(f"\n¡Entrenamiento completado! El modelo ha sido guardado en '{model_save_path}'")