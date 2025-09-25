import cv2
import numpy as np
import tensorflow as tf

# --- Documentación del Código ---
# Versión final y estable que utiliza TensorFlow para la inferencia.
# Carga directamente el modelo .keras y aplica optimizaciones de fluidez.

# --- 1. Cargar los modelos ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargamos directamente el modelo Keras original guardado desde Colab
try:
    emotion_model = tf.keras.models.load_model('modelo_emociones_final.keras')
except Exception as e:
    print("Error al cargar el modelo 'modelo_emociones.keras'.")
    print("Asegúrate de haberlo descargado de Colab y puesto en esta carpeta.")
    print(f"Error específico: {e}")
    exit()

# --- 2. Definir etiquetas de emociones ---
emotion_labels = {0: 'Enojo', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}

# --- 3. Iniciar captura de video ---
cap = cv2.VideoCapture(0)
WINDOW_NAME = 'Reconocimiento de Emociones (Version Estable)'

frame_count = 0
FRAMES_TO_SKIP = 4
last_known_faces = []
last_known_emotions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Bloque de procesamiento (se ejecuta cada N fotogramas)
    if frame_count % FRAMES_TO_SKIP == 0:
        last_known_faces = []
        last_known_emotions = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float32') / 255.0
            roi = np.asarray(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # La predicción se hace con el modelo de TensorFlow
            prediction = emotion_model.predict(roi, verbose=0)
            
            max_index = np.argmax(prediction[0])
            predicted_emotion = emotion_labels[max_index]

            last_known_faces.append((x, y, w, h))
            last_known_emotions.append(predicted_emotion)

    # Bloque de dibujo (se ejecuta en CADA fotograma)
    for i, (x, y, w, h) in enumerate(last_known_faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, last_known_emotions[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()