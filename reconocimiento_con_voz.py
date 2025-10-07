import cv2
import numpy as np
import tensorflow as tf
import requests
import json
import time
import os  # NUEVO: Para manejar archivos del sistema (borrar el MP3)
from gtts import gTTS  # NUEVO: La librería de Texto a Voz
from playsound import playsound  # NUEVO: La librería para reproducir audio

# --- 1. Cargar modelos ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
try:
    emotion_model = tf.keras.models.load_model('modelo_emociones_final.keras')
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# --- 2. Definir etiquetas y configuración de la API ---
emotion_labels = {0: 'Enojo', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}
OPENROUTER_API_KEY = "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"
YOUR_SITE_URL = "http://localhost:5000"

def obtener_respuesta_llm(emocion):
    # ... (Esta función es exactamente la misma que en la fase anterior) ...
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "HTTP-Referer": f"{YOUR_SITE_URL}", "X-Title": "Reconocimiento de Emociones"},
            json={"model": "google/gemma-2-9b-it:free", "messages": [{"role": "system", "content": "Eres un asistente amigable y empático. Responde con una frase corta, positiva y reconfortante (de no más de 15 palabras) basada en la emoción que siente el usuario. Habla en español."}, {"role": "user", "content": f"La emoción que siento ahora es: {emocion}"}]}
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error de API: {e}")
        return "No pude conectar con la IA."

# --- NUEVO: Función para convertir texto a voz y reproducirlo ---
def reproducir_respuesta(texto):
    """
    Convierte una cadena de texto a voz, la guarda como un MP3 temporal,
    la reproduce y luego elimina el archivo.
    """
    try:
        print("Generando audio...")
        tts = gTTS(text=texto, lang='es', slow=False)
        nombre_archivo = "respuesta_ia.mp3"
        tts.save(nombre_archivo)
        
        print("Reproduciendo respuesta...")
        playsound(nombre_archivo)
        
        # Eliminar el archivo después de reproducirlo
        os.remove(nombre_archivo)
        print("Audio reproducido y archivo temporal eliminado.")

    except Exception as e:
        print(f"Error al reproducir el audio: {e}")

# --- 3. Lógica Principal de Captura de Video ---
cap = cv2.VideoCapture(0)
WINDOW_NAME = 'Asistente de IA con Voz'
last_detected_emotion = None
last_api_call_time = 0
COOLDOWN_SECONDS = 10

while True:
    # ... (El bucle de detección de rostros es el mismo) ...
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    current_emotion = None
    for (x, y, w, h) in faces:
        # ... (código de predicción de emoción) ...
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.asarray(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = emotion_model.predict(roi, verbose=0)
        max_index = np.argmax(prediction[0])
        predicted_emotion = emotion_labels[max_index]
        current_emotion = predicted_emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow(WINDOW_NAME, frame)

    current_time = time.time()
    if current_emotion and current_emotion != last_detected_emotion and (current_time - last_api_call_time > COOLDOWN_SECONDS):
        print(f"Emoción detectada: {current_emotion}. Pidiendo consejo a la IA...")
        respuesta_ia = obtener_respuesta_llm(current_emotion)
        print(f"Respuesta de la IA: {respuesta_ia}")
        
        # --- NUEVO: Llamada a la función para reproducir la respuesta ---
        reproducir_respuesta(respuesta_ia)
        
        print("-" * 30)
        last_detected_emotion = current_emotion
        last_api_call_time = current_time

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()