import cv2
import numpy as np
import tensorflow as tf
import requests
import json
import time
import os
from gtts import gTTS
from playsound import playsound
import threading  # NUEVO: Importamos la librería de hilos

# --- 1. Cargar modelos y configuraciones (sin cambios) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = tf.keras.models.load_model('modelo_emociones_final.keras')
emotion_labels = {0: 'Enojo', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}
OPENROUTER_API_KEY = "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"
YOUR_SITE_URL = "http://localhost:5000"

respuestas_cache = {}

def limpiar_respuesta_ia(texto_crudo):
    """
    Elimina los tokens de control y otros artefactos de la respuesta del LLM.
    """
    # Lista de tokens de control a eliminar. Puedes añadir más si descubres otros.
    tokens_a_eliminar = ["<s>", "</s>", "[INST]", "[/INST]", "[ASSISTANT]", "[OUT]", "[/OUT]"]
    texto_limpio = texto_crudo
    for token in tokens_a_eliminar:
        texto_limpio = texto_limpio.replace(token, "")
    
    # Eliminar espacios en blanco al principio y al final
    return texto_limpio.strip()

def obtener_respuesta_llm(emocion):
    # ... (Esta función es la misma, no necesita cambios) ...
    try:
        # 1. Definimos un prompt de sistema mucho más estricto con ejemplos
        # Un prompt de sistema mucho más corto y directo
        system_prompt = """
        Tu rol es Aura, un asistente empático. Tu única tarea es generar una respuesta en español, corta (10-20 palabras), que sea reconfortante y se relacione directamente con la emoción del usuario. No uses saludos. 
        Ejemplo para 'Triste': "Veo que algo te pesa. Recuerda que todos los momentos pasan."
        Ahora, responde a la siguiente emoción:
        """

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "HTTP-Referer": f"{YOUR_SITE_URL}", "X-Title": "Reconocimiento de Emociones"},
            json={
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [
                    # Prompt del sistema más estricto
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"La emoción que siento ahora es: {emocion}"}
                ],
                "temperature": 0.7, # Controla la creatividad (más bajo = más predecible)
                "max_tokens": 50    # Límite estricto de longitud de la respuesta
            }
        )
        response.raise_for_status()
        data = response.json()

        respuesta_cruda = data['choices'][0]['message']['content']
        respuesta_limpia = limpiar_respuesta_ia(respuesta_cruda)

        return respuesta_limpia
    except Exception as e:
        print(f"Error de API: {e}")
        if e.response is not None:
            print(f"Detalles del error del servidor: {e.response.text}")
        return "No pude conectar con la IA."

def reproducir_respuesta(texto):
    # ... (Esta función es la misma, no necesita cambios) ...
    try:
        print("Generando audio en segundo plano...")
        tts = gTTS(text=texto, lang='es', slow=False)
        nombre_archivo = "respuesta_ia.mp3"
        tts.save(nombre_archivo)
        print("Reproduciendo respuesta...")
        playsound(nombre_archivo)
        os.remove(nombre_archivo)
        print("Audio reproducido y archivo temporal eliminado.")
    except Exception as e:
        print(f"Error al reproducir el audio: {e}")

# --- NUEVO: Función que agrupa las tareas lentas para el hilo secundario ---
def manejar_interaccion_ia(emocion):
    print(f"Emoción detectada: {emocion}.")
    
    # 1. Revisar si ya tenemos una respuesta guardada en nuestra "libreta"
    if emocion in respuestas_cache:
        print("Respuesta encontrada en el caché.")
        respuesta_ia = respuestas_cache[emocion]
    else:
        # 2. Si no la tenemos, vamos a la "biblioteca" (API)
        print("Respuesta no encontrada en caché. Llamando a la API...")
        respuesta_ia = obtener_respuesta_llm(emocion)
        
        # 3. Y anotamos la nueva respuesta en nuestra libreta para el futuro
        if "No pude conectar" not in respuesta_ia:
             respuestas_cache[emocion] = respuesta_ia

    print(f"Respuesta de la IA: {respuesta_ia}")
    reproducir_respuesta(respuesta_ia)
    print("-" * 30)

# --- 3. Lógica Principal de Captura de Video ---
cap = cv2.VideoCapture(0)
WINDOW_NAME = 'Asistente de IA Fluido'
last_detected_emotion = None
last_api_call_time = 0
COOLDOWN_SECONDS = 30 
hilo_ia = None

# Variable para controlar el hilo de la IA
hilo_ia = None

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # bucle de detección de rostros
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
    
    # Lógica para lanzar el hilo secundario ---
    # Verificamos las mismas condiciones de antes, pero también nos aseguramos
    # de que no haya ya un "repartidor" (hilo) trabajando.
    if current_emotion and current_emotion != last_detected_emotion and \
       (current_time - last_api_call_time > COOLDOWN_SECONDS) and \
       (hilo_ia is None or not hilo_ia.is_alive()):
        
        last_detected_emotion = current_emotion
        last_api_call_time = current_time
        
        # Creamos y lanzamos el hilo secundario
        hilo_ia = threading.Thread(target=manejar_interaccion_ia, args=(current_emotion,))
        hilo_ia.daemon = True  # Hilo demonio para que no bloquee la salida del programa
        hilo_ia.start()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()