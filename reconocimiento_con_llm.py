import cv2
import numpy as np
import tensorflow as tf
import requests # NUEVO: Para hacer llamadas a la API
import json     # NUEVO: Para manejar los datos de la API
import time     # NUEVO: Para controlar la frecuencia de las llamadas

# --- 1. Cargar los modelos (igual que antes) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
try:
    # Asegúrate de usar tu mejor modelo entrenado
    emotion_model = tf.keras.models.load_model('modelo_emociones_final.keras')
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# --- 2. Definir etiquetas y configuración de la API ---
emotion_labels = {0: 'Enojo', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}

# --- NUEVO: Configuración de OpenRouter ---
OPENROUTER_API_KEY = "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" # ¡IMPORTANTE!
YOUR_SITE_URL = "http://localhost:5000"  # Puede ser cualquier URL, es para el header

# --- NUEVO: Función para obtener respuesta del LLM ---
def obtener_respuesta_llm(emocion):
    """
    Envía la emoción detectada a un LLM a través de OpenRouter y devuelve su respuesta.
    """
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": f"{YOUR_SITE_URL}", 
                "X-Title": "Reconocimiento de Emociones", 
            },
            data=json.dumps({
                "model": "google/gemma-2-9b-it:free", # Un excelente modelo gratuito en OpenRouter
                "messages": [
                    {"role": "system", "content": "Eres un asistente amigable y empático. Responde con una frase corta, positiva y reconfortante (de no más de 15 palabras) basada en la emoción que siente el usuario. Habla en español."},
                    {"role": "user", "content": f"La emoción que siento ahora es: {emocion}"}
                ]
            })
        )
        response.raise_for_status() # Lanza un error si la petición falló
        
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error de API: {e}")
        return "No pude conectar con la IA."
    except (KeyError, IndexError) as e:
        print(f"Error al procesar la respuesta de la API: {e}")
        return "Hubo un problema con la respuesta de la IA."

# --- 3. Lógica Principal de Captura de Video ---
cap = cv2.VideoCapture(0)
WINDOW_NAME = 'Reconocimiento Facial con Asistente de IA'

# --- NUEVO: Variables para controlar la frecuencia de llamadas al LLM ---
last_detected_emotion = None
last_api_call_time = 0
COOLDOWN_SECONDS = 10 # Esperar 10 segundos entre llamadas

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_emotion = None
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.asarray(roi)
        roi = np.expand_dims(roi, axis=0)
        
        prediction = emotion_model.predict(roi, verbose=0)
        max_index = np.argmax(prediction[0])
        predicted_emotion = emotion_labels[max_index]
        current_emotion = predicted_emotion # Guardamos la emoción actual

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    # --- NUEVO: Lógica para llamar al LLM de forma controlada ---
    # Solo llamamos a la API si hay una cara detectada, si la emoción es diferente a la última
    # y si han pasado más de 10 segundos desde la última llamada.
    current_time = time.time()
    if current_emotion and current_emotion != last_detected_emotion and (current_time - last_api_call_time > COOLDOWN_SECONDS):
        print(f"Emoción detectada: {current_emotion}. Pidiendo consejo a la IA...")
        
        # Llamamos a nuestra nueva función
        respuesta_ia = obtener_respuesta_llm(current_emotion)
        
        print(f"Respuesta de la IA: {respuesta_ia}")
        print("-" * 30)

        # Actualizamos nuestras variables de control
        last_detected_emotion = current_emotion
        last_api_call_time = current_time

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()