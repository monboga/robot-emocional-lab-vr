import cv2
import numpy as np
import tensorflow as tf
import requests # Ya lo tenemos, pero ahora lo usaremos diferente
import time
import os
from gtts import gTTS
from playsound import playsound
import threading

# --- 1. Cargar modelos (sin cambios) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = tf.keras.models.load_model('modelo_emociones_final.keras')
emotion_labels = {0: 'Enojo', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}

# --- MODIFICADO: La función para llamar a tu API local ---
def obtener_respuesta_llm(emocion):
    """
    Envía la emoción detectada a la API local de Spring Boot que se conecta con Ollama.
    """
    # 1. Definimos la URL de tu servicio local
    api_url = "http://localhost:8080/api/v1/generate"

    # 2. Construimos un prompt único y completo para Llama3
    #    Incluimos todas las instrucciones en una sola cadena de texto.
    prompt_completo = f"""
    Eres un asistente empático llamado Aura. Tu única tarea es generar una respuesta en español, corta (10-20 palabras), que sea reconfortante y se relacione con la emoción humana detectada. No uses saludos.
    Ejemplo para 'Triste': "Veo que algo te pesa. Recuerda que todos los momentos pasan."
    Ahora, responde a la siguiente emoción detectada: {emocion}
    """
    
    try:
        # 3. Hacemos una petición GET con el prompt como parámetro
        response = requests.get(api_url, params={"promptMessage": prompt_completo})
        response.raise_for_status() # Lanza un error si la petición falló (ej. 404, 500)
        
        # 4. Tu API devuelve texto plano, así que lo limpiamos y lo retornamos
        return limpiar_respuesta_ia(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con la API local: {e}")
        return "No pude conectar con mi servicio de IA local."

def limpiar_respuesta_ia(texto_crudo):
    # ... (esta función de limpieza sigue siendo útil y no cambia) ...
    tokens_a_eliminar = ["<s>", "</s>", "[INST]", "[/INST]", "[ASSISTANT]", "[OUT]", "[/OUT]", "*", '"']
    texto_limpio = texto_crudo
    for token in tokens_a_eliminar:
        texto_limpio = texto_limpio.replace(token, "")
    return texto_limpio.strip()

# --- El resto del código (reproducir_respuesta, manejar_interaccion_ia, bucle principal) no necesita cambios ---
def reproducir_respuesta(texto):
    # ... (código sin cambios) ...
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
def manejar_interaccion_ia(emocion):
    # ... (código sin cambios) ...
    print(f"Emoción detectada: {emocion}. Pidiendo consejo a la IA local...")
    respuesta_ia = obtener_respuesta_llm(emocion)
    print(f"Respuesta de la IA: {respuesta_ia}")
    reproducir_respuesta(respuesta_ia)
    print("-" * 30)
cap = cv2.VideoCapture(0)
WINDOW_NAME = 'Asistente de IA Local'
last_detected_emotion = None
last_api_call_time = 0
COOLDOWN_SECONDS = 30 
hilo_ia = None
while True:
    # ... (código del bucle while sin cambios) ...
    ret, frame = cap.read()
    if not ret: break
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
        current_emotion = predicted_emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow(WINDOW_NAME, frame)
    current_time = time.time()
    if current_emotion and current_emotion != last_detected_emotion and \
       (current_time - last_api_call_time > COOLDOWN_SECONDS) and \
       (hilo_ia is None or not hilo_ia.is_alive()):
        last_detected_emotion = current_emotion
        last_api_call_time = current_time
        hilo_ia = threading.Thread(target=manejar_interaccion_ia, args=(current_emotion,))
        hilo_ia.daemon = True
        hilo_ia.start()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()