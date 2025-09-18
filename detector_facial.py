# Importamos la librería OpenCV
import cv2

# --- Documentación del Código ---
# Este script utiliza la cámara web para detectar rostros en tiempo real.
# Versión mejorada que permite cerrar la ventana con el botón 'X'.
#
# 1. Carga un clasificador pre-entrenado de OpenCV para detectar rostros.
# 2. Inicia la captura de video desde la cámara web predeterminada.
# 3. En un bucle, lee cada fotograma de la cámara.
# 4. Convierte el fotograma a escala de grises para mejorar la detección.
# 5. Utiliza el clasificador para encontrar las coordenadas de los rostros.
# 6. Dibuja un rectángulo verde alrededor de cada rostro detectado.
# 7. Muestra el resultado en una ventana.
# 8. El programa se detiene al presionar la tecla 'q' o al cerrar la ventana.

# Cargamos el clasificador pre-entrenado para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciamos la captura de video. El '0' indica que es la cámara web predeterminada.
cap = cv2.VideoCapture(0)

# Damos un nombre a nuestra ventana para poder referirnos a ella
WINDOW_NAME = 'Detector de Rostros'

# Iniciamos un bucle infinito para procesar cada fotograma del video
while True:
    # Leemos un fotograma de la cámara. 'ret' es un booleano que indica si la lectura fue exitosa.
    ret, frame = cap.read()
    if not ret:
        break

    # La detección de rostros es más efectiva en imágenes en escala de grises.
    # Convertimos el fotograma de color (BGR) a gris.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Usamos el clasificador para detectar rostros en la imagen en gris.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Recorremos la lista de rostros encontrados.
    # Cada 'rostro' es una tupla con (x, y, w, h)
    for (x, y, w, h) in faces:
        # Dibujamos un rectángulo en el fotograma original (a color).
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostramos el fotograma con los rostros detectados en nuestra ventana con nombre.
    cv2.imshow(WINDOW_NAME, frame)

    # Revisamos dos condiciones para salir del bucle:
    
    # 1. Si se presiona la tecla 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Cerrando por teclado...")
        break

    # 2. Si la propiedad de la ventana indica que ya no es visible (se presionó la 'X')
    # getWindowProperty devuelve un valor < 1 si la ventana ya no existe.
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        print("Cerrando desde la ventana...")
        break

# Cuando salimos del bucle, liberamos la cámara y cerramos todas las ventanas.
cap.release()
cv2.destroyAllWindows()
print("Aplicación finalizada correctamente.")