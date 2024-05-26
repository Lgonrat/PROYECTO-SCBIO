import os
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import pygame
from PIL import Image, ImageDraw, ImageFont
import logging

# Configurar el registro
logging.basicConfig(level=logging.INFO)

# Inicializamos variables
scoreCPU = 0
scorePLAYER = 0
mode = 3  # El modo para ganar es llegar a 3 puntos
cpu_choices = ["Rock", "Paper", "Scissors"]
cpu_choice = "Nothing"
player_choice = "Nothing"
winner = "None"
fin = False
options_displayed = False  # Inicializar la variable
display_values = ["Rock", "Invalid", "Scissors", "Invalid", "Invalid", "Paper"]

# Cargando MediaPipe
mp_hands = mp.solutions.hands

# Inicializar pygame
pygame.init()
# Cargar la música
pygame.mixer.music.load('MusicaMenu.mp3')
pygame.mixer.music.play(-1)  # Reproducir la música del menú en bucle

# Funciones del juego
def calculate_winner(cpu_choice, player_choice):
    """Determina el ganador de cada ronda y actualiza el puntaje."""
    global scoreCPU, scorePLAYER
    if player_choice == "Invalid":
        return "Invalid!"
    elif player_choice == cpu_choice:
        return "Tie!"
    elif (player_choice == "Rock" and cpu_choice == "Scissors") or \
         (player_choice == "Scissors" and cpu_choice == "Paper") or \
         (player_choice == "Paper" and cpu_choice == "Rock"):
        scorePLAYER += 1
        return "You win!"
    else:
        scoreCPU += 1
        return "CPU wins!"

def compute_fingers(hand_landmarks):
    """Cuenta el número de dedos levantados."""
    count = 0
    if len(hand_landmarks) < 21:
        return count
    if hand_landmarks[8].y < hand_landmarks[6].y:  # Índice
        count += 1
    if hand_landmarks[12].y < hand_landmarks[10].y:  # Corazón
        count += 1
    if hand_landmarks[16].y < hand_landmarks[14].y:  # Anular
        count += 1
    if hand_landmarks[20].y < hand_landmarks[18].y:  # Meñique
        count += 1
    if hand_landmarks[4].x < hand_landmarks[3].x:  # Pulgar
        count += 1
    return count

def calculate_choice(count):
    """Determina la elección del jugador basado en el número de dedos levantados."""
    if count == 0:
        return "Rock"
    elif count == 2:
        return "Scissors"
    elif count == 5:
        return "Paper"
    else:
        return "Invalid"

def reset_game():
    """Reinicia el juego."""
    global scoreCPU, scorePLAYER, fin, start_time
    scoreCPU = 0
    scorePLAYER = 0
    fin = False
    start_time = time.time()  # Reinicia el temporizador

def draw_text_with_pil(frame, text, position, font, text_color=(255, 255, 255), outline_color=(144, 238, 144), outline_width=2):
    """Dibuja texto con un contorno en una imagen usando Pillow."""
    # Convertir la imagen de OpenCV a formato PIL
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    
    x, y = position
    # Dibujar el texto con el contorno
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    
    # Dibujar el texto principal
    draw.text(position, text, font=font, fill=text_color)
    
    # Convertir de nuevo la imagen a formato OpenCV
    frame = np.array(img_pil)
    return frame

# Cargar fuente personalizada
font_path = "FuenteJuego.ttf"  # Ruta a tu fuente Arcade Gamer
font = ImageFont.truetype(font_path, 24)
font_contador = ImageFont.truetype(font_path, 60)

def display_info(frame, final_choice, cpu_choice, winner, elapsed_time, countdown_text):
    """Muestra la información del juego en la ventana."""
    frame = draw_text_with_pil(frame, "PLAYER CHOICE:  " + final_choice, (10, 30), font, text_color=(255, 0, 255),outline_color=(144, 238, 144), outline_width=2)
    frame = draw_text_with_pil(frame, "CPU CHOICE:  " + cpu_choice, (10, 60), font, text_color=(255, 0, 255),outline_color=(144, 238, 144), outline_width=2)
    frame = draw_text_with_pil(frame, "WINNER:  " + winner, (10, 90), font, text_color=(255, 0, 255), outline_color=(144, 238, 144), outline_width=2)
    frame = draw_text_with_pil(frame, "SCORE - PLAYER: " + str(scorePLAYER) + " CPU: " + str(scoreCPU), (10, 120), font, text_color=(255, 0, 255), outline_color=(144, 238, 144), outline_width=2)
    #frame = draw_text_with_pil(frame, "Time left: " + str(round(countdown - elapsed_time, 1)), (10, 150), font, color=(0, 0, 0))
    return frame

def detect_thumbs_up(hand_landmarks):
    """Detecta si el gesto de 'thumbs up' está presente."""
    if len(hand_landmarks) < 21:
        return False
    # El pulgar está hacia arriba si el pulgar está extendido y los otros dedos no lo están
    thumb_up = hand_landmarks[4].y < hand_landmarks[3].y and all(
        hand_landmarks[i].y > hand_landmarks[i - 2].y for i in [8, 12, 16, 20]
    )
    return thumb_up

# Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    logging.error("Error al abrir la cámara.")
    exit()

# Ruta absoluta para las imágenes
presentation_path = os.path.abspath("presentation.jpeg")
reglas1_path = os.path.abspath("Reglas1.jpeg")
reglas2_path = os.path.abspath("Reglas2.jpeg")
rock_path = os.path.abspath("piedra.jpeg")
paper_path = os.path.abspath("papel.jpeg")
scissors_path = os.path.abspath("tijera.jpeg")
nothing_path = os.path.abspath("nothing.jpeg")
win_img_path = os.path.abspath("win_image.jpeg")
lose_img_path = os.path.abspath("lose_image.jpeg")

# Cargar las imágenes
presentation_img = cv2.imread(presentation_path)
reglas1_img = cv2.imread(reglas1_path)
reglas2_img = cv2.imread(reglas2_path)
rock_img = cv2.imread(rock_path)
paper_img = cv2.imread(paper_path)
scissors_img = cv2.imread(scissors_path)
nothing_img = cv2.imread(nothing_path)
win_img = cv2.imread(win_img_path)
lose_img = cv2.imread(lose_img_path)

# Verificar si las imágenes se cargaron correctamente
if presentation_img is None:
    logging.error(f"Error al cargar la imagen {presentation_path}")
    exit()
if reglas1_img is None:
    logging.error(f"Error al cargar la imagen {reglas1_path}")
    exit()
if reglas2_img is None:
    logging.error(f"Error al cargar la imagen {reglas2_path}")
    exit()
if rock_img is None:
    logging.error(f"Error al cargar la imagen {rock_path}")
    exit()
if paper_img is None:
    logging.error(f"Error al cargar la imagen {paper_path}")
    exit()
if scissors_img is None:
    logging.error(f"Error al cargar la imagen {scissors_path}")
    exit()
if nothing_img is None:
    logging.error(f"Error al cargar la imagen {nothing_path}")
    exit()
if win_img is None:
    logging.error(f"Error al cargar la imagen {win_img_path}")
    exit()
if lose_img is None:
    logging.error(f"Error al cargar la imagen {lose_img_path}")
    exit()

# Diccionario para mapear la elección de la CPU a la imagen correspondiente
cpu_images = {
    "Rock": rock_img,
    "Paper": paper_img,
    "Scissors": scissors_img,
    "Nothing": nothing_img
}

# Función para manejar los clics del ratón
def check_click(event, x, y, flags, param):
    global current_image, game_started, scoreCPU, scorePLAYER, countdown_finished
    
    # Definimos las coordenadas del área de clic
    click_area_x_start = [587, 812, 589, 808]  # Coordenadas en X
    click_area_y_start = [652, 650, 731, 728]  # Coordenadas en Y
    
    click_area_x_reglas = [1098, 1341, 1096, 1345]  # Coordenadas en X
    click_area_y_reglas = [643, 647, 730, 734]  # Coordenadas en Y
    
    click_area_x_reglas1 = [106, 600, 108, 600]  # Coordenadas en X
    click_area_y_reglas1 = [1000, 1000, 800, 800]  # Coordenadas en Y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_image == "presentation":
            if min(click_area_x_start) <= x <= max(click_area_x_start) and min(click_area_y_start) <= y <= max(click_area_y_start):  # Coordenadas para Start
                pygame.mixer.music.load('MusicaJuego.mp3')
                pygame.mixer.music.play(-1)  # -1 indica que se reproduzca en bucle, 0 indica que se reproduzca una vez
                
                for img_path in ["Cuenta3.jpeg", "Cuenta2.jpeg", "Cuenta1.jpeg"]:
                    countdown_img = cv2.imread(os.path.abspath(img_path))
                    if countdown_img is not None:
                        countdown_img = cv2.resize(countdown_img, (frame.shape[1], frame.shape[0]))
                        cv2.imshow('Webcam', countdown_img)
                        cv2.waitKey(1000)  # Esperar 1 segundo antes de mostrar la siguiente imagen
                
                countdown_finished = True
                game_started = True  # Ahora el juego ha comenzado

            elif min(click_area_x_reglas) <= x <= max(click_area_x_reglas) and min(click_area_y_reglas) <= y <= max(click_area_y_reglas):  # Coordenadas para Menu (Reglas)
                current_image = "reglas1"
        
        elif current_image == "reglas1":
            if min(click_area_x_reglas1) <= x <= max(click_area_x_reglas1) and min(click_area_y_reglas1) <= y <= max(click_area_y_reglas1):  # Coordenadas para la siguiente página
                current_image = "reglas2"
        
        elif current_image == "reglas2":
            if min(click_area_x_reglas1) <= x <= max(click_area_x_reglas1) and min(click_area_y_reglas1) <= y <= max(click_area_y_reglas1):  # Coordenadas para volver a la presentación
                current_image = "presentation"


# Función para mostrar las imágenes de ganar y perder
def display_result_images(frame, winner):
    """Muestra la imagen correspondiente al resultado del juego."""
    if winner == "You win!":
        result_image = cv2.imread("win_image.jpeg")
        pygame.mixer.music.stop()
        pygame.mixer.music.load('MusicaWin.mp3')
        pygame.mixer.music.play(-1)
        #pygame.quit()
        if result_image is not None:
            # Escalar la imagen para que coincida con el tamaño del fotograma
            result_image = cv2.resize(result_image, (frame.shape[1], frame.shape[0]))
            # Mostrar la imagen
            frame = result_image
    elif winner == "CPU wins!":
        lose_img = cv2.imread("lose_image.jpeg")
        pygame.mixer.music.stop()
        pygame.mixer.music.load('MusicaLoss.mp3')
        pygame.mixer.music.play(-1)
        #pygame.quit()
        if lose_img is not None:
            # Escalar la imagen para que coincida con el tamaño del fotograma
            lose_img = cv2.resize(lose_img, (frame.shape[1], frame.shape[0]))
            # Mostrar la imagen
            frame = lose_img
    return frame
    
    if result_image is not None:
        # Escalar la imagen para que coincida con el tamaño del fotograma
        result_image = cv2.resize(result_image, (frame.shape[1], frame.shape[0]))
        # Superponer la imagen del resultado
        frame = cv2.addWeighted(frame, 1, result_image, 0.5, 0)
    return frame

# Crear la ventana en pantalla completa
cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('Webcam', check_click)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    current_image = "presentation"

    game_started = False
    start_time = time.time()
    countdown = 4  # Duración del temporizador en segundos
    countdown_started = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        if game_started:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0].landmark
                count = compute_fingers(hand_landmarks)
                player_choice = calculate_choice(count)
            else:
                player_choice = "Nothing"

            elapsed_time = time.time() - start_time

            if not countdown_started:
                countdown_started = True
                start_time = time.time()

            countdown_text = ""
            # Al comienzo del script, define una variable para controlar el tiempo de visualización de las opciones
            OPTIONS_DISPLAY_TIME = 3  # Duración en segundos

            # En el bucle principal del juego...
            if elapsed_time < countdown:
                countdown_text = str(int(countdown - elapsed_time))
                final_choice = " "
                cpu_choice = " "
                winner = " "
                options_displayed = False  # Variable para rastrear si las opciones están siendo mostradas
            else:
                if elapsed_time > countdown + OPTIONS_DISPLAY_TIME:
                    # Cuando el tiempo de visualización haya pasado, comienza la cuenta regresiva nuevamente
                    final_choice = " "
                    cpu_choice = " "
                    winner = " "
                    options_displayed = False  # Reiniciar el estado de visualización de las opciones
                    start_time = time.time()
                    countdown_started = False
                elif elapsed_time > countdown:
                    # Cuando el contador llega a cero
                    if not options_displayed:
                        # Mostrar las opciones solo una vez cuando el contador llega a cero
                        final_choice = player_choice
                        cpu_choice = random.choice(cpu_choices)
                        winner = calculate_winner(cpu_choice, final_choice)
                        options_displayed = True  # Marcar que las opciones están siendo mostradas
                    else:
                        # Si las opciones ya están siendo mostradas, dejarlas en pantalla
                        final_choice = player_choice
                        cpu_choice = cpu_choice
                        winner = winner

            frame = display_info(frame, player_choice, cpu_choice, winner, elapsed_time, countdown_text)

            if cpu_choice in cpu_images:
                if countdown_finished:
                    scoreCPU = 0
                    scorePLAYER = 0
                    countdown_finished = False
                cpu_choice_img = cpu_images[cpu_choice]
            else:
                cpu_choice_img = nothing_img

            frame_height, frame_width = frame.shape[:2]
            cpu_choice_img = cv2.resize(cpu_choice_img, (frame_width, frame_height))
            combined_frame = cv2.hconcat([frame, cpu_choice_img])
            combined_height, combined_width = combined_frame.shape[:2]

            #Calcular la posicion del contador en el medio de las dos pantallas
            countdown_position = (combined_width // 2 - 28, combined_height // 2 - 50)
            combined_frame = draw_text_with_pil(combined_frame, countdown_text, countdown_position, font_contador, text_color=(255, 0, 255))

            if scoreCPU == mode or scorePLAYER == mode:
                winner_message = "CPU wins the game!" if scoreCPU == mode else "Player wins the game!"
                fin = True

            if fin:
                # Mostrar la partida de puntos antes de mostrar la imagen de ganar/perder
                cv2.imshow('Webcam', combined_frame)
                cv2.waitKey(3000)  # Mostrar la partida de puntos por 3 segundos
                # Mostrar la imagen de ganar o perder
                combined_frame = display_result_images(combined_frame, winner)
                cv2.imshow('Webcam', combined_frame)
                cv2.waitKey(10000)
                
                thumbs_up_detected = False
                thumbs_down_detected = False
                start_thumb_time = time.time()

                while time.time() - start_thumb_time < 10:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmark in results.multi_hand_landmarks:
                            hand_landmarks = hand_landmark.landmark
                            if detect_thumbs_up(hand_landmarks):
                                thumbs_up_detected = True
                                break
                            elif compute_fingers(hand_landmarks) == 5:  # Detectar si se muestran los cinco dedos
                                thumbs_down_detected = True
                                break
                    
                    if thumbs_up_detected:
                        # Reiniciar el juego
                        reset_game()
                        current_image = "presentation" 
                        game_started = False
                        # Detener la música
                        pygame.mixer.music.stop()
                        pygame.mixer.music.load('MusicaMenu.mp3')
                        pygame.mixer.music.play(-1)  # Reproducir la música del menú en bucle
                        break
                    
                    elif thumbs_down_detected:
                        # Cerrar el juego
                        break
                    
                    combined_frame = display_result_images(frame, winner)
                    cv2.imshow('Webcam', combined_frame)
                    
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q') or key == 27:
                        break
                if thumbs_up_detected:
                    reset_game()
                    current_image = "presentation"
                    game_started = False
                else:
                    break

            cv2.imshow('Webcam', combined_frame)
        else:
            if current_image == "presentation":
                cv2.imshow('Webcam', presentation_img)
            elif current_image == "reglas1":
                cv2.imshow('Webcam', reglas1_img)
            elif current_image == "reglas2":
                cv2.imshow('Webcam', reglas2_img)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

cap.release()
cv2.destroyAllWindows()