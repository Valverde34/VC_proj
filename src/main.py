# Algoritmo principal de contagem de flexoes
import os
import cv2
import numpy as np
import mediapipe as mp

# escolher webcam ou vídeo

# SOURCE = "webcam"  # Para usar a webcam
SOURCE = "video"   # Para usar vídeo

# Definir caminho:
VIDEO_PATH = r"C:\VC_proj\src\Copy of push up 165.mp4"

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

# Counter variables
counter = 0
stage = None
view = "unknown"
form_status = "Unknown"
warning_message = ""
in_pushup_position = False


# Calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def check_pushup_position(landmarks):
    """
    Verifica se a pessoa está realmente em posição de flexão.
    """
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_wrist_y = (left_wrist.y + right_wrist.y) / 2
        avg_hip_y = (left_hip.y + right_hip.y) / 2
        avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
        
        wrists_below_shoulders = avg_wrist_y > avg_shoulder_y + 0.05
        body_horizontal = abs(avg_shoulder_y - avg_hip_y) < 0.2
        feet_extended = abs(avg_hip_y - avg_ankle_y) < 0.15
        not_standing = abs(avg_shoulder_y - avg_ankle_y) < 0.4
        
        return wrists_below_shoulders and body_horizontal and feet_extended and not_standing
        
    except:
        return False


def draw_ui(image, counter, stage, in_pushup_position, form_status, elbow_angle=0, source="VIDEO"):
    h, w = image.shape[:2] 
    
    # Painel lateral direito
    panel_width = 400
    overlay = image.copy()
    cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
    # Linha separadora
    cv2.line(image, (w - panel_width, 0), (w - panel_width, h), (100, 100, 100), 3)
    
    y_offset = 60
    x_margin = w - panel_width + 30
    
    # Título
    cv2.putText(image, "PUSHUP TRACKER", (x_margin, y_offset), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)
    cv2.line(image, (x_margin, y_offset + 10), (w - 30, y_offset + 10), (0, 255, 0), 2)
    
    y_offset += 80
    
    # CONTADOR - DESTAQUE
    cv2.rectangle(image, (x_margin - 15, y_offset - 50), (w - 30, y_offset + 30), (0, 100, 255), -1)
    cv2.putText(image, "REPS", (x_margin, y_offset - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, str(counter), (x_margin, y_offset + 25), 
                cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 4)
    
    y_offset += 100
    
    # STAGE
    stage_text = stage if stage else "N/A"
    stage_color = (0, 255, 0) if stage == "up" else (0, 165, 255) if stage == "down" else (200, 200, 200)
    cv2.putText(image, "STAGE:", (x_margin, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(image, stage_text.upper(), (x_margin, y_offset + 35), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, stage_color, 3)
    
    y_offset += 100
    
    # POSITION STATUS - Removido o ícone +/X confuso
    position_text = "IN POSITION" if in_pushup_position else "NOT IN POSITION"
    position_color = (0, 255, 0) if in_pushup_position else (0, 0, 255)
    
    cv2.rectangle(image, (x_margin - 15, y_offset - 10), (w - 30, y_offset + 50), 
                  position_color, 3)
    cv2.putText(image, position_text, (x_margin + 10, y_offset + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, position_color, 2)
    
    y_offset += 90
    
    # ELBOW ANGLE (se em posição) - CORRIGIDO o símbolo de grau
    if in_pushup_position and elbow_angle > 0:
        cv2.putText(image, "ELBOW ANGLE:", (x_margin, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        angle_color = (0, 255, 0) if 90 <= elbow_angle <= 170 else (0, 165, 255)
        # Removido o símbolo º que causava problemas de encoding
        cv2.putText(image, f"{int(elbow_angle)} deg", (x_margin, y_offset + 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, angle_color, 3)
        
        # Barra de progresso do ângulo
        bar_width = 300
        bar_x = x_margin
        bar_y = y_offset + 60
        progress = max(0, min(1, (elbow_angle - 80) / 100))
        
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (100, 100, 100), -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), 
                      angle_color, -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (200, 200, 200), 2)
        
        y_offset += 100
    
    # FORM STATUS
    y_offset += 20
    form_color = (0, 255, 0) if form_status == "Good form" else (255, 255, 255) if "Not in" in form_status else (0, 165, 255)
    cv2.putText(image, "FORM:", (x_margin, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Quebrar texto longo em múltiplas linhas
    words = form_status.split()
    line = ""
    line_y = y_offset + 35
    for word in words:
        if len(line + word) < 18:
            line += word + " "
        else:
            cv2.putText(image, line, (x_margin, line_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, form_color, 2)
            line = word + " "
            line_y += 30
    if line:
        cv2.putText(image, line, (x_margin, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, form_color, 2)
    
    # Footer com controles
    footer_y = h - 80
    cv2.rectangle(image, (w - panel_width, footer_y - 20), (w, h), (30, 30, 30), -1)
    cv2.putText(image, "CONTROLS:", (x_margin, footer_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(image, "Q: Quit  |  P: Pause  |  R: Reset", (x_margin, footer_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Badge de fonte no canto superior esquerdo
    cv2.rectangle(image, (10, 10), (180, 50), (60, 60, 60), -1)
    cv2.putText(image, f"SOURCE: {source}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    
# ====== INICIALIZAR CAPTURA ======
if SOURCE == "webcam":
    print("[INFO] Usando WEBCAM")
    cap = cv2.VideoCapture(0)
    is_video = False
    delay = 1
    
elif SOURCE == "video":
    print(f"[INFO] Usando VÍDEO: {VIDEO_PATH}")
    
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"Vídeo não encontrado em: {VIDEO_PATH}")
    
    cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Falha ao abrir o vídeo: {VIDEO_PATH}")
    
    is_video = True
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000 / fps))
    print(f"[INFO] FPS do vídeo: {fps}")
    
else:
    raise ValueError("SOURCE deve ser 'webcam' ou 'video'")

if not cap.isOpened():
    raise RuntimeError("Erro ao abrir a fonte de vídeo")

print("[INFO] Pressione 'q' para sair, 'p' para pausar/continuar, 'r' para reset")
if is_video:
    print("[INFO] O vídeo vai reiniciar automaticamente quando terminar")

# Criar janela em fullscreen
cv2.namedWindow("Pushup Counter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Pushup Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, image = cap.read()
    
    if not ret or image is None:
        if is_video:
            print("[INFO] Fim do vídeo. Recomeçando...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            print("[ERRO] Não foi possível ler da webcam")
            break

    # Redimensionar para tela cheia mantendo proporção
    screen_res = 1920, 1080
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    
    image = cv2.resize(image, (window_width, window_height), interpolation=cv2.INTER_AREA)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    elbow_angle = 0
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        in_pushup_position = check_pushup_position(landmarks)

        left_vis = (
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility +
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility +
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility
        ) / 3

        right_vis = (
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility +
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility +
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility
        ) / 3

        if left_vis > 0.6 and right_vis > 0.6:
            view = "front"
        elif left_vis > 0.6:
            view = "side_left"
        elif right_vis > 0.6:
            view = "side_right"
        else:
            view = "unknown"

        good_form = False
        if view == "front":
            left_angle = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            )
            right_angle = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            )
            if abs(left_angle - 180) < 30 and abs(right_angle - 180) < 30:
                good_form = True
        else:
            prefix = "LEFT" if view == "side_left" else "RIGHT"
            sh = mp_pose.PoseLandmark[f"{prefix}_SHOULDER"].value
            hp = mp_pose.PoseLandmark[f"{prefix}_HIP"].value
            an = mp_pose.PoseLandmark[f"{prefix}_ANKLE"].value
            angle = calculate_angle(
                [landmarks[sh].x, landmarks[sh].y],
                [landmarks[hp].x, landmarks[hp].y],
                [landmarks[an].x, landmarks[an].y]
            )
            if abs(angle - 180) < 30:
                good_form = True

        if not in_pushup_position:
            form_status = "Not in pushup position"
        elif good_form:
            form_status = "Good form"
        else:
            form_status = "Bad - Straighten body"

        if in_pushup_position:
            l_sh, l_el, l_wr = (mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                                mp_pose.PoseLandmark.LEFT_WRIST.value)
            r_sh, r_el, r_wr = (mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                                mp_pose.PoseLandmark.RIGHT_WRIST.value)

            left_elbow_angle = calculate_angle(
                [landmarks[l_sh].x, landmarks[l_sh].y],
                [landmarks[l_el].x, landmarks[l_el].y],
                [landmarks[l_wr].x, landmarks[l_wr].y],
            )
            right_elbow_angle = calculate_angle(
                [landmarks[r_sh].x, landmarks[r_sh].y],
                [landmarks[r_el].x, landmarks[r_el].y],
                [landmarks[r_wr].x, landmarks[r_wr].y],
            )

            use_right = right_vis >= left_vis
            elbow_angle = right_elbow_angle if use_right else left_elbow_angle

            DOWN_ANGLE = 95
            UP_ANGLE = 160

            if elbow_angle <= DOWN_ANGLE:
                stage = "down"
            if elbow_angle >= UP_ANGLE and stage == "down":
                stage = "up"
                if good_form:
                    counter += 1
        else:
            stage = None

        # Desenhar landmarks com estilo melhorado
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2)
        )

    # Desenhar UI
    source_text = "WEBCAM" if SOURCE == "webcam" else "VIDEO"
    draw_ui(image, counter, stage, in_pushup_position, form_status, elbow_angle, source_text)

    cv2.imshow("Pushup Counter", image)
    
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print("[INFO] Pausado. Pressione 'p' novamente para continuar...")
        while True:
            if cv2.waitKey(30) & 0xFF == ord('p'):
                print("[INFO] Continuando...")
                break
    elif key == ord('r'):
        counter = 0
        stage = None
        print("[INFO] Contador resetado!")

cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\n[INFO] Sessão finalizada!")
print(f"[INFO] Total de repetições: {counter}")