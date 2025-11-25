"""
Contador de Lunges usando MediaPipe Pose
- Detecção robusta de ambas as pernas
- Funciona de qualquer ângulo (frente, lado, costas)
- Thresholds adaptativos
- Análise de qualidade da forma
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# ===== CONFIGURAÇÃO =====
SOURCE = "video"  # "webcam" ou "video"
VIDEO_PATH = r"C:\VC_proj\src\v_Lunges_g05_c02.mp4"

# ===== INICIALIZAÇÃO MEDIAPIPE =====
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

# ===== VARIÁVEIS DE CONTAGEM =====
counter = 0
stage = None  # "up" ou "down"
current_leg = None  # "left" ou "right"
form_status = "Unknown"
in_position = False

# ===== THRESHOLDS ADAPTATIVOS =====
USE_ADAPTIVE_THRESHOLDS = True
KNEE_ANGLE_DOWN_THRESHOLD = 90   # Joelho flexionado (quanto menor, mais profundo)
KNEE_ANGLE_UP_THRESHOLD = 160    # Joelho estendido
HIP_ANGLE_DOWN_THRESHOLD = 100   # Quadril flexionado
HIP_ANGLE_UP_THRESHOLD = 160     # Quadril estendido

# Estado de calibração
calib_active = USE_ADAPTIVE_THRESHOLDS
calib_frames = 0
CALIB_MIN_FRAMES = 60
knee_angle_min = float('inf')
knee_angle_max = float('-inf')
hip_angle_min = float('inf')
hip_angle_max = float('-inf')

# ===== ANÁLISE DE QUALIDADE =====
good_reps = 0
incomplete_reps = 0
last_rep_feedback = ""
rep_min_knee_angle = None
rep_max_knee_angle = None
rep_min_hip_angle = None

# Tracking de qual perna está trabalhando
left_knee_angle_history = deque(maxlen=10)
right_knee_angle_history = deque(maxlen=10)


def calculate_angle(a, b, c):
    """Calcula ângulo entre 3 pontos"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_distance(point1, point2):
    """Distância euclidiana entre dois pontos"""
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def detect_camera_angle(landmarks):
    """
    Detecta se câmera está de frente, lado ou costas
    Retorna: "front", "side", "back"
    """
    L = mp_pose.PoseLandmark
    
    left_shoulder = landmarks[L.LEFT_SHOULDER.value]
    right_shoulder = landmarks[L.RIGHT_SHOULDER.value]
    left_hip = landmarks[L.LEFT_HIP.value]
    right_hip = landmarks[L.RIGHT_HIP.value]
    
    # Distância horizontal entre ombros e quadris
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    hip_width = abs(left_hip.x - right_hip.x)
    
    # Visibilidade média de pontos frontais vs laterais
    left_vis = np.mean([landmarks[L.LEFT_SHOULDER.value].visibility,
                        landmarks[L.LEFT_HIP.value].visibility,
                        landmarks[L.LEFT_KNEE.value].visibility])
    
    right_vis = np.mean([landmarks[L.RIGHT_SHOULDER.value].visibility,
                         landmarks[L.RIGHT_HIP.value].visibility,
                         landmarks[L.RIGHT_KNEE.value].visibility])
    
    # Vista lateral: largura pequena
    if shoulder_width < 0.15 and hip_width < 0.15:
        return "side"
    
    # Vista frontal/traseira: ambos os lados visíveis
    if left_vis > 0.5 and right_vis > 0.5:
        # Verificar se é frente ou costas pela posição do nariz/olhos
        nose = landmarks[L.NOSE.value]
        nose_vis = nose.visibility
        if nose_vis > 0.6:
            return "front"
        else:
            return "back"
    
    return "side"


def check_lunge_position(landmarks):
    """
    Verifica se está em posição válida para lunge
    Deve detectar corpo completo com pernas visíveis
    """
    try:
        L = mp_pose.PoseLandmark
        
        # Pontos essenciais para lunge
        required_points = [
            L.LEFT_HIP, L.RIGHT_HIP,
            L.LEFT_KNEE, L.RIGHT_KNEE,
            L.LEFT_ANKLE, L.RIGHT_ANKLE,
            L.LEFT_SHOULDER, L.RIGHT_SHOULDER
        ]
        
        # Verificar visibilidade individual - MAIS PERMISSIVO
        visible_count = 0
        for p in required_points:
            if landmarks[p.value].visibility > 0.5:  # era 0.6
                visible_count += 1
        
        # Pelo menos 6 dos 8 pontos devem estar visíveis
        if visible_count < 6:
            return False
        
        # Verificar se pernas estão no frame - MAIS PERMISSIVO
        left_knee = landmarks[L.LEFT_KNEE.value]
        right_knee = landmarks[L.RIGHT_KNEE.value]
        left_ankle = landmarks[L.LEFT_ANKLE.value]
        right_ankle = landmarks[L.RIGHT_ANKLE.value]
        
        # Pelo menos 3 dos 4 pontos devem estar no frame
        points_in_frame = 0
        for point in [left_knee, right_knee, left_ankle, right_ankle]:
            if 0.05 < point.y < 0.98:  # era 0.1 a 0.95 - MAIS MARGEM
                points_in_frame += 1
        
        if points_in_frame < 3:
            return False
        
        # Verificar altura mínima - MAIS PERMISSIVO
        left_hip = landmarks[L.LEFT_HIP.value]
        right_hip = landmarks[L.RIGHT_HIP.value]
        
        # Usar o quadril mais visível
        if left_hip.visibility > right_hip.visibility:
            hip_y = left_hip.y
            ankle_y = left_ankle.y if left_ankle.visibility > 0.5 else right_ankle.y
        else:
            hip_y = right_hip.y
            ankle_y = right_ankle.y if right_ankle.visibility > 0.5 else left_ankle.y
        
        vertical_span = abs(hip_y - ankle_y)
        if vertical_span < 0.2:  # era 0.25 - MAIS PERMISSIVO
            return False
        
        return True
        
    except Exception as e:
        print(f"[DEBUG] Erro em check_lunge_position: {e}")
        return False


def determine_working_leg(left_knee_angle, right_knee_angle, left_knee_hist, right_knee_hist):
    """
    Determina qual perna está trabalhando (fazendo lunge)
    A perna que trabalha tem maior variação de ângulo
    """
    # Calcular variação de cada perna
    if len(left_knee_hist) >= 5 and len(right_knee_hist) >= 5:
        left_variance = np.std(left_knee_hist)
        right_variance = np.std(right_knee_hist)
        
        # A perna com maior variação está trabalhando
        if left_variance > right_variance + 5:
            return "left"
        elif right_variance > left_variance + 5:
            return "right"
    
    # Se não houver histórico suficiente, usar perna mais flexionada
    if left_knee_angle < right_knee_angle - 10:
        return "left"
    elif right_knee_angle < left_knee_angle - 10:
        return "right"
    
    return None


def draw_ui(image, counter, stage, in_position, form_status,
            knee_angle=0, hip_angle=0, current_leg=None, camera_angle="front",
            source="VIDEO", thresholds=(90, 160), calib_active=False,
            good_reps=0, incomplete_reps=0, feedback_list=None, last_rep_feedback=""):
    """Desenha interface de usuário com métricas"""
    
    if feedback_list is None:
        feedback_list = []
    h, w = image.shape[:2]
    
    # Painel lateral
    panel_width = 400
    overlay = image.copy()
    cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.line(image, (w - panel_width, 0), (w - panel_width, h), (100, 100, 100), 3)
    
    y_offset = 60
    x_margin = w - panel_width + 30
    
    # Título
    cv2.putText(image, "LUNGE COUNTER", (x_margin, y_offset), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 3)
    cv2.line(image, (x_margin, y_offset + 10), (w - 30, y_offset + 10), (0,255,0), 2)
    
    # Ângulo da câmera
    y_offset += 40
    cam_colors = {"front": (0,255,0), "side": (0,165,255), "back": (255,100,0)}
    cam_color = cam_colors.get(camera_angle, (200,200,200))
    cv2.putText(image, f"VIEW: {camera_angle.upper()}", (x_margin, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cam_color, 2)
    
    # Status adaptativo
    y_offset += 35
    adapt_text = "ADAPTIVE: ON" if USE_ADAPTIVE_THRESHOLDS else "ADAPTIVE: OFF"
    adapt_color = (0,255,0) if USE_ADAPTIVE_THRESHOLDS else (0,0,255)
    cv2.putText(image, adapt_text, (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, adapt_color, 2)
    
    if USE_ADAPTIVE_THRESHOLDS:
        y_offset += 28
        calib_text = "CALIBRATING..." if calib_active else "CALIBRATION: OK"
        calib_color = (0,165,255) if calib_active else (0,255,0)
        cv2.putText(image, calib_text, (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, calib_color, 2)
    
    # Contador
    y_offset += 80
    cv2.rectangle(image, (x_margin - 15, y_offset - 50), (w - 30, y_offset + 30), (0,100,255), -1)
    cv2.putText(image, "REPS", (x_margin, y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(image, str(counter), (x_margin, y_offset + 25), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255,255,255), 4)
    
    # Perna atual
    y_offset += 70
    if current_leg:
        leg_color = (0,255,255)
        cv2.putText(image, "WORKING LEG:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(image, current_leg.upper(), (x_margin, y_offset + 35), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, leg_color, 3)
        y_offset += 70
    
    # Estado
    stage_text = stage if stage else "N/A"
    stage_color = (0,255,0) if stage == "down" else (0,165,255) if stage == "up" else (200,200,200)
    cv2.putText(image, "STAGE:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(image, stage_text.upper(), (x_margin, y_offset + 35), cv2.FONT_HERSHEY_DUPLEX, 1.2, stage_color, 3)
    
    # Thresholds
    y_offset += 80
    thr_down, thr_up = thresholds
    cv2.putText(image, "KNEE THRESHOLDS:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(image, f"DOWN {int(thr_down)} | UP {int(thr_up)}", (x_margin, y_offset + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Qualidade
    y_offset += 70
    cv2.putText(image, "QUALITY:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(image, f"GOOD {good_reps} | INCOMP {incomplete_reps}", (x_margin, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0) if incomplete_reps==0 else (0,165,255), 2)
    
    # Status de posição
    y_offset += 70
    position_text = "IN POSITION" if in_position else "NOT IN POSITION"
    position_color = (0,255,0) if in_position else (0,0,255)
    cv2.rectangle(image, (x_margin - 15, y_offset - 10), (w - 30, y_offset + 50), position_color, 3)
    cv2.putText(image, position_text, (x_margin + 10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, position_color, 2)
    
    # Ângulos com barras de progresso
    y_offset += 80
    if in_position and knee_angle > 0:
        cv2.putText(image, "KNEE ANGLE:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        angle_color = (0,255,0) if knee_angle < 110 else (0,165,255)
        cv2.putText(image, f"{int(knee_angle)} deg", (x_margin, y_offset + 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, angle_color, 3)
        
        bar_width = 300
        bar_x = x_margin
        bar_y = y_offset + 60
        progress = max(0, min(1, (180 - knee_angle) / 180))
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (100,100,100), -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), angle_color, -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (200,200,200), 2)
        y_offset += 100
        
        # Hip angle
        cv2.putText(image, "HIP ANGLE:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        hip_color = (0,255,0) if hip_angle < 120 else (0,165,255)
        cv2.putText(image, f"{int(hip_angle)} deg", (x_margin, y_offset + 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, hip_color, 3)
        y_offset += 60
    
    # Análise de forma
    y_offset += 20
    form_color = (0,255,0) if form_status == "Good form" else (255,255,255) if "Not in" in form_status else (0,165,255)
    cv2.putText(image, "FORM:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    
    words = form_status.split()
    line = ""
    line_y = y_offset + 35
    for word in words:
        if len(line + word) < 18:
            line += word + " "
        else:
            cv2.putText(image, line, (x_margin, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, form_color, 2)
            line = word + " "
            line_y += 30
    if line:
        cv2.putText(image, line, (x_margin, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, form_color, 2)
    
    # Feedback
    fb_y = line_y + 50
    cv2.putText(image, "FEEDBACK:", (x_margin, fb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)
    fb_y += 25
    
    if not feedback_list and last_rep_feedback:
        feedback_list = [last_rep_feedback]
    if not feedback_list:
        cv2.putText(image, "OK", (x_margin, fb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        for fb in feedback_list[:4]:
            color = (0,165,255) if "INCOMPLETE" in fb else (0,0,255) if any(x in fb for x in ["KNEE", "HIP", "DEPTH"]) else (0,255,255)
            cv2.putText(image, fb, (x_margin, fb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            fb_y += 25
    
    if last_rep_feedback:
        cv2.putText(image, f"LAST: {last_rep_feedback}", (x_margin, fb_y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    
    # Controles
    footer_y = h - 80
    cv2.rectangle(image, (w - panel_width, footer_y - 20), (w, h), (30,30,30), -1)
    cv2.putText(image, "CONTROLS:", (x_margin, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    cv2.putText(image, "Q Quit | P Pause | R Reset | C Recalib", (x_margin, footer_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,200,200), 1)
    
    # Indicador de fonte
    cv2.rectangle(image, (10,10), (180,50), (60,60,60), -1)
    cv2.putText(image, f"SOURCE: {source}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


# ===== INICIALIZAÇÃO DE CAPTURA =====
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

print("[INFO] Pressione 'q' para sair, 'p' para pausar, 'r' para reset, 'c' para recalibrar")

cv2.namedWindow("Lunge Counter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Lunge Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Buffers para suavização
knee_angle_buffer = deque(maxlen=5)
hip_angle_buffer = deque(maxlen=5)

# ===== LOOP PRINCIPAL =====
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

    # Redimensionar
    screen_res = 1920, 1080
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    
    image = cv2.resize(image, (window_width, window_height), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    knee_angle = 0
    hip_angle = 0
    feedback_list = []
    camera_angle = "front"
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        in_position = check_lunge_position(landmarks)
        camera_angle = detect_camera_angle(landmarks)
        
        # DEBUG: Mostrar visibilidade dos pontos
        L = mp_pose.PoseLandmark
        vis_avg = np.mean([
            landmarks[L.LEFT_KNEE.value].visibility,
            landmarks[L.RIGHT_KNEE.value].visibility,
            landmarks[L.LEFT_HIP.value].visibility,
            landmarks[L.RIGHT_HIP.value].visibility
        ])
        print(f"[DEBUG] Visibilidade média: {vis_avg:.2f} | In Position: {in_position}")
        
        # Calcular ângulos de ambas as pernas
        left_hip = landmarks[L.LEFT_HIP.value]
        left_knee = landmarks[L.LEFT_KNEE.value]
        left_ankle = landmarks[L.LEFT_ANKLE.value]
        left_shoulder = landmarks[L.LEFT_SHOULDER.value]
        
        right_hip = landmarks[L.RIGHT_HIP.value]
        right_knee = landmarks[L.RIGHT_KNEE.value]
        right_ankle = landmarks[L.RIGHT_ANKLE.value]
        right_shoulder = landmarks[L.RIGHT_SHOULDER.value]
        
        # Ângulos dos joelhos
        left_knee_angle = calculate_angle(
            [left_hip.x, left_hip.y],
            [left_knee.x, left_knee.y],
            [left_ankle.x, left_ankle.y]
        )
        
        right_knee_angle = calculate_angle(
            [right_hip.x, right_hip.y],
            [right_knee.x, right_knee.y],
            [right_ankle.x, right_ankle.y]
        )
        
        # Ângulos dos quadris
        left_hip_angle = calculate_angle(
            [left_shoulder.x, left_shoulder.y],
            [left_hip.x, left_hip.y],
            [left_knee.x, left_knee.y]
        )
        
        right_hip_angle = calculate_angle(
            [right_shoulder.x, right_shoulder.y],
            [right_hip.x, right_hip.y],
            [right_knee.x, right_knee.y]
        )
        
        # Adicionar ao histórico
        left_knee_angle_history.append(left_knee_angle)
        right_knee_angle_history.append(right_knee_angle)
        
        # Determinar qual perna está trabalhando
        working_leg = determine_working_leg(left_knee_angle, right_knee_angle, 
                                           left_knee_angle_history, right_knee_angle_history)
        
        if working_leg:
            current_leg = working_leg
            
            # Usar ângulos da perna que trabalha
            if current_leg == "left":
                raw_knee_angle = left_knee_angle
                raw_hip_angle = left_hip_angle
            else:
                raw_knee_angle = right_knee_angle
                raw_hip_angle = right_hip_angle
            
            knee_angle_buffer.append(raw_knee_angle)
            hip_angle_buffer.append(raw_hip_angle)
            
            knee_angle = np.mean(knee_angle_buffer)
            hip_angle = np.mean(hip_angle_buffer)

        # Análise de forma
        if not in_position:
            form_status = "Not in position"
        elif stage == "down" and knee_angle > 120:
            form_status = "Go deeper"
            feedback_list.append("INCREASE DEPTH")
        elif stage == "down" and hip_angle > 140:
            form_status = "Bend hip more"
            feedback_list.append("BEND HIP")
        else:
            form_status = "Good form"

        if in_position and current_leg:
            # Tracking de métricas
            if rep_min_knee_angle is None:
                rep_min_knee_angle = knee_angle
                rep_max_knee_angle = knee_angle
                rep_min_hip_angle = hip_angle
            else:
                rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)
                rep_max_knee_angle = max(rep_max_knee_angle, knee_angle)
                rep_min_hip_angle = min(rep_min_hip_angle, hip_angle)

            # Calibração adaptativa
            if USE_ADAPTIVE_THRESHOLDS and calib_active:
                calib_frames += 1
                knee_angle_min = min(knee_angle_min, knee_angle)
                knee_angle_max = max(knee_angle_max, knee_angle)
                hip_angle_min = min(hip_angle_min, hip_angle)
                hip_angle_max = max(hip_angle_max, hip_angle)

                if calib_frames >= CALIB_MIN_FRAMES:
                    if (knee_angle_max - knee_angle_min) >= 40:
                        KNEE_ANGLE_DOWN_THRESHOLD = knee_angle_min + 15
                        KNEE_ANGLE_UP_THRESHOLD = knee_angle_max - 10
                        
                        if (hip_angle_max - hip_angle_min) >= 30:
                            HIP_ANGLE_DOWN_THRESHOLD = hip_angle_min + 15
                            HIP_ANGLE_UP_THRESHOLD = hip_angle_max - 10
                        
                        calib_active = False
                        print(f"[INFO] Calibração concluída:")
                        print(f"       KNEE_DOWN={KNEE_ANGLE_DOWN_THRESHOLD:.1f} | KNEE_UP={KNEE_ANGLE_UP_THRESHOLD:.1f}")
                        print(f"       HIP_DOWN={HIP_ANGLE_DOWN_THRESHOLD:.1f} | HIP_UP={HIP_ANGLE_UP_THRESHOLD:.1f}")

            # Máquina de estados
            if stage is None:
                stage = "up"

            # Condições para transição
            knee_down_condition = knee_angle < KNEE_ANGLE_DOWN_THRESHOLD
            hip_down_condition = hip_angle < HIP_ANGLE_DOWN_THRESHOLD
            knee_up_condition = knee_angle > KNEE_ANGLE_UP_THRESHOLD
            hip_up_condition = hip_angle > HIP_ANGLE_UP_THRESHOLD

            # UP -> DOWN (descendo no lunge)
            if stage == "up" and knee_down_condition and hip_down_condition:
                stage = "down"
                print(f"[INFO] ✓ Estado DOWN detectado! Joelho: {knee_angle:.1f}°")

            # DOWN -> UP (subindo - rep completa)
            elif stage == "down" and knee_up_condition and hip_up_condition:
                stage = "up"
                counter += 1
                
                # Avaliar qualidade
                depth_ok = rep_min_knee_angle is not None and rep_min_knee_angle <= (KNEE_ANGLE_DOWN_THRESHOLD + 20)
                hip_ok = rep_min_hip_angle is not None and rep_min_hip_angle <= (HIP_ANGLE_DOWN_THRESHOLD + 20)
                range_ok = (rep_max_knee_angle - rep_min_knee_angle) >= 40
                
                if depth_ok and hip_ok and range_ok:
                    good_reps += 1
                    last_rep_feedback = "GOOD"
                    print(f"[INFO] ✓ REP #{counter} ({current_leg.upper()}) - BOA!")
                else:
                    incomplete_reps += 1
                    issues = []
                    if not depth_ok:
                        issues.append("DEPTH")
                    if not hip_ok:
                        issues.append("HIP")
                    if not range_ok:
                        issues.append("RANGE")
                    last_rep_feedback = "INCOMPLETE - " + " | ".join(issues)
                    print(f"[WARN] ✗ REP #{counter} ({current_leg.upper()}) - INCOMPLETA: {' | '.join(issues)}")
                
                # Reset tracking
                rep_min_knee_angle = None
                rep_max_knee_angle = None
                rep_min_hip_angle = None

        # Desenhar skeleton
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2)
        )

    # Desenhar UI
    source_text = "WEBCAM" if SOURCE == "webcam" else "VIDEO"
    draw_ui(
        image, counter, stage, in_position, form_status,
        knee_angle, hip_angle, current_leg, camera_angle,
        source_text,
        thresholds=(KNEE_ANGLE_DOWN_THRESHOLD, KNEE_ANGLE_UP_THRESHOLD),
        calib_active=calib_active,
        good_reps=good_reps,
        incomplete_reps=incomplete_reps,
        feedback_list=feedback_list,
        last_rep_feedback=last_rep_feedback
    )
    
    cv2.imshow("Lunge Counter", image)
    
    # Processar teclas
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
        current_leg = None
        good_reps = 0
        incomplete_reps = 0
        last_rep_feedback = ""
        rep_min_knee_angle = None
        rep_max_knee_angle = None
        rep_min_hip_angle = None
        left_knee_angle_history.clear()
        right_knee_angle_history.clear()
        print("[INFO] Contador resetado!")
    elif key == ord('c'):
        calib_active = True
        calib_frames = 0
        knee_angle_min = float('inf')
        knee_angle_max = float('-inf')
        hip_angle_min = float('inf')
        hip_angle_max = float('-inf')
        KNEE_ANGLE_DOWN_THRESHOLD = 90
        KNEE_ANGLE_UP_THRESHOLD = 160
        HIP_ANGLE_DOWN_THRESHOLD = 100
        HIP_ANGLE_UP_THRESHOLD = 160
        print("[INFO] Recalibrando...")

# ===== FINALIZAÇÃO =====
cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\n[INFO] Sessão finalizada!")
print(f"[INFO] Total: {counter} | Boas: {good_reps} | Incompletas: {incomplete_reps}")