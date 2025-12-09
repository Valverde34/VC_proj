import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

SOURCE = "video"  # "webcam" ou "video"
VIDEO_PATH = r"C:\Uni\1_ano\1_semestre\VC\VC_proj\src\lunges3.mp4"

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

# Variáveis de contagem
counter = 0
stage = None  
current_leg = None  
pending_leg_change = None  
leg_change_confirmation_frames = 0  
MIN_LEG_CHANGE_FRAMES = 2  
form_status = "Unknown"
in_position = False
frames_since_last_rep = 0  
MIN_FRAMES_BETWEEN_REPS = 15  

# Thresholds adaptativos
USE_ADAPTIVE_THRESHOLDS = False  
KNEE_ANGLE_DOWN_THRESHOLD = 110  
KNEE_ANGLE_UP_THRESHOLD = 160     
HIP_ANGLE_DOWN_THRESHOLD = 120    
HIP_ANGLE_UP_THRESHOLD = 145      

# Estado de calibração
calib_active = USE_ADAPTIVE_THRESHOLDS
calib_frames = 0
CALIB_MIN_FRAMES = 60
knee_angle_min = float('inf')
knee_angle_max = float('-inf')
hip_angle_min = float('inf')
hip_angle_max = float('-inf')

good_reps = 0
incomplete_reps = 0
last_rep_feedback = ""
rep_min_knee_angle = None
rep_max_knee_angle = None
rep_min_hip_angle = None

left_knee_angle_history = deque(maxlen=10)
right_knee_angle_history = deque(maxlen=10)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vetores partindo do ponto central (joelho)
    ba = a - b  
    bc = c - b  
    
    # Calcular ângulo usando produto escalar
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    # Evitar divisão por zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 180.0
    
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.degrees(np.arccos(cosine_angle))
    
    if len(c) >= 2 and len(b) >= 2:
        ankle_above_knee = c[1] < b[1]
        
        if ankle_above_knee and angle < 90:
            angle = 180.0 - angle
    
    if angle < 1.0:
        return 1.0
    
    return angle


def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def detect_camera_angle(landmarks):
    L = mp_pose.PoseLandmark
    
    left_shoulder = landmarks[L.LEFT_SHOULDER.value]
    right_shoulder = landmarks[L.RIGHT_SHOULDER.value]
    left_hip = landmarks[L.LEFT_HIP.value]
    right_hip = landmarks[L.RIGHT_HIP.value]
    
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    hip_width = abs(left_hip.x - right_hip.x)
    
    left_vis = np.mean([landmarks[L.LEFT_SHOULDER.value].visibility,
                        landmarks[L.LEFT_HIP.value].visibility,
                        landmarks[L.LEFT_KNEE.value].visibility])
    
    right_vis = np.mean([landmarks[L.RIGHT_SHOULDER.value].visibility,
                         landmarks[L.RIGHT_HIP.value].visibility,
                         landmarks[L.RIGHT_KNEE.value].visibility])
    
    if shoulder_width < 0.15 and hip_width < 0.15:
        return "side"
    
    if left_vis > 0.5 and right_vis > 0.5:
        nose = landmarks[L.NOSE.value]
        nose_vis = nose.visibility
        if nose_vis > 0.6:
            return "front"
        else:
            return "back"
    
    return "side"


def check_lunge_position(landmarks):
    try:
        L = mp_pose.PoseLandmark
        
        critical_points = [
            L.LEFT_HIP, L.RIGHT_HIP,
            L.LEFT_KNEE, L.RIGHT_KNEE,
        ]
        
        visible_count = 0
        for p in critical_points:
            if landmarks[p.value].visibility > 0.3:  
                visible_count += 1
        
        if visible_count < 3:
            return False
        
        left_knee = landmarks[L.LEFT_KNEE.value]
        right_knee = landmarks[L.RIGHT_KNEE.value]
        
        if left_knee.visibility < 0.3 and right_knee.visibility < 0.3:
            return False
        
        return True
        
    except Exception as e:
        return False


def determine_working_leg(landmarks, left_knee_angle, right_knee_angle):
    L = mp_pose.PoseLandmark
    
    # Coordenadas
    left_knee = landmarks[L.LEFT_KNEE.value]
    left_ankle = landmarks[L.LEFT_ANKLE.value]
    right_knee = landmarks[L.RIGHT_KNEE.value]
    right_ankle = landmarks[L.RIGHT_ANKLE.value]
    
    left_score = 0
    right_score = 0

    if abs(left_knee_angle - right_knee_angle) > 30:
        if left_knee_angle < right_knee_angle:
            left_score += 10
        else:
            right_score += 10
    
    left_shin_horiz_dist = abs(left_knee.x - left_ankle.x)
    right_shin_horiz_dist = abs(right_knee.x - right_ankle.x)
    
    if abs(left_shin_horiz_dist - right_shin_horiz_dist) > 0.02:
        if left_shin_horiz_dist < right_shin_horiz_dist:
            left_score += 5
        else:
            right_score += 5

    if abs(left_knee.y - right_knee.y) > 0.05:
        if left_knee.y < right_knee.y: 
            left_score += 2
        else:
            right_score += 2

    if abs(left_knee.z - right_knee.z) > 0.1: 
        if left_knee.z < right_knee.z:
            left_score += 1
        else:
            right_score += 1
    
    if left_score > right_score:
        return "left"
    elif right_score > left_score:
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
    
    if w < 800:
        panel_width = 250  
    elif w < 1200:
        panel_width = 300  
    else:
        panel_width = 400  
    
    overlay = image.copy()
    cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.line(image, (w - panel_width, 0), (w - panel_width, h), (100, 100, 100), 3)
    
    y_offset = 60
    x_margin = w - panel_width + 20  
    
    title_scale = 0.8 if panel_width < 350 else 1.0
    text_scale = 0.5 if panel_width < 350 else 0.6
    number_scale = 1.2 if panel_width < 350 else 1.8
    
    cv2.putText(image, "LUNGE COUNTER", (x_margin, y_offset), cv2.FONT_HERSHEY_DUPLEX, title_scale, (255,255,255), 2 if panel_width < 350 else 3)
    cv2.line(image, (x_margin, y_offset + 10), (w - 20, y_offset + 10), (0,255,0), 2)

    y_offset += 40
    cam_colors = {"front": (0,255,0), "side": (0,165,255), "back": (255,100,0)}
    cam_color = cam_colors.get(camera_angle, (200,200,200))
    cv2.putText(image, f"VIEW: {camera_angle.upper()}", (x_margin, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, cam_color, 2)
    
    y_offset += 30
    adapt_text = "ADAPTIVE: ON" if USE_ADAPTIVE_THRESHOLDS else "ADAPTIVE: OFF"
    adapt_color = (0,255,0) if USE_ADAPTIVE_THRESHOLDS else (0,0,255)
    cv2.putText(image, adapt_text, (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_scale, adapt_color, 2)
    
    if USE_ADAPTIVE_THRESHOLDS:
        y_offset += 28
        calib_text = "CALIBRATING..." if calib_active else "CALIBRATION: OK"
        calib_color = (0,165,255) if calib_active else (0,255,0)
        cv2.putText(image, calib_text, (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, calib_color, 2)
    
    y_offset += 60
    cv2.rectangle(image, (x_margin - 10, y_offset - 40), (w - 20, y_offset + 25), (0,100,255), -1)
    cv2.putText(image, "REPS", (x_margin, y_offset - 15), cv2.FONT_HERSHEY_SIMPLEX, text_scale * 1.2, (255,255,255), 2)
    cv2.putText(image, str(counter), (x_margin, y_offset + 20), cv2.FONT_HERSHEY_DUPLEX, number_scale, (255,255,255), 3 if panel_width < 350 else 4)
    
    y_offset += 70
    if current_leg:
        leg_color = (0,255,255)
        cv2.putText(image, "WORKING LEG:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(image, current_leg.upper(), (x_margin, y_offset + 35), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, leg_color, 3)
        y_offset += 70
    
    stage_text = stage if stage else "N/A"
    stage_color = (0,255,0) if stage == "down" else (0,165,255) if stage == "up" else (200,200,200)
    cv2.putText(image, "STAGE:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(image, stage_text.upper(), (x_margin, y_offset + 35), cv2.FONT_HERSHEY_DUPLEX, 1.2, stage_color, 3)
    
    y_offset += 80
    thr_down, thr_up = thresholds
    cv2.putText(image, "KNEE THRESHOLDS:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(image, f"DOWN {int(thr_down)} | UP {int(thr_up)}", (x_margin, y_offset + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    y_offset += 70
    cv2.putText(image, "QUALITY:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(image, f"GOOD {good_reps} | INCOMP {incomplete_reps}", (x_margin, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0) if incomplete_reps==0 else (0,165,255), 2)
    
    y_offset += 70
    position_text = "IN POSITION" if in_position else "NOT IN POSITION"
    position_color = (0,255,0) if in_position else (0,0,255)
    cv2.rectangle(image, (x_margin - 15, y_offset - 10), (w - 30, y_offset + 50), position_color, 3)
    cv2.putText(image, position_text, (x_margin + 10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, position_color, 2)
    
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
        
        cv2.putText(image, "HIP ANGLE:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        hip_color = (0,255,0) if hip_angle < 120 else (0,165,255)
        cv2.putText(image, f"{int(hip_angle)} deg", (x_margin, y_offset + 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, hip_color, 3)
        y_offset += 60
    
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
    
    footer_y = h - 80
    cv2.rectangle(image, (w - panel_width, footer_y - 20), (w, h), (30,30,30), -1)
    cv2.putText(image, "CONTROLS:", (x_margin, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    cv2.putText(image, "Q Quit | P Pause | R Reset | C Recalib", (x_margin, footer_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,200,200), 1)
    
    cv2.rectangle(image, (10,10), (180,50), (60,60,60), -1)
    cv2.putText(image, f"SOURCE: {source}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


# Configurar fonte de vídeo
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

# Buffers 
knee_angle_buffer = deque(maxlen=5)
hip_angle_buffer = deque(maxlen=5)


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

    screen_res = 1920, 1080
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    
    image = cv2.resize(image, (window_width, window_height), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    frames_since_last_rep += 1
    
    knee_angle = 0
    hip_angle = 0
    feedback_list = []
    camera_angle = "front"
    display_stage = stage if stage else "up" 
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        in_position = check_lunge_position(landmarks)
        camera_angle = detect_camera_angle(landmarks)
        
        if results.pose_world_landmarks:
            landmarks_3d = results.pose_world_landmarks.landmark
            use_3d = True
        else:
            landmarks_3d = landmarks
            use_3d = False
        
        L = mp_pose.PoseLandmark
        
        l_hip_3d = [landmarks_3d[L.LEFT_HIP.value].x, landmarks_3d[L.LEFT_HIP.value].y, landmarks_3d[L.LEFT_HIP.value].z]
        l_knee_3d = [landmarks_3d[L.LEFT_KNEE.value].x, landmarks_3d[L.LEFT_KNEE.value].y, landmarks_3d[L.LEFT_KNEE.value].z]
        l_ankle_3d = [landmarks_3d[L.LEFT_ANKLE.value].x, landmarks_3d[L.LEFT_ANKLE.value].y, landmarks_3d[L.LEFT_ANKLE.value].z]
        l_shoulder_3d = [landmarks_3d[L.LEFT_SHOULDER.value].x, landmarks_3d[L.LEFT_SHOULDER.value].y, landmarks_3d[L.LEFT_SHOULDER.value].z]
        
        r_hip_3d = [landmarks_3d[L.RIGHT_HIP.value].x, landmarks_3d[L.RIGHT_HIP.value].y, landmarks_3d[L.RIGHT_HIP.value].z]
        r_knee_3d = [landmarks_3d[L.RIGHT_KNEE.value].x, landmarks_3d[L.RIGHT_KNEE.value].y, landmarks_3d[L.RIGHT_KNEE.value].z]
        r_ankle_3d = [landmarks_3d[L.RIGHT_ANKLE.value].x, landmarks_3d[L.RIGHT_ANKLE.value].y, landmarks_3d[L.RIGHT_ANKLE.value].z]
        r_shoulder_3d = [landmarks_3d[L.RIGHT_SHOULDER.value].x, landmarks_3d[L.RIGHT_SHOULDER.value].y, landmarks_3d[L.RIGHT_SHOULDER.value].z]
        
        left_knee_angle = calculate_angle(l_hip_3d, l_knee_3d, l_ankle_3d)
        right_knee_angle = calculate_angle(r_hip_3d, r_knee_3d, r_ankle_3d)
        
        left_hip_angle = calculate_angle(l_shoulder_3d, l_hip_3d, l_knee_3d)
        right_hip_angle = calculate_angle(r_shoulder_3d, r_hip_3d, r_knee_3d)
        
        left_hip = landmarks[L.LEFT_HIP.value]
        left_knee = landmarks[L.LEFT_KNEE.value]
        left_ankle = landmarks[L.LEFT_ANKLE.value]
        right_hip = landmarks[L.RIGHT_HIP.value]
        right_knee = landmarks[L.RIGHT_KNEE.value]
        right_ankle = landmarks[L.RIGHT_ANKLE.value]
        
        left_knee_angle_history.append(left_knee_angle)
        right_knee_angle_history.append(right_knee_angle)
        
        working_leg = determine_working_leg(landmarks, left_knee_angle, right_knee_angle)
        
        if working_leg:
            if working_leg != current_leg:
                if working_leg == pending_leg_change:
                    leg_change_confirmation_frames += 1
                    if leg_change_confirmation_frames >= MIN_LEG_CHANGE_FRAMES:
                        current_leg = working_leg
                        leg_change_confirmation_frames = 0
                        pending_leg_change = None
                        print(f"[INFO] Working leg: {current_leg.upper()}")
                else:
                    pending_leg_change = working_leg
                    leg_change_confirmation_frames = 1
            else:
                pending_leg_change = None
                leg_change_confirmation_frames = 0
        
        if left_knee_angle > 10 and right_knee_angle > 10:
            if current_leg == "left":
                raw_knee_angle = left_knee_angle
                raw_hip_angle = left_hip_angle
            elif current_leg == "right":
                raw_knee_angle = right_knee_angle
                raw_hip_angle = right_hip_angle
            else:
                raw_knee_angle = min(left_knee_angle, right_knee_angle)
                raw_hip_angle = left_hip_angle if left_knee_angle < right_knee_angle else right_hip_angle
            
            # Validar ângulos antes de adicionar ao buffer
            if raw_knee_angle > 10 and raw_hip_angle > 10:
                knee_angle_buffer.append(raw_knee_angle)
                hip_angle_buffer.append(raw_hip_angle)
                
                if len(knee_angle_buffer) > 0:
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
            if left_knee_angle < right_knee_angle:
                knee_x = left_knee.x
                ankle_x = left_ankle.x
            else:
                knee_x = right_knee.x
                ankle_x = right_ankle.x
            
            shin_horizontal_offset = abs(knee_x - ankle_x)
            if shin_horizontal_offset > 0.15 and stage == "down":
                form_status = "Knee too forward"
                feedback_list.append("KEEP SHIN VERTICAL")
            else:
                form_status = "Good form"
        
        display_stage = stage if stage else "up"
        if knee_angle > 0:  
            if knee_angle < 130:
                display_stage = "down"
            elif knee_angle > 145:
                display_stage = "up"
        else:
            display_stage = stage if stage else "up"

        if in_position and current_leg:
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
                if knee_angle > 20 and hip_angle > 20:
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

            if stage == "up" and knee_down_condition and frames_since_last_rep > MIN_FRAMES_BETWEEN_REPS:
                stage = "down"
                print(f"[INFO] ✓ Estado DOWN detectado! Joelho: {knee_angle:.1f}° | Quadril: {hip_angle:.1f}°")

            elif stage == "down" and knee_up_condition and frames_since_last_rep > MIN_FRAMES_BETWEEN_REPS:
                stage = "up"
                counter += 1
                frames_since_last_rep = 0  
                
                leg_for_print = current_leg.upper() if current_leg else "UNKNOWN"
                
                current_leg = None
                pending_leg_change = None
                leg_change_confirmation_frames = 0
                
                depth_ok = rep_min_knee_angle is not None and rep_min_knee_angle <= 130
                hip_ok = rep_min_hip_angle is not None and rep_min_hip_angle <= (HIP_ANGLE_DOWN_THRESHOLD + 20)
                range_ok = (rep_max_knee_angle - rep_min_knee_angle) >= 50 if rep_min_knee_angle is not None and rep_max_knee_angle is not None else False
                
                if depth_ok and hip_ok and range_ok:
                    good_reps += 1
                    last_rep_feedback = "GOOD"
                    print(f"[INFO] ✓ REP #{counter} ({leg_for_print}) - BOA!")
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
                    print(f"[WARN] ✗ REP #{counter} ({leg_for_print}) - INCOMPLETA: {' | '.join(issues)}")
                
                rep_min_knee_angle = None
                rep_max_knee_angle = None
                rep_min_hip_angle = None
                
                left_knee_angle_history.clear()
                right_knee_angle_history.clear()
                knee_angle_buffer.clear()
                hip_angle_buffer.clear()

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2)
        )

    source_text = "WEBCAM" if SOURCE == "webcam" else "VIDEO"
    draw_ui(
        image, counter, display_stage, in_position, form_status,
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
        pending_leg_change = None  
        leg_change_confirmation_frames = 0
        good_reps = 0
        incomplete_reps = 0
        last_rep_feedback = ""
        rep_min_knee_angle = None
        rep_max_knee_angle = None
        rep_min_hip_angle = None
        frames_since_last_rep = 0
        left_knee_angle_history.clear()
        right_knee_angle_history.clear()
        knee_angle_buffer.clear()  
        hip_angle_buffer.clear()   
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

cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\n[INFO] Sessão finalizada!")
print(f"[INFO] Total: {counter} | Boas: {good_reps} | Incompletas: {incomplete_reps}")