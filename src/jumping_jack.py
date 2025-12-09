import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

SOURCE = "video"  # "webcam" ou "video"
VIDEO_PATH = r"C:\Uni\1_ano\1_semestre\VC\VC_proj\src\jumpingjack1.mp4"

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
form_status = "Unknown"
in_position = False

# Thresholds adaptativos
USE_ADAPTIVE_THRESHOLDS = True
ARM_DOWN_THRESHOLD = 40
ARM_UP_THRESHOLD = 130
FEET_TOGETHER_THRESHOLD = 0.10
FEET_APART_THRESHOLD = 0.20

# Calibração
calib_active = USE_ADAPTIVE_THRESHOLDS
calib_frames = 0
CALIB_MIN_FRAMES = 60
arm_angle_min = float('inf')
arm_angle_max = float('-inf')
feet_dist_min = float('inf')
feet_dist_max = float('-inf')

# Análise de qualidade
good_reps = 0
incomplete_reps = 0
last_rep_feedback = ""
rep_max_arm_angle = None
rep_min_feet_dist = None
rep_max_feet_dist = None


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_arm_angle_v2(landmarks):
    L = mp_pose.PoseLandmark
    
    left_shoulder = landmarks[L.LEFT_SHOULDER.value]
    left_wrist = landmarks[L.LEFT_WRIST.value]
    right_shoulder = landmarks[L.RIGHT_SHOULDER.value]
    right_wrist = landmarks[L.RIGHT_WRIST.value]
    
    left_y_diff = left_shoulder.y - left_wrist.y
    left_x_diff = abs(left_wrist.x - left_shoulder.x)
    left_angle = np.arctan2(left_y_diff, left_x_diff) * 180 / np.pi
    
    right_y_diff = right_shoulder.y - right_wrist.y
    right_x_diff = abs(right_wrist.x - right_shoulder.x)
    right_angle = np.arctan2(right_y_diff, right_x_diff) * 180 / np.pi
    
    left_angle = max(0, min(180, left_angle + 90))
    right_angle = max(0, min(180, right_angle + 90))
    
    return (left_angle + right_angle) / 2, min(left_angle, right_angle)


def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def check_jumping_jack_position(landmarks):
    try:
        L = mp_pose.PoseLandmark
        
        left_points = [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE]
        right_points = [L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE]
        
        left_visibility = np.mean([landmarks[p.value].visibility for p in left_points])
        right_visibility = np.mean([landmarks[p.value].visibility for p in right_points])
        
        if left_visibility < 0.5 or right_visibility < 0.5:
            return False
        
        left_knee_vis = landmarks[L.LEFT_KNEE.value].visibility
        right_knee_vis = landmarks[L.RIGHT_KNEE.value].visibility
        
        if left_knee_vis < 0.6 or right_knee_vis < 0.6:
            return False
        
        left_shoulder = landmarks[L.LEFT_SHOULDER.value]
        right_shoulder = landmarks[L.RIGHT_SHOULDER.value]
        left_knee = landmarks[L.LEFT_KNEE.value]
        right_knee = landmarks[L.RIGHT_KNEE.value]
        left_ankle = landmarks[L.LEFT_ANKLE.value]
        right_ankle = landmarks[L.RIGHT_ANKLE.value]
        
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_knee_y = (left_knee.y + right_knee.y) / 2
        avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
        
        shoulder_to_knee = abs(avg_shoulder_y - avg_knee_y)
        if shoulder_to_knee < 0.25:
            return False
        
        vertical_span = abs(avg_shoulder_y - avg_ankle_y)
        if vertical_span < 0.4:
            return False
        
        if avg_knee_y < 0.1 or avg_knee_y > 0.95:
            return False
        
        knee_position = (avg_knee_y - avg_shoulder_y) / vertical_span if vertical_span > 0 else 0
        if knee_position < 0.5 or knee_position > 0.85:
            return False
            
        return True
        
    except Exception:
        return False


def draw_ui(image, counter, stage, in_position, form_status, 
            arm_angle=0, feet_distance=0, source="VIDEO",
            thresholds=(60, 140), calib_active=False,
            good_reps=0, incomplete_reps=0, feedback_list=None, last_rep_feedback=""):
    
    if feedback_list is None:
        feedback_list = []
    h, w = image.shape[:2] 
   
    panel_width = 400
    overlay = image.copy()
    cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.line(image, (w - panel_width, 0), (w - panel_width, h), (100, 100, 100), 3)
    
    y_offset = 60
    x_margin = w - panel_width + 30
    
    cv2.putText(image, "JUMPING JACK", (x_margin, y_offset), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 3)
    cv2.line(image, (x_margin, y_offset + 10), (w - 30, y_offset + 10), (0,255,0), 2)
    
    y_offset += 40
    adapt_text = "ADAPTIVE: ON" if USE_ADAPTIVE_THRESHOLDS else "ADAPTIVE: OFF"
    adapt_color = (0,255,0) if USE_ADAPTIVE_THRESHOLDS else (0,0,255)
    cv2.putText(image, adapt_text, (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, adapt_color, 2)
    
    if USE_ADAPTIVE_THRESHOLDS:
        y_offset += 28
        calib_text = "CALIBRATING..." if calib_active else "CALIBRATION: OK"
        calib_color = (0,165,255) if calib_active else (0,255,0)
        cv2.putText(image, calib_text, (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, calib_color, 2)
    
    y_offset += 90
    cv2.rectangle(image, (x_margin - 15, y_offset - 50), (w - 30, y_offset + 30), (0,100,255), -1)
    cv2.putText(image, "REPS", (x_margin, y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(image, str(counter), (x_margin, y_offset + 25), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255,255,255), 4)
    
    y_offset += 100
    stage_text = stage if stage else "N/A"
    stage_color = (0,255,0) if stage == "open" else (0,165,255) if stage == "closed" else (200,200,200)
    cv2.putText(image, "STAGE:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(image, stage_text.upper(), (x_margin, y_offset + 35), cv2.FONT_HERSHEY_DUPLEX, 1.2, stage_color, 3)
    
    y_offset += 80
    thr_down, thr_up = thresholds
    cv2.putText(image, "ARM THRESHOLDS:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(image, f"DOWN {int(thr_down)} | UP {int(thr_up)}", (x_margin, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    y_offset += 60
    cv2.putText(image, "FEET DISTANCE:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    feet_percent = int(feet_distance * 100)
    cv2.putText(image, f"{feet_percent}%", (x_margin, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    y_offset += 70
    cv2.putText(image, "QUALITY:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(image, f"GOOD {good_reps} | INCOMP {incomplete_reps}", (x_margin, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0) if incomplete_reps==0 else (0,165,255), 2)
    
    y_offset += 70
    position_text = "IN POSITION" if in_position else "NOT IN POSITION"
    position_color = (0,255,0) if in_position else (0,0,255)
    cv2.rectangle(image, (x_margin - 15, y_offset - 10), (w - 30, y_offset + 50), position_color, 3)
    cv2.putText(image, position_text, (x_margin + 10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, position_color, 2)
    
    y_offset += 90
    if in_position and arm_angle > 0:
        cv2.putText(image, "ARM ANGLE:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        angle_color = (0,255,0) if arm_angle > 100 else (0,165,255)
        cv2.putText(image, f"{int(arm_angle)} deg", (x_margin, y_offset + 40), cv2.FONT_HERSHEY_DUPLEX, 1.5, angle_color, 3)
        
        bar_width = 300
        bar_x = x_margin
        bar_y = y_offset + 60
        progress = max(0, min(1, arm_angle / 180))
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (100,100,100), -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), angle_color, -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (200,200,200), 2)
        y_offset += 100
    
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
        fb_y += 25
    else:
        for fb in feedback_list[:4]:
            color = (0,165,255) if "INCOMPLETE" in fb else (0,0,255) if "ARMS" in fb or "FEET" in fb else (0,255,255)
            cv2.putText(image, fb, (x_margin, fb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            fb_y += 25
    
    if last_rep_feedback:
        cv2.putText(image, f"LAST: {last_rep_feedback}", (x_margin, fb_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    
    footer_y = h - 80
    cv2.rectangle(image, (w - panel_width, footer_y - 20), (w, h), (30,30,30), -1)
    cv2.putText(image, "CONTROLS:", (x_margin, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    cv2.putText(image, "Q Quit | P Pause | R Reset | C Recalib", (x_margin, footer_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,200,200), 1)
    
    cv2.rectangle(image, (10,10), (180,50), (60,60,60), -1)
    cv2.putText(image, f"SOURCE: {source}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


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

cv2.namedWindow("Jumping Jack Counter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Jumping Jack Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Buffers 
arm_angle_buffer = deque(maxlen=5)
feet_dist_buffer = deque(maxlen=5)

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
    
    arm_angle = 0
    feet_distance = 0
    feedback_list = []
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        in_position = check_jumping_jack_position(landmarks)

        L = mp_pose.PoseLandmark
        
        left_shoulder = landmarks[L.LEFT_SHOULDER.value]
        left_wrist = landmarks[L.LEFT_WRIST.value]
        right_shoulder = landmarks[L.RIGHT_SHOULDER.value]
        right_wrist = landmarks[L.RIGHT_WRIST.value]
        
        left_elevation = calculate_angle(
            [left_shoulder.x, left_shoulder.y + 0.3],
            [left_shoulder.x, left_shoulder.y],
            [left_wrist.x, left_wrist.y]
        )
        
        right_elevation = calculate_angle(
            [right_shoulder.x, right_shoulder.y + 0.3],
            [right_shoulder.x, right_shoulder.y],
            [right_wrist.x, right_wrist.y]
        )
        
        arm_angle_v2, min_arm_v2 = calculate_arm_angle_v2(landmarks)
        
        raw_arm_angle = max((left_elevation + right_elevation) / 2, arm_angle_v2)
        min_arm_angle = max(min(left_elevation, right_elevation), min_arm_v2)
        
        arm_angle_buffer.append(raw_arm_angle)
        arm_angle = np.mean(arm_angle_buffer)
        
        left_ankle = landmarks[L.LEFT_ANKLE.value]
        right_ankle = landmarks[L.RIGHT_ANKLE.value]
        raw_feet_distance = calculate_distance(left_ankle, right_ankle)
        
        feet_dist_buffer.append(raw_feet_distance)
        feet_distance = np.mean(feet_dist_buffer)

        both_arms_up = left_elevation > 100 and right_elevation > 100
        both_arms_down = left_elevation < 80 and right_elevation < 80
        
        if not in_position:
            form_status = "Not in position"
        elif stage == "open" and not both_arms_up:
            form_status = "Both arms must be up"
            feedback_list.append("RAISE BOTH ARMS")
        elif stage == "closed" and not both_arms_down:
            form_status = "Lower both arms"
        else:
            form_status = "Good form"

        if in_position:
            if rep_min_feet_dist is None:
                rep_min_feet_dist = feet_distance
                rep_max_feet_dist = feet_distance
                rep_max_arm_angle = arm_angle
            else:
                rep_min_feet_dist = min(rep_min_feet_dist, feet_distance)
                rep_max_feet_dist = max(rep_max_feet_dist, feet_distance)
                rep_max_arm_angle = max(rep_max_arm_angle, arm_angle)

            if USE_ADAPTIVE_THRESHOLDS and calib_active:
                calib_frames += 1
                arm_angle_min = min(arm_angle_min, arm_angle)
                arm_angle_max = max(arm_angle_max, arm_angle)
                feet_dist_min = min(feet_dist_min, feet_distance)
                feet_dist_max = max(feet_dist_max, feet_distance)

                if calib_frames >= CALIB_MIN_FRAMES:
                    if (arm_angle_max - arm_angle_min) >= 40:
                        ARM_DOWN_THRESHOLD = arm_angle_min + 10
                        ARM_UP_THRESHOLD = arm_angle_max - 20
                        
                        if (feet_dist_max - feet_dist_min) >= 0.08:
                            FEET_TOGETHER_THRESHOLD = feet_dist_min + 0.03
                            FEET_APART_THRESHOLD = feet_dist_max - 0.05
                        
                        calib_active = False
                        print(f"[INFO] Calibração concluída:")
                        print(f"       ARM_DOWN={ARM_DOWN_THRESHOLD:.1f} | ARM_UP={ARM_UP_THRESHOLD:.1f}")
                        print(f"       FEET_TOGETHER={FEET_TOGETHER_THRESHOLD:.2f} | FEET_APART={FEET_APART_THRESHOLD:.2f}")

            if stage is None:
                stage = "closed"

            arms_up_condition = arm_angle > ARM_UP_THRESHOLD and min_arm_angle > (ARM_UP_THRESHOLD - 30)
            feet_apart_condition = feet_distance > FEET_APART_THRESHOLD
            arms_down_condition = arm_angle < ARM_DOWN_THRESHOLD and min_arm_angle < (ARM_DOWN_THRESHOLD + 30)
            feet_together_condition = feet_distance < FEET_TOGETHER_THRESHOLD

            if stage == "closed" and arms_up_condition and feet_apart_condition:
                stage = "open"
                print(f"[INFO] ✓ Estado OPEN detectado! Ângulo: {arm_angle:.1f}°")

            elif stage == "open" and arms_down_condition and feet_together_condition:
                stage = "closed"
                counter += 1
                
                arms_ok = rep_max_arm_angle is not None and rep_max_arm_angle >= (ARM_UP_THRESHOLD - 30)
                feet_range = (rep_max_feet_dist - rep_min_feet_dist) if (rep_max_feet_dist and rep_min_feet_dist) else 0
                feet_moved = feet_range >= 0.06
                
                if arms_ok and feet_moved:
                    good_reps += 1
                    last_rep_feedback = "GOOD"
                    print(f"[INFO] ✓ REP #{counter} - BOA!")
                else:
                    incomplete_reps += 1
                    issues = []
                    if not arms_ok:
                        issues.append("ARMS")
                    if not feet_moved:
                        issues.append("FEET")
                    last_rep_feedback = "INCOMPLETE - " + " | ".join(issues)
                    print(f"[WARN] ✗ REP #{counter} - INCOMPLETA: {' | '.join(issues)}")
                
                rep_max_arm_angle = None
                rep_min_feet_dist = None
                rep_max_feet_dist = None

        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2)
        )

    source_text = "WEBCAM" if SOURCE == "webcam" else "VIDEO"
    draw_ui(
        image, counter, stage, in_position, form_status,
        arm_angle, feet_distance, source_text,
        thresholds=(ARM_DOWN_THRESHOLD, ARM_UP_THRESHOLD),
        calib_active=calib_active,
        good_reps=good_reps,
        incomplete_reps=incomplete_reps,
        feedback_list=feedback_list,
        last_rep_feedback=last_rep_feedback
    )
    
    cv2.imshow("Jumping Jack Counter", image)
    
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
        good_reps = 0
        incomplete_reps = 0
        last_rep_feedback = ""
        rep_max_arm_angle = None
        rep_min_feet_dist = None
        rep_max_feet_dist = None
        print("[INFO] Contador resetado!")
    elif key == ord('c'):
        calib_active = True
        calib_frames = 0
        arm_angle_min = float('inf')
        arm_angle_max = float('-inf')
        feet_dist_min = float('inf')
        feet_dist_max = float('-inf')
        ARM_DOWN_THRESHOLD = 40
        ARM_UP_THRESHOLD = 130
        FEET_TOGETHER_THRESHOLD = 0.10
        FEET_APART_THRESHOLD = 0.20
        print("[INFO] Recalibrando...")

cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\n[INFO] Sessão finalizada!")
print(f"[INFO] Total: {counter} | Boas: {good_reps} | Incompletas: {incomplete_reps}")