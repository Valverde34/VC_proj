# Algoritmo principal de contagem de flexoes
import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# escolher webcam ou vídeo
# SOURCE = "webcam"  # Para usar a webcam
SOURCE = "video"   # Para usar vídeo

# Definir caminho:
VIDEO_PATH = r"C:\VC_proj\src\Copy of push up 168.mp4"

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

# ===== ADAPTIVE THRESHOLDS (user-based) =====
USE_ADAPTIVE_THRESHOLDS = True
# Limiar base (fallback) até calibrar
DOWN_THRESHOLD = 90
UP_THRESHOLD = 155

# Estado de calibração
calib_active = USE_ADAPTIVE_THRESHOLDS
calib_frames = 0
CALIB_MIN_FRAMES = 120         # ~4s a 30 FPS em posicao
CALIB_MIN_SPREAD = 35          # amplitude minima entre topo e fundo para aceitar
MARGIN_DOWN = 5                # margem acima do minimo para DOWN
MARGIN_UP = 5                  # margem abaixo do maximo para UP
angle_min = float('inf')
angle_max = float('-inf')

# FORM ANALYSIS 
HIP_SAG_ANGLE_WARN = 165      # abaixo disto começa aviso
HIP_SAG_ANGLE_BAD = 150       # abaixo disto é grave
DEPTH_TOLERANCE = 6           # margem sobre DOWN_THRESHOLD
LOCKOUT_TOLERANCE = 25        # margem abaixo de UP_THRESHOLD
good_reps = 0
incomplete_reps = 0
rep_min_angle = None
rep_max_angle = None
last_rep_feedback = ""        # feedback da última repetição concluída

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
    Versão tolerante (vista mais lateral):
    - Usa tornozelo ou joelho (fallback) para o eixo do corpo.
    - Afrouxa ângulo de tronco e tolerância pulso<ombro.
    - Aceita também quando a orientação ombro→quadril está quase horizontal.
    """
    try:
        L = mp_pose.PoseLandmark

        def side_vis(side):
            ids = [
                getattr(L, f"{side}_SHOULDER").value,
                getattr(L, f"{side}_HIP").value,
                getattr(L, f"{side}_KNEE").value,
                getattr(L, f"{side}_ANKLE").value,
                getattr(L, f"{side}_WRIST").value,
            ]
            return float(np.mean([landmarks[i].visibility for i in ids]))

        left_v = side_vis("LEFT")
        right_v = side_vis("RIGHT")
        side = "LEFT" if left_v >= right_v else "RIGHT"

        sh = landmarks[getattr(L, f"{side}_SHOULDER").value]
        hp = landmarks[getattr(L, f"{side}_HIP").value]
        wr = landmarks[getattr(L, f"{side}_WRIST").value]
        kn = landmarks[getattr(L, f"{side}_KNEE").value]
        an = landmarks[getattr(L, f"{side}_ANKLE").value]

        # Fallback: usar joelho se tornozelo pouco visível
        foot = an if an.visibility >= 0.5 else kn

        # Escala vertical do corpo (ombro->pé/joelho)
        dy_sa = abs(foot.y - sh.y)
        dy_sa = max(dy_sa, 1e-6)

        # Pulso abaixo do ombro com tolerância mais baixa
        wrists_below_shoulders = wr.y > (sh.y + 0.008)

        # Tronco quase alinhado (ombro-quad-pezinho)
        trunk_angle = calculate_angle([sh.x, sh.y], [hp.x, hp.y], [foot.x, foot.y])
        trunk_ok = trunk_angle >= 140

        # Orientação ombro->quadril quase horizontal
        orient_deg = abs(np.degrees(np.arctan2(hp.y - sh.y, hp.x - sh.x)))
        orientation_ok = orient_deg <= 35

        # Fallback quando o sujeito ocupa pouca altura vertical (câmara inclinada)
        if dy_sa < 0.06:
            return wrists_below_shoulders and (orientation_ok or trunk_ok)

        # Regra principal (mais permissiva para vista lateral)
        return wrists_below_shoulders and (trunk_ok or orientation_ok)

    except Exception:
        return False


def draw_ui(image, counter, stage, in_pushup_position, form_status, elbow_angle=0, source="VIDEO",
            thresholds=(90, 155), calib_active=False,
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
    cv2.putText(image, "PUSHUP TRACKER", (x_margin, y_offset), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 3)
    cv2.line(image, (x_margin, y_offset + 10), (w - 30, y_offset + 10), (0,255,0), 2)
    y_offset += 40
    adapt_text = "ADAPTIVE: ON" if USE_ADAPTIVE_THRESHOLDS else "ADAPTIVE: OFF"
    adapt_color = (0,255,0) if USE_ADAPTIVE_THRESHOLDS else (0,0,255)
    cv2.putText(image, adapt_text, (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, adapt_color, 2)
    if USE_ADAPTIVE_THRESHOLDS:
        y_offset += 28
        calib_text = "CALIBRATING... do 1-2 reps" if calib_active else "CALIBRATION: OK"
        calib_color = (0,165,255) if calib_active else (0,255,0)
        cv2.putText(image, calib_text, (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, calib_color, 2)
    y_offset += 90
    cv2.rectangle(image, (x_margin - 15, y_offset - 50), (w - 30, y_offset + 30), (0,100,255), -1)
    cv2.putText(image, "REPS", (x_margin, y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(image, str(counter), (x_margin, y_offset + 25), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255,255,255), 4)
    y_offset += 100
    stage_text = stage if stage else "N/A"
    stage_color = (0,255,0) if stage == "up" else (0,165,255) if stage == "down" else (200,200,200)
    cv2.putText(image, "STAGE:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(image, stage_text.upper(), (x_margin, y_offset + 35), cv2.FONT_HERSHEY_DUPLEX, 1.2, stage_color, 3)
    y_offset += 80
    thr_down, thr_up = thresholds
    cv2.putText(image, "THRESHOLDS:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(image, f"DOWN {int(thr_down)} | UP {int(thr_up)}", (x_margin, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    y_offset += 60
    
    # Qualidade de reps
    cv2.putText(image, "QUALITY:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(image, f"GOOD {good_reps} | INCOMP {incomplete_reps}", (x_margin, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0) if incomplete_reps==0 else (0,165,255), 2)
    
    y_offset += 70
    position_text = "IN POSITION" if in_pushup_position else "NOT IN POSITION"
    position_color = (0,255,0) if in_pushup_position else (0,0,255)
    cv2.rectangle(image, (x_margin - 15, y_offset - 10), (w - 30, y_offset + 50), position_color, 3)
    cv2.putText(image, position_text, (x_margin + 10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, position_color, 2)
    y_offset += 90
    
    if in_pushup_position and elbow_angle > 0:
        cv2.putText(image, "ELBOW ANGLE:", (x_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        angle_color = (0,255,0) if 90 <= elbow_angle <= 170 else (0,165,255)
        cv2.putText(image, f"{int(elbow_angle)} deg", (x_margin, y_offset + 40), cv2.FONT_HERSHEY_DUPLEX, 1.5, angle_color, 3)
        bar_width = 300
        bar_x = x_margin
        bar_y = y_offset + 60
        progress = max(0, min(1, (elbow_angle - 80) / 100))
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
    
    # Feedback dinâmico
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
            color = (0,165,255) if "INCOMPLETE" in fb or "LOCKOUT" in fb else (0,0,255) if "HIP" in fb else (0,255,255)
            cv2.putText(image, fb, (x_margin, fb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            fb_y += 25
    
    # Última repetição
    if last_rep_feedback:
        cv2.putText(image, f"LAST: {last_rep_feedback}", (x_margin, fb_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    footer_y = h - 80
    cv2.rectangle(image, (w - panel_width, footer_y - 20), (w, h), (30,30,30), -1)
    cv2.putText(image, "CONTROLS:", (x_margin, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    cv2.putText(image, "Q Quit | P Pause | R Reset | C Recalib", (x_margin, footer_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,200,200), 1)
    cv2.rectangle(image, (10,10), (180,50), (60,60,60), -1)
    cv2.putText(image, f"SOURCE: {source}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
 
    
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

print("[INFO] Pressione 'q' para sair, 'p' para pausar/continuar, 'r' para reset, 'c' para recalibrar")
if is_video:
    print("[INFO] O vídeo vai reiniciar automaticamente quando terminar")

# Criar janela em fullscreen
cv2.namedWindow("Pushup Counter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Pushup Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

angle_buffer = deque(maxlen=5)

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
    feedback_list = []  # inicializar sempre
    
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
        
        if not in_pushup_position and good_form:
            in_pushup_position = True
            
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
            raw_angle = right_elbow_angle if use_right else left_elbow_angle

            # Suavização por média móvel
            angle_buffer.append(raw_angle)
            elbow_angle = np.mean(angle_buffer)

            # HIP SAG + lista de feedbacks (dentro de in_pushup_position)
            side_prefix = "LEFT" if left_vis >= right_vis else "RIGHT"
            sh_i = mp_pose.PoseLandmark[f"{side_prefix}_SHOULDER"].value
            hp_i = mp_pose.PoseLandmark[f"{side_prefix}_HIP"].value
            an_i = mp_pose.PoseLandmark[f"{side_prefix}_ANKLE"].value
            hip_angle = calculate_angle(
                [landmarks[sh_i].x, landmarks[sh_i].y],
                [landmarks[hp_i].x, landmarks[hp_i].y],
                [landmarks[an_i].x, landmarks[an_i].y]
            )

            if hip_angle < HIP_SAG_ANGLE_BAD:
                feedback_list.append("HIP SAG SEVERE")
            elif hip_angle < HIP_SAG_ANGLE_WARN:
                feedback_list.append("HIP SAG")

            # Range tracking (atualiza mínimos/máximos durante a rep)
            if stage == "down":
                rep_min_angle = elbow_angle if rep_min_angle is None else min(rep_min_angle, elbow_angle)
            # Atualizar sempre o máximo quando estamos na fase UP
            if stage == "up":
                rep_max_angle = elbow_angle if rep_max_angle is None else max(rep_max_angle, elbow_angle)
            else:
                # Mesmo na fase down, guardar máximos (para evitar perda por delay)
                rep_max_angle = elbow_angle if rep_max_angle is None else max(rep_max_angle, elbow_angle)

            # ===== Calibração adaptativa (coleta min/max) =====
            if USE_ADAPTIVE_THRESHOLDS:
                calib_frames += 1
                angle_min = min(angle_min, elbow_angle)
                angle_max = max(angle_max, elbow_angle)

                if calib_active and calib_frames >= CALIB_MIN_FRAMES and (angle_max - angle_min) >= CALIB_MIN_SPREAD:
                    new_down = angle_min + MARGIN_DOWN
                    new_up = angle_max - MARGIN_UP

                    # Garante histerese mínima
                    if new_up - new_down < 15:
                        new_up = new_down + 20

                    # Limites razoáveis
                    new_down = float(np.clip(new_down, 40, 140))
                    new_up = float(np.clip(new_up, 120, 175))

                    DOWN_THRESHOLD = new_down
                    UP_THRESHOLD = new_up
                    calib_active = False
                    print(f"[INFO] Calibração concluída: DOWN={DOWN_THRESHOLD:.1f} | UP={UP_THRESHOLD:.1f}")

            # Máquina de estados + avaliação de repetição
            if stage is None:
                stage = "up"

            if elbow_angle < DOWN_THRESHOLD and stage == "up":
                stage = "down"
                # iniciar novo ciclo de rep
                rep_min_angle = elbow_angle
                rep_max_angle = elbow_angle

            elif elbow_angle > UP_THRESHOLD and stage == "down":
                stage = "up"
                # final da repetição
                if good_form:
                    counter += 1
                    # Avaliação de amplitude
                    depth_ok = (rep_min_angle is not None) and (rep_min_angle <= (DOWN_THRESHOLD + DEPTH_TOLERANCE))

                    top_reached = rep_max_angle if rep_max_angle is not None else elbow_angle
                    lockout_ok = top_reached >= (UP_THRESHOLD - LOCKOUT_TOLERANCE)
                    
                    if depth_ok and lockout_ok and hip_angle >= HIP_SAG_ANGLE_WARN:
                        good_reps += 1
                        last_rep_feedback = "OK"
                    else:
                        issues = []
                        if not depth_ok: issues.append("INCOMPLETE DEPTH")
                        if not lockout_ok: issues.append("INCOMPLETE LOCKOUT")
                        if hip_angle < HIP_SAG_ANGLE_WARN: issues.append("HIP SAG")
                        last_rep_feedback = " | ".join(issues)
                        incomplete_reps += 1
                    # reset para próxima
                    rep_min_angle = None
                    rep_max_angle = None
                    
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
    draw_ui(
        image, counter, stage, in_pushup_position, form_status,
        elbow_angle, source_text,
        thresholds=(DOWN_THRESHOLD, UP_THRESHOLD),
        calib_active=calib_active,
        good_reps=good_reps,
        incomplete_reps=incomplete_reps,
        feedback_list=feedback_list,
        last_rep_feedback=last_rep_feedback
    )
        
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
        good_reps = 0
        incomplete_reps = 0
        last_rep_feedback = ""
        rep_min_angle = None
        rep_max_angle = None
        print("[INFO] Contador e métricas resetados!")
    elif key == ord('c'):
        # Reiniciar calibração
        calib_active = True
        calib_frames = 0
        angle_min = float('inf')
        angle_max = float('-inf')
        DOWN_THRESHOLD = 90
        UP_THRESHOLD = 155
        print("[INFO] Calibração reiniciada. Limiar reset para 90/155.")

cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\n[INFO] Sessão finalizada!")
print(f"[INFO] Total de repetições: {counter}")
print(f"[INFO] Boas: {good_reps} | Incompletas: {incomplete_reps}")