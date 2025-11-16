# so para testar a camera

import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    cap = cv2.VideoCapture(0)  # 0 = webcam principal

    # Otimizações importantes
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,         # 0 = mais leve, 2 = mais preciso
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        prev_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro a captar imagem da webcam.")
                break

            # Converter BGR → RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Processar com MediaPipe
            results = pose.process(rgb_frame)

            # Desenhar landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # FPS counter
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Mostrar janela
            cv2.imshow("MediaPipe Pose - Real-Time", frame)

            # Tecla 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
