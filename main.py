import cv2
import mediapipe as mp
import numpy as np

# Inicializar o Face Mesh do MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Inclui landmarks da íris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)  # Alterar o índice se necessárientao o

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível capturar a imagem da câmera.")
        break

    # Converter a imagem para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a imagem e encontrar os pontos faciais
    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Desenhar todos os pontos faciais
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Pontos para medir a largura do rosto
            left_face_point = face_landmarks.landmark[234]  # Ponto no lado esquerdo do rosto
            right_face_point = face_landmarks.landmark[454]  # Ponto no lado direito do rosto

            left_face_coords = (int(left_face_point.x * w), int(left_face_point.y * h))
            right_face_coords = (int(right_face_point.x * w), int(right_face_point.y * h))

            # Desenhar pontos nos lados do rosto
            cv2.circle(frame, left_face_coords, 5, (0, 0, 255), -1)
            cv2.circle(frame, right_face_coords, 5, (0, 0, 255), -1)

            # Calcular a largura do rosto em pixels
            face_width_px = np.linalg.norm(np.array(left_face_coords) - np.array(right_face_coords))

            # Assumir largura média do rosto em milímetros
            face_width_mm = 140  # 140 mm ou 14 cm

            # Calcular a proporção pixels/mm
            scale = face_width_px / face_width_mm

            # Obter coordenadas das pupilas (íris)
            if len(face_landmarks.landmark) > 473:
                left_eye = face_landmarks.landmark[468]
                right_eye = face_landmarks.landmark[473]

                left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
                right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

                # Desenhar círculos nas pupilas
                cv2.circle(frame, left_eye_coords, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_eye_coords, 5, (255, 0, 0), -1)

                # Calcular a distância entre as pupilas em pixels
                eye_distance_px = np.linalg.norm(np.array(left_eye_coords) - np.array(right_eye_coords))

                # Converter a distância para milímetros
                eye_distance_mm = eye_distance_px / scale

                cv2.putText(frame, f'Distância entre pupilas: {eye_distance_mm:.2f} mm', (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Feedback no console
                print(f"Largura do rosto em pixels: {face_width_px:.2f}px")
                print(f"Distância entre pupilas em pixels: {eye_distance_px:.2f}px")
                print(f"Distância entre pupilas em mm: {eye_distance_mm:.2f}mm")
            else:
                print("Pontos da íris não disponíveis.")
    else:
        print("Nenhum rosto detectado no frame atual.")

    # Mostrar o frame
    cv2.imshow('Estimativa da Distância entre Pupilas', frame)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
