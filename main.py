import cv2
import mediapipe as mp
import math
import time
import winsound

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

def calculate_eye_height(landmarks, top_idx, bottom_idx, img_w, img_h):
    top = landmarks.landmark[top_idx]
    bottom = landmarks.landmark[bottom_idx]
    x1, y1 = int(top.x * img_w), int(top.y * img_h)
    x2, y2 = int(bottom.x * img_w), int(bottom.y * img_h)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def average_eye_height(landmarks, w, h):
    rh = calculate_eye_height(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, w, h)
    lh = calculate_eye_height(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, w, h)
    return (rh + lh) / 2

def calibrate(cap, message, duration=3):
    winsound.Beep(1000, 300)
    heights = []
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                height = average_eye_height(face_landmarks, w, h)
                heights.append(height)
        cv2.putText(frame, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Calibrating", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    winsound.Beep(800, 300)
    return sum(heights) / len(heights) if heights else 0

cap = cv2.VideoCapture(0)
closed_eye_height = calibrate(cap, "Now close your eyes")
EYE_CLOSED_THRESHOLD = closed_eye_height + 2

eye_closed_start = None
eye_open_start = time.time()
eye_closed_duration = 0
eye_open_duration = 0
eye_closed = False
alarm_playing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            avg_eye_height = average_eye_height(face_landmarks, w, h)
            current_time = time.time()
            if avg_eye_height < EYE_CLOSED_THRESHOLD:
                if not eye_closed:
                    eye_closed_start = current_time
                    eye_open_duration = 0
                    eye_closed = True
                eye_closed_duration = current_time - eye_closed_start
                cv2.putText(frame, "ALERT: EYES CLOSED!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                if eye_closed_duration > 3 and not alarm_playing:
                    winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
                    alarm_playing = True
            else:
                if eye_closed:
                    eye_open_start = current_time
                    eye_closed_duration = 0
                    eye_closed = False
                eye_open_duration = current_time - eye_open_start
                if eye_open_duration > 5 and alarm_playing:
                    winsound.PlaySound(None, winsound.SND_ASYNC)
                    alarm_playing = False
            cv2.putText(frame, f"Eye height: {avg_eye_height:.2f} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Eyes Open: {eye_open_duration:.2f} s", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Eyes Closed: {eye_closed_duration:.2f} s", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, 'Press "q" to quit', (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
