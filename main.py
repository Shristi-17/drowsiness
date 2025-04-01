import cv2
import dlib
import numpy as np
import pygame
from tensorflow.keras.models import load_model
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load pre-trained drowsiness detection model
model = load_model("drowsiness_model.h5")

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize beep sound
pygame.mixer.init()
beep = pygame.mixer.Sound("beep.wav")

# Define EAR (Eye Aspect Ratio) function
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Live video stream
cap = cv2.VideoCapture(0)
drowsy_frames = 0
THRESHOLD_EAR = 0.25
CONSEC_FRAMES = 20  # Number of frames for drowsiness detection

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Draw a square around the detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect facial landmarks
        landmarks = predictor(gray, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        if avg_EAR < THRESHOLD_EAR:
            drowsy_frames += 1
            if drowsy_frames >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSY!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                beep.play()  # Play beep sound
        else:
            drowsy_frames = 0
            cv2.putText(frame, "AWAKE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            beep.stop()  # Stop beep sound when awake

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
