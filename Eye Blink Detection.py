# Install first if needed:
# pip install nltk

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Example text import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# Function to calculate Eye Aspect Ratio (EAR) using numpy
def eye_aspect_ratio(landmarks, eye_indices):
    pts = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C)
    return ear

# EAR threshold and consecutive frame requirements
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 3

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Video capture
cap = cv2.VideoCapture(0)

blink_counter = 0
total_blinks = 0

# Pandas DataFrame for logging
log_data = pd.DataFrame(columns=["Time", "EAR", "Total_Blinks"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])

            # EAR for both eyes
            leftEAR = eye_aspect_ratio(landmarks, LEFT_EYE)
            rightEAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (leftEAR + rightEAR) / 2.0

            # Blink detection
            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EAR_CONSEC_FRAMES:
                    total_blinks += 1
                    # Log blink event in pandas DataFrame
                    log_data.loc[len(log_data)] = [time.strftime("%H:%M:%S"), round(ear, 3), total_blinks]
                blink_counter = 0

            # Display blink count
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Eye Blink Detection - NumPy & Pandas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save results to CSV
log_data.to_csv("blink_log.csv", index=False)
print("Blink log saved to blink_log.csv")
import cv2

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Start webcam
cap = cv2.VideoCapture(0)

blink_count = 0
eye_open = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            eye_open = True
        else:
            if eye_open:   # transition from open → closed = blink
                blink_count += 1
                eye_open = False

        cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Eye Blink Detection - OpenCV", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

texts = [
    "I love this product! It's amazing ❤️",
    "This is the worst experience ever. Totally disappointed.",
    "The movie was okay, not too good, not too bad."
]

for text in texts:
    score = sia.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Sentiment Score: {score}")
    if score['compound'] >= 0.05:
        print("Overall Sentiment: Positive 😃\n")
    elif score['compound'] <= -0.05:
        print("Overall Sentiment: Negative 😡\n")
    else:
        print("Overall Sentiment: Neutral 😐\n")
