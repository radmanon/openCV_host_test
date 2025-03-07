import urllib.request
import cv2
import os

MODEL_URL = "https://raw.githubusercontent.com/radmanon/openCV_host_test/refs/heads/main/haarcascades/haarcascade_frontalface_default.xml"
LOCAL_MODEL_PATH = "haarcascade_frontalface_default.xml"

if not os.path.exists(LOCAL_MODEL_PATH):
    print("Downloading face detection model from GitHub...")
    urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL_PATH)
    print("Download complete!")

face_cascade = cv2.CascadeClassifier(LOCAL_MODEL_PATH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detection from GitHub Model", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
