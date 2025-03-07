import urllib.request
import cv2
import os

# Replace this with your actual GitHub raw file URL
MODEL_URL = "https://raw.githubusercontent.com/radmanon/openCV_host_test/refs/heads/main/haarcascades/haarcascade_frontalface_default.xml"
LOCAL_MODEL_PATH = "haarcascade_frontalface_default.xml"

# Check if the model exists locally; if not, download it from GitHub
if not os.path.exists(LOCAL_MODEL_PATH):
    print("Downloading face detection model from GitHub...")
    urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL_PATH)
    print("Download complete!")

# Load the face detection model
face_cascade = cv2.CascadeClassifier(LOCAL_MODEL_PATH)

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the output
    cv2.imshow("Face Detection from GitHub Model", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
