import cv2
import face_recognition
import numpy as np
import os
import csv
import pickle
from datetime import datetime
from flask import Flask, render_template, Response

app = Flask(__name__)

# File paths
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

# Load known faces and encodings from disk
def load_known_faces():
    if os.path.exists(ENCODINGS_FILE):  # Load precomputed encodings
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)

    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)

        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)

                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)

                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person_name)

    # Save encodings for faster future use
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()

# Check if attendance is already marked for today
def is_attendance_marked(name):
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(ATTENDANCE_FILE):
        return False

    with open(ATTENDANCE_FILE, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == name and row[1].startswith(today):
                return True
    return False

# Mark attendance in a CSV file
def mark_attendance(name):
    with open(ATTENDANCE_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# Video Stream Generator (Optimized)
def video_stream():
    video_capture = cv2.VideoCapture(0)

    # Reduce webcam frame size for faster processing
    video_capture.set(3, 640)  # Width
    video_capture.set(4, 480)  # Height

    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.5:
                    name = known_face_names[best_match_index]

                    if not is_attendance_marked(name):
                        mark_attendance(name)
                        print(f"✅ Attendance marked for {name}.")
                        cv2.putText(frame, "Attendance Taken!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Stop webcam after attendance is taken
                        video_capture.release()
                        return

            # Draw a rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
