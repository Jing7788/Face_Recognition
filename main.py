import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

wong_image = face_recognition.load_image_file("photos/wong.jpg")
wong_encoding = face_recognition.face_encodings(wong_image)[0]

daniel_image = face_recognition.load_image_file("photos/daniel.jpg")
daniel_encoding = face_recognition.face_encodings(daniel_image)[0]

ong_image = face_recognition.load_image_file("photos/ong.jpg")
ong_encoding = face_recognition.face_encodings(ong_image)[0]

known_face_encoding = [wong_encoding, daniel_encoding, ong_encoding]
known_face_names = ["wong", "daniel", "ong"]

attendance = known_face_names.copy()

face_locations = []
face_encodings = []
faces_names = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

try:
    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        faces_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            faces_names.append(name)

            if name in known_face_names:
                if name in attendance:
                    attendance.remove(name)
                    print(attendance)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

        cv2.imshow("attendance system", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_capture.release()
    cv2.destroyAllWindows()
    f.close()
