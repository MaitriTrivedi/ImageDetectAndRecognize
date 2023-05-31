import face_recognition
import os
import cv2

import face_recognition
known_image = face_recognition.load_image_file("known_faces/elon_musk.jpeg")
unknown_image = face_recognition.load_image_file("unknown_faces/7.jpeg")

elonMusk_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([elonMusk_encoding], unknown_encoding)

print(results)