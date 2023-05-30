import face_recognition
import os
import cv2

img = cv2.imread("known_faces/em2.jpeg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("known_faces/elon_musk.jpeg")
rgb_img = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
img_encoding2 = face_recognition.face_encodings(rgb_img)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print(result)

cv2.imshow("Img",img)
cv2.imshow("Img2",img2)
cv2.waitKey(0)


# known_image = face_recognition.load_image_file("known_faces/elon_musk.jpeg")
# unknown_image = face_recognition.load_image_file("known_faces/em2.jpeg")

# elon_encoding = face_recognition.face_encodings(known_image)[0]
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# result = face_recognition.compare_faces([elon_encoding], unknown_encoding)
# print(result)

# cv2.imshow("KnownImg",known_image)
# cv2.imshow("UnknownImg2",unknown_image)
# cv2.imshow("Img",elon_encoding)
# cv2.imshow("Img2",unknown_encoding)
# cv2.waitKey(0)
