import face_recognition
import os
import cv2
# import matplotlib.pyplot as plt

# Read Image :
img = cv2.imread("known_faces/em2.jpeg")

# print dimension of image array :
print(img.shape)

# convert image to gray image :
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print dimension of grayimage array :
print(gray_image.shape)

# Haar Cascade classifier : is a pre trained classifier and built into openCV
# load Haar Cascade classifier :
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# face detecting..
# detectMultiScale() : to identify faces of different sizes in the input image
# gray_image : source of gray image
# scaleFactor : used to scale down the size of the input image to make it easier for the algorithm to detect larger faces
# minSize : sets the minimum size of the object to be detected
face = face_classifier.detectMultiScale( 
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# Drawing a Bounding Box
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Displaying the Image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    

cv2.imshow("Img",img)
cv2.imshow("Img2",gray_image)
cv2.waitKey(0)


# # using matplotlib
# plt.figure(figsize=(20,10))
# plt.imshow(img_rgb)
# plt.axis('off')