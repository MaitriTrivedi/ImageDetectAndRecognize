import face_recognition
import cv2

video_capture = cv2.VideoCapture("video/elon_musk_2.mp4")

img = cv2.imread("known_faces/elon_musk.jpeg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

if video_capture.isOpened()==False :
    print("Error")
while video_capture.isOpened() :
    # ret = A boolean value indicating whether a frame was successfully read or not.
    # frame =  The actual frame data that was read from the video source.
    ret, frame = video_capture.read()

    if ret==True :
        # Display the resulting frame
        cv2.imshow("Frame", frame)
        cv2.imwrite("current_frame.jpeg", frame)

        img2 = cv2.imread("current_frame.jpeg")
        rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        try :
            img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]
        except  IndexError:
            continue
        # print(img_encoding2)
        result = face_recognition.compare_faces([img_encoding], img_encoding2)
        # print(result)
        if result[0] == True:
            gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            face_classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            
            face = face_classifier.detectMultiScale( 
                gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            for (x, y, w, h) in face:
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 4)

            img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)    
            cv2.imshow("Video",img2)
     

          
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else :
        break

print("Finished")
