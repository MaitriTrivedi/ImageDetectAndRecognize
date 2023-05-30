import face_recognition
import cv2
from PIL import Image, ImageDraw

# Open video file
video_capture = cv2.VideoCapture("video/testing.mp4")

frames = []
frame_count = 0
i = 0

while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Bail out when the video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    frame = frame[:, :, ::-1]

    # Save each frame of the video to a list
    frame_count += 1
    frames.append(frame)

    # Every 128 frames (or when the video ends), batch process the list of frames to find faces
    if len(frames) == 128 or not ret:
        batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)

        # Now let's process the list of frames
        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame = len(face_locations)

            frame_number = frame_count - len(frames) + frame_number_in_batch
            print("I found {} face(s) in frame #{}.".format(number_of_faces_in_frame, frame_number))

            for face_location in face_locations:
                # Print the location of each face in this frame
                top, right, bottom, left = face_location
                print(" - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                face_image = frame[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                d = ImageDraw.Draw(pil_image)
                
                # Find facial features in the face image
                face_landmarks_list = face_recognition.face_landmarks(face_image)

                # Draw facial features on the face image
                for face_landmarks in face_landmarks_list:
                    for facial_feature in face_landmarks.keys():
                        d.line(face_landmarks[facial_feature], width=5)
                
                    # Save the face image with facial landmarks
                    pil_image.save('/home/maitri/FaceDetectRecognize/video/frames/{}_landmarks.jpeg'.format(i), 'JPEG')
                    i += 1

        # Clear the frames array to start the next batch
        frames = []

video_capture.release()
