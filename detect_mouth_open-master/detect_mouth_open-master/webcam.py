"""
Detect a face in webcam video and check if mouth is open.
"""
import face_recognition
import cv2
from mouth_open_algorithm import get_lip_height, get_mouth_height
from datetime import datetime
import numpy as np

def is_mouth_open(face_landmarks):
    top_lip = face_landmarks['top_lip']
    bottom_lip = face_landmarks['bottom_lip']

    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)
    
    # if mouth is open more than lip height * ratio, return true.
    ratio = 0.5
    print('top_lip_height: %.2f, bottom_lip_height: %.2f, mouth_height: %.2f, min*ratio: %.2f' 
          % (top_lip_height,bottom_lip_height,mouth_height, min(top_lip_height, bottom_lip_height) * ratio))
          
    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        return True
    else:
        return False


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object to save video to local
fourcc = cv2.VideoWriter_fourcc(*'XVID') # codec
# cv2.VideoWriter( filename, fourcc, fps, frameSize )
out = cv2.VideoWriter('output.avi', fourcc, 7, (640, 480)) 

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg") # replace peter.jpg with your own image !!
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

mohamed_image = face_recognition.load_image_file("mohamed.jpg") # replace peter.jpg with your own image !!
mohamed_face_encoding = face_recognition.face_encodings(mohamed_image)[0]

ahmed_image = face_recognition.load_image_file("ahmed.jpg") # replace peter.jpg with your own image !!
ahmed_face_encoding = face_recognition.face_encodings(ahmed_image)[0]





# while True:
#     # Grab a single frame of video
#     ret, frame = video_capture.read()
#     # Find all the faces and face enqcodings in the frame of video
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#     face_landmarks_list = face_recognition.face_landmarks(frame)

#     # Loop through each face in this frame of video
#     for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
#         #  See if the face is a match for the known face(s)
#         match = face_recognition.compare_faces([mohamed_face_encoding], face_encoding)

#         name = "Unknown"
#         if match[0]:
#             name = "mohamed"
       

#         # Draw a box around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         # Draw a label with a name below the face
#         cv2.rectangle(frame, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom + 25), font, 1.0, (255, 255, 255), 1)


#         # Display text for mouth open / close
#         ret_mouth_open = is_mouth_open(face_landmarks)
#         if ret_mouth_open is True:
#             text = 'talking'
#         else:
#             text = ''
#         cv2.putText(frame, text, (left, top - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


#     # Display the resulting image
#     cv2.imshow('Video', frame)
#     out.write(frame)

#     # Hit 'q' on the keyboard to quit!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()









# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    mohamed_face_encoding,
    ahmed_face_encoding
]
known_face_names = [
    "mohamed",
    "mohamed",
    "ahmed"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(frame)

        face_names = []
        for(top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        # Display text for mouth open / close
        ret_mouth_open = is_mouth_open(face_landmarks)
        if ret_mouth_open is True:
            text = 'talking'
        else:
            text = ''
        cv2.putText(frame, text, (left, top - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()