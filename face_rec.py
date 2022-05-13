# Import OpenCV module
# 1) A module that includes many functions to interact with the file system.
import os
# 2) For computer vision applications,focuses on image processing and other features like face detection
import cv2
# 3) Built using the “dlib” library, helps to recognize the face of a person with 99.38% Accuracy.
import face_recognition
import face_recognition as fr
# 4) A highly optimized library for numerical operations with a MATLAB-style syntax
import numpy as np


def get_encoded_faces():
    """
    looks through the faces' folder (our small database) and encodes all
    the files
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    """
    Encodes a face upon given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    Finds all the faces in a given image then label all of them upon recognition
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    img_half = cv2.resize(img, (0, 0), fx=0.7, fy=0.6)

    face_locations = face_recognition.face_locations(img_half)
    unknown_face_encodings = face_recognition.face_encodings(img_half, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "???"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img_half, (left - 40, top - 40), (right + 10, bottom + 30), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img_half, (left - 40, bottom - 10), (right + 10, bottom + 30), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img_half, name, (left - 38, bottom + 20), font, 0.8, (255, 255, 255), 2)

    # Display the resulting image in Open cv output terminal
    while True:
        cv2.namedWindow("Image0")
        cv2.startWindowThread()
        cv2.imshow('Image0', img_half)
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            return face_names


# Get the number of files in our database and display them in a loop
items = os.listdir("./test")
numImg = len(items)

for i in range(0, numImg):
    print(classify_face("./test/" + items[i]))
    cv2.destroyWindow("Image0")
    i = i+1


# for testing a specific image
# print(classify_face("./test/image name.jpg"))
