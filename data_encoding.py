# -*- coding: utf-8 -*-
import os
import face_recognition
import cv2
import numpy as np
import pickle
from PIL import Image
from imutils import paths
from recognize_face import recognize_face_from_image_cv, save_image_cv

'''serialize faces
'''

def load_images_cv(dataset, resize=True):
    '''load image from directory, extract faces

    Args:
        dataset: direction with images in
    Returns:
        i: index of image file
        k: index of face in image
        dir_name: directory name where image in
        file_name: image file name
        face(Numpy): face data(cv)
    '''
    imagePaths = list(paths.list_images(dataset))
    for i, imagePath in enumerate(imagePaths):
        imagepaths = imagePath.split(os.path.sep)
        dir_name = imagepaths[-2]
        file_name = imagepaths[-1].split(".")[0]
        print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
        for k, face in enumerate(recognize_face_from_image_cv(imagePath, resize)):
            yield i, k, dir_name, file_name, face

def serialize_faces(dataset, encoding_output):
    '''serilize image in directory, save the serialize output to local

    Args:
        dataset: direction with images in
        encoding_output: file path to save serialized data
    '''
    knownEncodings = []
    knownNames = []

    for encoding, name in encode_images(dataset):
        knownEncodings.append(encoding)
        knownNames.append(name)

    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open(encoding_output, "wb+") as f:
        f.write(pickle.dumps(data))

def unserialize_faces(data_path):
    '''unserialize face data
    '''
    with open(data_path, "rb") as f:
        data = pickle.loads(f.read())
    return data

def encode_images(dataset, resize=True):
    '''encode face
    '''
    for _, _, dir_name, _, face in load_images_cv(dataset, resize):
        encoding = face_recognition.face_encodings(face)[0]
        yield encoding, dir_name

def save_face_cv(dataset, save_dir, resize=True):
    '''save face images
    '''
    for i, k, _, file_name, face in load_images_cv(dataset, resize):
        save_image_cv(face, "{}_{}_{}.jpg".format(save_dir, i, k))

if __name__ == "__main__":
    serialize_faces("dataset", "encodings.pickle")
    print(unserialize_faces("encodings.pickle"))
    