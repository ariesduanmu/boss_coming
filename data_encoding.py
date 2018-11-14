# -*- coding: utf-8 -*-
import os
import face_recognition
import cv2
from PIL import Image
from imutils import paths

IMAGE_SIZE = 64

def face_locations(imagePath):
    image = face_recognition.load_image_file(imagePath)
    face_locations = face_recognition.face_locations(image)
    return [(l,t,r,b) for t,r,b,l in face_locations]

def resize_width_padding(image, width=IMAGE_SIZE):
    w, h = image.size
    s = max(w, h)
    resized_image = Image.new("RGB", (s, s))
    resized_image.paste(image, ((s-w)//2, (s-h)//2))
    resized_image.thumbnail((width,width))

    return resized_image

def save_face(dataset, save_dir):
    imagePaths = list(paths.list_images(dataset))
    for i, imagePath in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-1].split(".")[0]
        imagePath = os.path.abspath(imagePath)
        faces = face_locations(imagePath)
        image = Image.open(imagePath)
        for k, face in enumerate(faces):
            img = image.crop(face)
            img = resize_width_padding(img)
            img.save(os.path.join(os.path.abspath(save_dir), "{}_{}.jpg".format(name, k+1)))

if __name__ == "__main__":
    save_face("images","emma")