# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import face_recognition

MAX_PIXEL = 256
IMAGE_WIDTH = 64

def _resize_ratio(w, h):
    if max(w, h) <= MAX_PIXEL:
        return 1
    else:
        if w >= h:
            return w // MAX_PIXEL
        else:
            return h // MAX_PIXEL

def _resize_image_cv(frame, width=IMAGE_WIDTH):
    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(frame)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (width, width))

    return resized_image

def _resize_image_pil(frame, width=IMAGE_WIDTH):
    w, h = frame.size
    s = max(w, h)
    resized_image = Image.new("RGB", (s, s))
    resized_image.paste(frame, ((s-w)//2, (s-h)//2))
    resized_image.thumbnail((width,width))

    return resized_image

def _recognize_face(frame, ratio):
    face_locations = face_recognition.face_locations(frame)
    for top, right, bottom, left in face_locations:
        yield (top*ratio, right*ratio, bottom*ratio, left*ratio)

def recognize_face_from_cv(frame, resize=False):
    h, w, k = frame.shape
    ratio = _resize_ratio(w, h)
    small_frame = cv2.resize(frame, (0, 0), fx=1/ratio, fy=1/ratio)
    small_frame = small_frame[:, :, ::-1]
    for top, right, bottom, left in _recognize_face(small_frame, ratio):
        cropped = frame[top:bottom, left:right]
        if resize:
            cropped = _resize_image_cv(cropped)
        yield cropped

def recognize_face_from_pil(frame, resize=False):
    w, h = frame.size
    ratio = _resize_ratio(w, h)
    small_frame = frame.resize((w//ratio, h//ratio))
    small_frame = np.array(small_frame)
    for top, right, bottom, left in _recognize_face(small_frame, ratio):
        cropped = frame.crop((left,top,right,bottom))
        if resize:
            cropped = _resize_image_pil(cropped)
        yield cropped

def recognize_face_from_image_cv(image_path, output_path, resize=False):
    frame = cv2.imread(image_path)
    for cropped in recognize_face_from_cv(frame, resize):
        cv2.imwrite(output_path, cropped)

def recognize_face_from_image_pil(image_path, output_path, resize=False):
    frame = Image.open(image_path)
    for cropped in recognize_face_from_pil(frame, resize):
        cropped.save(output_path)

if __name__ == "__main__":
    recognize_face_from_image_cv("lmq.jpg", "images/image_1.jpg", True)
    recognize_face_from_image_pil("lmq.jpg", "images/image_2.jpg", True)


