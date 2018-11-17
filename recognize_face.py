# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import face_recognition

'''recognize face in images and save to file
'''

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

def save_image_cv(frame, output_path):
    cv2.imwrite(output_path, frame)

def save_image_pil(frame, output_path):
    frame.save(output_path)

def resize_image_cv(frame, width=IMAGE_WIDTH):
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

def resize_image_pil(frame, width=IMAGE_WIDTH):
    w, h = frame.size
    s = max(w, h)
    resized_image = Image.new("RGB", (s, s))
    resized_image.paste(frame, ((s-w)//2, (s-h)//2))
    resized_image.thumbnail((width,width))

    return resized_image

def recognize_face(frame, ratio=1):
    '''
    Args:
        frame(Numpy): 图片的数据信息，三维矩阵(w, h, 3)
        ratio: 缩放比例
    '''
    face_locations = face_recognition.face_locations(frame)
    for top, right, bottom, left in face_locations:
        yield (top*ratio, right*ratio, bottom*ratio, left*ratio)

def draw_face_location_cv2(frame, location, name):
    '''使用cv绘制识别到的人脸名称
    '''
    top, right, bottom, left = location
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

def draw_face_location_pil(frame, location, name):
    '''使用pil绘制识别到的人脸名称(未完成)
    '''
    top, right, bottom, left = location
    draw.rectangle((left, top, right, bottom), outline="red")
    #TODO: draw name on it

def recognize_face_from_cv(frame, resize=True):
    '''获取脸部数据，图片压缩后进行脸部识别，或对识别后的脸部进行压缩
    Args:
        frame: 图片数据，读取方式cv
        resize: 是否进行改变大小(64, 64)
    Returns:
        cropped: 脸部数据(cv)
    '''
    h, w, k = frame.shape
    ratio = _resize_ratio(w, h)
    small_frame = cv2.resize(frame, (0, 0), fx=1/ratio, fy=1/ratio)
    small_frame = small_frame[:, :, ::-1]
    for top, right, bottom, left in recognize_face(small_frame, ratio):
        cropped = frame[top:bottom, left:right]
        if resize:
            cropped = resize_image_cv(cropped)
        yield cropped

def recognize_face_from_pil(frame, resize=True):
    '''获取脸部数据，图片压缩后进行脸部识别，或对识别后的脸部进行压缩
    Args:
        frame: 图片数据，读取方式pil
        resize: 是否进行改变大小(64, 64)
    Returns:
        cropped: 脸部数据(pil)
    '''
    w, h = frame.size
    ratio = _resize_ratio(w, h)
    small_frame = frame.resize((w//ratio, h//ratio))
    small_frame = np.array(small_frame)
    for top, right, bottom, left in recognize_face(small_frame, ratio):
        cropped = frame.crop((left,top,right,bottom))
        if resize:
            cropped = resize_image_pil(cropped)
        yield cropped

def recognize_face_from_image_cv(image_path, resize=True):
    '''从图片路径获取脸部信息(cv)
    '''
    frame = cv2.imread(image_path)
    for cropped in recognize_face_from_cv(frame, resize):
        yield cropped
        

def recognize_face_from_image_pil(image_path, resize=True):
    '''从图片路径获取脸部信息(pil)
    '''
    frame = Image.open(image_path)
    for cropped in recognize_face_from_pil(frame, resize):
        yield cropped
