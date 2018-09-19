# -*- coding: utf-8 -*-
import numpy as np
import cv2
import moviepy.editor as mpy
import face_recognition
from PIL import Image, ImageDraw
from numpy import array


def capture_video(output_file="output.avi"):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, -1, 20.0, (640,480))

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def compress_video(file_path):
    pass

def split_video2flames(file_path="output.avi"):
    vid = mpy.VideoFileClip(file_path)
    return [frame for frame in vid.iter_frames(dtype="uint8")]
    
def save_frames(frames):
    for i in range(len(frames)):
        path = f"images/{i}.jpg"
        mark_face(array(frames[i]), path)

def locate_face(image, output_file):
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        for t,r,b,l in face_locations:
            draw.rectangle((l,t,r,b), outline="red")
        img.save(output_file)

def gather_face_data(image):
    face_landmarks_list = face_recognition.face_landmarks(image)
    # normalize data
    return face_landmarks_list



if __name__ == "__main__":
    capture_video()
    frames = split_video2flames()
    save_frames(frames)

