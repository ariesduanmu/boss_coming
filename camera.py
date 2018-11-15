# -*- coding: utf-8 -*-
import os
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

def save_face_from_video():
    print("STRAT")
    cap = cv2.VideoCapture(0)
    k = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            if k % 3 == 0:

                face_locations = face_recognition.face_locations(rgb_small_frame)
                if len(face_locations) > 0:
                    print(f"[*] {k} | with face: {len(face_locations)}")
                for top, right, bottom, left in face_locations:
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cropped = frame[left:right, top:bottom]
                    # cv2.imshow("cropped", cropped)
                    cv2.imwrite("images/image_{}.jpg".format(k), cropped)
            k += 1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def split_video2flames(file_path="output.avi"):
    vid = mpy.VideoFileClip(file_path)
    for frame in vid.iter_frames(dtype="uint8"):
        yield frame
    
def save_frame(frame, output):
    image = Image.fromarray(frame)
    image.save(output)
        
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
    # dst = "my_images"
    # i = 0
    # for frame in split_video2flames("video/video_2.mov"):
    #     save_frame(frame, os.path.join(dst, "me_{:02d}.jpg".format(i)))
    #     i += 1

    # save_face_from_video()

    



