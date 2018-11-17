# -*- coding: utf-8 -*-
import moviepy.editor as mpy
from PIL import Image

def split_video2flames(file_path="output.avi"):
    vid = mpy.VideoFileClip(file_path)
    for frame in vid.iter_frames(dtype="uint8"):
        yield frame
    
def save_frame(frame, output):
    image = Image.fromarray(frame)
    image.save(output)

def save_video2images(video_file, output):
    for i, frame in enumerate(split_video2flames("video/video_2.mov")):
        save_frame(frame, "{}_{:02d}.jpg".format(output,i))

def compress_video(file_path):
    pass