# -*- coding: utf-8 -*-
import os
import face_recognition
from PIL import Image, ImageDraw

def draw_face_location(image_file, output_file="location.png"):
    image = face_recognition.load_image_file(image_file)
    face_locations = face_recognition.face_locations(image)
    original_image = Image.open(image_file).convert("RGBA")
    draw = ImageDraw.Draw(original_image)
    for t,r,b,l in face_locations:
        draw.rectangle((l,t,r,b), outline="red")
    original_image.save(output_file, "PNG")

def draw_face_landmarks(image_file, output_file="landmark.png"):
    image = face_recognition.load_image_file(image_file)
    face_landmarks_list = face_recognition.face_landmarks(image)
    original_image = Image.open(image_file).convert("RGBA")
    draw = ImageDraw.Draw(original_image)
    for face_landmarks in face_landmarks_list:
        # Make the eyebrows into a nightmare
        draw.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        draw.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        draw.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        draw.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

        # Gloss the lips
        draw.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        draw.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        draw.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        draw.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # Sparkle the eyes
        draw.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        draw.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # Apply some eyeliner
        draw.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        draw.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
    original_image.save(output_file, "PNG")

