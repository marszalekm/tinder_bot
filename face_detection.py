#!/usr/bin/env python3
from PIL import Image
import face_recognition
import os
import torch


def convert_to_face(input, output):
    """
    Finds a face in the image and saves it to a file.
    """
    face = face_recognition.load_image_file(input)
    face_locations = face_recognition.face_locations(face, number_of_times_to_upsample=0, model="cnn")

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = face[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(output)


input = "./raw_images/female/"
output = "./training_images/female/"

for filename in os.listdir(input):
    print("Finding face for: ", filename)
    file_output = filename.split('.')[0] + '-face.' + filename.split('.')[1]
    convert_to_face(input + filename, output + file_output)
    torch.cuda.empty_cache()

