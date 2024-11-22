import os
import face_recognition
import cv2
import numpy as np
from pathlib import Path

def calculate_brightness(image_path):
  image = cv2.imread(image_path)
  gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  mean_brightness = np.mean(gray_image)
  return mean_brightness

def find_best_matching(image_name,image_cont_name,folder_cont_name,folder_name):
  image_path = 'https://console.cloud.google.com/'+image_cont_name+'/'+image_name
  folder_path = 'https://console.cloud.google.com/'+folder_cont_name+'/'+folder_name
#   image_path= image_name
#   folder_path = folder_cont_name
  if calculate_brightness(image_path)<60:
    return "face image is not enough bright as needed",1,1
  known_image=face_recognition.load_image_file(image_path)
  if not face_recognition.face_encodings(known_image):
      return "face is not present.",2,1
  else:
    known_image_encoding = face_recognition.face_encodings(known_image)[0]
    # print(known_image_encoding)
  # Variables to keep track of the most similar face match we've found
    best_face_distance = 0.56
    best_face_image_name = None
    name_show = "no matching"
    best_distance=100

    # Loop over all the images we want to check for similar people
    for image_path in Path(folder_path).glob("*.jpg"):
        # Load an image to check
        unknown_image = face_recognition.load_image_file(image_path)

        # Get the location of faces and face encodings for the current image
        face_encodings = face_recognition.face_encodings(unknown_image)

        # Get the face distance between the known person and all the faces in this image
        if not face_recognition.face_distance(face_encodings, known_image_encoding):
            continue
        face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)[0]
        #print(face_distance)
        #print(f"image path = {image_path}")

        # If this face is more similar to our known image than we've seen so far, save it
        if face_distance < best_face_distance:
            # Save the new best face distance
            best_face_distance = face_distance
            best_distance = face_distance
            best_face_image_name = image_path.stem
    if best_distance==100:
      return "no matching found",3,1
    else:
      return "matching found",best_face_image_name,folder_cont_name
if __name__ == "__main__":
    image_name = "test.jpg"  # Example image name
    image_cont_name = "/mnt/data"  # Example image container name
    folder_cont_name ='attendence_system' # Example folder container name
    folder_name = "images_folder"  # Example folder name where the images are stored

    result = find_best_matching(image_name, image_cont_name, folder_cont_name, folder_name)
    print(result)
