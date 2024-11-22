import os
import face_recognition
import cv2
import numpy as np
from pathlib import Path
from io import BytesIO
from google.cloud import storage # type: ignore


def calculate_brightness(bucket_name, image_blob_name):
    
  # Initialize the GCS client
  client = storage.Client()

  # Get the bucket and blob
  bucket = client.get_bucket(bucket_name)
  blob = bucket.blob(image_blob_name)

  # Download image data
  image_data = BytesIO(blob.download_as_bytes())

  if image_data is None:
      return None, None

  # Convert image data to a numpy array
  image_array = np.frombuffer(image_data.getvalue(), dtype=np.uint8)

  image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

  if image is None:
      print("Error: Unable to decode the image")
      return None, None

  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
  mean_brightness = np.mean(gray_image)

  return mean_brightness, np.array(image_rgb,dtype=np.uint8)

def find_best_matching(image_name,image_cont_name,folder_cont_name,folder_name):
  image_path = 'https://console.cloud.google.com/'+image_cont_name+'/'+image_name
  folder_path = 'https://console.cloud.google.com/'+folder_cont_name+'/'+folder_name
#   image_path= image_name
#   folder_path = folder_cont_name
  brightness, known_image = calculate_brightness(image_cont_name, image_name)
  if brightness<60:
    return "face image is not enough bright as needed",1,1
  # known_image=face_recognition.load_image_file(known_image)
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
    files = list_files_in_folder( folder_cont_name, folder_name)
    print(files)

    # Loop over all the images we want to check for similar people
    for image_path in files[1:]:
        # Load an image to check
        brightness_, unknown_image = calculate_brightness(folder_cont_name , image_path)
        # unknown_image = face_recognition.load_image_file(image_path)

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
            best_face_image_name = image_path
    if best_distance==100:
      return "no matching found",3,1
    else:
      return "matching found",best_face_image_name,folder_cont_name

def list_files_in_folder(bucket_name, folder_name):
    # Initialize the GCS client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # List all files with the prefix (folder name)
    blobs = bucket.list_blobs(prefix=folder_name)

    # Iterate over the blobs and print their names
    file_names = []
    for blob in blobs:
        file_names.append(blob.name)
        print(f"File: {blob.name}")

    return file_names