import cv2
import numpy as np
from pathlib import Path

# from google.colab.patches import cv2_imshow


def crop_image(image, crop_percent=0.27):
    height, width, _ = image.shape
    start_row, start_col = int(height * crop_percent), int(width * crop_percent)
    end_row, end_col = int(height * (1 - crop_percent)), int(width * (1 - crop_percent))
    cropped_image = image[start_row:end_row, start_col:end_col]
    return cropped_image


def increase_contrast(image):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Split the LAB image into its channels
    l, a, b = cv2.split(lab)
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    # Merge the CLAHE enhanced L channel back with a and b channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    # Convert the LAB image back to BGR
    contrast_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    # cv2_imshow(contrast_image)
    return contrast_image


def get_average_color(image):
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    # Compute the average color
    mean_color = np.mean(pixels, axis=0)
    return mean_color


def classify_color(mean_color):

    # Define color values in RGB
    colors = {
        "black": np.array([0, 0, 0]),
        "white": np.array([255, 255, 255]),
        "blue": np.array([0, 0, 255]),
        "red": np.array([255, 0, 0]),
        "green": np.array([0, 255, 0]),
        "yellow": np.array([255, 255, 0]),
    }
    # Find the closest color
    min_dist = float("inf")
    dominant_color = "unknown"
    for color_name, color_value in colors.items():
        dist = np.linalg.norm(mean_color - color_value)
        if dist < min_dist:
            min_dist = dist
            dominant_color = color_name
    return dominant_color
